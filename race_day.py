"""Race-day lap log parsing, validation, forecasting, and Google Sheets sync."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from simulation_engine import (
    FINAL_CUTOFF_MINUTE,
    LAST_START_MINUTE,
    SimulationSettings,
    simulate_many_from_state,
)


RACE_LOG_COLUMNS = [
    "lap_number",
    "runner",
    "start_minute",
    "finish_minute",
    "lap_duration_minutes",
    "notes",
    "official",
]


@dataclass(frozen=True)
class RaceState:
    current_laps: int
    elapsed_minute: float
    laps_by_noon: int
    next_runner: str | None
    next_runner_index: int
    runner_lap_counts: dict[str, int]
    latest_lap: dict[str, Any] | None
    remaining_to_target: int
    average_needed_to_target: float | None
    average_needed_to_start_target_lap: float | None


def empty_race_log() -> pd.DataFrame:
    return pd.DataFrame(columns=RACE_LOG_COLUMNS)


def normalise_race_log(raw_log: pd.DataFrame | None, roster: pd.DataFrame) -> pd.DataFrame:
    if raw_log is None or raw_log.empty:
        return empty_race_log()

    source = raw_log.copy()
    source.columns = [_normalise_column_name(column) for column in source.columns]

    rows: list[dict[str, Any]] = []
    previous_finish = 0.0
    for index, row in source.reset_index(drop=True).iterrows():
        lap_number = _number_or_default(row.get("lap_number"), index + 1)
        runner = str(row.get("runner", "")).strip()
        notes = str(row.get("notes", "")).strip() if pd.notna(row.get("notes", "")) else ""

        start_minute = _first_parsed_time(row, ["start_minute", "start_time", "start"])
        finish_minute = _first_parsed_time(row, ["finish_minute", "finish_time", "finish"])
        duration = _first_number(row, ["lap_duration_minutes", "duration_minutes", "lap_duration", "duration"])

        if start_minute is None and finish_minute is None and duration is not None:
            start_minute = previous_finish
            finish_minute = start_minute + duration
        elif start_minute is None and finish_minute is not None and duration is not None:
            start_minute = finish_minute - duration
        elif start_minute is None:
            start_minute = previous_finish
        elif finish_minute is None and duration is not None:
            finish_minute = start_minute + duration

        if duration is None and start_minute is not None and finish_minute is not None:
            duration = finish_minute - start_minute

        official = (
            start_minute is not None
            and finish_minute is not None
            and start_minute <= LAST_START_MINUTE
            and finish_minute <= FINAL_CUTOFF_MINUTE
        )

        rows.append(
            {
                "lap_number": int(lap_number),
                "runner": runner,
                "start_minute": np.nan if start_minute is None else float(start_minute),
                "finish_minute": np.nan if finish_minute is None else float(finish_minute),
                "lap_duration_minutes": np.nan if duration is None else float(duration),
                "notes": notes,
                "official": bool(official),
            }
        )
        if finish_minute is not None:
            previous_finish = float(finish_minute)

    log = pd.DataFrame(rows, columns=RACE_LOG_COLUMNS)
    log = log.sort_values("lap_number", kind="stable").reset_index(drop=True)
    return log


def parse_pasted_laps(text: str, roster: pd.DataFrame) -> pd.DataFrame:
    if not text.strip():
        return empty_race_log()

    try:
        pasted = pd.read_csv(StringIO(text), sep=None, engine="python")
    except Exception:
        pasted = pd.read_csv(StringIO(text), sep="\t", header=None)

    if not set(_normalise_column_name(column) for column in pasted.columns).intersection({"runner", "finish_time", "finish_minute", "lap_duration_minutes", "duration"}):
        pasted = pasted.iloc[:, :4].copy()
        pasted.columns = ["runner", "finish_time", "lap_duration_minutes", "notes"][: len(pasted.columns)]

    return normalise_race_log(pasted, roster)


def append_manual_lap(
    current_log: pd.DataFrame,
    roster: pd.DataFrame,
    runner: str,
    finish_minute: float | None = None,
    duration_minutes: float | None = None,
    notes: str = "",
) -> pd.DataFrame:
    log = normalise_race_log(current_log, roster)
    start_minute = float(log["finish_minute"].dropna().max()) if not log["finish_minute"].dropna().empty else 0.0
    if finish_minute is None and duration_minutes is not None:
        finish_minute = start_minute + float(duration_minutes)
    if finish_minute is None:
        finish_minute = start_minute
    duration = float(finish_minute) - start_minute
    next_lap = int(log["lap_number"].max()) + 1 if not log.empty else 1
    added = pd.DataFrame(
        [
            {
                "lap_number": next_lap,
                "runner": runner,
                "start_minute": start_minute,
                "finish_minute": float(finish_minute),
                "lap_duration_minutes": duration,
                "notes": notes,
            }
        ]
    )
    if log.empty:
        return normalise_race_log(added, roster)
    return normalise_race_log(pd.concat([log, added], ignore_index=True), roster)


def validate_race_log(log: pd.DataFrame, roster: pd.DataFrame) -> tuple[list[str], list[str]]:
    clean = normalise_race_log(log, roster)
    warnings: list[str] = []
    errors: list[str] = []
    if clean.empty:
        return warnings, errors

    known_runners = set(roster["runner"].astype(str).str.strip())
    unknown = sorted(set(clean["runner"].astype(str).str.strip()).difference(known_runners).difference({""}))
    if unknown:
        errors.append(f"Unknown runner names in race log: {', '.join(unknown)}.")

    if clean["runner"].astype(str).str.strip().eq("").any():
        errors.append("Every race-log row needs a runner.")
    if clean["lap_number"].duplicated().any():
        errors.append("Duplicate lap numbers found. Each lap number must be unique.")
    if clean.duplicated(["runner", "finish_minute"]).any():
        errors.append("Duplicate runner/finish-time entries found.")
    if clean[["start_minute", "finish_minute", "lap_duration_minutes"]].isna().any(axis=None):
        errors.append("Every race-log row needs enough timing data to calculate start, finish, and duration.")

    impossible = clean[
        clean["start_minute"].lt(0)
        | clean["finish_minute"].lt(0)
        | clean["finish_minute"].le(clean["start_minute"])
        | clean["lap_duration_minutes"].le(0)
    ]
    if not impossible.empty:
        errors.append("One or more laps have impossible timing: finish must be after start and duration must be positive.")

    if clean["finish_minute"].gt(FINAL_CUTOFF_MINUTE).any():
        warnings.append("At least one logged finish is after Sunday 13:00, so it will not count as an official lap.")
    if clean["start_minute"].gt(LAST_START_MINUTE).any():
        warnings.append("At least one logged start is after Sunday 12:00, so it will not count as an official lap.")

    ordered = clean.sort_values("lap_number")
    if ordered["finish_minute"].dropna().diff().le(0).any():
        errors.append("Logged finish times must increase lap by lap.")
    if ordered["start_minute"].dropna().diff().lt(0).any():
        errors.append("Logged start times must not move backwards.")

    caps = roster.copy()
    caps["max_laps"] = pd.to_numeric(caps["max_laps"], errors="coerce")
    official_counts = clean[clean["official"].fillna(False).astype(bool)]["runner"].value_counts()
    cap_messages = []
    for row in caps.dropna(subset=["max_laps"]).itertuples(index=False):
        max_laps = int(row.max_laps)
        if max_laps > 0 and official_counts.get(str(row.runner), 0) > max_laps:
            cap_messages.append(f"{row.runner} has {official_counts.get(str(row.runner), 0)} logged laps against cap {max_laps}")
    if cap_messages:
        warnings.append("Cap warning: " + "; ".join(cap_messages) + ".")

    return warnings, errors


def race_state_from_log(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    target_laps: int,
    caps_enabled: bool = True,
) -> RaceState:
    clean = normalise_race_log(log, roster)
    official = clean[clean["official"].fillna(False).astype(bool)].copy()
    elapsed_minute = float(clean["finish_minute"].dropna().max()) if not clean["finish_minute"].dropna().empty else 0.0
    current_laps = int(len(official))
    laps_by_noon = int(official["finish_minute"].le(LAST_START_MINUTE).sum()) if not official.empty else 0

    order = _available_order(roster)
    counts = {runner: 0 for runner in order}
    for runner, count in official["runner"].value_counts().items():
        if str(runner) in counts:
            counts[str(runner)] = int(count)

    if not clean.empty and str(clean.iloc[-1]["runner"]) in order:
        start_index = (order.index(str(clean.iloc[-1]["runner"])) + 1) % len(order)
    else:
        start_index = current_laps % len(order) if order else 0
    next_runner, next_index = next_eligible_runner(roster, counts, start_index, caps_enabled=caps_enabled)
    latest_lap = clean.iloc[-1].to_dict() if not clean.empty else None

    remaining_to_target = max(0, int(target_laps) - current_laps)
    time_remaining_to_cutoff = max(0.0, FINAL_CUTOFF_MINUTE - elapsed_minute)
    if remaining_to_target:
        average_needed_to_target = time_remaining_to_cutoff / remaining_to_target
    else:
        average_needed_to_target = None

    laps_needed_before_target_start = max(0, remaining_to_target - 1)
    time_remaining_to_last_start = max(0.0, LAST_START_MINUTE - elapsed_minute)
    if laps_needed_before_target_start:
        average_needed_to_start_target_lap = time_remaining_to_last_start / laps_needed_before_target_start
    else:
        average_needed_to_start_target_lap = None

    return RaceState(
        current_laps=current_laps,
        elapsed_minute=elapsed_minute,
        laps_by_noon=laps_by_noon,
        next_runner=next_runner,
        next_runner_index=next_index,
        runner_lap_counts=counts,
        latest_lap=latest_lap,
        remaining_to_target=remaining_to_target,
        average_needed_to_target=average_needed_to_target,
        average_needed_to_start_target_lap=average_needed_to_start_target_lap,
    )


def forecast_from_race_log(
    roster: pd.DataFrame,
    settings: SimulationSettings,
    log: pd.DataFrame,
    caps_enabled: bool = True,
) -> dict[str, Any]:
    state = race_state_from_log(log, roster, settings.target_laps, caps_enabled=caps_enabled)
    return simulate_many_from_state(
        roster,
        settings,
        elapsed_minute=state.elapsed_minute,
        next_runner_index=state.next_runner_index,
        runner_lap_counts=state.runner_lap_counts,
        official_laps_so_far=state.current_laps,
        laps_by_noon_so_far=state.laps_by_noon,
        caps_enabled=caps_enabled,
    )


def next_eligible_runner(
    roster: pd.DataFrame,
    runner_lap_counts: dict[str, int],
    start_index: int,
    caps_enabled: bool = True,
) -> tuple[str | None, int]:
    order = _available_order(roster)
    if not order:
        return None, 0

    caps = roster.copy()
    caps["max_laps"] = pd.to_numeric(caps["max_laps"], errors="coerce")
    cap_lookup = {
        str(row.runner): None if pd.isna(row.max_laps) else int(row.max_laps)
        for row in caps.itertuples(index=False)
    }
    for offset in range(len(order)):
        idx = (int(start_index) + offset) % len(order)
        runner = order[idx]
        cap = cap_lookup.get(runner)
        if caps_enabled and cap is not None and runner_lap_counts.get(runner, 0) >= cap:
            continue
        return runner, idx
    return None, int(start_index) % len(order)


def google_sheets_configured(secrets: Mapping[str, Any], sheet_id: str | None = None) -> bool:
    return bool((sheet_id or _secret_value(secrets, "google_sheet_id", "race_log_google_sheet_id")) and _service_account_info(secrets))


def read_race_log_from_google_sheet(
    secrets: Mapping[str, Any],
    sheet_id: str | None = None,
    worksheet_name: str = "race_log",
) -> pd.DataFrame:
    sheet = _open_sheet(secrets, sheet_id, worksheet_name)
    records = sheet.get_all_records()
    return pd.DataFrame(records)


def write_race_log_to_google_sheet(
    log: pd.DataFrame,
    secrets: Mapping[str, Any],
    sheet_id: str | None = None,
    worksheet_name: str = "race_log",
) -> None:
    sheet = _open_sheet(secrets, sheet_id, worksheet_name)
    clean = log.copy()
    clean = clean[RACE_LOG_COLUMNS]
    values = [clean.columns.tolist()] + clean.replace({np.nan: ""}).astype(str).values.tolist()
    sheet.clear()
    if values:
        sheet.update(values)


def parse_race_time_to_minute(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric):
        return float(numeric)

    match = re.search(r"(?:(sat(?:urday)?|sun(?:day)?)\s+)?(\d{1,2}):(\d{2})", text, flags=re.IGNORECASE)
    if not match:
        return None

    day_text = (match.group(1) or "").lower()
    hour = int(match.group(2))
    minute = int(match.group(3))
    if hour > 23 or minute > 59:
        return None

    clock_minutes = hour * 60 + minute
    if day_text.startswith("sun") or (not day_text and hour < 12):
        absolute = 24 * 60 + clock_minutes
    else:
        absolute = clock_minutes
    return float(absolute - 12 * 60)


def _normalise_column_name(column: Any) -> str:
    text = str(column).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    aliases = {
        "lap": "lap_number",
        "lap_no": "lap_number",
        "lap_num": "lap_number",
        "name": "runner",
        "athlete": "runner",
        "time": "finish_time",
        "finish": "finish_time",
        "duration": "lap_duration_minutes",
        "lap_time": "lap_duration_minutes",
        "lap_time_minutes": "lap_duration_minutes",
    }
    return aliases.get(text, text)


def _first_parsed_time(row: pd.Series, columns: list[str]) -> float | None:
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            parsed = parse_race_time_to_minute(row[column])
            if parsed is not None:
                return parsed
    return None


def _first_number(row: pd.Series, columns: list[str]) -> float | None:
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            number = pd.to_numeric(row[column], errors="coerce")
            if pd.notna(number):
                return float(number)
    return None


def _number_or_default(value: Any, default: int) -> int:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return int(default)
    return int(number)


def _available_order(roster: pd.DataFrame) -> list[str]:
    clean = roster.copy()
    clean["available"] = clean["available"].astype(bool)
    clean["max_laps"] = pd.to_numeric(clean["max_laps"], errors="coerce")
    clean = clean[clean["available"] & (clean["max_laps"].isna() | clean["max_laps"].ne(0))]
    return clean.sort_values(["running_order", "runner"])["runner"].astype(str).tolist()


def _secret_value(secrets: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        try:
            value = secrets.get(key)
        except Exception:
            value = None
        if value:
            return value
    return None


def _service_account_info(secrets: Mapping[str, Any]) -> dict[str, Any] | None:
    value = _secret_value(secrets, "gcp_service_account", "google_service_account")
    if not value:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _open_sheet(secrets: Mapping[str, Any], sheet_id: str | None, worksheet_name: str):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as exc:  # pragma: no cover - depends on optional cloud packages.
        raise RuntimeError("Install gspread and google-auth to use Google Sheets sync.") from exc

    service_account = _service_account_info(secrets)
    resolved_sheet_id = sheet_id or _secret_value(secrets, "google_sheet_id", "race_log_google_sheet_id")
    if not service_account or not resolved_sheet_id:
        raise RuntimeError("Google Sheets sync needs google_sheet_id and gcp_service_account in Streamlit secrets.")

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_info(service_account, scopes=scopes)
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(str(resolved_sheet_id))
    try:
        return spreadsheet.worksheet(worksheet_name)
    except Exception:
        return spreadsheet.add_worksheet(title=worksheet_name, rows=200, cols=len(RACE_LOG_COLUMNS))
