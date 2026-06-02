"""Race-day lap log parsing, validation, forecasting, and Google Sheets sync."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
import re
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

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

DRINKS_COLUMNS = ["runner", "count", "unit", "notes"]

GOOGLE_SHEET_TEMPLATE_COLUMNS = [
    "lap_number",
    "runner",
    "start_time",
    "finish_time",
    "lap_duration_minutes",
    "notes",
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
    current_lap_in_progress: bool
    in_progress_runner: str | None
    in_progress_start_minute: float | None
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
    start_minute: float | None = None,
    finish_minute: float | None = None,
    duration_minutes: float | None = None,
    notes: str = "",
) -> pd.DataFrame:
    log = normalise_race_log(current_log, roster)
    previous_finish = float(log["finish_minute"].dropna().max()) if not log["finish_minute"].dropna().empty else 0.0
    resolved_start = previous_finish if start_minute is None else float(start_minute)
    resolved_finish = None if finish_minute is None else float(finish_minute)
    resolved_duration = None if duration_minutes is None else float(duration_minutes)

    if start_minute is None and resolved_finish is not None and resolved_duration is not None:
        resolved_start = resolved_finish - resolved_duration
    elif resolved_finish is None and resolved_duration is not None:
        resolved_finish = resolved_start + resolved_duration

    if resolved_duration is None and resolved_finish is not None:
        resolved_duration = resolved_finish - resolved_start

    next_lap = int(log["lap_number"].max()) + 1 if not log.empty else 1
    added = pd.DataFrame(
        [
            {
                "lap_number": next_lap,
                "runner": runner,
                "start_minute": resolved_start,
                "finish_minute": np.nan if resolved_finish is None else resolved_finish,
                "lap_duration_minutes": np.nan if resolved_duration is None else resolved_duration,
                "notes": notes,
            }
        ]
    )
    if log.empty:
        return normalise_race_log(added, roster)
    return normalise_race_log(pd.concat([log, added], ignore_index=True), roster)


def complete_in_progress_lap(
    current_log: pd.DataFrame,
    roster: pd.DataFrame,
    finish_minute: float | None = None,
    duration_minutes: float | None = None,
    notes: str = "",
) -> pd.DataFrame:
    log = normalise_race_log(current_log, roster)
    if log.empty:
        raise ValueError("There is no in-progress lap to complete.")

    latest_idx = log.sort_values("lap_number").index[-1]
    latest = log.loc[latest_idx]
    if pd.isna(latest.get("start_minute")) or pd.notna(latest.get("finish_minute")):
        raise ValueError("The latest row is not an in-progress lap.")

    start_minute = float(latest["start_minute"])
    resolved_finish = None if finish_minute is None else float(finish_minute)
    resolved_duration = None if duration_minutes is None else float(duration_minutes)
    if resolved_finish is None and resolved_duration is not None:
        resolved_finish = start_minute + resolved_duration
    if resolved_duration is None and resolved_finish is not None:
        resolved_duration = resolved_finish - start_minute
    if resolved_finish is None or resolved_duration is None:
        raise ValueError("Enter either the finish time or the lap duration.")

    log.loc[latest_idx, "finish_minute"] = resolved_finish
    log.loc[latest_idx, "lap_duration_minutes"] = resolved_duration
    if notes.strip():
        existing_notes = str(log.loc[latest_idx, "notes"] or "").strip()
        log.loc[latest_idx, "notes"] = f"{existing_notes}; {notes.strip()}" if existing_notes else notes.strip()
    return normalise_race_log(log, roster)


def race_order_review(log: pd.DataFrame, roster: pd.DataFrame, caps_enabled: bool = True) -> pd.DataFrame:
    columns = ["lap_number", "runner", "expected_runner", "order_status", "order_exception"]
    clean = normalise_race_log(log, roster)
    if clean.empty:
        return pd.DataFrame(columns=columns)

    try:
        order = _available_order(roster)
    except Exception:
        return pd.DataFrame(columns=columns)
    if not order:
        return pd.DataFrame(columns=columns)

    counts = {runner: 0 for runner in order}
    pointer = 0
    rows: list[dict[str, Any]] = []
    for row in clean.sort_values("lap_number").itertuples(index=False):
        actual = str(row.runner).strip()
        expected, expected_index = next_eligible_runner(roster, counts, pointer, caps_enabled=caps_enabled)

        if not actual:
            status = "Missing runner"
            exception = True
        elif actual not in order:
            status = "Unknown runner"
            exception = True
        elif expected and actual != expected:
            status = f"Expected {expected}"
            exception = True
        else:
            status = "On plan"
            exception = False

        rows.append(
            {
                "lap_number": int(row.lap_number),
                "runner": actual,
                "expected_runner": expected,
                "order_status": status,
                "order_exception": exception,
            }
        )

        if actual in order:
            pointer = (order.index(actual) + 1) % len(order)
            if bool(row.official):
                counts[actual] = counts.get(actual, 0) + 1
        elif expected_index is not None:
            pointer = expected_index

    return pd.DataFrame(rows, columns=columns)


def validate_race_log(log: pd.DataFrame, roster: pd.DataFrame, caps_enabled: bool = True) -> tuple[list[str], list[str]]:
    clean = normalise_race_log(log, roster)
    warnings: list[str] = []
    errors: list[str] = []
    if clean.empty:
        return warnings, errors

    known_runners = set(roster["runner"].astype(str).str.strip())
    unknown = sorted(set(clean["runner"].astype(str).str.strip()).difference(known_runners).difference({""}))
    if unknown:
        errors.append(f"Unknown runner in the race log: {', '.join(unknown)}. Use the roster spelling or correct the Sheet.")

    if clean["runner"].astype(str).str.strip().eq("").any():
        errors.append("Every race-log row needs a runner.")
    if clean["lap_number"].duplicated().any():
        errors.append("Duplicate lap numbers found. Each lap number must be unique.")
    completed = clean.dropna(subset=["start_minute", "finish_minute", "lap_duration_minutes"]).copy()
    incomplete = clean[clean[["start_minute", "finish_minute", "lap_duration_minutes"]].isna().any(axis=1)].copy()
    completed_finish_rows = clean.dropna(subset=["runner", "finish_minute"])
    if completed_finish_rows.duplicated(["runner", "finish_minute"]).any():
        errors.append("Duplicate runner/finish-time entries found.")

    if not incomplete.empty:
        latest_lap_number = int(clean["lap_number"].max())
        incomplete_laps = incomplete["lap_number"].astype(int).tolist()
        if len(incomplete) == 1 and int(incomplete.iloc[0]["lap_number"]) == latest_lap_number:
            row = incomplete.iloc[0]
            if pd.notna(row.get("start_minute")) and pd.isna(row.get("finish_minute")):
                warnings.append(
                    f"Lap {latest_lap_number} is in progress. It will not count as official until a finish time or duration is added."
                )
            else:
                errors.append(f"Lap {latest_lap_number} needs a start time plus either finish time or duration.")
        else:
            errors.append(
                "Incomplete timing is only allowed for the latest lap. Check laps "
                + ", ".join(str(lap) for lap in incomplete_laps)
                + "."
            )

    impossible = clean[
        clean["start_minute"].lt(0)
    ]
    if not impossible.empty:
        errors.append(f"Start time is before Saturday 12:00 on lap(s) {_format_laps(impossible['lap_number'])}.")

    impossible_completed = completed[
        completed["finish_minute"].lt(0)
        | completed["finish_minute"].le(completed["start_minute"])
        | completed["lap_duration_minutes"].le(0)
    ]
    if not impossible_completed.empty:
        errors.append(
            f"Lap timing is impossible on lap(s) {_format_laps(impossible_completed['lap_number'])}: "
            "finish must be after start and duration must be positive."
        )

    if clean["finish_minute"].gt(FINAL_CUTOFF_MINUTE).any():
        late_finish = clean[clean["finish_minute"].gt(FINAL_CUTOFF_MINUTE)]
        warnings.append(f"Lap(s) {_format_laps(late_finish['lap_number'])} finish after Sunday 13:00 and will not count.")
    if clean["start_minute"].gt(LAST_START_MINUTE).any():
        late_start = clean[clean["start_minute"].gt(LAST_START_MINUTE)]
        warnings.append(f"Lap(s) {_format_laps(late_start['lap_number'])} start after Sunday 12:00 and will not count.")

    ordered = clean.sort_values("lap_number")
    completed_ordered = ordered.dropna(subset=["start_minute", "finish_minute"])
    overlap_laps = []
    gap_messages = []
    previous_lap = None
    previous_finish = None
    for row in completed_ordered.itertuples(index=False):
        if previous_finish is not None:
            if float(row.start_minute) < previous_finish - 0.01:
                overlap_laps.append(int(row.lap_number))
            elif float(row.start_minute) > previous_finish + 10:
                gap_messages.append(f"lap {int(row.lap_number)} has a {float(row.start_minute) - previous_finish:.1f} min gap after lap {previous_lap}")
        previous_lap = int(row.lap_number)
        previous_finish = float(row.finish_minute)
    if overlap_laps:
        errors.append(f"Lap(s) {_format_laps(overlap_laps)} start before the previous lap has finished.")

    finish_regressions = ordered.dropna(subset=["finish_minute"])
    if finish_regressions["finish_minute"].diff().le(0).any():
        bad_laps = finish_regressions.loc[finish_regressions["finish_minute"].diff().le(0), "lap_number"]
        errors.append(f"Finish times must increase lap by lap. Check lap(s) {_format_laps(bad_laps)}.")
    start_regressions = ordered.dropna(subset=["start_minute"])
    if start_regressions["start_minute"].diff().lt(0).any():
        bad_laps = start_regressions.loc[start_regressions["start_minute"].diff().lt(0), "lap_number"]
        errors.append(f"Start times move backwards. Check lap(s) {_format_laps(bad_laps)}.")

    if gap_messages:
        warnings.append("Timing gap check: " + "; ".join(gap_messages[:3]) + ("." if len(gap_messages) <= 3 else "; more gaps hidden."))

    if {"projected_mean_minutes", "projected_sd_minutes"}.issubset(roster.columns):
        pace_source = roster.copy()
        pace_source["_runner_key"] = pace_source["runner"].astype(str).str.strip()
        pace_lookup = pace_source.drop_duplicates("_runner_key").set_index("_runner_key")
        pace_messages = []
        for row in completed.itertuples(index=False):
            runner = str(row.runner).strip()
            if runner not in pace_lookup.index:
                continue
            pace_row = pace_lookup.loc[runner]
            mean = pd.to_numeric(pace_row.get("projected_mean_minutes"), errors="coerce")
            sd = pd.to_numeric(pace_row.get("projected_sd_minutes"), errors="coerce")
            if pd.isna(mean) or pd.isna(sd) or float(sd) <= 0:
                continue
            duration = float(row.lap_duration_minutes)
            low = max(10.0, float(mean) - 3 * float(sd))
            high = float(mean) + 3 * float(sd)
            if duration < low or duration > high:
                pace_messages.append(f"lap {int(row.lap_number)} {runner} is {duration:.1f} min vs usual {float(mean):.1f}")
        if pace_messages:
            warnings.append("Pace check: " + "; ".join(pace_messages[:4]) + ("." if len(pace_messages) <= 4 else "; more pace checks hidden."))

    order_review = race_order_review(clean, roster, caps_enabled=caps_enabled)
    if not order_review.empty:
        exceptions = order_review[
            order_review["order_exception"].fillna(False)
            & order_review["runner"].astype(str).ne("")
            & order_review["expected_runner"].astype(str).ne("")
            & ~order_review["order_status"].isin(["Unknown runner", "Missing runner"])
        ]
        if not exceptions.empty:
            details = [
                f"lap {int(row.lap_number)} {row.runner} ran, expected {row.expected_runner}"
                for row in exceptions.head(4).itertuples(index=False)
            ]
            suffix = "." if len(exceptions) <= 4 else "; more order exceptions hidden."
            warnings.append(
                "Order exception: "
                + "; ".join(details)
                + suffix
                + " This is allowed; the forecast resumes the fixed order from the actual runner logged."
            )

    if caps_enabled:
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


def _format_laps(values: Any) -> str:
    laps = [str(int(value)) for value in pd.Series(values).dropna().tolist()]
    if not laps:
        return "-"
    return ", ".join(laps)


def build_progress_actuals(log: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    clean = normalise_race_log(log, roster)
    official = clean[clean["official"].fillna(False).astype(bool)].copy()
    official = official.dropna(subset=["finish_minute"]).sort_values("finish_minute")

    rows = [{"time_minute": 0.0, "completed_laps": 0}]
    for lap_count, row in enumerate(official.itertuples(index=False), start=1):
        rows.append({"time_minute": float(row.finish_minute), "completed_laps": lap_count})
    return pd.DataFrame(rows, columns=["time_minute", "completed_laps"])


def target_pace_series(target_laps: int) -> pd.DataFrame:
    target = max(0, int(target_laps))
    return pd.DataFrame(
        [
            {"time_minute": 0.0, "completed_laps": 0.0},
            {"time_minute": float(LAST_START_MINUTE), "completed_laps": float(max(target - 1, 0))},
            {"time_minute": float(FINAL_CUTOFF_MINUTE), "completed_laps": float(target)},
        ]
    )


def build_runner_queue(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    caps_enabled: bool = True,
    queue_size: int = 3,
) -> pd.DataFrame:
    clean = normalise_race_log(log, roster)
    state = race_state_from_log(clean, roster, target_laps=1, caps_enabled=caps_enabled)
    order = _available_order(roster)
    columns = ["role", "runner", "completed_laps", "cap_remaining", "last_lap_minutes", "rest_minutes", "order_status"]
    if not order:
        return pd.DataFrame(columns=columns)

    roster_lookup = _roster_lookup(roster)
    completed = clean.dropna(subset=["finish_minute", "lap_duration_minutes"]).copy()
    order_review = race_order_review(clean, roster, caps_enabled=caps_enabled)
    latest_order_status = ""
    if state.latest_lap is not None and not order_review.empty:
        latest_lap_number = int(state.latest_lap["lap_number"])
        matched = order_review[order_review["lap_number"].astype(int) == latest_lap_number]
        if not matched.empty:
            latest_order_status = str(matched.iloc[-1]["order_status"])

    rows: list[dict[str, Any]] = []
    queued = 0
    pointer = int(state.next_runner_index)

    if state.current_lap_in_progress and state.in_progress_runner in order:
        rows.append(
            _runner_queue_row(
                role="Current",
                runner=state.in_progress_runner,
                state=state,
                completed=completed,
                roster_lookup=roster_lookup,
                caps_enabled=caps_enabled,
                order_status=latest_order_status or "Running",
            )
        )
        pointer = (order.index(state.in_progress_runner) + 1) % len(order)

    counts = dict(state.runner_lap_counts)
    while queued < int(queue_size):
        runner, runner_index = next_eligible_runner(roster, counts, pointer, caps_enabled=caps_enabled)
        if runner is None:
            break
        rows.append(
            _runner_queue_row(
                role="Next" if queued == 0 and not state.current_lap_in_progress else f"+{queued + 1}",
                runner=runner,
                state=state,
                completed=completed,
                roster_lookup=roster_lookup,
                caps_enabled=caps_enabled,
                order_status="Planned",
            )
        )
        pointer = (int(runner_index) + 1) % len(order)
        queued += 1

    return pd.DataFrame(rows, columns=columns)


def build_runner_status_table(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    caps_enabled: bool = True,
) -> pd.DataFrame:
    clean = normalise_race_log(log, roster)
    state = race_state_from_log(clean, roster, target_laps=1, caps_enabled=caps_enabled)
    completed = clean.dropna(subset=["finish_minute", "lap_duration_minutes"]).copy()
    roster_lookup = _roster_lookup(roster)

    rows = []
    for runner in _available_order(roster):
        row = _runner_queue_row(
            role="Running" if state.current_lap_in_progress and state.in_progress_runner == runner else "Available",
            runner=runner,
            state=state,
            completed=completed,
            roster_lookup=roster_lookup,
            caps_enabled=caps_enabled,
            order_status="",
        )
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=["role", "runner", "completed_laps", "cap_remaining", "last_lap_minutes", "rest_minutes", "order_status"],
    )


def race_state_from_log(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    target_laps: int,
    caps_enabled: bool = True,
) -> RaceState:
    clean = normalise_race_log(log, roster)
    completed = clean.dropna(subset=["start_minute", "finish_minute", "lap_duration_minutes"]).copy()
    official = completed[completed["official"].fillna(False).astype(bool)].copy()

    latest_lap = clean.sort_values("lap_number").iloc[-1].to_dict() if not clean.empty else None
    current_lap_in_progress = False
    in_progress_runner = None
    in_progress_start_minute = None
    if latest_lap is not None:
        has_start = pd.notna(latest_lap.get("start_minute"))
        has_finish = pd.notna(latest_lap.get("finish_minute"))
        if has_start and not has_finish:
            current_lap_in_progress = True
            in_progress_runner = str(latest_lap.get("runner") or "").strip() or None
            in_progress_start_minute = float(latest_lap["start_minute"])

    last_completed_finish = float(completed["finish_minute"].max()) if not completed.empty else 0.0
    elapsed_minute = max(last_completed_finish, in_progress_start_minute or 0.0)
    current_laps = int(len(official))
    laps_by_noon = int(official["finish_minute"].le(LAST_START_MINUTE).sum()) if not official.empty else 0

    order = _available_order(roster)
    counts = {runner: 0 for runner in order}
    for runner, count in official["runner"].value_counts().items():
        if str(runner) in counts:
            counts[str(runner)] = int(count)

    if current_lap_in_progress and in_progress_runner in order:
        start_index = order.index(in_progress_runner)
        next_runner = in_progress_runner
        next_index = start_index
    elif not completed.empty and str(completed.iloc[-1]["runner"]) in order:
        start_index = (order.index(str(completed.iloc[-1]["runner"])) + 1) % len(order)
        next_runner, next_index = next_eligible_runner(roster, counts, start_index, caps_enabled=caps_enabled)
    else:
        start_index = current_laps % len(order) if order else 0
        next_runner, next_index = next_eligible_runner(roster, counts, start_index, caps_enabled=caps_enabled)

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
        current_lap_in_progress=current_lap_in_progress,
        in_progress_runner=in_progress_runner,
        in_progress_start_minute=in_progress_start_minute,
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
        first_transition_already_applied=state.current_lap_in_progress,
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


def read_drinks_from_google_sheet(
    secrets: Mapping[str, Any],
    sheet_id: str | None = None,
    worksheet_name: str = "drinks",
) -> pd.DataFrame:
    sheet = _open_sheet(secrets, sheet_id, worksheet_name, columns=DRINKS_COLUMNS)
    records = sheet.get_all_records()
    return pd.DataFrame(records)


def write_drinks_to_google_sheet(
    drinks: pd.DataFrame,
    secrets: Mapping[str, Any],
    sheet_id: str | None = None,
    worksheet_name: str = "drinks",
) -> None:
    sheet = _open_sheet(secrets, sheet_id, worksheet_name, columns=DRINKS_COLUMNS)
    clean = drinks.copy()
    for column in DRINKS_COLUMNS:
        if column not in clean:
            clean[column] = ""
    clean = clean[DRINKS_COLUMNS]
    values = [DRINKS_COLUMNS] + clean.replace({np.nan: ""}).astype(str).values.tolist()
    sheet.clear()
    sheet.update(values)


def google_sheet_url_to_csv_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    if "output=csv" in text or "format=csv" in text:
        return text

    parsed = urlparse(text)
    if "docs.google.com" not in parsed.netloc or "/spreadsheets/d/" not in parsed.path:
        return text

    match = re.search(r"/spreadsheets/d/([^/]+)", parsed.path)
    if not match:
        return text

    query_gid = parse_qs(parsed.query).get("gid")
    fragment_gid = parse_qs(parsed.fragment).get("gid")
    gid = (query_gid or fragment_gid or ["0"])[0]
    sheet_id = match.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def read_race_log_from_google_sheet_url(url: str, roster: pd.DataFrame) -> pd.DataFrame:
    csv_url = google_sheet_url_to_csv_url(url)
    if not csv_url:
        return empty_race_log()
    raw = pd.read_csv(csv_url)
    return normalise_race_log(raw, roster)


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


def parse_duration_minutes(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().lower()
    if not text:
        return None

    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric):
        return float(numeric)

    clock_match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{2})", text)
    if clock_match:
        hours = int(clock_match.group(1) or 0)
        minutes = int(clock_match.group(2))
        seconds = int(clock_match.group(3))
        if minutes > 59 or seconds > 59:
            return None
        return hours * 60 + minutes + seconds / 60

    words_match = re.fullmatch(
        r"(?:(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\s*)?"
        r"(?:(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\s*)?"
        r"(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\s*)?",
        text,
    )
    if words_match and any(words_match.groups()):
        hours = float(words_match.group(1) or 0)
        minutes = float(words_match.group(2) or 0)
        seconds = float(words_match.group(3) or 0)
        return hours * 60 + minutes + seconds / 60

    number_match = re.search(r"\d+(?:\.\d+)?", text)
    if number_match:
        return float(number_match.group(0))
    return None


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
            number = parse_duration_minutes(row[column])
            if number is not None:
                return float(number)
    return None


def _number_or_default(value: Any, default: int) -> int:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return int(default)
    return int(number)


def _roster_lookup(roster: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in roster.itertuples(index=False):
        runner = str(getattr(row, "runner", "")).strip()
        if runner:
            rows[runner] = row._asdict()
    return rows


def _runner_queue_row(
    role: str,
    runner: str,
    state: RaceState,
    completed: pd.DataFrame,
    roster_lookup: dict[str, dict[str, Any]],
    caps_enabled: bool,
    order_status: str,
) -> dict[str, Any]:
    runner_laps = completed[completed["runner"].astype(str).str.strip() == str(runner)].sort_values("finish_minute")
    last_lap_minutes = np.nan
    rest_minutes = np.nan
    if not runner_laps.empty:
        latest = runner_laps.iloc[-1]
        last_lap_minutes = float(latest["lap_duration_minutes"])
        rest_minutes = max(0.0, float(state.elapsed_minute) - float(latest["finish_minute"]))

    completed_laps = int(state.runner_lap_counts.get(str(runner), 0))
    cap_remaining: str | int = "Open"
    cap = pd.to_numeric(roster_lookup.get(str(runner), {}).get("max_laps"), errors="coerce")
    if caps_enabled and pd.notna(cap):
        cap_remaining = max(0, int(cap) - completed_laps)

    return {
        "role": role,
        "runner": runner,
        "completed_laps": completed_laps,
        "cap_remaining": cap_remaining,
        "last_lap_minutes": np.nan if pd.isna(last_lap_minutes) else round(float(last_lap_minutes), 1),
        "rest_minutes": np.nan if pd.isna(rest_minutes) else round(float(rest_minutes), 0),
        "order_status": order_status,
    }


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
    value = _secret_value(
        secrets,
        "gcp_service_account",
        "google_service_account",
        "gcp_service_account_json",
        "google_service_account_json",
    )
    if not value:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _open_sheet(
    secrets: Mapping[str, Any],
    sheet_id: str | None,
    worksheet_name: str,
    columns: list[str] | None = None,
):
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
        return spreadsheet.add_worksheet(title=worksheet_name, rows=200, cols=len(columns or RACE_LOG_COLUMNS))
