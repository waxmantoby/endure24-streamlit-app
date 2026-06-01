"""Workbook parsing and Endure24 roster defaults.

The simulation engine is intentionally team-agnostic. Team aliases and
projection defaults live here because they are specific to this workbook.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import datetime as dt
import re
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("data/endure_info_for_lake_dwellers_to_review.xlsx")
DEFAULT_NIGHT_START_MINUTE = 9 * 60
DEFAULT_NIGHT_END_MINUTE = 18 * 60


DEFAULT_ROSTER = [
    {
        "runner": "Toby",
        "projected_mean_minutes": 33.0,
        "projected_sd_minutes": np.nan,
        "running_order": 1,
        "max_laps": np.nan,
        "available": True,
        "fatigue_pct_per_lap": 0.0,
        "night_penalty_pct": 0.0,
        "notes": "Override from 33.9 to 33.0",
    },
    {
        "runner": "Rory",
        "projected_mean_minutes": 34.0,
        "projected_sd_minutes": np.nan,
        "running_order": 2,
        "max_laps": np.nan,
        "available": True,
        "fatigue_pct_per_lap": 0.0,
        "night_penalty_pct": 0.0,
        "notes": "",
    },
    {
        "runner": "Alex",
        "projected_mean_minutes": 34.0,
        "projected_sd_minutes": np.nan,
        "running_order": 3,
        "max_laps": np.nan,
        "available": True,
        "fatigue_pct_per_lap": 0.0,
        "night_penalty_pct": 0.0,
        "notes": "New runner. Defaults to Toby's projected SD.",
    },
    {
        "runner": "Jared",
        "projected_mean_minutes": 36.0,
        "projected_sd_minutes": np.nan,
        "running_order": 4,
        "max_laps": 4,
        "available": True,
        "fatigue_pct_per_lap": 0.0,
        "night_penalty_pct": 0.0,
        "notes": "Capped at 4 laps",
    },
    {
        "runner": "James",
        "projected_mean_minutes": 37.6,
        "projected_sd_minutes": np.nan,
        "running_order": 5,
        "max_laps": np.nan,
        "available": True,
        "fatigue_pct_per_lap": 0.0,
        "night_penalty_pct": 0.0,
        "notes": "",
    },
    {
        "runner": "Becs",
        "projected_mean_minutes": 38.0,
        "projected_sd_minutes": np.nan,
        "running_order": 6,
        "max_laps": 4,
        "available": True,
        "fatigue_pct_per_lap": 0.0,
        "night_penalty_pct": 0.0,
        "notes": "Capped at 4 laps. Becca and Becs are treated as the same runner.",
    },
    {
        "runner": "Craig",
        "projected_mean_minutes": 41.0,
        "projected_sd_minutes": np.nan,
        "running_order": 7,
        "max_laps": np.nan,
        "available": True,
        "fatigue_pct_per_lap": 0.0,
        "night_penalty_pct": 0.0,
        "notes": "",
    },
    {
        "runner": "Ellie",
        "projected_mean_minutes": 44.0,
        "projected_sd_minutes": np.nan,
        "running_order": 8,
        "max_laps": np.nan,
        "available": True,
        "fatigue_pct_per_lap": 0.0,
        "night_penalty_pct": 0.0,
        "notes": "",
    },
]


TEAM_ALIASES = {
    "toby": "Toby",
    "toby waxman": "Toby",
    "rory": "Rory",
    "rory myner": "Rory",
    "alex": "Alex",
    "jared": "Jared",
    "jared taylor": "Jared",
    "james": "James",
    "james ponsford": "James",
    "becs": "Becs",
    "becca": "Becs",
    "becca hawkins": "Becs",
    "craig": "Craig",
    "craig losh": "Craig",
    "ellie": "Ellie",
    "ellie dorsman": "Ellie",
}


@dataclass
class LoadedEndureData:
    source_label: str
    sheet_info: pd.DataFrame
    last_year_laps: pd.DataFrame
    last_year_stats: pd.DataFrame
    team_stats: dict[str, Any]
    roster: pd.DataFrame
    name_map: pd.DataFrame
    warnings: list[str]
    errors: list[str]
    detected_tables: list[str]


def load_endure_workbook(source: str | Path | bytes, source_label: str | None = None) -> LoadedEndureData:
    """Load the workbook and return cleaned lap data plus default roster assumptions."""

    workbook_bytes = _read_source_bytes(source)
    label = source_label or getattr(source, "name", None) or "uploaded workbook"
    raw_sheets = pd.read_excel(BytesIO(workbook_bytes), sheet_name=None, header=None, engine="openpyxl")
    sheet_info = _sheet_info(raw_sheets)

    warnings: list[str] = []
    errors: list[str] = []
    detected_tables: list[str] = []

    last_year_laps = _find_clean_last_year_sheet(workbook_bytes)
    if last_year_laps is not None:
        detected_tables.append("Detected clean last_year_laps-style table.")
    else:
        last_year_laps = _find_dashboard_lap_table(raw_sheets)
        if last_year_laps is not None:
            detected_tables.append("Detected dashboard lap table from the workbook.")

    if last_year_laps is None or last_year_laps.empty:
        last_year_laps = pd.DataFrame(
            columns=[
                "lap",
                "runner_raw",
                "runner",
                "lap_time_minutes",
                "start_minute",
                "finish_minute",
                "lap_number_for_runner",
                "is_night",
            ]
        )
        errors.append("Could not find a usable last-year lap table in the workbook.")
    else:
        last_year_laps = clean_last_year_laps(last_year_laps)
        if last_year_laps.empty:
            errors.append("The last-year lap table was detected but no valid lap times could be parsed.")

    roster_input = _find_clean_roster_sheet(workbook_bytes)
    if roster_input is not None and not roster_input.empty:
        detected_tables.append("Detected this_year_roster-style table.")
    else:
        warnings.append(
            "No clean this_year_roster sheet was found; using the projected roster defaults from the brief."
        )

    last_year_stats, team_stats = compute_last_year_stats(last_year_laps)
    roster = build_roster_defaults(last_year_stats, team_stats, roster_input)
    roster_warnings, roster_errors = validate_roster(roster)
    warnings.extend(roster_warnings)
    errors.extend(roster_errors)

    name_map = _build_name_map(last_year_laps, roster)
    if "Becca" in name_map["original"].astype(str).str.title().tolist() or any(
        name_map["original"].astype(str).str.contains("Becca", case=False, na=False)
    ):
        warnings.append("Becca from last year has been matched to Becs for this year's roster.")

    return LoadedEndureData(
        source_label=label,
        sheet_info=sheet_info,
        last_year_laps=last_year_laps,
        last_year_stats=last_year_stats,
        team_stats=team_stats,
        roster=roster,
        name_map=name_map,
        warnings=warnings,
        errors=errors,
        detected_tables=detected_tables,
    )


def clean_last_year_laps(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [_canonical_column_name(c) for c in cleaned.columns]
    col_map = _map_columns(cleaned.columns)
    cleaned = cleaned.rename(columns={source: target for target, source in col_map.items() if source})

    if "runner" not in cleaned.columns:
        if "runner_raw" in cleaned.columns:
            cleaned["runner"] = cleaned["runner_raw"]
        else:
            return pd.DataFrame()

    cleaned["runner_raw"] = cleaned["runner"].astype(str).str.strip()
    cleaned["runner"] = cleaned["runner_raw"].map(normalize_runner_name)

    if "lap_time_minutes" not in cleaned.columns:
        if "lap_time" in cleaned.columns:
            cleaned["lap_time_minutes"] = cleaned["lap_time"].map(parse_lap_minutes)
        else:
            return pd.DataFrame()
    else:
        cleaned["lap_time_minutes"] = cleaned["lap_time_minutes"].map(parse_lap_minutes)

    cleaned = cleaned[pd.to_numeric(cleaned["lap_time_minutes"], errors="coerce").notna()].copy()
    cleaned["lap_time_minutes"] = cleaned["lap_time_minutes"].astype(float)
    cleaned = cleaned[cleaned["lap_time_minutes"] > 0].copy()

    if "lap" not in cleaned.columns:
        cleaned["lap"] = np.arange(1, len(cleaned) + 1)
    cleaned["lap"] = pd.to_numeric(cleaned["lap"], errors="coerce")
    cleaned = cleaned.sort_values("lap", na_position="last").reset_index(drop=True)

    if "start_minute" in cleaned.columns:
        cleaned["start_minute"] = cleaned["start_minute"].map(parse_lap_minutes)
    elif "start_time" in cleaned.columns:
        cleaned["start_minute"] = cleaned["start_time"].map(parse_lap_minutes)
    else:
        cleaned["start_minute"] = cleaned["lap_time_minutes"].cumsum().shift(fill_value=0)

    cleaned["finish_minute"] = cleaned["start_minute"] + cleaned["lap_time_minutes"]

    if "lap_number_for_runner" in cleaned.columns:
        cleaned["lap_number_for_runner"] = pd.to_numeric(cleaned["lap_number_for_runner"], errors="coerce")
    else:
        cleaned["lap_number_for_runner"] = cleaned.groupby("runner").cumcount() + 1

    if "is_night" in cleaned.columns:
        cleaned["is_night"] = cleaned["is_night"].map(_to_bool)
    else:
        cleaned["is_night"] = cleaned["start_minute"].map(
            lambda minute: DEFAULT_NIGHT_START_MINUTE <= float(minute) < DEFAULT_NIGHT_END_MINUTE
        )

    columns = [
        "lap",
        "runner_raw",
        "runner",
        "lap_time_minutes",
        "start_minute",
        "finish_minute",
        "lap_number_for_runner",
        "is_night",
    ]
    return cleaned[columns].reset_index(drop=True)


def compute_last_year_stats(last_year_laps: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if last_year_laps.empty:
        return pd.DataFrame(), {
            "total_laps": 0,
            "total_time_minutes": np.nan,
            "team_mean_minutes": np.nan,
            "team_sd_minutes": np.nan,
            "team_cv": 0.05,
            "team_average_runner_cv": 0.05,
            "fastest_lap_minutes": np.nan,
            "slowest_lap_minutes": np.nan,
        }

    rows: list[dict[str, Any]] = []
    for runner, group in last_year_laps.groupby("runner", sort=False):
        lap_times = group["lap_time_minutes"].astype(float)
        mean = float(lap_times.mean())
        sd = float(lap_times.std(ddof=1)) if len(group) > 1 else np.nan
        cv = sd / mean if mean and not np.isnan(sd) else np.nan

        day = group.loc[~group["is_night"].astype(bool), "lap_time_minutes"]
        night = group.loc[group["is_night"].astype(bool), "lap_time_minutes"]
        day_mean = float(day.mean()) if not day.empty else np.nan
        night_mean = float(night.mean()) if not night.empty else np.nan
        night_slowdown_pct = (
            (night_mean / day_mean - 1) * 100 if day_mean and not np.isnan(day_mean) and not np.isnan(night_mean) else np.nan
        )

        fatigue_slope = np.nan
        if len(group) >= 2:
            x = pd.to_numeric(group["lap_number_for_runner"], errors="coerce").to_numpy(dtype=float)
            y = lap_times.to_numpy(dtype=float)
            if np.isfinite(x).sum() >= 2:
                fatigue_slope = float(np.polyfit(x, y, 1)[0])

        rows.append(
            {
                "runner": runner,
                "last_year_laps": int(len(group)),
                "last_year_mean_minutes": mean,
                "last_year_sd_minutes": sd,
                "last_year_cv": cv,
                "fastest_lap_minutes": float(lap_times.min()),
                "slowest_lap_minutes": float(lap_times.max()),
                "day_mean_minutes": day_mean,
                "night_mean_minutes": night_mean,
                "night_slowdown_pct": night_slowdown_pct,
                "fatigue_slope_minutes_per_runner_lap": fatigue_slope,
            }
        )

    stats = pd.DataFrame(rows).sort_values("last_year_mean_minutes").reset_index(drop=True)
    runner_cvs = stats["last_year_cv"].replace([np.inf, -np.inf], np.nan).dropna()
    team_average_runner_cv = float(runner_cvs.mean()) if not runner_cvs.empty else 0.05
    all_laps = last_year_laps["lap_time_minutes"].astype(float)
    team_mean = float(all_laps.mean())
    team_sd = float(all_laps.std(ddof=1))
    team_stats = {
        "total_laps": int(len(last_year_laps)),
        "total_time_minutes": float(all_laps.sum()),
        "team_mean_minutes": team_mean,
        "team_sd_minutes": team_sd,
        "team_cv": float(team_sd / team_mean) if team_mean else team_average_runner_cv,
        "team_average_runner_cv": team_average_runner_cv,
        "fastest_lap_minutes": float(all_laps.min()),
        "slowest_lap_minutes": float(all_laps.max()),
    }
    return stats, team_stats


def build_roster_defaults(
    last_year_stats: pd.DataFrame,
    team_stats: dict[str, Any],
    roster_input: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if roster_input is None or roster_input.empty:
        roster = pd.DataFrame(DEFAULT_ROSTER)
        roster["source"] = "brief default"
    else:
        roster = _clean_roster_input(roster_input)
        roster["source"] = "workbook"

    stats_by_runner = last_year_stats.set_index("runner") if not last_year_stats.empty else pd.DataFrame()
    default_by_runner = pd.DataFrame(DEFAULT_ROSTER).set_index("runner")

    # If a workbook roster exists, keep its runners but backfill the columns the app expects.
    for column in pd.DataFrame(DEFAULT_ROSTER).columns:
        if column not in roster.columns:
            roster[column] = np.nan

    roster["runner"] = roster["runner"].map(normalize_runner_name)
    roster = roster.dropna(subset=["runner"]).drop_duplicates("runner", keep="first").copy()

    for runner, default_row in default_by_runner.iterrows():
        if runner not in set(roster["runner"]):
            roster = pd.concat([roster, pd.DataFrame([{**default_row.to_dict(), "runner": runner, "source": "brief default"}])])

    roster = roster.reset_index(drop=True)
    roster["runner"] = roster["runner"].map(normalize_runner_name)

    roster["projected_mean_minutes"] = pd.to_numeric(roster["projected_mean_minutes"], errors="coerce")
    roster["projected_sd_minutes"] = pd.to_numeric(roster["projected_sd_minutes"], errors="coerce")
    roster["running_order"] = pd.to_numeric(roster["running_order"], errors="coerce")
    roster["max_laps"] = pd.to_numeric(roster["max_laps"], errors="coerce")
    roster["available"] = roster["available"].map(_to_bool).fillna(True)
    roster["fatigue_pct_per_lap"] = pd.to_numeric(roster["fatigue_pct_per_lap"], errors="coerce").fillna(0.0)
    roster["night_penalty_pct"] = pd.to_numeric(roster["night_penalty_pct"], errors="coerce").fillna(0.0)
    roster["notes"] = roster["notes"].fillna("").astype(str)

    # Team-specific acceptance defaults.
    roster.loc[roster["runner"] == "Toby", "projected_mean_minutes"] = 33.0
    roster.loc[roster["runner"] == "Alex", "projected_mean_minutes"] = 34.0
    roster.loc[roster["runner"] == "Jared", "max_laps"] = roster.loc[
        roster["runner"] == "Jared", "max_laps"
    ].fillna(4)
    roster.loc[roster["runner"] == "Becs", "max_laps"] = roster.loc[roster["runner"] == "Becs", "max_laps"].fillna(4)

    for idx, row in roster.iterrows():
        runner = row["runner"]
        if pd.isna(row["projected_mean_minutes"]):
            roster.at[idx, "projected_mean_minutes"] = _default_mean_for_runner(runner, stats_by_runner, team_stats)

    for idx, row in roster.iterrows():
        if pd.isna(row["projected_sd_minutes"]) or float(row["projected_sd_minutes"]) <= 0:
            roster.at[idx, "projected_sd_minutes"] = _estimate_sd_for_runner(
                row["runner"],
                float(row["projected_mean_minutes"]),
                stats_by_runner,
                team_stats,
            )

    toby_sd = roster.loc[roster["runner"] == "Toby", "projected_sd_minutes"]
    if not toby_sd.empty and np.isfinite(float(toby_sd.iloc[0])):
        roster.loc[roster["runner"] == "Alex", "projected_sd_minutes"] = float(toby_sd.iloc[0])

    if roster["running_order"].isna().any():
        order_defaults = pd.DataFrame(DEFAULT_ROSTER)[["runner", "running_order"]].set_index("runner")["running_order"]
        roster["running_order"] = roster.apply(
            lambda row: order_defaults.get(row["runner"], np.nan)
            if pd.isna(row["running_order"])
            else row["running_order"],
            axis=1,
        )
    if roster["running_order"].isna().any():
        missing = roster["running_order"].isna()
        roster.loc[missing, "running_order"] = np.arange(1, missing.sum() + 1) + roster["running_order"].max(skipna=True)

    roster.loc[roster["max_laps"].fillna(-1).astype(float) == 0, "available"] = False

    ordered_columns = [
        "runner",
        "projected_mean_minutes",
        "projected_sd_minutes",
        "running_order",
        "max_laps",
        "available",
        "fatigue_pct_per_lap",
        "night_penalty_pct",
        "notes",
        "source",
    ]
    return roster[ordered_columns].sort_values("running_order").reset_index(drop=True)


def validate_roster(roster: pd.DataFrame) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    if roster.empty:
        errors.append("The roster is empty.")
        return warnings, errors

    required = ["runner", "projected_mean_minutes", "projected_sd_minutes", "running_order", "available"]
    for column in required:
        if column not in roster.columns:
            errors.append(f"Roster is missing required column: {column}.")

    bad_mean = roster[pd.to_numeric(roster["projected_mean_minutes"], errors="coerce") <= 0]
    if not bad_mean.empty:
        errors.append("Every available runner needs a positive projected mean lap time.")

    bad_sd = roster[pd.to_numeric(roster["projected_sd_minutes"], errors="coerce") <= 0]
    if not bad_sd.empty:
        warnings.append("Some projected standard deviations are non-positive; the simulator will use a fallback SD.")

    if "Toby" in set(roster["runner"]):
        toby_mean = float(roster.loc[roster["runner"] == "Toby", "projected_mean_minutes"].iloc[0])
        if abs(toby_mean - 33.0) > 1e-9:
            warnings.append("Toby's mean should default to 33.0 minutes.")

    if "Alex" in set(roster["runner"]) and "Toby" in set(roster["runner"]):
        alex_sd = float(roster.loc[roster["runner"] == "Alex", "projected_sd_minutes"].iloc[0])
        toby_sd = float(roster.loc[roster["runner"] == "Toby", "projected_sd_minutes"].iloc[0])
        if abs(alex_sd - toby_sd) > 1e-6:
            warnings.append("Alex defaults to Toby's projected SD, but it has been manually changed.")

    return warnings, errors


def parse_lap_minutes(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, pd.Timestamp):
        return float(value.hour * 60 + value.minute + value.second / 60 + value.microsecond / 60_000_000)
    if isinstance(value, dt.datetime):
        return float(value.hour * 60 + value.minute + value.second / 60 + value.microsecond / 60_000_000)
    if isinstance(value, dt.time):
        return float(value.hour * 60 + value.minute + value.second / 60 + value.microsecond / 60_000_000)
    if isinstance(value, dt.timedelta):
        return value.total_seconds() / 60
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if 0 < number < 1.5:
            return number * 24 * 60
        return number

    text = str(value).strip()
    if not text:
        return np.nan
    text = text.replace(",", ".")
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return parse_lap_minutes(float(text))

    parts = text.split(":")
    try:
        nums = [float(part) for part in parts]
    except ValueError:
        parsed = pd.to_timedelta(text, errors="coerce")
        if pd.notna(parsed):
            return float(parsed.total_seconds() / 60)
        return np.nan

    if len(nums) == 2:
        minutes, seconds = nums
        return minutes + seconds / 60
    if len(nums) == 3:
        hours, minutes, seconds = nums
        return hours * 60 + minutes + seconds / 60
    return np.nan


def normalize_runner_name(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return None
    key = text.casefold()
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    return text.title()


def _read_source_bytes(source: str | Path | bytes) -> bytes:
    if isinstance(source, bytes):
        return source
    if hasattr(source, "getvalue"):
        return source.getvalue()
    return Path(source).read_bytes()


def _sheet_info(raw_sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sheet_name, raw in raw_sheets.items():
        candidate_headers = []
        for _, row in raw.head(8).iterrows():
            non_blank = [str(x).strip() for x in row.tolist() if pd.notna(x) and str(x).strip()]
            if len(non_blank) >= 2:
                candidate_headers = non_blank
                break
        rows.append(
            {
                "sheet": sheet_name,
                "rows": int(raw.shape[0]),
                "columns": int(raw.shape[1]),
                "detected_header_preview": ", ".join(candidate_headers[:12]),
            }
        )
    return pd.DataFrame(rows)


def _find_clean_last_year_sheet(workbook_bytes: bytes) -> pd.DataFrame | None:
    sheets = pd.read_excel(BytesIO(workbook_bytes), sheet_name=None, engine="openpyxl")
    for sheet_name, df in sheets.items():
        normalized_sheet = _normalize_token(sheet_name)
        renamed = df.copy()
        renamed.columns = [_canonical_column_name(c) for c in renamed.columns]
        mapped = _map_columns(renamed.columns)
        if mapped.get("runner") and (mapped.get("lap_time_minutes") or mapped.get("lap_time")):
            if "roster" not in normalized_sheet and "runner" in "".join(renamed.columns):
                return renamed
    return None


def _find_clean_roster_sheet(workbook_bytes: bytes) -> pd.DataFrame | None:
    sheets = pd.read_excel(BytesIO(workbook_bytes), sheet_name=None, engine="openpyxl")
    for sheet_name, df in sheets.items():
        normalized_sheet = _normalize_token(sheet_name)
        renamed = df.copy()
        renamed.columns = [_canonical_column_name(c) for c in renamed.columns]
        mapped = _map_roster_columns(renamed.columns)
        if mapped.get("runner") and (mapped.get("projected_mean_minutes") or "roster" in normalized_sheet):
            return renamed.rename(columns={source: target for target, source in mapped.items() if source})
    return None


def _find_dashboard_lap_table(raw_sheets: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    for sheet_name, raw in raw_sheets.items():
        for row_idx in range(min(len(raw), 30)):
            row = raw.iloc[row_idx].tolist()
            normalized = [_normalize_token(value) for value in row]
            for start_col, token in enumerate(normalized):
                if token != "lap":
                    continue
                window = normalized[start_col : start_col + 8]
                if "runner" in window and ("lapminutes" in window or "lapmin" in window) and (
                    "laptime" in window or "cumulativeminutes" in window
                ):
                    headers = []
                    for value in row[start_col:]:
                        if pd.isna(value) or not str(value).strip():
                            break
                        headers.append(_canonical_column_name(value))
                    if len(headers) < 3:
                        continue
                    end_row = row_idx + 1
                    while end_row < len(raw):
                        first = raw.iat[end_row, start_col]
                        runner = raw.iat[end_row, start_col + 1] if start_col + 1 < raw.shape[1] else None
                        if (pd.isna(first) or str(first).strip() == "") and (
                            pd.isna(runner) or str(runner).strip() == ""
                        ):
                            break
                        end_row += 1

                    table = raw.iloc[row_idx + 1 : end_row, start_col : start_col + len(headers)].copy()
                    table.columns = headers
                    table["source_sheet"] = sheet_name
                    return table
    return None


def _clean_roster_input(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [_canonical_column_name(c) for c in clean.columns]
    mapped = _map_roster_columns(clean.columns)
    clean = clean.rename(columns={source: target for target, source in mapped.items() if source})
    if "runner" not in clean.columns:
        return pd.DataFrame(DEFAULT_ROSTER)
    return clean


def _map_columns(columns: list[str] | pd.Index) -> dict[str, str | None]:
    columns = list(columns)

    def pick(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    return {
        "lap": pick("lap", "lapnumber", "officiallap"),
        "runner": pick("runner", "name", "athlete"),
        "lap_time": pick("laptime", "split", "time"),
        "lap_time_minutes": pick("laptimeminutes", "lapminutes", "lapmin", "minutes"),
        "start_time": pick("starttime", "lapstarttime"),
        "start_minute": pick("startminute", "startminutes", "lapstartminute"),
        "lap_number_for_runner": pick("lapnumberforrunner", "runnerlap", "runnerlapnumber"),
        "is_night": pick("isnight", "night", "nightlap"),
    }


def _map_roster_columns(columns: list[str] | pd.Index) -> dict[str, str | None]:
    columns = list(columns)

    def pick(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    return {
        "runner": pick("runner", "name", "athlete"),
        "projected_mean_minutes": pick("projectedmeanminutes", "projectedmean", "meanminutes", "meanlapminutes"),
        "projected_sd_minutes": pick("projectedsdminutes", "projectedsd", "sdminutes", "stddevminutes"),
        "running_order": pick("runningorder", "order", "position"),
        "max_laps": pick("maxlaps", "lapcap", "cap"),
        "available": pick("available", "isavailable"),
        "fatigue_pct_per_lap": pick("fatiguepctperlap", "fatiguepercentperlap", "fatigue"),
        "night_penalty_pct": pick("nightpenaltypct", "nightpenaltypercent", "nightpenalty"),
        "notes": pick("notes", "note", "comments"),
    }


def _canonical_column_name(value: Any) -> str:
    return _normalize_token(value)


def _normalize_token(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _to_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "yes", "y", "1", "available"}:
        return True
    if text in {"false", "no", "n", "0", "unavailable"}:
        return False
    return None


def _default_mean_for_runner(runner: str, stats_by_runner: pd.DataFrame, team_stats: dict[str, Any]) -> float:
    if runner in stats_by_runner.index:
        return float(stats_by_runner.loc[runner, "last_year_mean_minutes"])
    defaults = pd.DataFrame(DEFAULT_ROSTER).set_index("runner")
    if runner in defaults.index and pd.notna(defaults.loc[runner, "projected_mean_minutes"]):
        return float(defaults.loc[runner, "projected_mean_minutes"])
    return float(team_stats.get("team_mean_minutes") or 40.0)


def _estimate_sd_for_runner(
    runner: str,
    projected_mean_minutes: float,
    stats_by_runner: pd.DataFrame,
    team_stats: dict[str, Any],
) -> float:
    team_cv = float(team_stats.get("team_average_runner_cv") or team_stats.get("team_cv") or 0.05)
    if runner in stats_by_runner.index:
        sd = stats_by_runner.loc[runner, "last_year_sd_minutes"]
        cv = stats_by_runner.loc[runner, "last_year_cv"]
        if pd.notna(sd) and float(sd) > 0:
            return float(projected_mean_minutes * float(cv))
    return float(projected_mean_minutes * team_cv)


def _build_name_map(last_year_laps: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not last_year_laps.empty:
        for original, normalized in (
            last_year_laps[["runner_raw", "runner"]].drop_duplicates().sort_values("runner").itertuples(index=False)
        ):
            rows.append({"source": "last_year_laps", "original": original, "normalised": normalized})
    for runner in roster["runner"].dropna().drop_duplicates():
        rows.append({"source": "this_year_roster", "original": runner, "normalised": normalize_runner_name(runner)})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
