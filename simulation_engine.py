"""Team-agnostic Monte Carlo engine for Endure24 relay simulations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import truncnorm
except Exception:  # pragma: no cover - the NumPy fallback is used when SciPy is unavailable.
    truncnorm = None


LAST_START_MINUTE = 24 * 60
FINAL_CUTOFF_MINUTE = 25 * 60
FINAL_HOUR_START_MINUTE = 22.5 * 60
FINAL_HOUR_END_MINUTE = 24 * 60


@dataclass(frozen=True)
class SimulationSettings:
    n_simulations: int = 10_000
    target_laps: int = 40
    random_seed: int | None = 42
    night_start_minute: float = 9 * 60
    night_end_minute: float = 18 * 60
    distribution: str = "truncated_normal"
    fatigue_enabled: bool = True
    night_penalty_enabled: bool = True
    transition_mistakes_enabled: bool = True
    base_transition_seconds: float = 0.0
    transition_mistake_probability_per_lap: float = 0.01
    transition_mistake_min_seconds: float = 30.0
    transition_mistake_max_seconds: float = 300.0
    major_mistake_probability_per_lap: float = 0.002
    major_mistake_min_seconds: float = 300.0
    major_mistake_max_seconds: float = 900.0
    hard_min_lap_minutes: float = 15.0
    path_sample_size: int = 120


def simulate_many(
    roster: pd.DataFrame,
    settings: SimulationSettings,
    caps_enabled: bool = True,
) -> dict[str, Any]:
    """Run many independent relay simulations."""

    runners = _prepare_roster(roster)
    if not runners:
        raise ValueError("No runners are available to simulate.")

    n = int(settings.n_simulations)
    rng = np.random.default_rng(settings.random_seed)
    simulation_rows: list[dict[str, Any]] = []
    runner_count_rows: list[dict[str, Any]] = []
    cap_hit_rows: list[dict[str, Any]] = []
    final_hour_rows: list[dict[str, Any]] = []
    path_sample_rows: list[dict[str, Any]] = []

    time_grid = np.linspace(0, FINAL_CUTOFF_MINUTE, 61)
    path_grid = np.zeros((n, len(time_grid)), dtype=float)

    for sim_id in range(n):
        result = _simulate_one(runners, settings, rng, caps_enabled=caps_enabled)

        simulation_rows.append(
            {
                "simulation": sim_id + 1,
                "official_laps": result["official_laps"],
                "laps_by_noon": result["laps_by_noon"],
                "final_elapsed_minute": result["final_elapsed_minute"],
                "final_lap_start_minute": result["final_lap_start_minute"],
                "final_lap_finish_minute": result["final_lap_finish_minute"],
                "final_runner": result["final_runner"],
                "squeezed_final_lap_after_noon": result["squeezed_final_lap_after_noon"],
                "missed_final_cutoff": result["missed_final_cutoff"],
                "missed_start_cutoff": result["missed_start_cutoff"],
                "ran_out_of_eligible_runners": result["ran_out_of_eligible_runners"],
                "stop_reason": result["stop_reason"],
            }
        )

        runner_row = {"simulation": sim_id + 1}
        runner_row.update(result["runner_lap_counts"])
        runner_count_rows.append(runner_row)

        cap_row = {"simulation": sim_id + 1}
        cap_row.update(result["cap_hits"])
        cap_hit_rows.append(cap_row)

        for runner in result["final_hour_runners"]:
            final_hour_rows.append({"simulation": sim_id + 1, "runner": runner})

        finish_times = np.array(result["official_finish_times"], dtype=float)
        if finish_times.size:
            path_grid[sim_id, :] = np.searchsorted(finish_times, time_grid, side="right")
        if sim_id < settings.path_sample_size:
            path_sample_rows.extend(_path_points_to_rows(sim_id + 1, result["path_points"]))

    simulations = pd.DataFrame(simulation_rows)
    runner_laps = pd.DataFrame(runner_count_rows).fillna(0)
    cap_hits = pd.DataFrame(cap_hit_rows).fillna(False)
    final_hour = pd.DataFrame(final_hour_rows, columns=["simulation", "runner"])
    path_samples = pd.DataFrame(path_sample_rows, columns=["simulation", "time_minute", "completed_laps"])
    path_quantiles = pd.DataFrame(
        {
            "time_minute": time_grid,
            "p10": np.percentile(path_grid, 10, axis=0),
            "median": np.percentile(path_grid, 50, axis=0),
            "p90": np.percentile(path_grid, 90, axis=0),
        }
    )

    return {
        "settings": settings,
        "caps_enabled": caps_enabled,
        "simulations": simulations,
        "runner_laps": runner_laps,
        "cap_hits": cap_hits,
        "final_hour": final_hour,
        "path_samples": path_samples,
        "path_quantiles": path_quantiles,
    }


def summarize_results(sim_output: dict[str, Any], target_laps: int | None = None) -> dict[str, Any]:
    simulations = sim_output["simulations"]
    runner_laps = sim_output["runner_laps"]
    cap_hits = sim_output["cap_hits"]
    final_hour = sim_output["final_hour"]
    target = target_laps if target_laps is not None else sim_output["settings"].target_laps

    official = simulations["official_laps"].astype(float)
    counts = (
        official.astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("official_laps")
        .reset_index(name="simulations")
    )
    counts["probability"] = counts["simulations"] / len(simulations)
    counts["probability_at_least"] = counts.sort_values("official_laps", ascending=False)["probability"].cumsum().sort_index()

    mode_laps = int(counts.sort_values(["simulations", "official_laps"], ascending=[False, True]).iloc[0]["official_laps"])
    summary = {
        "simulations": int(len(simulations)),
        "expected_official_laps": float(official.mean()),
        "median_official_laps": float(official.median()),
        "mode_official_laps": mode_laps,
        "p05_official_laps": float(np.percentile(official, 5)),
        "p10_official_laps": float(np.percentile(official, 10)),
        "p90_official_laps": float(np.percentile(official, 90)),
        "p95_official_laps": float(np.percentile(official, 95)),
        "target_laps": int(target),
        "probability_target_laps": float((official >= target).mean()),
        "probability_target_plus_one_laps": float((official >= target + 1).mean()),
        "expected_laps_by_noon": float(simulations["laps_by_noon"].mean()),
        "probability_squeezed_final_lap_after_noon": float(simulations["squeezed_final_lap_after_noon"].mean()),
        "probability_missed_final_cutoff": float(simulations["missed_final_cutoff"].mean()),
        "probability_ran_out_of_eligible_runners": float(simulations["ran_out_of_eligible_runners"].mean()),
        "average_final_finish_minute": float(simulations["final_elapsed_minute"].dropna().mean()),
    }

    runner_columns = [c for c in runner_laps.columns if c != "simulation"]
    expected_runner_laps = (
        runner_laps[runner_columns]
        .mean()
        .rename_axis("runner")
        .reset_index(name="expected_laps")
        .sort_values("expected_laps", ascending=False)
    )

    cap_columns = [c for c in cap_hits.columns if c != "simulation"]
    if cap_columns:
        cap_hit_rates = (
            cap_hits[cap_columns]
            .astype(float)
            .mean()
            .rename_axis("runner")
            .reset_index(name="cap_hit_probability")
            .sort_values("cap_hit_probability", ascending=False)
        )
    else:
        cap_hit_rates = pd.DataFrame(columns=["runner", "cap_hit_probability"])

    final_runner_probs = (
        simulations["final_runner"]
        .dropna()
        .value_counts(normalize=True)
        .rename_axis("runner")
        .reset_index(name="probability")
    )

    if final_hour.empty:
        final_hour_probs = pd.DataFrame(columns=["runner", "probability"])
    else:
        final_hour_probs = (
            final_hour.drop_duplicates(["simulation", "runner"])["runner"]
            .value_counts()
            .rename_axis("runner")
            .reset_index(name="count")
        )
        final_hour_probs["probability"] = final_hour_probs["count"] / len(simulations)
        final_hour_probs = final_hour_probs.drop(columns=["count"])

    return {
        "summary": summary,
        "final_lap_distribution": counts,
        "expected_runner_laps": expected_runner_laps,
        "cap_hit_rates": cap_hit_rates,
        "final_runner_probs": final_runner_probs,
        "final_hour_runner_probs": final_hour_probs,
    }


def compare_capped_vs_uncapped(
    roster: pd.DataFrame,
    settings: SimulationSettings,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    capped = simulate_many(roster, settings, caps_enabled=True)
    uncapped_settings = replace(settings, random_seed=None if settings.random_seed is None else settings.random_seed + 1)
    uncapped = simulate_many(roster, uncapped_settings, caps_enabled=False)
    capped_summary = summarize_results(capped, settings.target_laps)
    uncapped_summary = summarize_results(uncapped, settings.target_laps)
    comparison = pd.DataFrame(
        [
            {
                "scenario": "Caps enabled",
                "expected_official_laps": capped_summary["summary"]["expected_official_laps"],
                "probability_target_laps": capped_summary["summary"]["probability_target_laps"],
                "probability_ran_out_of_eligible_runners": capped_summary["summary"][
                    "probability_ran_out_of_eligible_runners"
                ],
            },
            {
                "scenario": "Caps disabled",
                "expected_official_laps": uncapped_summary["summary"]["expected_official_laps"],
                "probability_target_laps": uncapped_summary["summary"]["probability_target_laps"],
                "probability_ran_out_of_eligible_runners": uncapped_summary["summary"][
                    "probability_ran_out_of_eligible_runners"
                ],
            },
        ]
    )
    comparison["expected_lap_impact_vs_uncapped"] = (
        comparison["expected_official_laps"] - comparison.loc[comparison["scenario"] == "Caps disabled", "expected_official_laps"].iloc[0]
    )
    return capped, uncapped, comparison


def _simulate_one(
    runners: list[dict[str, Any]],
    settings: SimulationSettings,
    rng: np.random.Generator,
    caps_enabled: bool,
) -> dict[str, Any]:
    elapsed = 0.0
    pointer = 0
    official_laps = 0
    laps_by_noon = 0
    missed_final_cutoff = False
    missed_start_cutoff = False
    ran_out = False
    stop_reason = "race_window_complete"

    runner_lap_counts = {runner["runner"]: 0 for runner in runners}
    cap_hits = {runner["runner"]: False for runner in runners if _finite_max_laps(runner) is not None}
    official_finish_times: list[float] = []
    path_points: list[tuple[float, int]] = [(0.0, 0)]
    final_hour_runners: set[str] = set()
    final_runner = None
    final_lap_start = np.nan
    final_lap_finish = np.nan

    while True:
        eligible_idx, next_pointer = _next_eligible_runner_index(runners, pointer, runner_lap_counts, caps_enabled)
        if eligible_idx is None:
            ran_out = True
            stop_reason = "ran_out_of_eligible_runners"
            break
        pointer = next_pointer
        runner = runners[eligible_idx]

        transition_delay = _sample_transition_delay(settings, rng)
        lap_start = elapsed + transition_delay
        if lap_start > LAST_START_MINUTE:
            missed_start_cutoff = True
            stop_reason = "start_after_sunday_noon"
            break

        previous_runner_laps = runner_lap_counts[runner["runner"]]
        night = is_night_minute(lap_start, settings.night_start_minute, settings.night_end_minute)
        run_minutes = _sample_run_minutes(runner, previous_runner_laps, night, settings, rng)
        lap_finish = lap_start + run_minutes

        # Endure24 cutoff logic: the runner must cross the start line by Sunday
        # 12:00, then finish that lap by Sunday 13:00 for it to count.
        if lap_start <= LAST_START_MINUTE and lap_finish <= FINAL_CUTOFF_MINUTE:
            official_laps += 1
            if lap_finish <= LAST_START_MINUTE:
                laps_by_noon += 1
            runner_lap_counts[runner["runner"]] += 1
            final_runner = runner["runner"]
            final_lap_start = lap_start
            final_lap_finish = lap_finish
            elapsed = lap_finish
            official_finish_times.append(lap_finish)
            path_points.append((lap_finish, official_laps))

            max_laps = _finite_max_laps(runner)
            if caps_enabled and max_laps is not None and runner_lap_counts[runner["runner"]] >= max_laps:
                cap_hits[runner["runner"]] = True

            if _overlaps(lap_start, lap_finish, FINAL_HOUR_START_MINUTE, FINAL_HOUR_END_MINUTE):
                final_hour_runners.add(runner["runner"])
        else:
            missed_final_cutoff = True
            stop_reason = "finish_after_sunday_13"
            break

    squeezed_after_noon = bool(
        official_laps > 0
        and np.isfinite(final_lap_start)
        and final_lap_start <= LAST_START_MINUTE
        and final_lap_finish > LAST_START_MINUTE
        and final_lap_finish <= FINAL_CUTOFF_MINUTE
    )

    return {
        "official_laps": official_laps,
        "laps_by_noon": laps_by_noon,
        "final_elapsed_minute": final_lap_finish if np.isfinite(final_lap_finish) else elapsed,
        "final_lap_start_minute": final_lap_start,
        "final_lap_finish_minute": final_lap_finish,
        "final_runner": final_runner,
        "squeezed_final_lap_after_noon": squeezed_after_noon,
        "missed_final_cutoff": missed_final_cutoff,
        "missed_start_cutoff": missed_start_cutoff,
        "ran_out_of_eligible_runners": ran_out,
        "stop_reason": stop_reason,
        "runner_lap_counts": runner_lap_counts,
        "cap_hits": cap_hits,
        "official_finish_times": official_finish_times,
        "path_points": path_points,
        "final_hour_runners": sorted(final_hour_runners),
    }


def is_night_minute(minute: float, night_start_minute: float, night_end_minute: float) -> bool:
    minute = float(minute)
    start = float(night_start_minute)
    end = float(night_end_minute)
    if start <= end:
        return start <= minute < end
    return minute >= start or minute < end


def _prepare_roster(roster: pd.DataFrame) -> list[dict[str, Any]]:
    required = {
        "runner",
        "projected_mean_minutes",
        "projected_sd_minutes",
        "running_order",
        "max_laps",
        "available",
        "fatigue_pct_per_lap",
        "night_penalty_pct",
    }
    missing = required.difference(roster.columns)
    if missing:
        raise ValueError(f"Roster is missing required columns: {', '.join(sorted(missing))}")

    clean = roster.copy()
    clean["runner"] = clean["runner"].astype(str).str.strip()
    clean["projected_mean_minutes"] = pd.to_numeric(clean["projected_mean_minutes"], errors="coerce")
    clean["projected_sd_minutes"] = pd.to_numeric(clean["projected_sd_minutes"], errors="coerce")
    clean["running_order"] = pd.to_numeric(clean["running_order"], errors="coerce")
    clean["max_laps"] = pd.to_numeric(clean["max_laps"], errors="coerce")
    clean["available"] = clean["available"].map(_to_bool).fillna(True)
    clean["fatigue_pct_per_lap"] = pd.to_numeric(clean["fatigue_pct_per_lap"], errors="coerce").fillna(0)
    clean["night_penalty_pct"] = pd.to_numeric(clean["night_penalty_pct"], errors="coerce").fillna(0)
    clean = clean[clean["runner"].ne("")].copy()
    clean = clean[clean["projected_mean_minutes"].gt(0)].copy()
    clean = clean.sort_values(["running_order", "runner"], na_position="last").reset_index(drop=True)

    records = []
    for _, row in clean.iterrows():
        records.append(
            {
                "runner": row["runner"],
                "projected_mean_minutes": float(row["projected_mean_minutes"]),
                "projected_sd_minutes": _positive_float_or_default(
                    row["projected_sd_minutes"], float(row["projected_mean_minutes"]) * 0.05
                ),
                "running_order": float(row["running_order"]),
                "max_laps": None if pd.isna(row["max_laps"]) else int(row["max_laps"]),
                "available": bool(row["available"]) and (pd.isna(row["max_laps"]) or int(row["max_laps"]) != 0),
                "fatigue_pct_per_lap": float(row["fatigue_pct_per_lap"]),
                "night_penalty_pct": float(row["night_penalty_pct"]),
            }
        )
    return records


def _next_eligible_runner_index(
    runners: list[dict[str, Any]],
    pointer: int,
    runner_lap_counts: dict[str, int],
    caps_enabled: bool,
) -> tuple[int | None, int]:
    # Max-lap logic: capped or unavailable runners are skipped in the rotation.
    n = len(runners)
    for offset in range(n):
        idx = (pointer + offset) % n
        runner = runners[idx]
        if not runner["available"]:
            continue
        max_laps = _finite_max_laps(runner)
        if caps_enabled and max_laps is not None and runner_lap_counts[runner["runner"]] >= max_laps:
            continue
        return idx, (idx + 1) % n
    return None, pointer


def _sample_run_minutes(
    runner: dict[str, Any],
    previous_runner_laps: int,
    is_night: bool,
    settings: SimulationSettings,
    rng: np.random.Generator,
) -> float:
    fatigue_factor = (
        1 + _pct_to_fraction(runner["fatigue_pct_per_lap"]) * previous_runner_laps if settings.fatigue_enabled else 1.0
    )
    night_factor = (
        1 + _pct_to_fraction(runner["night_penalty_pct"]) if settings.night_penalty_enabled and is_night else 1.0
    )
    mean = runner["projected_mean_minutes"] * fatigue_factor * night_factor
    sd = max(float(runner["projected_sd_minutes"]), mean * 0.02, 0.25)
    lower = max(settings.hard_min_lap_minutes, mean - 3 * sd)
    upper = max(lower + 0.1, mean + 3 * sd)

    if settings.distribution == "lognormal":
        sample = _sample_lognormal(mean, sd, rng)
        return float(np.clip(sample, lower, upper))

    # This rejection sampler is intentionally used instead of per-lap
    # scipy.stats.truncnorm calls. The statistical behavior is the same for
    # these narrow bounds, and it keeps 10,000+ Streamlit runs responsive.
    for _ in range(30):
        sample = rng.normal(mean, sd)
        if lower <= sample <= upper:
            return float(sample)
    return float(np.clip(rng.normal(mean, sd), lower, upper))


def _sample_lognormal(mean: float, sd: float, rng: np.random.Generator) -> float:
    variance = sd**2
    sigma2 = np.log(1 + variance / (mean**2))
    mu = np.log(mean) - sigma2 / 2
    return float(rng.lognormal(mu, np.sqrt(sigma2)))


def _sample_transition_delay(settings: SimulationSettings, rng: np.random.Generator) -> float:
    delay = max(0.0, float(settings.base_transition_seconds)) / 60
    if settings.transition_mistakes_enabled:
        if rng.random() < _probability(settings.transition_mistake_probability_per_lap):
            delay += rng.uniform(settings.transition_mistake_min_seconds, settings.transition_mistake_max_seconds) / 60
        if rng.random() < _probability(settings.major_mistake_probability_per_lap):
            delay += rng.uniform(settings.major_mistake_min_seconds, settings.major_mistake_max_seconds) / 60
    return float(delay)


def _path_points_to_rows(simulation: int, path_points: list[tuple[float, int]]) -> list[dict[str, Any]]:
    rows = []
    for minute, completed in path_points:
        rows.append({"simulation": simulation, "time_minute": minute, "completed_laps": completed})
    if not path_points or path_points[-1][0] < FINAL_CUTOFF_MINUTE:
        rows.append(
            {
                "simulation": simulation,
                "time_minute": FINAL_CUTOFF_MINUTE,
                "completed_laps": path_points[-1][1] if path_points else 0,
            }
        )
    return rows


def _finite_max_laps(runner: dict[str, Any]) -> int | None:
    max_laps = runner.get("max_laps")
    if max_laps is None or pd.isna(max_laps):
        return None
    return int(max_laps)


def _overlaps(start: float, end: float, window_start: float, window_end: float) -> bool:
    return start < window_end and end > window_start


def _pct_to_fraction(value: float) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value) / 100


def _probability(value: float) -> float:
    if value is None or pd.isna(value):
        return 0.0
    value = float(value)
    if value > 1:
        return value / 100
    return max(0.0, min(1.0, value))


def _positive_float_or_default(value: Any, default: float) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or float(number) <= 0:
        return float(default)
    return float(number)


def _to_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "yes", "y", "1", "available"}:
        return True
    if text in {"false", "no", "n", "0", "unavailable"}:
        return False
    return None
