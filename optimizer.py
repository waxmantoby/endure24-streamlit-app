"""Running-order search for the Endure24 simulator."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations
import math
from typing import Any

import numpy as np
import pandas as pd

from simulation_engine import SimulationSettings, simulate_many, summarize_results


def optimize_running_orders(
    roster: pd.DataFrame,
    settings: SimulationSettings,
    target_laps: int,
    objective: str = "max_expected_laps",
    max_candidates: int = 200,
    sims_per_order: int = 400,
    final_sims: int = 1_500,
    top_n: int = 10,
    brute_force_if_feasible: bool = False,
    locked_positions: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Rank running orders for the selected competition objective."""

    locked_positions = locked_positions or {}
    names = _available_runner_names(roster)
    current_order = _current_order(roster, names)
    candidate_orders = _candidate_orders(
        names,
        current_order,
        max_candidates=max_candidates,
        brute_force_if_feasible=brute_force_if_feasible,
        locked_positions=locked_positions,
        rng=np.random.default_rng(settings.random_seed),
    )

    first_pass_rows = []
    for idx, order in enumerate(candidate_orders):
        seed = None if settings.random_seed is None else settings.random_seed + idx * 17
        order_settings = replace(settings, n_simulations=int(sims_per_order), random_seed=seed, path_sample_size=0)
        first_pass_rows.append(_evaluate_order(roster, order, order_settings, target_laps, current_order))

    first_pass = _rank(pd.DataFrame(first_pass_rows), objective)
    rerun_orders = first_pass.head(min(20, len(first_pass)))["order_list"].tolist()

    final_rows = []
    for idx, order in enumerate(rerun_orders):
        seed = None if settings.random_seed is None else settings.random_seed + 10_000 + idx * 31
        order_settings = replace(settings, n_simulations=int(final_sims), random_seed=seed, path_sample_size=0)
        final_rows.append(_evaluate_order(roster, order, order_settings, target_laps, current_order))

    final = _rank(pd.DataFrame(final_rows), objective).head(top_n).reset_index(drop=True)
    final.insert(0, "rank", np.arange(1, len(final) + 1))
    return final.drop(columns=["order_list"])


def _evaluate_order(
    roster: pd.DataFrame,
    order: list[str],
    settings: SimulationSettings,
    target_laps: int,
    current_order: list[str],
) -> dict[str, Any]:
    ordered_roster = _apply_order(roster, order)
    result = simulate_many(ordered_roster, settings, caps_enabled=True)
    summary_bundle = summarize_results(result, target_laps)
    summary = summary_bundle["summary"]
    final_runner_probs = summary_bundle["final_runner_probs"]
    likely_final_runner = ""
    if not final_runner_probs.empty:
        likely_final_runner = str(final_runner_probs.iloc[0]["runner"])

    cap_rates = summary_bundle["cap_hit_rates"].copy()
    cap_text = ""
    if not cap_rates.empty:
        nonzero = cap_rates[cap_rates["cap_hit_probability"] > 0.01].head(4)
        cap_text = ", ".join(
            f"{row.runner}: {row.cap_hit_probability:.0%}" for row in nonzero.itertuples(index=False)
        )

    return {
        "order": " -> ".join(order),
        "order_list": order,
        "is_current_order": order == current_order,
        "expected_official_laps": summary["expected_official_laps"],
        "median_official_laps": summary["median_official_laps"],
        "p10_official_laps": summary["p10_official_laps"],
        "p90_official_laps": summary["p90_official_laps"],
        "probability_target_laps": summary["probability_target_laps"],
        "probability_target_plus_one_laps": summary["probability_target_plus_one_laps"],
        "probability_missed_final_cutoff": summary["probability_missed_final_cutoff"],
        "probability_ran_out_of_eligible_runners": summary["probability_ran_out_of_eligible_runners"],
        "likely_final_runner": likely_final_runner,
        "cap_hit_highlights": cap_text,
    }


def _rank(df: pd.DataFrame, objective: str) -> pd.DataFrame:
    if df.empty:
        return df
    if objective == "target_probability":
        sort_columns = [
            "probability_target_laps",
            "expected_official_laps",
            "probability_target_plus_one_laps",
            "probability_missed_final_cutoff",
            "probability_ran_out_of_eligible_runners",
        ]
        ascending = [False, False, False, True, True]
    else:
        sort_columns = [
            "expected_official_laps",
            "median_official_laps",
            "p10_official_laps",
            "probability_target_laps",
            "probability_missed_final_cutoff",
            "probability_ran_out_of_eligible_runners",
        ]
        ascending = [False, False, False, False, True, True]
    return df.sort_values(
        sort_columns,
        ascending=ascending,
    ).reset_index(drop=True)


def _available_runner_names(roster: pd.DataFrame) -> list[str]:
    clean = roster.copy()
    clean["available"] = clean["available"].astype(bool)
    clean["max_laps"] = pd.to_numeric(clean["max_laps"], errors="coerce")
    clean = clean[clean["available"] & (clean["max_laps"].isna() | clean["max_laps"].ne(0))]
    return clean.sort_values(["running_order", "runner"])["runner"].astype(str).tolist()


def _current_order(roster: pd.DataFrame, names: list[str]) -> list[str]:
    current = roster[roster["runner"].isin(names)].sort_values(["running_order", "runner"])["runner"].astype(str).tolist()
    return current or names


def _apply_order(roster: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    ordered = roster.copy()
    order_lookup = {runner: idx + 1 for idx, runner in enumerate(order)}
    max_order = len(order_lookup)
    ordered["running_order"] = ordered.apply(
        lambda row: order_lookup.get(str(row["runner"]), max_order + float(row.get("running_order", max_order + 1))),
        axis=1,
    )
    return ordered.sort_values(["running_order", "runner"]).reset_index(drop=True)


def _candidate_orders(
    names: list[str],
    current_order: list[str],
    max_candidates: int,
    brute_force_if_feasible: bool,
    locked_positions: dict[int, str],
    rng: np.random.Generator,
) -> list[list[str]]:
    max_candidates = max(1, int(max_candidates))
    valid_locks = {
        int(position): runner
        for position, runner in locked_positions.items()
        if runner in names and 1 <= int(position) <= len(names)
    }
    locked_runners = set(valid_locks.values())
    free_names = [name for name in names if name not in locked_runners]

    orders: list[tuple[str, ...]] = []
    current_tuple = tuple(current_order)
    if _respects_locks(current_tuple, valid_locks):
        orders.append(current_tuple)

    possible_free_perms = math.factorial(len(free_names))
    if brute_force_if_feasible and possible_free_perms <= max_candidates:
        for perm in permutations(free_names):
            orders.append(tuple(_insert_locks(list(perm), valid_locks, len(names))))
    else:
        attempts = 0
        while len(set(orders)) < max_candidates and attempts < max_candidates * 50:
            attempts += 1
            perm = list(rng.permutation(free_names))
            orders.append(tuple(_insert_locks(perm, valid_locks, len(names))))

    deduped = list(dict.fromkeys(orders))
    return [list(order) for order in deduped[:max_candidates]]


def _insert_locks(free_order: list[str], locked_positions: dict[int, str], n: int) -> list[str]:
    order: list[str | None] = [None] * n
    for position, runner in locked_positions.items():
        order[position - 1] = runner
    free_iter = iter(free_order)
    for idx in range(n):
        if order[idx] is None:
            order[idx] = next(free_iter)
    return [str(item) for item in order if item is not None]


def _respects_locks(order: tuple[str, ...], locked_positions: dict[int, str]) -> bool:
    for position, runner in locked_positions.items():
        if position - 1 >= len(order) or order[position - 1] != runner:
            return False
    return True
