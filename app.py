from __future__ import annotations

from dataclasses import replace
import hashlib
import html
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover - optional dependency for live Sheet refresh.
    st_autorefresh = None

from data_loader import DEFAULT_DATA_PATH, load_endure_workbook, validate_roster
from optimizer import optimize_running_orders
import race_day as race_day_module

race_day_module = importlib.reload(race_day_module)
from race_day import (
    GOOGLE_SHEET_TEMPLATE_COLUMNS,
    append_manual_lap,
    build_progress_actuals,
    build_live_fair_queue,
    build_runner_queue,
    build_runner_status_table,
    complete_in_progress_lap,
    empty_race_log,
    forecast_from_race_log,
    google_sheets_configured,
    normalise_race_log,
    parse_duration_minutes,
    parse_pasted_laps,
    parse_race_time_to_minute,
    race_order_review,
    race_state_from_log,
    rank_live_next_runner_options,
    read_race_log_from_google_sheet_url,
    read_race_log_from_google_sheet,
    target_pace_series,
    validate_race_log,
    write_race_log_to_google_sheet,
)
from simulation_engine import (
    FINAL_CUTOFF_MINUTE,
    LAST_START_MINUTE,
    SimulationSettings,
    simulate_many,
    summarize_results,
)


st.set_page_config(page_title="Endure24 Relay Monte Carlo", layout="wide")

ASSUMPTION_VERSION = "zero-fatigue-midnight-night-defaults-v2"
DEFAULT_EDITABLE_GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1dKvzME6TL4EJ8u_f0-l2p7ZENBwj7TUZLt0cW0T7QHo/edit?usp=sharing"
)
RACE_TIMEZONE = "Europe/London"
RACE_TIMEZONE_LABEL = "BST"
DEFAULT_LIVE_REFRESH_SECONDS = 60
OLD_LIVE_REFRESH_DEFAULT_SECONDS = 15


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1240px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e3e8ef;
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
        }
        div[data-testid="stMetric"] label {
            color: #465467;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid #dde3ea;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            padding: 0.55rem 0.8rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 6px;
            min-height: 2.5rem;
        }
        .race-status {
            border: 1px solid #d7dde5;
            border-left-width: 8px;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.25rem 0 0.75rem 0;
            background: #ffffff;
        }
        .race-status h3 {
            font-size: 1.05rem;
            margin: 0 0 0.15rem 0;
        }
        .race-status p {
            margin: 0;
            color: #465467;
        }
        .race-status-good {
            border-left-color: #168251;
            background: #f3fbf7;
        }
        .race-status-watch {
            border-left-color: #c47a00;
            background: #fff8eb;
        }
        .race-status-risk {
            border-left-color: #c03535;
            background: #fff4f4;
        }
        .race-mini {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #ffffff;
            min-height: 5.25rem;
        }
        .race-mini-label {
            color: #617083;
            font-size: 0.78rem;
            font-weight: 650;
            text-transform: uppercase;
        }
        .race-mini-value {
            color: #162235;
            font-size: 1.35rem;
            font-weight: 760;
            line-height: 1.2;
            margin-top: 0.2rem;
        }
        .race-mini-help {
            color: #617083;
            font-size: 0.82rem;
            margin-top: 0.25rem;
        }
        @media (max-width: 760px) {
            .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }
            .race-mini {
                min-height: auto;
            }
            .race-mini-value {
                font-size: 1.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data_cached(workbook_bytes: bytes, source_label: str, assumption_version: str):
    return load_endure_workbook(workbook_bytes, source_label)


def format_minutes(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def format_probability(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def now_bst_label() -> str:
    return pd.Timestamp.now(tz=RACE_TIMEZONE).strftime("%H:%M:%S %Z")


def live_refresh_seconds_default(session_key: str) -> int:
    migration_key = f"{session_key}_default_migrated"
    current = st.session_state.get(session_key)
    if current is None:
        return DEFAULT_LIVE_REFRESH_SECONDS
    try:
        value = int(current)
    except Exception:
        return DEFAULT_LIVE_REFRESH_SECONDS
    if not st.session_state.get(migration_key) and value == OLD_LIVE_REFRESH_DEFAULT_SECONDS:
        st.session_state[migration_key] = True
        return DEFAULT_LIVE_REFRESH_SECONDS
    st.session_state[migration_key] = True
    return value


def format_race_clock(minute: float | int | None) -> str:
    if minute is None or pd.isna(minute):
        return "-"
    total_minutes = int(round(float(minute)))
    clock_minutes = (12 * 60 + total_minutes) % (24 * 60)
    day = "Saturday" if total_minutes < 12 * 60 else "Sunday"
    hour = clock_minutes // 60
    minute_part = clock_minutes % 60
    return f"{day} {hour:02d}:{minute_part:02d} {RACE_TIMEZONE_LABEL}"


def race_clock_options(step_minutes: int = 15) -> list[int]:
    return list(range(0, int(FINAL_CUTOFF_MINUTE) + 1, step_minutes))


def race_clock_tick_values() -> list[int]:
    values = set(range(0, int(FINAL_CUTOFF_MINUTE) + 1, 120))
    values.update({int(LAST_START_MINUTE), int(FINAL_CUTOFF_MINUTE)})
    return sorted(values)


def apply_race_clock_axis(fig: go.Figure, title: str = "Race time") -> go.Figure:
    ticks = race_clock_tick_values()
    fig.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        ticktext=[format_race_clock(value) for value in ticks],
        title_text=title,
    )
    return fig


def add_clock_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    display = df.copy()
    for column in columns:
        if column in display.columns:
            label = column.removesuffix("_minute") + "_time"
            display[label] = display[column].map(format_race_clock)
    return display


def final_lap_probability_figure(distribution: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=distribution["official_laps"],
            y=distribution["probability"],
            name="Probability",
            marker_color="#2f6f8f",
            hovertemplate="%{x} laps: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=distribution["official_laps"],
            y=distribution["probability"],
            mode="lines+markers",
            name="PMF line",
            line={"color": "#17324d", "width": 3},
            marker={"size": 8},
            hovertemplate="%{x} laps: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Final official lap number",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
        bargap=0.25,
    )
    return fig


def comparison_final_lap_probability_figure(
    baseline_distribution: pd.DataFrame,
    override_distribution: pd.DataFrame,
) -> go.Figure:
    baseline = baseline_distribution[["official_laps", "probability"]].copy()
    baseline["scenario"] = "Current assumptions"
    override = override_distribution[["official_laps", "probability"]].copy()
    override["scenario"] = "What-if override"
    combined = pd.concat([baseline, override], ignore_index=True)
    fig = px.line(
        combined,
        x="official_laps",
        y="probability",
        color="scenario",
        markers=True,
        title="What-if final lap number probability plot",
    )
    fig.update_layout(
        xaxis_title="Final official lap number",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
    )
    return fig


def final_lap_comparison_figure(
    baseline_distribution: pd.DataFrame,
    live_distribution: pd.DataFrame,
    baseline_label: str,
    live_label: str,
    title: str,
) -> go.Figure:
    baseline = baseline_distribution[["official_laps", "probability"]].copy()
    baseline["scenario"] = baseline_label
    live = live_distribution[["official_laps", "probability"]].copy()
    live["scenario"] = live_label
    combined = pd.concat([baseline, live], ignore_index=True)
    fig = px.line(
        combined,
        x="official_laps",
        y="probability",
        color="scenario",
        markers=True,
        title=title,
    )
    fig.update_layout(
        xaxis_title="Final official lap number",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
        legend_title_text="Forecast",
    )
    return fig


def race_progress_figure(
    actual_progress: pd.DataFrame,
    target_progress: pd.DataFrame,
    baseline_output: dict | None,
    live_output: dict | None,
    elapsed_minute: float,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=target_progress["time_minute"],
            y=target_progress["completed_laps"],
            mode="lines",
            name="Target pace",
            line={"color": "#7a5c00", "dash": "dash", "width": 2},
            hovertemplate="%{y:.1f} laps by %{x:.0f} min<extra></extra>",
        )
    )

    _add_forecast_band(fig, baseline_output, "Pre-race", "#2f6f8f", "rgba(47, 111, 143, 0.13)")
    _add_forecast_band(fig, live_output, "Live forecast", "#168251", "rgba(22, 130, 81, 0.14)")

    if not actual_progress.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_progress["time_minute"],
                y=actual_progress["completed_laps"],
                mode="lines+markers",
                name="Actual",
                line={"color": "#111827", "width": 4, "shape": "hv"},
                marker={"size": 7, "color": "#111827"},
                hovertemplate="%{y:.0f} official laps<extra></extra>",
            )
        )

    if elapsed_minute > 0:
        fig.add_vline(x=float(elapsed_minute), line_color="#64748b", line_dash="solid", opacity=0.45)
    fig.add_vline(x=LAST_START_MINUTE, line_dash="dash", line_color="#c03535")
    fig.add_vline(x=FINAL_CUTOFF_MINUTE, line_dash="dot", line_color="#c03535")

    max_laps = float(target_progress["completed_laps"].max()) if not target_progress.empty else 0.0
    for source in [actual_progress, _path_quantiles_or_empty(baseline_output), _path_quantiles_or_empty(live_output)]:
        if not source.empty:
            for column in ["completed_laps", "p90", "median"]:
                if column in source:
                    max_laps = max(max_laps, float(pd.to_numeric(source[column], errors="coerce").max()))
    fig.update_layout(
        title="Progress vs plan",
        height=390,
        margin={"l": 8, "r": 8, "t": 46, "b": 8},
        legend={"orientation": "h", "y": -0.18},
        yaxis_title="Official laps",
        yaxis_range=[0, max_laps + 1],
    )
    apply_race_clock_axis(fig)
    return fig


def _add_forecast_band(fig: go.Figure, sim_output: dict | None, label: str, color: str, fillcolor: str) -> None:
    quantiles = _path_quantiles_or_empty(sim_output)
    if quantiles.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=quantiles["time_minute"],
            y=quantiles["p90"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=quantiles["time_minute"],
            y=quantiles["p10"],
            mode="lines",
            fill="tonexty",
            fillcolor=fillcolor,
            line={"width": 0},
            name=f"{label} 10-90%",
            hovertemplate="%{y:.0f} laps<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=quantiles["time_minute"],
            y=quantiles["median"],
            mode="lines",
            name=f"{label} median",
            line={"color": color, "width": 2},
            hovertemplate="%{y:.0f} laps<extra></extra>",
        )
    )


def _path_quantiles_or_empty(sim_output: dict | None) -> pd.DataFrame:
    if not sim_output:
        return pd.DataFrame()
    quantiles = sim_output.get("path_quantiles")
    if not isinstance(quantiles, pd.DataFrame):
        return pd.DataFrame()
    return quantiles


def edited_roster_table(roster: pd.DataFrame) -> pd.DataFrame:
    display = roster.copy()
    display["max_laps"] = display["max_laps"].astype("Float64")
    return st.data_editor(
        display,
        hide_index=True,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "runner": st.column_config.TextColumn("Runner", required=True),
            "projected_mean_minutes": st.column_config.NumberColumn("Mean minutes", min_value=1.0, step=0.1),
            "projected_sd_minutes": st.column_config.NumberColumn("SD minutes", min_value=0.1, step=0.1),
            "running_order": st.column_config.NumberColumn("Order", min_value=1, step=1),
            "max_laps": st.column_config.NumberColumn("Max laps", min_value=0, step=1, help="Blank means unlimited."),
            "available": st.column_config.CheckboxColumn("Available"),
            "fatigue_pct_per_lap": st.column_config.NumberColumn("Fatigue % / lap", step=0.1),
            "night_penalty_pct": st.column_config.NumberColumn("Night penalty %", step=0.1),
            "notes": st.column_config.TextColumn("Notes"),
            "source": st.column_config.TextColumn("Source", disabled=True),
        },
    )


def make_settings() -> SimulationSettings:
    st.sidebar.header("Simulation Settings")
    n_simulations = st.sidebar.number_input("Number of simulations", min_value=100, max_value=100_000, value=10_000, step=1_000)
    target_laps = st.sidebar.number_input("Reference target laps", min_value=1, max_value=80, value=40, step=1)
    st.sidebar.caption("Used for probability readouts only. The default optimiser objective is maximum official laps.")
    random_seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)
    distribution = st.sidebar.selectbox("Lap time distribution", ["truncated_normal", "lognormal"], index=0)

    st.sidebar.subheader("Night Window")
    clock_options = race_clock_options()
    night_start_minute = st.sidebar.selectbox(
        "Night starts",
        clock_options,
        index=clock_options.index(12 * 60),
        format_func=format_race_clock,
    )
    night_end_minute = st.sidebar.selectbox(
        "Night ends",
        clock_options,
        index=clock_options.index(18 * 60),
        format_func=format_race_clock,
    )
    st.sidebar.caption(
        f"Night window: {format_race_clock(night_start_minute)} to {format_race_clock(night_end_minute)}."
    )

    st.sidebar.subheader("Model Toggles")
    fatigue_enabled = st.sidebar.checkbox("Apply fatigue", value=True)
    night_penalty_enabled = st.sidebar.checkbox("Apply night penalties", value=False)
    st.sidebar.caption("Night penalties only affect simulations when this toggle is on.")
    transition_mistakes_enabled = st.sidebar.checkbox("Model transition mistakes", value=True)

    st.sidebar.subheader("Transitions")
    base_transition_seconds = st.sidebar.number_input("Base transition seconds", min_value=0.0, value=0.0, step=5.0)
    transition_mistake_probability_per_lap = st.sidebar.number_input(
        "Mistake probability per lap",
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        step=0.005,
        format="%.3f",
    )
    transition_mistake_min_seconds = st.sidebar.number_input("Mistake min seconds", min_value=0.0, value=30.0, step=15.0)
    transition_mistake_max_seconds = st.sidebar.number_input("Mistake max seconds", min_value=0.0, value=300.0, step=30.0)
    major_mistake_probability_per_lap = st.sidebar.number_input(
        "Major mistake probability per lap",
        min_value=0.0,
        max_value=1.0,
        value=0.002,
        step=0.001,
        format="%.3f",
    )
    major_mistake_min_seconds = st.sidebar.number_input("Major mistake min seconds", min_value=0.0, value=300.0, step=60.0)
    major_mistake_max_seconds = st.sidebar.number_input("Major mistake max seconds", min_value=0.0, value=900.0, step=60.0)

    return SimulationSettings(
        n_simulations=int(n_simulations),
        target_laps=int(target_laps),
        random_seed=int(random_seed),
        night_start_minute=float(night_start_minute),
        night_end_minute=float(night_end_minute),
        distribution=distribution,
        fatigue_enabled=fatigue_enabled,
        night_penalty_enabled=night_penalty_enabled,
        transition_mistakes_enabled=transition_mistakes_enabled,
        base_transition_seconds=float(base_transition_seconds),
        transition_mistake_probability_per_lap=float(transition_mistake_probability_per_lap),
        transition_mistake_min_seconds=float(transition_mistake_min_seconds),
        transition_mistake_max_seconds=float(transition_mistake_max_seconds),
        major_mistake_probability_per_lap=float(major_mistake_probability_per_lap),
        major_mistake_min_seconds=float(major_mistake_min_seconds),
        major_mistake_max_seconds=float(major_mistake_max_seconds),
    )


def show_validation(loaded) -> None:
    st.subheader("Data Upload & Validation")
    st.dataframe(loaded.sheet_info, width="stretch", hide_index=True)
    if loaded.detected_tables:
        for item in loaded.detected_tables:
            st.success(item)
    for warning in loaded.warnings:
        st.warning(warning)
    for error in loaded.errors:
        st.error(error)

    with st.expander("Cleaned last-year lap preview", expanded=True):
        preview = add_clock_columns(loaded.last_year_laps, ["start_minute", "finish_minute"])
        st.dataframe(preview.head(50), width="stretch", hide_index=True)
    with st.expander("Runner name normalisation"):
        st.dataframe(loaded.name_map, width="stretch", hide_index=True)


def show_last_year_analysis(loaded) -> None:
    st.subheader("Last Year Analysis")
    stats = loaded.last_year_stats.copy()
    if stats.empty:
        st.info("No last-year stats are available.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total laps", f"{loaded.team_stats['total_laps']}")
    c2.metric("Team avg lap", format_minutes(loaded.team_stats["team_mean_minutes"]))
    c3.metric("Fastest lap", format_minutes(loaded.team_stats["fastest_lap_minutes"]))
    c4.metric("Slowest lap", format_minutes(loaded.team_stats["slowest_lap_minutes"]))

    st.dataframe(stats, width="stretch", hide_index=True)

    laps = loaded.last_year_laps.copy()
    laps = add_clock_columns(laps, ["start_minute", "finish_minute"])
    fig = px.line(
        laps,
        x="lap",
        y="lap_time_minutes",
        color="runner",
        markers=True,
        title="Last-year lap times by runner",
    )
    fig.update_layout(height=430, legend_title_text="Runner")
    st.plotly_chart(fig, width="stretch")

    box = px.box(laps, x="runner", y="lap_time_minutes", points="all", title="Lap time distribution by runner")
    box.update_layout(height=430, xaxis_title="")
    st.plotly_chart(box, width="stretch")


def show_result_metrics(summary: dict, comparison: pd.DataFrame | None = None) -> None:
    row1 = st.columns(5)
    row1[0].metric("Expected laps", format_minutes(summary["expected_official_laps"]))
    row1[1].metric("Median", format_minutes(summary["median_official_laps"]))
    row1[2].metric("Mode", f"{summary['mode_official_laps']}")
    row1[3].metric("P10-P90", f"{summary['p10_official_laps']:.0f} - {summary['p90_official_laps']:.0f}")
    row1[4].metric("Hit reference target", format_probability(summary["probability_target_laps"]))

    row2 = st.columns(5)
    row2[0].metric("Hit reference target + 1", format_probability(summary["probability_target_plus_one_laps"]))
    row2[1].metric("Laps by Sunday 12:00 BST", format_minutes(summary["expected_laps_by_noon"]))
    row2[2].metric("Final lap after noon", format_probability(summary["probability_squeezed_final_lap_after_noon"]))
    row2[3].metric("Miss Sunday 13:00 BST cutoff", format_probability(summary["probability_missed_final_cutoff"]))
    row2[4].metric("Run out of runners", format_probability(summary["probability_ran_out_of_eligible_runners"]))

    st.caption(f"Average final finish: {format_race_clock(summary['average_final_finish_minute'])}")

    if comparison is not None and not comparison.empty:
        capped = comparison.loc[comparison["scenario"] == "Caps enabled", "expected_official_laps"].iloc[0]
        uncapped = comparison.loc[comparison["scenario"] == "Caps disabled", "expected_official_laps"].iloc[0]
        st.info(f"Caps impact: {capped - uncapped:+.2f} expected laps versus the uncapped simulation.")


def show_charts(sim_output: dict, summary_bundle: dict, uncapped_summary: dict | None, comparison: pd.DataFrame | None) -> None:
    simulations = sim_output["simulations"]
    distribution = summary_bundle["final_lap_distribution"]

    st.subheader("Results Dashboard")
    show_result_metrics(summary_bundle["summary"], comparison)

    c1, c2 = st.columns(2)
    with c1:
        bar = final_lap_probability_figure(
            distribution,
            "Final lap number probability plot",
        )
        st.plotly_chart(bar, width="stretch")
        st.caption("This is a discrete probability mass function for the final official lap count.")
    with c2:
        cumulative = distribution.sort_values("official_laps")
        cum_fig = px.line(
            cumulative,
            x="official_laps",
            y="probability_at_least",
            markers=True,
            title="Probability of at least N laps",
        )
        cum_fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(cum_fig, width="stretch")

    st.subheader("Race Path")
    path_fig = go.Figure()
    samples = sim_output["path_samples"]
    if not samples.empty:
        for sim_id, group in samples.groupby("simulation"):
            path_fig.add_trace(
                go.Scatter(
                    x=group["time_minute"],
                    y=group["completed_laps"],
                    mode="lines",
                    line={"width": 1, "color": "rgba(90, 110, 130, 0.18)"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    quantiles = sim_output["path_quantiles"]
    path_fig.add_trace(
        go.Scatter(
            x=quantiles["time_minute"],
            y=quantiles["p90"],
            mode="lines",
            line={"width": 0},
            name="90th percentile",
            hovertemplate="%{y:.0f} laps<extra></extra>",
        )
    )
    path_fig.add_trace(
        go.Scatter(
            x=quantiles["time_minute"],
            y=quantiles["p10"],
            mode="lines",
            fill="tonexty",
            line={"width": 0},
            fillcolor="rgba(40, 120, 180, 0.18)",
            name="10th-90th band",
            hovertemplate="%{y:.0f} laps<extra></extra>",
        )
    )
    path_fig.add_trace(
        go.Scatter(
            x=quantiles["time_minute"],
            y=quantiles["median"],
            mode="lines",
            line={"width": 3, "color": "#1f77b4"},
            name="Median path",
        )
    )
    path_fig.add_vline(x=LAST_START_MINUTE, line_dash="dash", line_color="firebrick")
    path_fig.add_vline(x=FINAL_CUTOFF_MINUTE, line_dash="dot", line_color="firebrick")
    path_fig.update_layout(height=480, yaxis_title="Completed official laps")
    apply_race_clock_axis(path_fig)
    st.plotly_chart(path_fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        start_hist = px.histogram(
            simulations.dropna(subset=["final_lap_start_minute"]),
            x="final_lap_start_minute",
            nbins=40,
            title="Final lap start-time distribution",
        )
        start_hist.add_vline(x=LAST_START_MINUTE, line_dash="dash", line_color="firebrick")
        apply_race_clock_axis(start_hist, "Final lap start time")
        st.plotly_chart(start_hist, width="stretch")
    with c4:
        finish_hist = px.histogram(
            simulations.dropna(subset=["final_lap_finish_minute"]),
            x="final_lap_finish_minute",
            nbins=40,
            title="Final lap finish-time distribution",
        )
        finish_hist.add_vline(x=FINAL_CUTOFF_MINUTE, line_dash="dash", line_color="firebrick")
        apply_race_clock_axis(finish_hist, "Final lap finish time")
        st.plotly_chart(finish_hist, width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        runner_fig = px.bar(
            summary_bundle["expected_runner_laps"],
            x="runner",
            y="expected_laps",
            title="Expected laps per runner",
        )
        st.plotly_chart(runner_fig, width="stretch")
    with c6:
        cap_fig = px.bar(
            summary_bundle["cap_hit_rates"],
            x="runner",
            y="cap_hit_probability",
            title="Probability each capped runner hits their cap",
        )
        cap_fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(cap_fig, width="stretch")

    c7, c8 = st.columns(2)
    with c7:
        final_runner_fig = px.bar(
            summary_bundle["final_runner_probs"],
            x="runner",
            y="probability",
            title="Likely final-lap runner",
        )
        final_runner_fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(final_runner_fig, width="stretch")
    with c8:
        final_hour_fig = px.bar(
            summary_bundle["final_hour_runner_probs"],
            x="runner",
            y="probability",
            title="Likely running between Sunday 10:30 and 12:00 BST",
        )
        final_hour_fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(final_hour_fig, width="stretch")

    if comparison is not None:
        compare_fig = px.bar(
            comparison,
            x="scenario",
            y="expected_official_laps",
            color="scenario",
            title="Capped vs uncapped expected laps",
        )
        st.plotly_chart(compare_fig, width="stretch")
        st.dataframe(comparison, width="stretch", hide_index=True)


def show_downloads(sim_output: dict, summary_bundle: dict, comparison: pd.DataFrame | None) -> None:
    st.subheader("Export Results")
    summary_df = pd.DataFrame([summary_bundle["summary"]])
    simulation_export = add_clock_columns(
        sim_output["simulations"],
        ["final_elapsed_minute", "final_lap_start_minute", "final_lap_finish_minute"],
    )
    st.download_button(
        "Download simulation rows CSV",
        simulation_export.to_csv(index=False),
        file_name="endure24_simulation_rows.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download summary CSV",
        summary_df.to_csv(index=False),
        file_name="endure24_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download runner expected laps CSV",
        summary_bundle["expected_runner_laps"].to_csv(index=False),
        file_name="endure24_runner_expected_laps.csv",
        mime="text/csv",
    )
    if comparison is not None:
        st.download_button(
            "Download capped comparison CSV",
            comparison.to_csv(index=False),
            file_name="endure24_capped_vs_uncapped.csv",
            mime="text/csv",
        )


def get_streamlit_secret(*keys: str):
    for key in keys:
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            return value
    return None


def format_race_log_for_display(log: pd.DataFrame, roster: pd.DataFrame, caps_enabled: bool) -> pd.DataFrame:
    display = normalise_race_log(log, roster)
    display = add_clock_columns(display, ["start_minute", "finish_minute"])
    review = race_order_review(display, roster, caps_enabled=caps_enabled)
    if not review.empty:
        display = display.merge(
            review[["lap_number", "expected_runner", "order_status"]],
            on="lap_number",
            how="left",
        )
    else:
        display["expected_runner"] = ""
        display["order_status"] = ""
    display["status"] = display.apply(
        lambda row: "In progress"
        if pd.notna(row.get("start_minute")) and pd.isna(row.get("finish_minute"))
        else ("Official" if bool(row.get("official")) else "Not official"),
        axis=1,
    )
    display["lap_duration_minutes"] = pd.to_numeric(display["lap_duration_minutes"], errors="coerce").round(1)
    columns = [
        "lap_number",
        "runner",
        "status",
        "order_status",
        "start_time",
        "finish_time",
        "lap_duration_minutes",
        "notes",
    ]
    return display[[column for column in columns if column in display.columns]]


def race_log_digest(log: pd.DataFrame) -> str:
    clean = log.copy()
    return hashlib.sha256(clean.to_csv(index=False).encode("utf-8")).hexdigest()


def set_race_log(log: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    clean = normalise_race_log(log, roster)
    digest = race_log_digest(clean)
    changed = digest != st.session_state.get("race_log_digest")
    st.session_state["race_log"] = clean
    st.session_state["race_log_digest"] = digest
    st.session_state["race_log_changed"] = changed
    if changed:
        st.session_state.pop("race_day_forecast", None)
        st.session_state.pop("race_day_plan_ranking", None)
    return clean


def show_race_day_metrics(state, live_bundle: dict | None, target_laps: int) -> None:
    live_summary = live_bundle["summary"] if live_bundle else None
    cols = st.columns(5)
    cols[0].metric("Official laps", f"{state.current_laps}")
    cols[1].metric("Target probability", format_probability(live_summary["probability_target_laps"]) if live_summary else "-")
    cols[2].metric("Projected final laps", format_minutes(live_summary["expected_official_laps"]) if live_summary else "-")
    cols[3].metric("Elapsed", format_race_clock(state.elapsed_minute))
    runner_label = "Current runner" if state.current_lap_in_progress else "Next runner"
    cols[4].metric(runner_label, state.next_runner or "No eligible runner")

    if state.current_lap_in_progress:
        st.info(
            f"Lap {int(state.latest_lap['lap_number'])} is in progress: "
            f"{state.in_progress_runner} started at {format_race_clock(state.in_progress_start_minute)}. "
            "Add the finish time or lap duration when they come in."
        )

    if state.remaining_to_target <= 0:
        st.success(f"Target of {target_laps} official laps is already logged.")
    else:
        parts = [f"Need {state.remaining_to_target} more official laps to reach {target_laps}."]
        if state.average_needed_to_target is not None:
            parts.append(f"Average {state.average_needed_to_target:.1f} min/lap or faster by Sunday 13:00 BST.")
        if state.average_needed_to_start_target_lap is not None:
            parts.append(
                f"Average {state.average_needed_to_start_target_lap:.1f} min/lap or faster before Sunday 12:00 BST "
                "to start the target lap on time."
            )
        st.info(" ".join(parts))


def show_race_control_dashboard(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    settings: SimulationSettings,
    state,
    baseline_output: dict | None,
    live_output: dict | None,
    live_bundle: dict | None,
    warnings: list[str],
    errors: list[str],
    caps_enabled: bool,
) -> None:
    live_summary = live_bundle["summary"] if live_bundle else None
    status_level, status_title, status_body = race_day_status_summary(state, live_summary, warnings, errors)

    st.markdown("### Race Control")
    st.markdown(
        f"""
        <div class="race-status race-status-{status_level}">
            <h3>{html.escape(status_title)}</h3>
            <p>{html.escape(status_body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest_completed = latest_completed_lap(log, roster)
    last_lap_value = "-"
    last_lap_help = "No completed laps yet"
    if latest_completed is not None:
        last_lap_value = f"{latest_completed['runner']} {float(latest_completed['lap_duration_minutes']):.1f}m"
        last_lap_help = f"Finished {format_race_clock(latest_completed['finish_minute'])}"

    runner_label = "Current" if state.current_lap_in_progress else "Next"
    runner_value = state.next_runner or "None"
    runner_help = "On course now" if state.current_lap_in_progress else "Fixed planned order"

    pace_needed = state.average_needed_to_start_target_lap or state.average_needed_to_target
    pace_value = f"{pace_needed:.1f}m" if pace_needed is not None else "Hit"
    pace_help = (
        "Avg needed to start target lap by Sunday 12:00 BST"
        if state.average_needed_to_start_target_lap
        else "Avg needed by Sunday 13:00 BST"
    )
    if state.remaining_to_target <= 0:
        pace_help = "Target already logged"

    cards = [
        ("Official laps", str(state.current_laps), f"Target {settings.target_laps}"),
        ("Target chance", format_probability(live_summary["probability_target_laps"]) if live_summary else "-", "Run forecast for live probability"),
        ("Projected final", format_minutes(live_summary["expected_official_laps"]) if live_summary else "-", "Actual-adjusted forecast"),
        (runner_label, runner_value, runner_help),
        ("Last lap", last_lap_value, last_lap_help),
        ("Pace needed", pace_value, pace_help),
    ]
    for row_start in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, card in zip(cols, cards[row_start : row_start + 3]):
            with col:
                race_mini_card(*card)

    actual_progress = build_progress_actuals(log, roster)
    target_progress = target_pace_series(settings.target_laps)
    st.plotly_chart(
        race_progress_figure(actual_progress, target_progress, baseline_output, live_output, state.elapsed_minute),
        width="stretch",
    )

    with st.expander("Fixed-order queue and runner status", expanded=False):
        queue = build_runner_queue(log, roster, caps_enabled=caps_enabled, queue_size=5)
        st.markdown("#### Fixed Order Queue")
        if queue.empty:
            st.info("No available runners in the current setup.")
        else:
            st.dataframe(
                format_runner_queue_for_display(queue),
                width="stretch",
                hide_index=True,
                column_config={
                    "role": "Slot",
                    "runner": "Runner",
                    "completed_laps": st.column_config.NumberColumn("Laps", format="%d"),
                    "cap_remaining": "Cap left",
                    "last_lap": "Last",
                    "rest": "Rest",
                    "order_status": "Order",
                },
            )

        st.markdown("#### All Runner Status")
        status = build_runner_status_table(log, roster, caps_enabled=caps_enabled)
        st.dataframe(
            format_runner_queue_for_display(status),
            width="stretch",
            hide_index=True,
            column_config={
                "role": "Status",
                "runner": "Runner",
                "completed_laps": st.column_config.NumberColumn("Laps", format="%d"),
                "cap_remaining": "Cap left",
                "last_lap": "Last",
                "rest": "Rest",
            },
        )


def queue_next_runner(queue: pd.DataFrame) -> str:
    if queue.empty or "runner" not in queue:
        return ""
    if "role" in queue:
        next_rows = queue[queue["role"].astype(str).eq("Next")]
        if not next_rows.empty:
            return str(next_rows.iloc[0]["runner"])
        non_current = queue[~queue["role"].astype(str).eq("Current")]
        if not non_current.empty:
            return str(non_current.iloc[0]["runner"])
    return str(queue.iloc[0]["runner"])


def show_race_day_plan_assistant(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    settings: SimulationSettings,
    state,
    caps_enabled: bool,
) -> None:
    st.markdown("### Plan Assistant")
    st.caption("Use this as the race-day handover view. It keeps completed laps fixed and preserves a fair live rotation.")
    fixed_queue = build_runner_queue(log, roster, caps_enabled=caps_enabled, queue_size=5)
    fair_queue = build_live_fair_queue(log, roster, caps_enabled=caps_enabled, queue_size=5)
    fixed_next = queue_next_runner(fixed_queue) or state.next_runner or ""
    fair_next = queue_next_runner(fair_queue)

    cols = st.columns(3)
    cols[0].metric("Recommended next", fair_next or "No eligible runner")
    cols[1].metric("Fixed order next", fixed_next or "No eligible runner")
    cols[2].metric("Decision rule", "Fair live rotation")

    if fair_next and fixed_next and fair_next != fixed_next:
        st.warning(
            f"Live fair rotation recommends {fair_next}; the fixed pre-race order expects {fixed_next}. "
            "Use the recommendation if the team has already changed the order on the day."
        )
    elif fair_next:
        st.caption("Live fair rotation agrees with the fixed order for the next runner.")
    else:
        st.info("No eligible runner is available under the current roster and cap settings.")

    st.caption("The queue learns the order used on the day, then moves whoever ran most recently to the back.")
    if fair_queue.empty:
        st.info("The live fair queue is empty.")
    else:
        st.dataframe(
            format_runner_queue_for_display(fair_queue),
            width="stretch",
            hide_index=True,
            column_config={
                "role": "Slot",
                "runner": "Runner",
                "completed_laps": st.column_config.NumberColumn("Laps", format="%d"),
                "cap_remaining": "Cap left",
                "last_lap": "Last",
                "rest": "Rest",
                "order_status": "Basis",
                "reason": "Why",
            },
        )

    compare_cols = st.columns([1, 1.4])
    if compare_cols[0].button("Compare race-day plans", width="stretch"):
        with st.spinner("Comparing fixed order vs live fair queue..."):
            st.session_state["race_day_plan_ranking"] = rank_live_next_runner_options(
                log,
                roster,
                settings,
                caps_enabled=caps_enabled,
            )
    compare_cols[1].caption("Optional. Compares fixed order against the live fair queue from the current race state.")

    ranking = st.session_state.get("race_day_plan_ranking")
    if isinstance(ranking, pd.DataFrame) and not ranking.empty:
        st.dataframe(
            ranking.style.format(
                {
                    "expected_official_laps": "{:.2f}",
                    "median_official_laps": "{:.1f}",
                    "p10_official_laps": "{:.1f}",
                    "p90_official_laps": "{:.1f}",
                    "probability_target_laps": "{:.1%}",
                    "probability_target_plus_one_laps": "{:.1%}",
                    "probability_missed_final_cutoff": "{:.1%}",
                    "probability_ran_out_of_eligible_runners": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )


def show_race_day_sync_status(sync: Mapping[str, Any]) -> None:
    last_load = st.session_state.get("editable_google_sheet_last_loaded", "-")
    last_save = st.session_state.get("race_day_sheet_last_saved", "-")
    live_reload = st.session_state.get("race_day_live_sheet_enabled", bool(sync.get("configured")))
    refresh_seconds = live_refresh_seconds_default("race_day_live_refresh_seconds")
    with st.container(border=True):
        st.markdown("#### Sheet Activity")
        cols = st.columns(4)
        cols[0].metric("Write access", "Ready" if sync.get("configured") else "Off")
        cols[1].metric("Auto reload", f"On ({refresh_seconds}s)" if live_reload else "Off")
        cols[2].metric("Last load", str(last_load))
        cols[3].metric("Last save", str(last_save))


def show_race_day_readiness(
    sync: Mapping[str, Any],
    warnings: list[str],
    errors: list[str],
    live_bundle: dict | None,
    roster: pd.DataFrame,
    settings: SimulationSettings,
    log: pd.DataFrame,
    caps_enabled: bool,
) -> None:
    sheet_state = "Ready" if sync.get("configured") else "Check setup"
    live_enabled = st.session_state.get("race_day_live_sheet_enabled", bool(sync.get("configured")))
    live_state = f"On ({live_refresh_seconds_default('race_day_live_refresh_seconds')}s)" if live_enabled else "Off"
    if errors:
        log_state = f"{len(errors)} fix"
    elif warnings:
        log_state = f"{len(warnings)} watch"
    else:
        log_state = "Clean"
    forecast_state = "Live" if live_bundle else "Run forecast"

    with st.container(border=True):
        st.markdown("#### Race-Day Checklist")
        cols = st.columns(4)
        cols[0].metric("Google Sheet", sheet_state)
        cols[1].metric("Auto reload", live_state)
        cols[2].metric("Race log", log_state)
        cols[3].metric("Forecast", forecast_state)

        action_cols = st.columns([1, 2])
        forecast_disabled = bool(errors)
        if action_cols[0].button(
            "Refresh live forecast",
            type="primary",
            width="stretch",
            disabled=forecast_disabled,
        ):
            with st.spinner("Refreshing live forecast..."):
                update_race_day_forecast(roster, settings, log, caps_enabled)
                st.session_state["race_day_notice"] = "Live forecast refreshed."
                st.rerun()
        if forecast_disabled:
            action_cols[1].caption("Fix race-log errors before refreshing the forecast.")
        else:
            action_cols[1].caption("Updates target chance, projected final laps, and the Race Control plot.")


def show_lap_handover_strip(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    state,
    caps_enabled: bool,
) -> None:
    latest_completed = latest_completed_lap(log, roster)
    if latest_completed is None:
        last_value = "None logged"
        last_help = "Use Add Race Entry when the first lap comes in."
    else:
        last_value = f"{latest_completed['runner']} {float(latest_completed['lap_duration_minutes']):.1f}m"
        last_help = f"Finished {format_race_clock(latest_completed['finish_minute'])}"

    if state.current_lap_in_progress:
        on_course_value = str(state.in_progress_runner or "-")
        on_course_help = f"Started {format_race_clock(state.in_progress_start_minute)}"
        action_value = "Complete lap"
        action_help = "Enter finish time or lap duration."
    else:
        on_course_value = "None recorded"
        on_course_help = "Record a start only if you want to track an in-progress lap."
        action_value = "Log lap"
        action_help = "Add completed lap, or record the next start."

    fair_queue = build_live_fair_queue(log, roster, caps_enabled=caps_enabled, queue_size=2)
    recommended_next = queue_next_runner(fair_queue) or state.next_runner or "No eligible runner"

    st.markdown("#### Handover")
    cols = st.columns(4)
    cards = [
        ("Last completed", last_value, last_help),
        ("On course", on_course_value, on_course_help),
        ("Recommended next", recommended_next, "Used as the default runner below."),
        ("Logger action", action_value, action_help),
    ]
    for col, card in zip(cols, cards):
        with col:
            race_mini_card(*card)


def build_team_brief(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    settings: SimulationSettings,
    state,
    live_bundle: dict | None,
    warnings: list[str],
    errors: list[str],
    caps_enabled: bool,
    sync: Mapping[str, Any],
) -> str:
    latest_completed = latest_completed_lap(log, roster)
    fair_queue = build_live_fair_queue(log, roster, caps_enabled=caps_enabled, queue_size=1)
    recommended_next = queue_next_runner(fair_queue) or state.next_runner or "No eligible runner"
    live_summary = live_bundle["summary"] if live_bundle else None

    if errors:
        status = f"Fix now: {len(errors)} issue(s)"
    elif warnings:
        status = f"Watch: {len(warnings)} item(s)"
    else:
        status = "Clean"

    if state.current_lap_in_progress:
        runner_line = (
            f"On course: {state.in_progress_runner or '-'} "
            f"since {format_race_clock(state.in_progress_start_minute)}"
        )
    else:
        runner_line = f"Next up: {recommended_next}"

    if latest_completed is None:
        last_line = "Last lap: none logged"
    else:
        last_line = (
            f"Last lap: {latest_completed['runner']} "
            f"{float(latest_completed['lap_duration_minutes']):.1f}m, "
            f"finished {format_race_clock(latest_completed['finish_minute'])}"
        )

    if live_summary:
        forecast_line = (
            "Forecast: "
            f"{float(live_summary['expected_official_laps']):.2f} projected laps, "
            f"{format_probability(live_summary['probability_target_laps'])} chance of {settings.target_laps}"
        )
    else:
        forecast_line = "Forecast: refresh pending"

    pace_needed = state.average_needed_to_start_target_lap or state.average_needed_to_target
    if state.remaining_to_target <= 0:
        pace_line = f"Pace needed: target {settings.target_laps} already logged"
    elif pace_needed is None:
        pace_line = "Pace needed: not available"
    else:
        pace_line = f"Pace needed: {pace_needed:.1f} min/lap"

    sheet_line = (
        "Sheet: "
        f"{'write ready' if sync.get('configured') else 'read/local only'}, "
        f"auto reload {'on' if st.session_state.get('race_day_live_sheet_enabled', bool(sync.get('configured'))) else 'off'}, "
        f"load {st.session_state.get('editable_google_sheet_last_loaded', '-')}, "
        f"save {st.session_state.get('race_day_sheet_last_saved', '-')}"
    )

    return "\n".join(
        [
            f"Endure24 update: {state.current_laps}/{settings.target_laps} official laps - {status}",
            runner_line,
            f"Recommended next: {recommended_next}",
            last_line,
            forecast_line,
            pace_line,
            sheet_line,
        ]
    )


def show_team_brief(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    settings: SimulationSettings,
    state,
    live_bundle: dict | None,
    warnings: list[str],
    errors: list[str],
    caps_enabled: bool,
    sync: Mapping[str, Any],
) -> None:
    brief = build_team_brief(log, roster, settings, state, live_bundle, warnings, errors, caps_enabled, sync)
    with st.container(border=True):
        st.markdown("#### Team Brief")
        st.code(brief, language="text")
        st.download_button(
            "Download brief",
            brief,
            file_name="endure24_team_brief.txt",
            mime="text/plain",
            width="stretch",
        )


def race_mini_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="race-mini">
            <div class="race-mini-label">{html.escape(str(label))}</div>
            <div class="race-mini-value">{html.escape(str(value))}</div>
            <div class="race-mini-help">{html.escape(str(help_text))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def race_day_status_summary(
    state,
    live_summary: dict | None,
    warnings: list[str],
    errors: list[str],
) -> tuple[str, str, str]:
    if errors:
        return "risk", "Fix now", f"{len(errors)} issue(s) block forecasting or saving."
    if state.remaining_to_target <= 0:
        return "good", "Target logged", "The selected target lap count is already in the race log."

    probability = None if live_summary is None else live_summary.get("probability_target_laps")
    if probability is not None and not pd.isna(probability):
        probability = float(probability)
        if probability >= 0.7 and not warnings:
            return "good", "On track", f"Live target probability is {probability:.0%}."
        if probability < 0.35:
            return "risk", "Target at risk", f"Live target probability is {probability:.0%}."
        return "watch", "Watch closely", f"Live target probability is {probability:.0%}."

    if warnings:
        return "watch", "Watch closely", f"{len(warnings)} race-log check(s) need attention."
    return "good", "Ready", "No race-log issues detected. Run a race-day forecast for live probability."


def latest_completed_lap(log: pd.DataFrame, roster: pd.DataFrame) -> dict | None:
    clean = normalise_race_log(log, roster)
    completed = clean.dropna(subset=["finish_minute", "lap_duration_minutes"]).sort_values("finish_minute")
    if completed.empty:
        return None
    return completed.iloc[-1].to_dict()


def format_runner_queue_for_display(queue: pd.DataFrame) -> pd.DataFrame:
    display = queue.copy()
    display["last_lap"] = display["last_lap_minutes"].map(lambda value: "-" if pd.isna(value) else f"{float(value):.1f}m")
    display["rest"] = display["rest_minutes"].map(lambda value: "-" if pd.isna(value) else f"{float(value):.0f}m")
    columns = ["role", "runner", "completed_laps", "cap_remaining", "last_lap", "rest", "order_status", "reason"]
    return display[[column for column in columns if column in display.columns]]


def show_race_day_alerts(warnings: list[str], errors: list[str], detail_expanded: bool = False) -> None:
    warnings = prioritize_alerts(warnings)
    errors = prioritize_alerts(errors)
    if not warnings and not errors:
        st.success("Race log looks clean.")
        return

    cols = st.columns(2)
    if errors:
        cols[0].error(f"Fix now: {len(errors)}")
    else:
        cols[0].success("Fix now: 0")
    if warnings:
        cols[1].warning(f"Watch: {len(warnings)}")
    else:
        cols[1].success("Watch: 0")

    with st.expander("Alert details", expanded=detail_expanded):
        if errors:
            st.markdown("**Fix now**")
            st.markdown("- " + "\n- ".join(errors[:8]))
            if len(errors) > 8:
                st.caption(f"{len(errors) - 8} more errors hidden.")
        if warnings:
            st.markdown("**Watch**")
            st.markdown("- " + "\n- ".join(warnings[:10]))
            if len(warnings) > 10:
                st.caption(f"{len(warnings) - 10} more checks hidden.")


def prioritize_alerts(messages: list[str]) -> list[str]:
    priority_terms = [
        "Unknown runner",
        "Every race-log row",
        "Duplicate",
        "overlap",
        "start before",
        "Finish times",
        "Start times",
        "in progress",
        "Order exception",
        "Sunday 13:00 BST",
        "Sunday 12:00 BST",
        "Pace check",
        "Cap warning",
        "Timing gap",
    ]

    def key(message: str) -> tuple[int, str]:
        for index, term in enumerate(priority_terms):
            if term.lower() in str(message).lower():
                return index, str(message)
        return len(priority_terms), str(message)

    return sorted(messages, key=key)


def race_day_sheet_context() -> dict[str, Any]:
    default_sheet_id = get_streamlit_secret("google_sheet_id", "race_log_google_sheet_id") or ""
    st.session_state.setdefault("race_day_sheet_id", str(default_sheet_id))
    st.session_state.setdefault("race_day_worksheet_name", "race_log")
    st.session_state.setdefault("editable_google_sheet_url", DEFAULT_EDITABLE_GOOGLE_SHEET_URL)

    sheet_id = str(st.session_state.get("race_day_sheet_id") or default_sheet_id).strip()
    worksheet_name = str(st.session_state.get("race_day_worksheet_name") or "race_log").strip() or "race_log"
    sheet_url = str(st.session_state.get("editable_google_sheet_url") or "").strip()
    return {
        "sheet_id": sheet_id,
        "worksheet_name": worksheet_name,
        "sheet_url": sheet_url,
        "configured": google_sheets_configured(st.secrets, sheet_id or None),
    }


def save_race_log_live(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    caps_enabled: bool,
    sync: Mapping[str, Any],
    notice: str | None = None,
) -> pd.DataFrame:
    warnings, errors = validate_race_log(log, roster, caps_enabled=caps_enabled)
    if errors:
        raise ValueError("; ".join(errors[:3]))

    clean = set_race_log(log, roster)
    if sync.get("configured"):
        write_race_log_to_google_sheet(
            clean,
            st.secrets,
            sheet_id=sync.get("sheet_id") or None,
            worksheet_name=str(sync.get("worksheet_name") or "race_log"),
        )
        saved_at = now_bst_label()
        st.session_state["race_day_sheet_last_saved"] = saved_at
        st.session_state["race_day_notice"] = notice or f"Race log saved to Google Sheet at {saved_at}."
    else:
        st.session_state["race_day_notice"] = "Race log saved locally. Google Sheets write access is not configured."
    if warnings:
        st.session_state["race_day_warnings_after_save"] = warnings
    return clean


def show_google_sheet_controls(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    caps_enabled: bool,
    sync: Mapping[str, Any],
) -> pd.DataFrame:
    sheet_id = str(sync["sheet_id"])
    worksheet_name = str(sync["worksheet_name"])
    sheet_url = str(sync["sheet_url"])
    configured = bool(sync["configured"])

    with st.container(border=True):
        top_cols = st.columns([1.5, 1, 1])
        with top_cols[0]:
            st.markdown("#### Live Sheet")
            if configured:
                st.caption("App changes save immediately. Sheet auto reload defaults to once per minute.")
            else:
                st.caption("Write access is not configured. Public-link loading and CSV backup are still available.")

        live_reload = top_cols[1].checkbox(
            "Auto reload",
            value=st.session_state.get("race_day_live_sheet_enabled", configured),
            help="Polls the Sheet while this Race Day tab is open. App saves still write immediately.",
        )
        refresh_seconds = top_cols[2].number_input(
            "Reload every seconds",
            min_value=10,
            max_value=300,
            value=live_refresh_seconds_default("race_day_live_refresh_seconds"),
            step=5,
        )
        st.session_state["race_day_live_sheet_enabled"] = live_reload
        st.session_state["race_day_live_refresh_seconds"] = int(refresh_seconds)

        if live_reload and st_autorefresh is not None:
            st_autorefresh(
                interval=int(refresh_seconds) * 1000,
                key="race_day_live_sheet_autorefresh",
            )
        elif live_reload and st_autorefresh is None:
            st.warning("Live reload needs the streamlit-autorefresh package. Manual loading still works.")

        action_cols = st.columns([1, 1, 1.2])
        load_now = action_cols[0].button("Load Sheet", width="stretch")
        save_now = action_cols[1].button("Save Sheet", width="stretch", disabled=not configured)
        if configured:
            action_cols[2].success("Google sync ready.")
        else:
            action_cols[2].warning("Save disabled until secrets are set.")

        should_load_sheet = load_now or live_reload
        if should_load_sheet:
            try:
                if configured:
                    loaded = read_race_log_from_google_sheet(
                        st.secrets,
                        sheet_id=sheet_id or None,
                        worksheet_name=worksheet_name,
                    )
                else:
                    loaded = read_race_log_from_google_sheet_url(sheet_url, roster)
                warnings, errors = validate_race_log(loaded, roster, caps_enabled=caps_enabled)
                if errors:
                    show_race_day_alerts(warnings, errors)
                else:
                    log = set_race_log(loaded, roster)
                    loaded_at = now_bst_label()
                    st.session_state["editable_google_sheet_last_loaded"] = loaded_at
                    if st.session_state.get("race_log_changed"):
                        st.session_state["race_day_notice"] = f"Race log loaded at {loaded_at}."
                        st.rerun()
                    else:
                        st.caption(f"Sheet checked at {loaded_at}; no race-log changes found.")
                    if warnings:
                        show_race_day_alerts(warnings, [])
            except Exception as exc:
                st.error(
                    "Could not load the Sheet. Check sharing, credentials, and worksheet name. "
                    f"Details: {exc}"
                )
        if st.session_state.get("editable_google_sheet_last_loaded"):
            st.caption(f"Last load: {st.session_state['editable_google_sheet_last_loaded']}.")
        if st.session_state.get("race_day_sheet_last_saved"):
            st.caption(f"Last save: {st.session_state['race_day_sheet_last_saved']}.")

        if save_now:
            try:
                log = save_race_log_live(log, roster, caps_enabled, sync, "Race log saved to the Sheet.")
                st.success("Race log saved to the Sheet.")
                if st.session_state.get("race_day_warnings_after_save"):
                    show_race_day_alerts(st.session_state.pop("race_day_warnings_after_save"), [])
            except ValueError:
                warnings, errors = validate_race_log(log, roster, caps_enabled=caps_enabled)
                show_race_day_alerts(warnings, errors)
            except Exception as exc:
                st.error(f"Could not save the Sheet: {exc}")

        with st.expander("Sheet settings", expanded=False):
            settings_cols = st.columns([2, 1])
            settings_cols[0].text_input("Sheet ID", key="race_day_sheet_id", placeholder="Google Sheet ID")
            settings_cols[1].text_input("Worksheet", key="race_day_worksheet_name")
            st.text_input(
                "Public Sheet link fallback",
                key="editable_google_sheet_url",
                placeholder="https://docs.google.com/spreadsheets/d/.../edit#gid=0",
            )
            template = pd.DataFrame(columns=GOOGLE_SHEET_TEMPLATE_COLUMNS)
            st.download_button(
                "Download Sheet template CSV",
                template.to_csv(index=False),
                file_name="endure24_google_sheet_race_log_template.csv",
                mime="text/csv",
                width="stretch",
            )
    return log


def show_race_log_entry(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    caps_enabled: bool,
    sync: Mapping[str, Any],
) -> pd.DataFrame:
    state = race_state_from_log(log, roster, target_laps=1, caps_enabled=caps_enabled)
    runner_options = roster.sort_values("running_order")["runner"].astype(str).tolist()
    if not runner_options:
        st.warning("Add at least one available runner before logging race laps.")
        return log

    fair_queue = build_live_fair_queue(log, roster, caps_enabled=caps_enabled, queue_size=1)
    recommended_runner = queue_next_runner(fair_queue) or state.next_runner
    next_runner_index = runner_options.index(recommended_runner) if recommended_runner in runner_options else 0

    st.markdown("#### Add Race Entry")
    with st.form("manual_lap_form", clear_on_submit=True):
        action_options = ["Add completed lap", "Record start only"]
        if state.current_lap_in_progress:
            action_options = ["Complete current lap"] + action_options

        action = st.radio("Action", action_options, horizontal=True)
        top_cols = st.columns([1.2, 1.4, 1.2])
        runner = top_cols[0].selectbox(
            "Runner",
            runner_options,
            index=next_runner_index,
            disabled=action == "Complete current lap",
        )
        if action == "Complete current lap" and state.in_progress_runner in runner_options:
            runner = state.in_progress_runner
            top_cols[1].caption(
                f"Completing lap {int(state.latest_lap['lap_number'])}, started {format_race_clock(state.in_progress_start_minute)}."
            )
        notes = top_cols[2].text_input("Notes", value="")
        if action != "Complete current lap" and recommended_runner and runner != recommended_runner:
            st.warning(
                f"Plan Assistant recommends {recommended_runner}. Saving {runner} is allowed if the team has changed the plan."
            )
        elif action != "Complete current lap" and state.next_runner and runner != state.next_runner:
            st.info(
                f"Live plan uses {runner}; fixed pre-race order expected {state.next_runner}. "
                "The lap will be saved and tracked as an order exception."
            )
        elif action != "Complete current lap" and recommended_runner:
            top_cols[1].caption(f"Recommended by Plan Assistant: {recommended_runner}")

        start_minute = None
        finish_minute = None
        duration_minutes = None
        start_text = ""
        finish_text = ""
        duration_text = ""

        if action == "Complete current lap":
            detail_mode = st.radio("Known timing", ["Finish time", "Lap duration"], horizontal=True)
            if detail_mode == "Finish time":
                finish_text = st.text_input("Finish time", placeholder="Sat 13:04, Sun 00:12, or 724")
            else:
                duration_text = st.text_input("Lap duration", placeholder="41.5, 41 min, or 41:30")
        elif action == "Record start only":
            start_text = st.text_input("Start time", placeholder="Sat 13:04, Sun 00:12, or 724")
        else:
            detail_mode = st.selectbox(
                "Known timing",
                ["Lap duration only", "Finish time only", "Start + finish", "Start + duration"],
            )
            detail_cols = st.columns(2)
            if detail_mode in {"Start + finish", "Start + duration"}:
                start_text = detail_cols[0].text_input("Start time", placeholder="Sat 13:04, Sun 00:12, or 724")
            if detail_mode in {"Finish time only", "Start + finish"}:
                target_col = detail_cols[1] if detail_mode == "Start + finish" else detail_cols[0]
                finish_text = target_col.text_input("Finish time", placeholder="Sat 13:45, Sun 00:52, or 765")
            if detail_mode in {"Lap duration only", "Start + duration"}:
                target_col = detail_cols[1] if detail_mode == "Start + duration" else detail_cols[0]
                duration_text = target_col.text_input("Lap duration", placeholder="41.5, 41 min, or 41:30")

        submitted = st.form_submit_button("Save race entry", type="primary")

    if submitted:
        if start_text:
            start_minute = parse_race_time_to_minute(start_text)
        if finish_text:
            finish_minute = parse_race_time_to_minute(finish_text)
        if duration_text:
            duration_minutes = parse_duration_minutes(duration_text)

        form_errors = []
        if state.current_lap_in_progress and action != "Complete current lap":
            form_errors.append("Complete the in-progress lap before adding another row.")
        if action == "Complete current lap":
            if finish_text and finish_minute is None:
                form_errors.append("Enter a valid finish time.")
            if duration_text and duration_minutes is None:
                form_errors.append("Enter a valid lap duration.")
            if not finish_text and not duration_text:
                form_errors.append("Enter either the finish time or lap duration.")
        elif action == "Record start only":
            if start_minute is None:
                form_errors.append("Enter a valid start time.")
        else:
            if start_text and start_minute is None:
                form_errors.append("Enter a valid start time.")
            if finish_text and finish_minute is None:
                form_errors.append("Enter a valid finish time.")
            if duration_text and duration_minutes is None:
                form_errors.append("Enter a valid lap duration.")
            if not finish_text and not duration_text:
                form_errors.append("Enter a finish time or lap duration.")

        if form_errors:
            for error in form_errors:
                st.error(error)
        else:
            try:
                if action == "Complete current lap":
                    candidate = complete_in_progress_lap(
                        log,
                        roster,
                        finish_minute=finish_minute,
                        duration_minutes=duration_minutes,
                        notes=notes,
                    )
                else:
                    candidate = append_manual_lap(
                        log,
                        roster,
                        runner=runner,
                        start_minute=start_minute,
                        finish_minute=finish_minute,
                        duration_minutes=duration_minutes,
                        notes=notes,
                    )
            except ValueError as exc:
                st.error(str(exc))
                candidate = None

        if not form_errors and candidate is not None:
            warnings, errors = validate_race_log(candidate, roster, caps_enabled=caps_enabled)
            if errors:
                show_race_day_alerts(warnings, errors)
            else:
                try:
                    log = save_race_log_live(candidate, roster, caps_enabled, sync, "Race log updated and saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Race log was valid, but could not save to the Sheet: {exc}")

    with st.expander("Bulk paste", expanded=False):
        pasted = st.text_area(
            "Paste rows",
            height=120,
            placeholder="runner,start_time,finish_time,lap_duration_minutes,notes\nToby,Sat 12:00 BST,Sat 12:41 BST,41,clean lap",
        )
        paste_col_1, paste_col_2 = st.columns(2)
        if paste_col_1.button("Append pasted rows", width="stretch"):
            try:
                pasted_log = parse_pasted_laps(pasted, roster)
                if not pasted_log.empty:
                    offset = int(log["lap_number"].max()) if not log.empty else 0
                    pasted_log["lap_number"] = pasted_log["lap_number"] + offset
                candidate = normalise_race_log(pd.concat([log, pasted_log], ignore_index=True), roster)
                warnings, errors = validate_race_log(candidate, roster, caps_enabled=caps_enabled)
                if errors:
                    show_race_day_alerts(warnings, errors)
                else:
                    try:
                        log = save_race_log_live(candidate, roster, caps_enabled, sync, "Pasted rows appended and saved.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Pasted rows were valid, but could not save to the Sheet: {exc}")
            except Exception as exc:
                st.error(f"Could not import pasted rows: {exc}")

        if paste_col_2.button("Replace with pasted table", width="stretch"):
            try:
                candidate = parse_pasted_laps(pasted, roster)
                warnings, errors = validate_race_log(candidate, roster, caps_enabled=caps_enabled)
                if errors:
                    show_race_day_alerts(warnings, errors)
                else:
                    try:
                        log = save_race_log_live(candidate, roster, caps_enabled, sync, "Race log replaced and saved.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Pasted table was valid, but could not save to the Sheet: {exc}")
            except Exception as exc:
                st.error(f"Could not import pasted table: {exc}")

    return log


def show_race_log_backup(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    caps_enabled: bool,
    sync: Mapping[str, Any],
) -> pd.DataFrame:
    c1, c2, c3, c4 = st.columns(4)
    uploaded_log = c1.file_uploader("Upload race-log CSV", type=["csv"], key="race_log_csv")
    if c2.button("Load CSV backup", width="stretch"):
        if uploaded_log is None:
            st.warning("Choose a CSV backup first.")
        else:
            try:
                log = save_race_log_live(
                    pd.read_csv(uploaded_log),
                    roster,
                    caps_enabled,
                    sync,
                    "CSV backup loaded and saved.",
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load and save CSV backup: {exc}")

    c3.download_button(
        "Download CSV backup",
        normalise_race_log(log, roster).to_csv(index=False),
        file_name="endure24_race_log.csv",
        mime="text/csv",
        width="stretch",
    )

    if c4.button("Undo last entry", width="stretch"):
        clean = normalise_race_log(log, roster)
        if clean.empty:
            st.warning("There is no race-log entry to undo.")
        else:
            last_lap = int(clean["lap_number"].max())
            try:
                log = save_race_log_live(
                    clean[clean["lap_number"].astype(int) != last_lap],
                    roster,
                    caps_enabled,
                    sync,
                    f"Removed lap {last_lap} and saved.",
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save undo to the Sheet: {exc}")

    if st.button("Clear race log"):
        try:
            log = save_race_log_live(empty_race_log(), roster, caps_enabled, sync, "Race log cleared and saved.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not clear and save the race log: {exc}")
    return log


def update_race_day_forecast(
    roster: pd.DataFrame,
    settings: SimulationSettings,
    log: pd.DataFrame,
    caps_enabled: bool,
) -> tuple[dict, dict]:
    forecast_settings = replace(settings, path_sample_size=60)
    baseline_output = simulate_many(roster, forecast_settings, caps_enabled=caps_enabled)
    live_output = forecast_from_race_log(roster, forecast_settings, log, caps_enabled=caps_enabled)
    baseline_bundle = summarize_results(baseline_output, settings.target_laps)
    live_bundle = summarize_results(live_output, settings.target_laps)
    st.session_state["race_day_forecast"] = {
        "baseline_bundle": baseline_bundle,
        "live_bundle": live_bundle,
        "baseline_output": baseline_output,
        "live_output": live_output,
    }
    st.session_state["race_log_changed"] = False
    return baseline_bundle, live_bundle


def show_race_day(roster: pd.DataFrame, settings: SimulationSettings, caps_enabled: bool) -> None:
    st.subheader("Race Day")
    if st.session_state.get("race_day_notice"):
        st.success(st.session_state.pop("race_day_notice"))
    if "race_log" not in st.session_state:
        st.session_state["race_log"] = empty_race_log()
    log = normalise_race_log(st.session_state["race_log"], roster)
    st.session_state["race_log"] = log
    race_sync = race_day_sheet_context()

    warnings, errors = validate_race_log(log, roster, caps_enabled=caps_enabled)
    state = race_state_from_log(log, roster, settings.target_laps, caps_enabled=caps_enabled)
    forecast_state = st.session_state.get("race_day_forecast", {})
    live_bundle = forecast_state.get("live_bundle")
    baseline_bundle = forecast_state.get("baseline_bundle")
    baseline_output = forecast_state.get("baseline_output")
    live_output = forecast_state.get("live_output")

    show_race_control_dashboard(
        log,
        roster,
        settings,
        state,
        baseline_output,
        live_output,
        live_bundle,
        warnings,
        errors,
        caps_enabled,
    )
    show_race_day_plan_assistant(log, roster, settings, state, caps_enabled)
    show_race_day_readiness(race_sync, warnings, errors, live_bundle, roster, settings, log, caps_enabled)
    show_race_day_sync_status(race_sync)
    show_race_day_alerts(warnings, errors)
    show_team_brief(log, roster, settings, state, live_bundle, warnings, errors, caps_enabled, race_sync)

    st.markdown("### Lap Desk")
    show_lap_handover_strip(log, roster, state, caps_enabled)
    c1, c2 = st.columns([1, 1])
    with c1:
        log = show_race_log_entry(log, roster, caps_enabled, race_sync)
    with c2:
        st.markdown("#### Latest Laps")
        display_log = format_race_log_for_display(log, roster, caps_enabled)
        if display_log.empty:
            st.info("No race laps logged yet.")
        else:
            st.dataframe(
                display_log.tail(12).sort_values("lap_number", ascending=False),
                width="stretch",
                hide_index=True,
                column_config={
                    "lap_number": st.column_config.NumberColumn("Lap", format="%d"),
                    "runner": "Runner",
                    "status": "Status",
                    "order_status": "Order",
                    "start_time": "Start",
                    "finish_time": "Finish",
                    "lap_duration_minutes": st.column_config.NumberColumn("Mins", format="%.1f"),
                    "notes": "Notes",
                },
            )
        with st.expander("Backup and undo", expanded=False):
            log = show_race_log_backup(log, roster, caps_enabled, race_sync)

    with st.expander("Live Sheet controls", expanded=False):
        log = show_google_sheet_controls(log, roster, caps_enabled, race_sync)

    with st.expander("Forecast controls and probability plot", expanded=bool(baseline_bundle and live_bundle)):
        forecast_col, auto_col = st.columns([1.2, 1])
        with forecast_col:
            if st.button("Update race-day forecast", type="primary", width="stretch"):
                if errors:
                    st.error("Fix race-log errors before forecasting.")
                else:
                    with st.spinner("Updating race-day forecast..."):
                        update_race_day_forecast(roster, settings, log, caps_enabled)
                        st.session_state["race_day_notice"] = "Race-day forecast updated."
                        st.rerun()
        auto_forecast = auto_col.checkbox(
            "Update after Sheet changes",
            value=st.session_state.get("race_day_live_auto_forecast", True),
            help="When live Sheet reload changes the log, refresh the forecast automatically.",
        )
        st.session_state["race_day_live_auto_forecast"] = auto_forecast
        if auto_forecast and st.session_state.get("race_log_changed") and not errors:
            with st.spinner("Live Sheet changed; updating race-day forecast..."):
                update_race_day_forecast(roster, settings, log, caps_enabled)
                st.session_state["race_day_notice"] = "Live Sheet changed; race-day forecast updated."
                st.rerun()

        if st.session_state.get("race_day_live_sheet_enabled"):
            st.caption("Live Sheet reload is on.")
        else:
            st.caption("Live Sheet reload is off.")

        if baseline_bundle and live_bundle:
            fig = final_lap_comparison_figure(
                baseline_bundle["final_lap_distribution"],
                live_bundle["final_lap_distribution"],
                "Pre-race forecast",
                "Actual-adjusted forecast",
                "Pre-race vs actual-adjusted final lap probability",
            )
            st.plotly_chart(fig, width="stretch")
        elif errors:
            st.info("Forecast plot is paused until race-log errors are fixed.")
        else:
            st.info("Run the race-day forecast to show the live probability plot.")

    # Keep the latest validated log in session after any nested controls changed it.
    st.session_state["race_log"] = normalise_race_log(log, roster)


def show_exports_tab(sim_output: dict | None, summary_bundle: dict | None, comparison: pd.DataFrame | None) -> None:
    if sim_output is None or summary_bundle is None:
        st.info("Run a forecast first to enable simulation exports.")
    else:
        show_downloads(sim_output, summary_bundle, comparison)

    if "race_log" in st.session_state:
        st.download_button(
            "Download current race log CSV",
            normalise_race_log(st.session_state["race_log"], pd.DataFrame({"runner": []})).to_csv(index=False),
            file_name="endure24_current_race_log.csv",
            mime="text/csv",
        )


def split_order(order: str) -> list[str]:
    return [part.strip() for part in str(order).replace("→", "->").split("->") if part.strip()]


def build_route_commentary(
    ranking: pd.DataFrame,
    roster: pd.DataFrame,
    ranking_objective: str,
    target_laps: int,
) -> list[str]:
    if ranking.empty:
        return []

    best = ranking.iloc[0]
    best_order = split_order(best["order"])
    if not best_order:
        return []

    clean_roster = roster.copy()
    clean_roster["projected_mean_minutes"] = pd.to_numeric(clean_roster["projected_mean_minutes"], errors="coerce")
    clean_roster["projected_sd_minutes"] = pd.to_numeric(clean_roster["projected_sd_minutes"], errors="coerce")
    clean_roster["running_order"] = pd.to_numeric(clean_roster["running_order"], errors="coerce")
    clean_roster["max_laps"] = pd.to_numeric(clean_roster["max_laps"], errors="coerce")
    roster_by_runner = clean_roster.set_index("runner")
    pace_rank = clean_roster.sort_values("projected_mean_minutes")["runner"].astype(str).tolist()
    top_routes = [split_order(order) for order in ranking["order"].head(10)]

    lines = []
    if ranking_objective == "target_probability":
        lines.append(
            f"This route was selected because it had the highest simulated chance of reaching {target_laps}+ laps "
            f"among the tested orders: {best['probability_target_laps']:.1%}, with {best['expected_official_laps']:.2f} expected laps."
        )
    else:
        lines.append(
            f"This route was selected because it had the highest simulated expected lap total among the tested orders: "
            f"{best['expected_official_laps']:.2f} expected laps, with {best['probability_target_laps']:.1%} chance of {target_laps}+ laps."
        )

    lines.append(
        "The explanation below is an interpretation of the Monte Carlo results. The simulator tests the whole 24-hour rotation, "
        "so a position can be useful because of late-race timing, caps, and who becomes available near Sunday noon."
    )

    for position, runner in enumerate(best_order, start=1):
        if runner not in roster_by_runner.index:
            lines.append(f"{position}. {runner}: selected in this slot by the simulation.")
            continue

        row = roster_by_runner.loc[runner]
        mean = row["projected_mean_minutes"]
        sd = row["projected_sd_minutes"]
        cap = row["max_laps"]
        current_position = row["running_order"]
        pace_position = pace_rank.index(runner) + 1 if runner in pace_rank else None
        top_route_same_slot = sum(1 for order in top_routes if len(order) >= position and order[position - 1] == runner)
        top_route_frequency = top_route_same_slot / max(len(top_routes), 1)

        reasons = []
        if position == 1:
            reasons.append("starts the race cleanly with no cutoff pressure")
        elif position <= 3:
            reasons.append("is placed early enough to shape the first several rotations")
        elif position >= len(best_order) - 1:
            reasons.append("is placed late in the cycle, which affects who is available around the Sunday noon cutoff")

        if pace_position is not None:
            if pace_position <= 2:
                reasons.append(f"is one of the two fastest projected runners ({mean:.1f} min mean)")
            elif pace_position <= 4:
                reasons.append(f"is in the faster half of the roster ({mean:.1f} min mean)")
            else:
                reasons.append(f"has a slower projected mean ({mean:.1f} min), so the route is using timing/cap effects rather than pure pace")

        if pd.notna(sd):
            reasons.append(f"uses a {sd:.1f} min SD in the simulations")

        if pd.notna(cap) and int(cap) > 0:
            reasons.append(f"is capped at {int(cap)} laps, so the rotation skips them after that cap is reached")
        else:
            reasons.append("is uncapped, so they remain eligible late if capped runners are skipped")

        if pd.notna(current_position) and int(current_position) != position:
            reasons.append(f"moves from current slot {int(current_position)} to slot {position}")
        else:
            reasons.append("stays in the same slot as the current order")

        if top_route_frequency >= 0.5:
            reasons.append(f"also appears in this exact slot in {top_route_frequency:.0%} of the top tested routes")

        lines.append(f"{position}. {runner}: " + "; ".join(reasons) + ".")

    cap_rows = clean_roster[pd.to_numeric(clean_roster["max_laps"], errors="coerce").fillna(-1).gt(0)]
    if not cap_rows.empty:
        capped_names = ", ".join(
            f"{row.runner} ({int(row.max_laps)})" for row in cap_rows.itertuples(index=False)
        )
        lines.append(
            f"Cap effect: capped runners are {capped_names}. Their placement matters because once they hit the cap, "
            "the simulator skips them and the later rotation compresses around the uncapped runners."
        )

    if pd.notna(best.get("likely_final_runner", pd.NA)) and str(best["likely_final_runner"]).strip():
        lines.append(
            f"Late-race effect: the most likely final-lap runner for this route is {best['likely_final_runner']}, "
            "which is part of why the order can differ from simply sorting by fastest mean pace."
        )

    return lines


def show_route_commentary(
    ranking: pd.DataFrame,
    roster: pd.DataFrame,
    ranking_objective: str,
    target_laps: int,
) -> None:
    commentary = build_route_commentary(ranking, roster, ranking_objective, target_laps)
    if not commentary:
        return
    with st.expander("Why this route?", expanded=True):
        st.markdown(commentary[0])
        st.markdown(commentary[1])
        for line in commentary[2:]:
            st.markdown(f"- {line}")


def show_optimizer(roster: pd.DataFrame, settings: SimulationSettings) -> None:
    st.subheader("Running Order Optimiser")
    target_route_laps = st.number_input(
        "Target laps for route finder",
        min_value=1,
        max_value=80,
        value=int(settings.target_laps),
        step=1,
    )
    objective_label = st.selectbox(
        "Optimiser objective",
        ["Maximise expected official laps", "Maximise chance of reference target"],
        index=0,
    )
    objective = "target_probability" if objective_label == "Maximise chance of reference target" else "max_expected_laps"
    st.caption(
        "The default objective matches the competition goal: complete as many official laps as possible. "
        "All available runners can move anywhere in the order unless you add locked positions below."
    )

    c1, c2, c3, c4 = st.columns(4)
    max_candidates = c1.number_input("Candidate orders", min_value=5, max_value=5_000, value=200, step=25)
    sims_per_order = c2.number_input("First-pass sims/order", min_value=50, max_value=5_000, value=400, step=50)
    final_sims = c3.number_input("Final sims/order", min_value=100, max_value=20_000, value=1_500, step=100)
    brute_force = c4.checkbox("Brute force if feasible", value=False)

    st.caption("Optional locked positions. Leave every position blank for a fully unconstrained order search.")
    lock_table = pd.DataFrame({"runner": roster.sort_values("running_order")["runner"], "locked_position": pd.NA})
    edited_locks = st.data_editor(
        lock_table,
        hide_index=True,
        width="stretch",
        column_config={
            "runner": st.column_config.TextColumn("Runner", disabled=True),
            "locked_position": st.column_config.NumberColumn("Locked position", min_value=1, max_value=len(lock_table), step=1),
        },
        key="locked_positions_editor",
    )
    locked_positions = {
        int(row["locked_position"]): str(row["runner"])
        for _, row in edited_locks.dropna(subset=["locked_position"]).iterrows()
    }

    button_col_1, button_col_2 = st.columns(2)
    optimise_most_laps = button_col_1.button("Optimise for most laps")
    find_target_route = button_col_2.button("Find route for target")
    run_selected_objective = st.button("Optimise selected objective")

    if optimise_most_laps or find_target_route or run_selected_objective:
        objective_to_run = objective
        if optimise_most_laps:
            objective_to_run = "max_expected_laps"
        if find_target_route:
            objective_to_run = "target_probability"

        with st.spinner("Testing running orders..."):
            opt_settings = replace(settings, path_sample_size=0)
            ranking = optimize_running_orders(
                roster,
                opt_settings,
                target_laps=int(target_route_laps),
                objective=objective_to_run,
                max_candidates=int(max_candidates),
                sims_per_order=int(sims_per_order),
                final_sims=int(final_sims),
                top_n=10,
                brute_force_if_feasible=brute_force,
                locked_positions=locked_positions,
            )
        st.session_state["order_ranking"] = ranking
        st.session_state["order_ranking_context"] = {
            "objective": objective_to_run,
            "target_laps": int(target_route_laps),
        }

    if "order_ranking" in st.session_state:
        ranking = st.session_state["order_ranking"]
        context = st.session_state.get(
            "order_ranking_context",
            {"objective": objective, "target_laps": int(target_route_laps)},
        )
        ranking_objective = context["objective"]
        ranking_target_laps = int(context["target_laps"])
        if not ranking.empty:
            best = ranking.iloc[0]
            if ranking_objective == "target_probability":
                st.success(
                    f"Best route for {ranking_target_laps} laps: {best['order']} "
                    f"({best['probability_target_laps']:.1%} estimated probability; "
                    f"{best['expected_official_laps']:.2f} expected laps)."
                )
            else:
                st.success(
                    f"Best route for maximum laps: {best['order']} "
                    f"({best['expected_official_laps']:.2f} expected laps; "
                    f"{best['probability_target_laps']:.1%} chance of {ranking_target_laps}+ laps)."
                )
        st.dataframe(
            ranking.style.format(
                {
                    "expected_official_laps": "{:.2f}",
                    "median_official_laps": "{:.1f}",
                    "p10_official_laps": "{:.1f}",
                    "p90_official_laps": "{:.1f}",
                    "probability_target_laps": "{:.1%}",
                    "probability_target_plus_one_laps": "{:.1%}",
                    "probability_missed_final_cutoff": "{:.1%}",
                    "probability_ran_out_of_eligible_runners": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        show_route_commentary(ranking, roster, ranking_objective, ranking_target_laps)
        fig = px.bar(
            ranking,
            x="rank",
            y="expected_official_laps" if ranking_objective == "max_expected_laps" else "probability_target_laps",
            hover_data=["order", "expected_official_laps", "probability_target_laps", "likely_final_runner"],
            title="Top orders by expected official laps"
            if ranking_objective == "max_expected_laps"
            else f"Top routes by chance of {ranking_target_laps}+ laps",
        )
        if ranking_objective == "target_probability":
            fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.download_button(
            "Download optimiser ranking CSV",
            ranking.to_csv(index=False),
            file_name="endure24_running_order_ranking.csv",
            mime="text/csv",
        )


def override_runner_mean(roster: pd.DataFrame, runner: str, new_mean_minutes: float) -> pd.DataFrame:
    adjusted = roster.copy()
    adjusted.loc[adjusted["runner"].astype(str) == runner, "projected_mean_minutes"] = float(new_mean_minutes)
    return adjusted


def what_if_summary_table(
    baseline_summary: dict,
    override_summary: dict,
    override_label: str,
) -> pd.DataFrame:
    rows = [
        {
            "scenario": "Current assumptions",
            "runner_override": "",
            "expected_laps": baseline_summary["expected_official_laps"],
            "probability_reference_target": baseline_summary["probability_target_laps"],
            "median_laps": baseline_summary["median_official_laps"],
            "p10_laps": baseline_summary["p10_official_laps"],
            "p90_laps": baseline_summary["p90_official_laps"],
            "average_final_finish": format_race_clock(baseline_summary["average_final_finish_minute"]),
        },
        {
            "scenario": "What-if override",
            "runner_override": override_label,
            "expected_laps": override_summary["expected_official_laps"],
            "probability_reference_target": override_summary["probability_target_laps"],
            "median_laps": override_summary["median_official_laps"],
            "p10_laps": override_summary["p10_official_laps"],
            "p90_laps": override_summary["p90_official_laps"],
            "average_final_finish": format_race_clock(override_summary["average_final_finish_minute"]),
        },
    ]
    table = pd.DataFrame(rows)
    table["expected_lap_delta"] = table["expected_laps"] - table.loc[0, "expected_laps"]
    table["target_probability_delta"] = (
        table["probability_reference_target"] - table.loc[0, "probability_reference_target"]
    )
    return table


def apply_runner_mean_overrides(roster: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    adjusted = roster.copy()
    for row in overrides.itertuples(index=False):
        adjusted.loc[
            adjusted["runner"].astype(str) == str(row.runner),
            "projected_mean_minutes",
        ] = float(row.what_if_mean_minutes)
    return adjusted


def show_what_if_pace_override(roster: pd.DataFrame, settings: SimulationSettings) -> None:
    st.subheader("What-if Pace Override")
    st.caption(
        "Use this for quick questions like: what if one or more runners average 1 minute faster? "
        "It does not permanently edit the roster above."
    )

    clean = roster.copy()
    clean["runner"] = clean["runner"].astype(str)
    clean["projected_mean_minutes"] = pd.to_numeric(clean["projected_mean_minutes"], errors="coerce")
    clean = clean.dropna(subset=["runner", "projected_mean_minutes"])
    if clean.empty:
        st.info("Add at least one runner with a projected mean before running a what-if.")
        return

    override_table = (
        clean.sort_values("running_order")[["runner", "projected_mean_minutes"]]
        .rename(columns={"projected_mean_minutes": "current_mean_minutes"})
        .reset_index(drop=True)
    )
    override_table["apply"] = False
    override_table["what_if_mean_minutes"] = override_table["current_mean_minutes"]
    edited_overrides = st.data_editor(
        override_table[["apply", "runner", "current_mean_minutes", "what_if_mean_minutes"]],
        hide_index=True,
        width="stretch",
        column_config={
            "apply": st.column_config.CheckboxColumn("Apply"),
            "runner": st.column_config.TextColumn("Runner", disabled=True),
            "current_mean_minutes": st.column_config.NumberColumn("Current mean", disabled=True, format="%.1f"),
            "what_if_mean_minutes": st.column_config.NumberColumn(
                "What-if mean",
                min_value=1.0,
                max_value=120.0,
                step=0.1,
                format="%.1f",
            ),
        },
        key="what_if_multi_runner_editor",
    )
    run_target_route = st.checkbox("Also find target route", value=False)

    if st.button("Run what-if recalculation"):
        active_overrides = edited_overrides[
            edited_overrides["apply"].astype(bool)
            & pd.to_numeric(edited_overrides["what_if_mean_minutes"], errors="coerce").notna()
        ].copy()
        active_overrides["current_mean_minutes"] = pd.to_numeric(
            active_overrides["current_mean_minutes"],
            errors="coerce",
        )
        active_overrides["what_if_mean_minutes"] = pd.to_numeric(
            active_overrides["what_if_mean_minutes"],
            errors="coerce",
        )
        if active_overrides.empty:
            st.warning("Tick Apply for at least one runner and enter a what-if mean time.")
        else:
            adjusted_roster = apply_runner_mean_overrides(roster, active_overrides)
            override_label = "; ".join(
                f"{row.runner}: {row.current_mean_minutes:.1f} -> {row.what_if_mean_minutes:.1f} min"
                for row in active_overrides.itertuples(index=False)
            )
            with st.spinner("Running what-if comparison..."):
                baseline_output = simulate_many(roster, settings, caps_enabled=True)
                override_output = simulate_many(adjusted_roster, settings, caps_enabled=True)
                baseline_bundle = summarize_results(baseline_output, settings.target_laps)
                baseline_summary = baseline_bundle["summary"]
                override_bundle = summarize_results(override_output, settings.target_laps)
                override_summary = override_bundle["summary"]

                target_ranking = None
                if run_target_route:
                    opt_settings = replace(settings, path_sample_size=0)
                    target_ranking = optimize_running_orders(
                        adjusted_roster,
                        opt_settings,
                        target_laps=settings.target_laps,
                        objective="target_probability",
                        max_candidates=200,
                        sims_per_order=400,
                        final_sims=1_500,
                        top_n=10,
                        brute_force_if_feasible=False,
                        locked_positions={},
                    )

            st.session_state["what_if_result"] = {
                "override_label": override_label,
                "adjusted_roster": adjusted_roster,
                "comparison": what_if_summary_table(
                    baseline_summary,
                    override_summary,
                    override_label,
                ),
                "baseline_bundle": baseline_bundle,
                "override_bundle": override_bundle,
                "target_ranking": target_ranking,
            }

    if "what_if_result" in st.session_state and "override_label" not in st.session_state["what_if_result"]:
        del st.session_state["what_if_result"]

    if "what_if_result" in st.session_state:
        result = st.session_state["what_if_result"]
        comparison = result["comparison"]
        override_row = comparison.loc[comparison["scenario"] == "What-if override"].iloc[0]
        st.success(
            f"What-if result for {result['override_label']}: "
            f"{override_row['expected_laps']:.2f} expected laps "
            f"({override_row['expected_lap_delta']:+.2f} vs current), "
            f"{override_row['probability_reference_target']:.1%} chance of {settings.target_laps}+ laps "
            f"({override_row['target_probability_delta']:+.1%})."
        )
        st.dataframe(
            comparison.style.format(
                {
                    "expected_laps": "{:.2f}",
                    "expected_lap_delta": "{:+.2f}",
                    "probability_reference_target": "{:.1%}",
                    "target_probability_delta": "{:+.1%}",
                    "median_laps": "{:.1f}",
                    "p10_laps": "{:.1f}",
                    "p90_laps": "{:.1f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        runner_laps = result["override_bundle"]["expected_runner_laps"]
        lap_pdf_fig = comparison_final_lap_probability_figure(
            result["baseline_bundle"]["final_lap_distribution"],
            result["override_bundle"]["final_lap_distribution"],
        )
        st.plotly_chart(lap_pdf_fig, width="stretch")
        st.caption("Final lap count is discrete, so this is a PMF/PDF-style probability plot.")

        fig = px.bar(
            runner_laps,
            x="runner",
            y="expected_laps",
            title="What-if expected laps per runner",
        )
        st.plotly_chart(fig, width="stretch")

        target_ranking = result.get("target_ranking")
        if target_ranking is not None and not target_ranking.empty:
            best = target_ranking.iloc[0]
            st.success(
                f"Best target route with this override: {best['order']} "
                f"({best['probability_target_laps']:.1%} chance of {settings.target_laps}+ laps; "
                f"{best['expected_official_laps']:.2f} expected laps)."
            )
            st.dataframe(
                target_ranking.style.format(
                    {
                        "expected_official_laps": "{:.2f}",
                        "median_official_laps": "{:.1f}",
                        "p10_official_laps": "{:.1f}",
                        "p90_official_laps": "{:.1f}",
                        "probability_target_laps": "{:.1%}",
                        "probability_target_plus_one_laps": "{:.1%}",
                        "probability_missed_final_cutoff": "{:.1%}",
                        "probability_ran_out_of_eligible_runners": "{:.1%}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            show_route_commentary(target_ranking, result["adjusted_roster"], "target_probability", settings.target_laps)


def runner_drink_unit(runner: str) -> str:
    return "Rubicons consumed" if str(runner).strip().lower() == "jared" else "Drinks"


DRINKS_SHEET_COLUMNS = ["runner", "count", "unit", "notes"]


def _drinks_secret_value(secrets: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        try:
            value = secrets.get(key)
        except Exception:
            value = None
        if value:
            return value
    return None


def _drinks_service_account_info(secrets: Mapping[str, Any]) -> dict[str, Any] | None:
    value = _drinks_secret_value(
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


def _open_drinks_sheet(secrets: Mapping[str, Any], sheet_id: str | None, worksheet_name: str):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as exc:  # pragma: no cover - depends on cloud packages.
        raise RuntimeError("Install gspread and google-auth to use Google Sheets sync.") from exc

    service_account = _drinks_service_account_info(secrets)
    resolved_sheet_id = sheet_id or _drinks_secret_value(secrets, "google_sheet_id", "race_log_google_sheet_id")
    if not service_account or not resolved_sheet_id:
        raise RuntimeError("Google Sheets sync needs google_sheet_id and gcp_service_account in Streamlit secrets.")

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_info(service_account, scopes=scopes)
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(str(resolved_sheet_id))
    try:
        return spreadsheet.worksheet(worksheet_name)
    except Exception:
        return spreadsheet.add_worksheet(title=worksheet_name, rows=200, cols=len(DRINKS_SHEET_COLUMNS))


def read_drinks_from_google_sheet(
    secrets: Mapping[str, Any],
    sheet_id: str | None = None,
    worksheet_name: str = "drinks",
) -> pd.DataFrame:
    sheet = _open_drinks_sheet(secrets, sheet_id, worksheet_name)
    return pd.DataFrame(sheet.get_all_records())


def write_drinks_to_google_sheet(
    drinks: pd.DataFrame,
    secrets: Mapping[str, Any],
    sheet_id: str | None = None,
    worksheet_name: str = "drinks",
) -> None:
    sheet = _open_drinks_sheet(secrets, sheet_id, worksheet_name)
    clean = drinks.copy()
    for column in DRINKS_SHEET_COLUMNS:
        if column not in clean:
            clean[column] = ""
    clean = clean[DRINKS_SHEET_COLUMNS]
    values = [DRINKS_SHEET_COLUMNS] + clean.fillna("").astype(str).values.tolist()
    sheet.clear()
    sheet.update(values)


def normalise_drinks_table(raw: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    runners = roster.sort_values("running_order")["runner"].astype(str).str.strip()
    runners = runners[runners.ne("")].drop_duplicates().tolist()
    current = raw.copy() if raw is not None and not raw.empty else pd.DataFrame(columns=["runner", "count", "notes"])
    current.columns = [str(column).strip().lower() for column in current.columns]
    if "runner" not in current:
        current["runner"] = ""
    if "count" not in current:
        current["count"] = 0
    if "notes" not in current:
        current["notes"] = ""
    current["runner"] = current["runner"].astype(str).str.strip()
    current["count"] = pd.to_numeric(current["count"], errors="coerce").fillna(0).clip(lower=0).round().astype(int)
    current["notes"] = current["notes"].fillna("").astype(str)
    current = current[current["runner"].ne("")].drop_duplicates("runner", keep="last")

    existing = set(current["runner"].astype(str))
    missing = [runner for runner in runners if runner not in existing]
    if missing:
        current = pd.concat(
            [
                current,
                pd.DataFrame({"runner": missing, "count": [0 for _ in missing], "notes": ["" for _ in missing]}),
            ],
            ignore_index=True,
        )

    current = current[current["runner"].isin(runners)].copy()
    order_lookup = {runner: index for index, runner in enumerate(runners)}
    current["_order"] = current["runner"].map(order_lookup)
    current = current.sort_values(["_order", "runner"]).drop(columns="_order").reset_index(drop=True)
    current["unit"] = current["runner"].map(runner_drink_unit)
    return current


def drinks_state_table(roster: pd.DataFrame) -> pd.DataFrame:
    if "drinks_tracker" not in st.session_state:
        st.session_state["drinks_tracker"] = pd.DataFrame(columns=["runner", "count", "notes"])
    current = normalise_drinks_table(st.session_state["drinks_tracker"], roster)
    st.session_state["drinks_tracker"] = current[["runner", "count", "notes"]].copy()
    return current


def set_drinks_state(table: pd.DataFrame) -> pd.DataFrame:
    roster = st.session_state.get("current_roster_for_drinks")
    if not isinstance(roster, pd.DataFrame) or roster.empty:
        roster = pd.DataFrame(
            {
                "runner": table["runner"].astype(str).str.strip().tolist() if "runner" in table else [],
                "running_order": list(range(1, len(table) + 1)),
            }
        )
    clean = normalise_drinks_table(table, roster)
    st.session_state["drinks_tracker"] = clean[["runner", "count", "notes"]].copy()
    return clean


def drinks_sheet_payload(table: pd.DataFrame) -> pd.DataFrame:
    clean = table.copy()
    clean["unit"] = clean["runner"].map(runner_drink_unit)
    return clean[["runner", "count", "unit", "notes"]]


def drinks_digest(table: pd.DataFrame) -> str:
    return hashlib.sha256(table[["runner", "count", "notes"]].to_csv(index=False).encode("utf-8")).hexdigest()


def drinks_sheet_context() -> dict[str, Any]:
    default_sheet_id = get_streamlit_secret("google_sheet_id", "race_log_google_sheet_id") or ""
    default_sheet_id = st.session_state.get("race_day_sheet_id", default_sheet_id)
    st.session_state.setdefault("drinks_sheet_id", str(default_sheet_id))
    st.session_state.setdefault("drinks_worksheet_name", "drinks")

    sheet_id = str(st.session_state.get("drinks_sheet_id") or default_sheet_id).strip()
    worksheet_name = str(st.session_state.get("drinks_worksheet_name") or "drinks").strip() or "drinks"
    return {
        "sheet_id": sheet_id,
        "worksheet_name": worksheet_name,
        "configured": google_sheets_configured(st.secrets, sheet_id or None),
    }


def save_drinks_live(table: pd.DataFrame, sync: Mapping[str, Any], notice: str | None = None) -> pd.DataFrame:
    clean = set_drinks_state(table)
    if not sync.get("configured"):
        st.session_state["drinks_sheet_notice"] = "Saved locally. Google Sheets is not configured."
        return clean

    write_drinks_to_google_sheet(
        drinks_sheet_payload(clean),
        st.secrets,
        sheet_id=sync.get("sheet_id") or None,
        worksheet_name=str(sync.get("worksheet_name") or "drinks"),
    )
    saved_at = now_bst_label()
    st.session_state["drinks_sheet_last_saved"] = saved_at
    st.session_state["drinks_sheet_notice"] = notice or f"Drinks saved to Google Sheet at {saved_at}."
    return clean


def show_drinks_sheet_controls(table: pd.DataFrame, roster: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sync = drinks_sheet_context()
    sheet_id = str(sync["sheet_id"])
    worksheet_name = str(sync["worksheet_name"])
    configured = bool(sync["configured"])

    with st.container(border=True):
        top_cols = st.columns([1.5, 1, 1])
        with top_cols[0]:
            st.markdown("#### Drinks Sheet")
            if configured:
                st.caption("App changes save immediately. Sheet auto reload defaults to once per minute.")
            else:
                st.caption("Google sync is not configured. Local tracking and CSV download still work.")

        live_reload = top_cols[1].checkbox(
            "Auto reload",
            value=st.session_state.get("drinks_live_sheet_enabled", configured),
            help="Polls the drinks worksheet while this tab is open. App saves still write immediately.",
        )
        refresh_seconds = top_cols[2].number_input(
            "Reload every seconds",
            min_value=10,
            max_value=300,
            value=live_refresh_seconds_default("drinks_live_refresh_seconds"),
            step=5,
            key="drinks_refresh_seconds_input",
        )
        st.session_state["drinks_live_sheet_enabled"] = live_reload
        st.session_state["drinks_live_refresh_seconds"] = int(refresh_seconds)

        if live_reload and st_autorefresh is not None:
            st_autorefresh(
                interval=int(refresh_seconds) * 1000,
                key="drinks_live_sheet_autorefresh",
            )
        elif live_reload and st_autorefresh is None:
            st.warning("Live reload needs the streamlit-autorefresh package. Manual loading still works.")

        action_cols = st.columns([1, 1, 1.2])
        load_now = action_cols[0].button("Load drinks", width="stretch", disabled=not configured)
        save_now = action_cols[1].button("Save drinks", width="stretch", disabled=not configured)
        if configured:
            action_cols[2].success(f"Worksheet: {worksheet_name}")
        else:
            action_cols[2].warning("Save disabled until secrets are set.")

        if (load_now or live_reload) and configured:
            try:
                loaded = read_drinks_from_google_sheet(
                    st.secrets,
                    sheet_id=sheet_id or None,
                    worksheet_name=worksheet_name,
                )
                before = drinks_digest(table)
                table = set_drinks_state(loaded)
                loaded_at = now_bst_label()
                st.session_state["drinks_sheet_last_loaded"] = loaded_at
                if drinks_digest(table) != before:
                    st.success(f"Drinks loaded at {loaded_at}.")
                else:
                    st.caption(f"Drinks checked at {loaded_at}; no changes found.")
            except Exception as exc:
                st.error(f"Could not load drinks worksheet: {exc}")

        if st.session_state.get("drinks_sheet_last_loaded"):
            st.caption(f"Last drinks load: {st.session_state['drinks_sheet_last_loaded']}.")
        if st.session_state.get("drinks_sheet_last_saved"):
            st.caption(f"Last drinks save: {st.session_state['drinks_sheet_last_saved']}.")
        if st.session_state.get("drinks_sheet_notice"):
            st.success(st.session_state.pop("drinks_sheet_notice"))

        if save_now and configured:
            try:
                table = save_drinks_live(table, sync, "Drinks saved to Google Sheet.")
                st.success("Drinks saved to Google Sheet.")
            except Exception as exc:
                st.error(f"Could not save drinks worksheet: {exc}")

        with st.expander("Drinks Sheet settings", expanded=False):
            settings_cols = st.columns([2, 1])
            settings_cols[0].text_input("Sheet ID", key="drinks_sheet_id", placeholder="Google Sheet ID")
            settings_cols[1].text_input("Worksheet", key="drinks_worksheet_name")

    return drinks_state_table(roster), sync


def show_drinks_tab(roster: pd.DataFrame) -> None:
    st.subheader("Drinks")
    st.caption("For morale monitoring only. This does not affect race forecasts, runner order, or pacing assumptions.")
    st.session_state["current_roster_for_drinks"] = roster.copy()

    table = drinks_state_table(roster)
    table, drinks_sync = show_drinks_sheet_controls(table, roster)
    if table.empty:
        st.info("Add runners in Setup before tracking drinks.")
        return

    total = int(table["count"].sum())
    leader = table.sort_values(["count", "runner"], ascending=[False, True]).iloc[0]
    jared_rows = table[table["runner"].astype(str).str.lower().eq("jared")]
    jared_count = int(jared_rows.iloc[0]["count"]) if not jared_rows.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total logged", f"{total}")
    c2.metric("Current leader", f"{leader['runner']} ({int(leader['count'])})")
    c3.metric("Jared", f"{jared_count} Rubicons")

    st.markdown("#### Quick Add")
    for start in range(0, len(table), 4):
        cols = st.columns(4)
        for col, row in zip(cols, table.iloc[start : start + 4].itertuples(index=False)):
            if col.button(
                f"+1 {row.runner}",
                key=f"drink_plus_{row.runner}",
                help=runner_drink_unit(row.runner),
                width="stretch",
            ):
                updated = table.copy()
                updated.loc[updated["runner"].astype(str) == str(row.runner), "count"] += 1
                try:
                    save_drinks_live(
                        updated,
                        drinks_sync,
                        f"Added 1 for {row.runner} and saved to Google Sheet.",
                    )
                except Exception as exc:
                    set_drinks_state(updated)
                    st.session_state["drinks_sheet_notice"] = f"Updated locally, but could not save to Sheet: {exc}"
                st.rerun()

    edited = st.data_editor(
        table[["runner", "count", "unit", "notes"]],
        hide_index=True,
        width="stretch",
        column_config={
            "runner": st.column_config.TextColumn("Runner", disabled=True),
            "count": st.column_config.NumberColumn("Count", min_value=0, max_value=200, step=1),
            "unit": st.column_config.TextColumn("Unit", disabled=True),
            "notes": st.column_config.TextColumn("Notes"),
        },
        key=f"drinks_tracker_editor_{drinks_digest(table)[:12]}",
    )
    edited_clean = normalise_drinks_table(edited, roster)
    if drinks_digest(edited_clean) != drinks_digest(table):
        try:
            save_drinks_live(edited_clean, drinks_sync, "Drink table edit saved to Google Sheet.")
        except Exception as exc:
            set_drinks_state(edited_clean)
            st.session_state["drinks_sheet_notice"] = f"Updated locally, but could not save to Sheet: {exc}"
        st.rerun()

    action_cols = st.columns(3)
    if action_cols[0].button(
        "Force save to Sheet",
        type="primary",
        width="stretch",
        disabled=not drinks_sync.get("configured"),
    ):
        try:
            saved = save_drinks_live(edited, drinks_sync, "Drinks forced saved to Google Sheet.")
            st.success(f"Saved {int(saved['count'].sum())} total logged items.")
        except Exception as exc:
            st.error(f"Could not save drinks worksheet: {exc}")
    if action_cols[1].button("Reset drinks", width="stretch"):
        reset = table.copy()
        reset["count"] = 0
        reset["notes"] = ""
        try:
            save_drinks_live(reset, drinks_sync, "Drinks reset and saved to Google Sheet.")
        except Exception as exc:
            set_drinks_state(reset)
            st.session_state["drinks_sheet_notice"] = f"Reset locally, but could not save to Sheet: {exc}"
        st.rerun()
    action_cols[2].download_button(
        "Download drinks CSV",
        drinks_state_table(roster).to_csv(index=False),
        file_name="endure24_drinks_tracker.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown("#### Silly Graphs")
    chart_table = drinks_state_table(roster)
    bar = px.bar(
        chart_table.sort_values("count", ascending=False),
        x="runner",
        y="count",
        color="runner",
        text="count",
        title="Questionable Refreshment Leaderboard",
    )
    bar.update_layout(showlegend=False, yaxis_title="Logged items", xaxis_title="")
    st.plotly_chart(bar, width="stretch")

    left, right = st.columns(2)
    with left:
        donut = px.pie(
            chart_table,
            values="count",
            names="runner",
            hole=0.55,
            title="Share of the Nonsense",
        )
        st.plotly_chart(donut, width="stretch")

    with right:
        jared_max = max(5, jared_count + 2, int(chart_table["count"].max()) + 1)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=jared_count,
                title={"text": "Jared Rubicons consumed"},
                gauge={
                    "axis": {"range": [0, jared_max]},
                    "bar": {"color": "#2f6f8f"},
                    "steps": [
                        {"range": [0, max(1, jared_max * 0.4)], "color": "#e9f5ee"},
                        {"range": [max(1, jared_max * 0.4), max(2, jared_max * 0.75)], "color": "#fff4d6"},
                        {"range": [max(2, jared_max * 0.75), jared_max], "color": "#ffe2e2"},
                    ],
                },
            )
        )
        gauge.update_layout(height=320, margin={"l": 20, "r": 20, "t": 50, "b": 20})
        st.plotly_chart(gauge, width="stretch")

    matrix = chart_table.copy()
    order_lookup = (
        roster.assign(runner=roster["runner"].astype(str).str.strip())
        .set_index("runner")["running_order"]
        .to_dict()
    )
    matrix["running_order"] = matrix["runner"].map(order_lookup)
    matrix["bubble_size"] = matrix["count"].clip(lower=1)
    scatter = px.scatter(
        matrix,
        x="running_order",
        y="count",
        size="bubble_size",
        color="runner",
        hover_data=["unit", "notes"],
        title="Pit Stop Philosophy Matrix",
    )
    scatter.update_layout(xaxis_title="Running order", yaxis_title="Logged items", showlegend=False)
    st.plotly_chart(scatter, width="stretch")


SLEEP_SHEET_COLUMNS = ["entry_id", "runner", "hours", "unit", "recorded_at", "notes"]


def _open_sleep_sheet(secrets: Mapping[str, Any], sheet_id: str | None, worksheet_name: str):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as exc:  # pragma: no cover - depends on cloud packages.
        raise RuntimeError("Install gspread and google-auth to use Google Sheets sync.") from exc

    service_account = _drinks_service_account_info(secrets)
    resolved_sheet_id = sheet_id or _drinks_secret_value(secrets, "google_sheet_id", "race_log_google_sheet_id")
    if not service_account or not resolved_sheet_id:
        raise RuntimeError("Google Sheets sync needs google_sheet_id and gcp_service_account in Streamlit secrets.")

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_info(service_account, scopes=scopes)
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(str(resolved_sheet_id))
    try:
        return spreadsheet.worksheet(worksheet_name)
    except Exception:
        return spreadsheet.add_worksheet(title=worksheet_name, rows=500, cols=len(SLEEP_SHEET_COLUMNS))


def read_sleep_from_google_sheet(
    secrets: Mapping[str, Any],
    sheet_id: str | None = None,
    worksheet_name: str = "sleep",
) -> pd.DataFrame:
    sheet = _open_sleep_sheet(secrets, sheet_id, worksheet_name)
    return pd.DataFrame(sheet.get_all_records())


def write_sleep_to_google_sheet(
    sleep: pd.DataFrame,
    secrets: Mapping[str, Any],
    sheet_id: str | None = None,
    worksheet_name: str = "sleep",
) -> None:
    sheet = _open_sleep_sheet(secrets, sheet_id, worksheet_name)
    clean = sleep.copy()
    for column in SLEEP_SHEET_COLUMNS:
        if column not in clean:
            clean[column] = ""
    clean = clean[SLEEP_SHEET_COLUMNS]
    values = [SLEEP_SHEET_COLUMNS] + clean.fillna("").astype(str).values.tolist()
    sheet.clear()
    sheet.update(values)


def _sleep_entry_id(row: pd.Series, index: int) -> str:
    existing = str(row.get("entry_id", "") or "").strip()
    if existing:
        return existing
    payload = "|".join(
        [
            str(index),
            str(row.get("runner", "")),
            str(row.get("hours", "")),
            str(row.get("recorded_at", "")),
            str(row.get("notes", "")),
        ]
    )
    return f"sleep_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def normalise_sleep_log(raw: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    runners = roster.sort_values("running_order")["runner"].astype(str).str.strip()
    runners = runners[runners.ne("")].drop_duplicates().tolist()
    current = raw.copy() if raw is not None and not raw.empty else pd.DataFrame(columns=SLEEP_SHEET_COLUMNS)
    column_aliases = {
        "id": "entry_id",
        "log_id": "entry_id",
        "sleep_id": "entry_id",
        "sleep": "hours",
        "sleep_hours": "hours",
        "hours_sleep": "hours",
        "hours_slept": "hours",
        "hrs": "hours",
        "count": "hours",
        "time": "recorded_at",
        "timestamp": "recorded_at",
        "recorded": "recorded_at",
        "recorded_time": "recorded_at",
        "recorded_bst": "recorded_at",
    }
    current.columns = [
        column_aliases.get(str(column).strip().lower().replace(" ", "_").replace("-", "_"), str(column).strip().lower())
        for column in current.columns
    ]
    if "entry_id" not in current:
        current["entry_id"] = ""
    if "runner" not in current:
        current["runner"] = ""
    if "hours" not in current:
        current["hours"] = 0.0
    if "unit" not in current:
        current["unit"] = "hours"
    if "recorded_at" not in current:
        current["recorded_at"] = "Initial total"
    if "notes" not in current:
        current["notes"] = ""
    current["entry_id"] = current["entry_id"].fillna("").astype(str).str.strip()
    current["runner"] = current["runner"].astype(str).str.strip()
    current["hours"] = pd.to_numeric(current["hours"], errors="coerce").fillna(0).clip(lower=0, upper=48).round(2)
    current["unit"] = "hours"
    current["recorded_at"] = current["recorded_at"].fillna("Initial total").astype(str).str.strip()
    current.loc[current["recorded_at"].eq(""), "recorded_at"] = "Initial total"
    current["notes"] = current["notes"].fillna("").astype(str)
    current = current[current["runner"].ne("") & current["runner"].isin(runners) & current["hours"].gt(0)].copy()
    current = current.reset_index(drop=True)

    seen: set[str] = set()
    entry_ids: list[str] = []
    for index, row in current.iterrows():
        entry_id = _sleep_entry_id(row, index)
        if entry_id in seen:
            entry_id = f"{entry_id}_{index + 1}"
        seen.add(entry_id)
        entry_ids.append(entry_id)
    current["entry_id"] = entry_ids
    return current[SLEEP_SHEET_COLUMNS].reset_index(drop=True)


def sleep_summary_table(log: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    runners = roster.sort_values("running_order")["runner"].astype(str).str.strip()
    runners = runners[runners.ne("")].drop_duplicates().tolist()
    clean = normalise_sleep_log(log, roster)
    summary = pd.DataFrame({"runner": runners})
    if clean.empty:
        summary["hours"] = 0.0
        summary["entries"] = 0
        summary["last_recorded"] = "-"
        summary["notes"] = ""
    else:
        grouped = clean.groupby("runner", as_index=False).agg(
            hours=("hours", "sum"),
            entries=("entry_id", "count"),
            last_recorded=("recorded_at", "last"),
            notes=("notes", "last"),
        )
        summary = summary.merge(grouped, on="runner", how="left")
        summary["hours"] = pd.to_numeric(summary["hours"], errors="coerce").fillna(0).round(2)
        summary["entries"] = pd.to_numeric(summary["entries"], errors="coerce").fillna(0).astype(int)
        summary["last_recorded"] = summary["last_recorded"].fillna("-").astype(str)
        summary["notes"] = summary["notes"].fillna("").astype(str)
    summary["unit"] = "hours"
    return summary


def sleep_state_log(roster: pd.DataFrame) -> pd.DataFrame:
    if "sleep_log" not in st.session_state:
        if "sleep_tracker" in st.session_state:
            st.session_state["sleep_log"] = st.session_state["sleep_tracker"]
        else:
            st.session_state["sleep_log"] = pd.DataFrame(columns=SLEEP_SHEET_COLUMNS)
    current = normalise_sleep_log(st.session_state["sleep_log"], roster)
    st.session_state["sleep_log"] = current.copy()
    return current


def set_sleep_state(table: pd.DataFrame) -> pd.DataFrame:
    roster = st.session_state.get("current_roster_for_sleep")
    if not isinstance(roster, pd.DataFrame) or roster.empty:
        roster = pd.DataFrame(
            {
                "runner": table["runner"].astype(str).str.strip().tolist() if "runner" in table else [],
                "running_order": list(range(1, len(table) + 1)),
            }
        )
    clean = normalise_sleep_log(table, roster)
    st.session_state["sleep_log"] = clean.copy()
    return clean


def sleep_sheet_payload(table: pd.DataFrame) -> pd.DataFrame:
    roster = st.session_state.get("current_roster_for_sleep")
    if not isinstance(roster, pd.DataFrame) or roster.empty:
        roster = pd.DataFrame(
            {
                "runner": table["runner"].astype(str).str.strip().tolist() if "runner" in table else [],
                "running_order": list(range(1, len(table) + 1)),
            }
        )
    return normalise_sleep_log(table, roster)


def sleep_digest(table: pd.DataFrame) -> str:
    clean = table.copy()
    for column in SLEEP_SHEET_COLUMNS:
        if column not in clean:
            clean[column] = ""
    clean["hours"] = pd.to_numeric(clean["hours"], errors="coerce").fillna(0).round(2)
    return hashlib.sha256(clean[SLEEP_SHEET_COLUMNS].to_csv(index=False).encode("utf-8")).hexdigest()


def append_sleep_log_entry(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    runner: str,
    hours: float,
    notes: str = "",
    recorded_at: str | None = None,
) -> pd.DataFrame:
    clean = normalise_sleep_log(log, roster)
    recorded = recorded_at or now_bst_label()
    entry = pd.DataFrame(
        [
            {
                "entry_id": "",
                "runner": str(runner).strip(),
                "hours": float(hours),
                "unit": "hours",
                "recorded_at": recorded,
                "notes": str(notes or "").strip(),
            }
        ]
    )
    return normalise_sleep_log(pd.concat([clean, entry], ignore_index=True), roster)


def sleep_sheet_context() -> dict[str, Any]:
    default_sheet_id = get_streamlit_secret("google_sheet_id", "race_log_google_sheet_id") or ""
    default_sheet_id = st.session_state.get("race_day_sheet_id", default_sheet_id)
    st.session_state.setdefault("sleep_sheet_id", str(default_sheet_id))
    st.session_state.setdefault("sleep_worksheet_name", "sleep")

    sheet_id = str(st.session_state.get("sleep_sheet_id") or default_sheet_id).strip()
    worksheet_name = str(st.session_state.get("sleep_worksheet_name") or "sleep").strip() or "sleep"
    return {
        "sheet_id": sheet_id,
        "worksheet_name": worksheet_name,
        "configured": google_sheets_configured(st.secrets, sheet_id or None),
    }


def save_sleep_live(table: pd.DataFrame, sync: Mapping[str, Any], notice: str | None = None) -> pd.DataFrame:
    clean = set_sleep_state(table)
    if not sync.get("configured"):
        st.session_state["sleep_sheet_notice"] = "Saved locally. Google Sheets is not configured."
        return clean

    write_sleep_to_google_sheet(
        sleep_sheet_payload(clean),
        st.secrets,
        sheet_id=sync.get("sheet_id") or None,
        worksheet_name=str(sync.get("worksheet_name") or "sleep"),
    )
    saved_at = now_bst_label()
    st.session_state["sleep_sheet_last_saved"] = saved_at
    st.session_state["sleep_sheet_notice"] = notice or f"Sleep saved to Google Sheet at {saved_at}."
    return clean


def show_sleep_sheet_controls(log: pd.DataFrame, roster: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sync = sleep_sheet_context()
    sheet_id = str(sync["sheet_id"])
    worksheet_name = str(sync["worksheet_name"])
    configured = bool(sync["configured"])

    with st.container(border=True):
        top_cols = st.columns([1.5, 1, 1])
        with top_cols[0]:
            st.markdown("#### Sleep Sheet")
            if configured:
                st.caption("App changes save immediately. Sheet auto reload defaults to once per minute.")
            else:
                st.caption("Google sync is not configured. Local tracking and CSV download still work.")

        live_reload = top_cols[1].checkbox(
            "Auto reload",
            value=st.session_state.get("sleep_live_sheet_enabled", configured),
            help="Polls the sleep worksheet while this tab is open. App saves still write immediately.",
            key="sleep_live_reload_checkbox",
        )
        refresh_seconds = top_cols[2].number_input(
            "Reload every seconds",
            min_value=10,
            max_value=300,
            value=live_refresh_seconds_default("sleep_live_refresh_seconds"),
            step=5,
            key="sleep_refresh_seconds_input",
        )
        st.session_state["sleep_live_sheet_enabled"] = live_reload
        st.session_state["sleep_live_refresh_seconds"] = int(refresh_seconds)

        if live_reload and st_autorefresh is not None:
            st_autorefresh(
                interval=int(refresh_seconds) * 1000,
                key="sleep_live_sheet_autorefresh",
            )
        elif live_reload and st_autorefresh is None:
            st.warning("Live reload needs the streamlit-autorefresh package. Manual loading still works.")

        action_cols = st.columns([1, 1, 1.2])
        load_now = action_cols[0].button("Load sleep", width="stretch", disabled=not configured, key="sleep_load_button")
        save_now = action_cols[1].button("Save sleep", width="stretch", disabled=not configured, key="sleep_save_button")
        if configured:
            action_cols[2].success(f"Worksheet: {worksheet_name}")
        else:
            action_cols[2].warning("Save disabled until secrets are set.")

        if (load_now or live_reload) and configured:
            try:
                loaded = read_sleep_from_google_sheet(
                    st.secrets,
                    sheet_id=sheet_id or None,
                    worksheet_name=worksheet_name,
                )
                before = sleep_digest(log)
                log = set_sleep_state(loaded)
                loaded_at = now_bst_label()
                st.session_state["sleep_sheet_last_loaded"] = loaded_at
                if sleep_digest(log) != before:
                    st.success(f"Sleep loaded at {loaded_at}.")
                else:
                    st.caption(f"Sleep checked at {loaded_at}; no changes found.")
            except Exception as exc:
                st.error(f"Could not load sleep worksheet: {exc}")

        if st.session_state.get("sleep_sheet_last_loaded"):
            st.caption(f"Last sleep load: {st.session_state['sleep_sheet_last_loaded']}.")
        if st.session_state.get("sleep_sheet_last_saved"):
            st.caption(f"Last sleep save: {st.session_state['sleep_sheet_last_saved']}.")
        if st.session_state.get("sleep_sheet_notice"):
            st.success(st.session_state.pop("sleep_sheet_notice"))

        if save_now and configured:
            try:
                log = save_sleep_live(log, sync, "Sleep saved to Google Sheet.")
                st.success("Sleep saved to Google Sheet.")
            except Exception as exc:
                st.error(f"Could not save sleep worksheet: {exc}")

        with st.expander("Sleep Sheet settings", expanded=False):
            settings_cols = st.columns([2, 1])
            settings_cols[0].text_input("Sheet ID", key="sleep_sheet_id", placeholder="Google Sheet ID")
            settings_cols[1].text_input("Worksheet", key="sleep_worksheet_name")

    return sleep_state_log(roster), sync


def show_sleep_tab(roster: pd.DataFrame) -> None:
    st.subheader("Sleep")
    st.caption("Log sleep in increments. Totals are calculated from the log and do not affect race forecasts or runner order.")
    st.session_state["current_roster_for_sleep"] = roster.copy()

    log = sleep_state_log(roster)
    log, sleep_sync = show_sleep_sheet_controls(log, roster)
    summary = sleep_summary_table(log, roster)
    if summary.empty:
        st.info("Add runners in Setup before tracking sleep.")
        return

    total_sleep = float(summary["hours"].sum())
    average_sleep = float(summary["hours"].mean()) if not summary.empty else 0.0
    most_rested = summary.sort_values(["hours", "runner"], ascending=[False, True]).iloc[0]
    least_rested = summary.sort_values(["hours", "runner"], ascending=[True, True]).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Team sleep bank", f"{total_sleep:.1f}h")
    c2.metric("Average", f"{average_sleep:.1f}h")
    c3.metric("Most rested", f"{most_rested['runner']} ({float(most_rested['hours']):.1f}h)")
    c4.metric("Lowest sleep", f"{least_rested['runner']} ({float(least_rested['hours']):.1f}h)")

    if log.empty:
        st.info("No sleep blocks logged yet. Add sleep below when someone gets a rest block.")

    st.markdown("#### Add Sleep Block")
    quick_cols = st.columns([1.4, 1, 1])
    runner = quick_cols[0].selectbox(
        "Runner",
        summary["runner"].astype(str).tolist(),
        key="sleep_quick_runner",
    )
    added_hours = quick_cols[1].number_input(
        "Hours slept",
        min_value=0.25,
        max_value=12.0,
        value=1.0,
        step=0.25,
        key="sleep_quick_hours",
    )
    sleep_note = st.text_input("Sleep note", value="", placeholder="tent, car, decent nap, interrupted", key="sleep_quick_note")
    if quick_cols[2].button("Add sleep", type="primary", width="stretch", key="sleep_add_button"):
        updated = append_sleep_log_entry(log, roster, runner, float(added_hours), sleep_note)
        try:
            save_sleep_live(updated, sleep_sync, f"Added {float(added_hours):.2f}h for {runner} and saved to Google Sheet.")
        except Exception as exc:
            set_sleep_state(updated)
            st.session_state["sleep_sheet_notice"] = f"Updated locally, but could not save to Sheet: {exc}"
        st.rerun()

    st.markdown("#### Sleep Log")
    if log.empty:
        display_log = pd.DataFrame(columns=SLEEP_SHEET_COLUMNS)
    else:
        display_log = log.sort_values("recorded_at", kind="stable").reset_index(drop=True)
    edited = st.data_editor(
        display_log[SLEEP_SHEET_COLUMNS],
        hide_index=True,
        width="stretch",
        column_config={
            "entry_id": None,
            "runner": st.column_config.SelectboxColumn("Runner", options=summary["runner"].astype(str).tolist()),
            "hours": st.column_config.NumberColumn("Hours", min_value=0.0, max_value=12.0, step=0.25, format="%.2f"),
            "unit": st.column_config.TextColumn("Unit", disabled=True),
            "recorded_at": st.column_config.TextColumn("Recorded"),
            "notes": st.column_config.TextColumn("Notes"),
        },
        column_order=["runner", "hours", "recorded_at", "notes"],
        num_rows="dynamic",
        key=f"sleep_log_editor_{sleep_digest(log)[:12]}",
    )
    edited_clean = normalise_sleep_log(edited, roster)
    if sleep_digest(edited_clean) != sleep_digest(log):
        try:
            save_sleep_live(edited_clean, sleep_sync, "Sleep log edit saved to Google Sheet.")
        except Exception as exc:
            set_sleep_state(edited_clean)
            st.session_state["sleep_sheet_notice"] = f"Updated locally, but could not save to Sheet: {exc}"
        st.rerun()

    action_cols = st.columns(3)
    if action_cols[0].button(
        "Force save to Sheet",
        type="primary",
        width="stretch",
        disabled=not sleep_sync.get("configured"),
        key="sleep_force_save_button",
    ):
        try:
            saved = save_sleep_live(edited_clean, sleep_sync, "Sleep log forced saved to Google Sheet.")
            st.success(f"Saved {len(saved)} sleep log entries.")
        except Exception as exc:
            st.error(f"Could not save sleep worksheet: {exc}")
    if action_cols[1].button("Clear sleep log", width="stretch", key="sleep_reset_button"):
        reset = pd.DataFrame(columns=SLEEP_SHEET_COLUMNS)
        try:
            save_sleep_live(reset, sleep_sync, "Sleep log cleared and saved to Google Sheet.")
        except Exception as exc:
            set_sleep_state(reset)
            st.session_state["sleep_sheet_notice"] = f"Cleared locally, but could not save to Sheet: {exc}"
        st.rerun()
    action_cols[2].download_button(
        "Download sleep log CSV",
        sleep_state_log(roster).to_csv(index=False),
        file_name="endure24_sleep_log.csv",
        mime="text/csv",
        width="stretch",
        key="sleep_download_csv",
    )

    st.markdown("#### Sleep Graphs")
    chart_table = sleep_summary_table(log, roster)
    chart_table["hours_label"] = chart_table["hours"].map(lambda value: f"{float(value):.1f}")
    bar = px.bar(
        chart_table.sort_values("hours", ascending=False),
        x="runner",
        y="hours",
        color="runner",
        text="hours_label",
        title="Sleep Bank Leaderboard",
    )
    bar.update_layout(showlegend=False, yaxis_title="Hours slept", xaxis_title="")
    st.plotly_chart(bar, width="stretch")

    target_cols = st.columns([1, 2])
    sleep_target = target_cols[0].slider("Sleep target hours", min_value=0.0, max_value=10.0, value=4.0, step=0.25)
    target_cols[1].caption("Used only for the sleep debt chart below.")
    chart_table["sleep_debt"] = (float(sleep_target) - chart_table["hours"]).clip(lower=0)
    chart_table["sleep_debt_label"] = chart_table["sleep_debt"].map(lambda value: f"{float(value):.1f}")

    left, right = st.columns(2)
    with left:
        debt = px.bar(
            chart_table.sort_values("sleep_debt", ascending=False),
            x="runner",
            y="sleep_debt",
            color="runner",
            text="sleep_debt_label",
            title=f"Sleep Debt to {float(sleep_target):.1f}h Target",
        )
        debt.update_layout(showlegend=False, yaxis_title="Hours short", xaxis_title="")
        st.plotly_chart(debt, width="stretch")

    with right:
        gauge_max = max(8.0, average_sleep + 2.0, float(chart_table["hours"].max()) + 1.0)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=average_sleep,
                title={"text": "Average team sleep"},
                number={"suffix": "h"},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar": {"color": "#356d8f"},
                    "steps": [
                        {"range": [0, min(2, gauge_max)], "color": "#ffe2e2"},
                        {"range": [min(2, gauge_max), min(4, gauge_max)], "color": "#fff4d6"},
                        {"range": [min(4, gauge_max), gauge_max], "color": "#e9f5ee"},
                    ],
                },
            )
        )
        gauge.update_layout(height=320, margin={"l": 20, "r": 20, "t": 50, "b": 20})
        st.plotly_chart(gauge, width="stretch")

    matrix = chart_table.copy()
    order_lookup = (
        roster.assign(runner=roster["runner"].astype(str).str.strip())
        .set_index("runner")["running_order"]
        .to_dict()
    )
    matrix["running_order"] = matrix["runner"].map(order_lookup)
    matrix["bubble_size"] = matrix["hours"].clip(lower=0.5)
    scatter = px.scatter(
        matrix,
        x="running_order",
        y="hours",
        size="bubble_size",
        color="runner",
        hover_data=["notes"],
        title="Sleep vs Running Order",
    )
    scatter.update_layout(xaxis_title="Running order", yaxis_title="Hours slept", showlegend=False)
    st.plotly_chart(scatter, width="stretch")

    if log.empty:
        st.info("Add sleep blocks to show the cumulative sleep log chart.")
    else:
        timeline = log.copy().reset_index(drop=True)
        timeline["entry"] = timeline.index + 1
        timeline["cumulative_hours"] = timeline.groupby("runner")["hours"].cumsum()
        line = px.line(
            timeline,
            x="entry",
            y="cumulative_hours",
            color="runner",
            markers=True,
            hover_data=["hours", "recorded_at", "notes"],
            title="Cumulative Sleep Blocks",
        )
        line.update_layout(xaxis_title="Sleep log entry", yaxis_title="Cumulative hours")
        st.plotly_chart(line, width="stretch")


def main() -> None:
    inject_app_styles()
    st.title("Endure24 Race Control")
    st.caption("Race-day monitoring, live Sheet logging, forecast, optimiser, and team tracking.")

    with st.expander("Admin setup", expanded=False):
        st.caption("Race rule model: starts through Sunday 12:00 BST count if the lap finishes by Sunday 13:00 BST.")
        uploaded = st.file_uploader("Upload Endure24 workbook", type=["xlsx", "xls"])
        if uploaded is not None:
            workbook_bytes = uploaded.getvalue()
            source_label = uploaded.name
        else:
            workbook_bytes = DEFAULT_DATA_PATH.read_bytes()
            source_label = str(DEFAULT_DATA_PATH)
        st.caption(f"Workbook source: {source_label}")

    loaded = load_data_cached(workbook_bytes, source_label, ASSUMPTION_VERSION)
    settings = make_settings()
    st.sidebar.subheader("Race Rules")
    caps_enabled = st.sidebar.checkbox("Apply max-lap caps", value=True)

    race_day_tab, forecast_tab, optimiser_tab, drinks_tab, sleep_tab, what_if_tab, exports_tab, setup_tab = st.tabs(
        ["Race Day", "Forecast", "Optimiser", "Drinks", "Sleep", "What-if", "Exports", "Setup"]
    )

    with setup_tab:
        show_validation(loaded)
        show_last_year_analysis(loaded)

        st.subheader("This Year Assumptions")
        st.info(
            "Per-runner fatigue and night penalty values default to 0.0%. Night penalties are also toggled off "
            "in the sidebar by default. You can edit these if you want to add a conservative slowdown assumption. "
            "Last-year day/night and fatigue stats are shown below where the workbook has enough data, but the app "
            "does not apply them automatically."
        )
        roster = edited_roster_table(loaded.roster)
        roster_warnings, roster_errors = validate_roster(roster)
        for warning in roster_warnings:
            st.warning(warning)
        for error in roster_errors:
            st.error(error)

        with st.expander("Last-year stats beside this-year assumptions", expanded=False):
            combined = roster.merge(loaded.last_year_stats, on="runner", how="left")
            st.dataframe(combined, width="stretch", hide_index=True)

    with forecast_tab:
        st.subheader("Forecast")
        run_uncapped_comparison = st.checkbox("Also run uncapped comparison", value=True)
        if st.button("Run Monte Carlo simulation", type="primary"):
            if roster_errors:
                st.error("Fix roster errors before running the simulation.")
            else:
                with st.spinner("Running simulations..."):
                    sim_output = simulate_many(roster, settings, caps_enabled=caps_enabled)
                    summary_bundle = summarize_results(sim_output, settings.target_laps)
                    uncapped_summary = None
                    comparison = None
                    if run_uncapped_comparison:
                        uncapped_settings = replace(
                            settings,
                            random_seed=None if settings.random_seed is None else settings.random_seed + 1,
                        )
                        uncapped_output = simulate_many(roster, uncapped_settings, caps_enabled=False)
                        uncapped_summary = summarize_results(uncapped_output, settings.target_laps)
                        comparison = pd.DataFrame(
                            [
                                {
                                    "scenario": "Caps enabled" if caps_enabled else "Current run",
                                    "expected_official_laps": summary_bundle["summary"]["expected_official_laps"],
                                    "probability_target_laps": summary_bundle["summary"]["probability_target_laps"],
                                    "probability_ran_out_of_eligible_runners": summary_bundle["summary"][
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
                    st.session_state["sim_output"] = sim_output
                    st.session_state["summary_bundle"] = summary_bundle
                    st.session_state["uncapped_summary"] = uncapped_summary
                    st.session_state["comparison"] = comparison

        if "sim_output" in st.session_state:
            show_charts(
                st.session_state["sim_output"],
                st.session_state["summary_bundle"],
                st.session_state.get("uncapped_summary"),
                st.session_state.get("comparison"),
            )
        else:
            st.info("Run the Monte Carlo forecast to populate the dashboard.")

    with optimiser_tab:
        show_optimizer(roster, settings)

    with race_day_tab:
        show_race_day(roster, settings, caps_enabled=caps_enabled)

    with drinks_tab:
        show_drinks_tab(roster)

    with sleep_tab:
        show_sleep_tab(roster)

    with what_if_tab:
        show_what_if_pace_override(roster, settings)

    with exports_tab:
        show_exports_tab(
            st.session_state.get("sim_output"),
            st.session_state.get("summary_bundle"),
            st.session_state.get("comparison"),
        )


if __name__ == "__main__":
    main()
