#!/usr/bin/env python3
"""Merge benchmark data and plot performance transformations."""
from __future__ import annotations

import csv
import math
import pathlib
from datetime import date, datetime
from typing import Callable, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_CSV = DATA_DIR / "model_performance_merged.csv"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
FIGURE_PATHS = {
    "actual_percent": FIGURES_DIR / "model_performance_actual_percent.png",
    "log_percent": FIGURES_DIR / "model_performance_log_percent.png",
    "odds": FIGURES_DIR / "model_performance_odds.png",
    "logit": FIGURES_DIR / "model_performance_logit.png",
}
PROJECTION_CSV = DATA_DIR / "model_performance_projections.csv"
PROJECTION_MD = DATA_DIR / "model_performance_projections.md"

MODEL_KEY_MAP = {
    "GPT-4o": "gpt_4o",
    "Grok 4": "grok_4",
    "Gemini 2.5 Pro": "gemini_2_5_pro_preview",
    "o4-mini high": "o4-mini",
    "o3 high": "o3",
    "GPT-5 high": "gpt_5",
    "Claude Opus 4.1": "claude_4_1_opus",
}

REFERENCE_P = 0.5
EPS_PROB = 1e-4  # Clamp to avoid singularities at exactly 0 or 1.


def load_benchmark_results() -> dict:
    yaml_path = DATA_DIR / "benchmark_results.yaml"
    with yaml_path.open() as fh:
        payload = yaml.safe_load(fh) or {}
    return payload.get("results", {})


def load_model_performance() -> list[dict[str, str]]:
    csv_path = DATA_DIR / "model_performance.csv"
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))


def parse_release_date(value: object) -> datetime:
    """Normalize YAML date values into naive datetime instances."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported release_date type: {type(value)!r}")


def build_dataset() -> list[tuple[str, datetime, float]]:
    benchmark = load_benchmark_results()
    rows = []
    for entry in load_model_performance():
        model_label = entry["Model"]
        benchmark_key = MODEL_KEY_MAP.get(model_label)
        if not benchmark_key:
            raise KeyError(f"No benchmark mapping defined for {model_label!r}")
        benchmark_entry = benchmark.get(benchmark_key)
        if not benchmark_entry:
            raise KeyError(f"Benchmark results missing for {benchmark_key!r}")
        release_date = parse_release_date(benchmark_entry["release_date"])
        percentage = float(entry["Win Rate (%)"]) / 100.0
        rows.append((model_label, release_date, percentage))
    rows.sort(key=lambda item: (item[1], item[0]))
    return rows


def write_merged_csv(records: list[tuple[str, datetime, float]]) -> None:
    OUTPUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model_name", "date", "model_performance"])
        for model, date, performance in records:
            writer.writerow([model, date.date().isoformat(), f"{performance:.4f}"])


def write_projection_csv(entries: list[dict[str, str]]) -> None:
    PROJECTION_CSV.parent.mkdir(exist_ok=True, parents=True)
    fieldnames = ["metric", "percent", "date", "line_type"]
    entries_sorted = sorted(
        entries,
        key=lambda item: (item["metric"], item["line_type"], float(item["percent"])),
    )
    with PROJECTION_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries_sorted)


def write_projection_markdown(entries: list[dict[str, str]]) -> None:
    PROJECTION_MD.parent.mkdir(exist_ok=True, parents=True)
    entries_sorted = sorted(
        entries,
        key=lambda item: (item["metric"], item["line_type"], float(item["percent"])),
    )
    header = "| Metric | Percent | Date | Line Type |\n"
    separator = "| --- | --- | --- | --- |\n"
    lines = [header, separator]
    for row in entries_sorted:
        lines.append(
            f"| {row['metric']} | {row['percent']}% | {row['date']} | {row['line_type']} |\n"
        )
    with PROJECTION_MD.open("w") as fh:
        fh.writelines(lines)


def compute_transforms(records: list[tuple[str, datetime, float]]):
    dates = [record[1] for record in records]
    probs = [record[2] for record in records]
    percents = [p * 100.0 for p in probs]
    log_percents = [math.log(pct) for pct in percents]
    odds = [p / (1.0 - p) for p in probs]
    logits = [math.log(o) for o in odds]
    labels = [record[0] for record in records]
    frontier_mask: list[bool] = []
    max_prob = -float("inf")
    for prob in probs:
        if prob > max_prob:
            frontier_mask.append(True)
            max_prob = prob
        else:
            frontier_mask.append(False)
    return {
        "dates": dates,
        "labels": labels,
        "actual_percent": percents,
        "log_percent": log_percents,
        "odds": odds,
        "logit": logits,
        "frontier_mask": frontier_mask,
    }


def annotate_points(ax, dates, values, labels):
    for date, value, label in zip(dates, values, labels):
        ax.annotate(label, (date, value), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)


def add_reference_line(ax, x_values, y_value):
    ax.axhline(y_value, color="tab:red", linestyle=":", linewidth=1.2)
    if x_values:
        ax.annotate(
            "Industry Expert",
            (x_values[-1], y_value),
            textcoords="offset points",
            xytext=(-5, 5),
            ha="right",
            va="bottom",
            fontsize=8,
            color="tab:red",
        )


def fit_regression(dates, values):
    base = mdates.date2num(dates[0])
    x = np.array([mdates.date2num(d) - base for d in dates], dtype=float)
    y = np.array(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return base, x, slope, intercept


def clamp_probability(prob: float) -> float:
    return min(max(prob, EPS_PROB), 1.0 - EPS_PROB)


def configure_probability_axis(
    ax, transform: Callable[[float], float], perc_ticks: Iterable[int] | None = None
) -> None:
    ticks = []
    labels = []
    if perc_ticks is None:
        perc_ticks = range(0, 101, 20)
    for percent in perc_ticks:
        prob = clamp_probability(percent / 100.0)
        ticks.append(transform(prob))
        labels.append(str(percent))
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def transform_actual(prob: float) -> float:
    return prob * 100.0


def transform_log_percent(prob: float) -> float:
    return math.log(clamp_probability(prob) * 100.0)


def transform_odds(prob: float) -> float:
    if prob <= 0.0:
        return 0.0
    prob = clamp_probability(prob)
    return prob / (1.0 - prob)


def transform_logit(prob: float) -> float:
    return math.log(transform_odds(prob))


def inverse_actual(value: float) -> float:
    return clamp_probability(value / 100.0)


def inverse_log_percent(value: float) -> float:
    return clamp_probability(math.exp(value) / 100.0)


def inverse_odds(value: float) -> float:
    if value <= 0.0:
        return EPS_PROB
    prob = value / (1.0 + value)
    return clamp_probability(prob)


def inverse_logit(value: float) -> float:
    odds = math.exp(value)
    return inverse_odds(odds)


def plot_metric(
    title,
    dates,
    values,
    labels,
    ylabel,
    reference,
    output_path,
    y_limits=None,
    probability_transform: Callable[[float], float] | None = None,
    probability_inverse: Callable[[float], float] | None = None,
    annotate_projection: bool = False,
    projection_targets: Iterable[float] | None = None,
    perc_ticks: Iterable[int] | None = None,
    frontier_mask: Iterable[bool] | None = None,
    metric_key: str = "",
    plot_title: str | None = None,
    y_axis_label: str | None = None,
):
    base, x_numeric, slope, intercept = fit_regression(dates, values)

    figure, ax = plt.subplots(figsize=(8, 5))

    legend_handles = []
    frontier_dates: list[datetime] = []
    frontier_values: list[float] = []
    other_dates: list[datetime] = []
    other_values: list[float] = []

    if frontier_mask is not None:
        for d, value, flag in zip(dates, values, frontier_mask):
            if flag:
                frontier_dates.append(d)
                frontier_values.append(value)
            else:
                other_dates.append(d)
                other_values.append(value)

        if other_dates:
            scatter = ax.scatter(other_dates, other_values, color="#b0b0b0", label="Other models")
            legend_handles.append(scatter)
        if frontier_dates:
            scatter = ax.scatter(frontier_dates, frontier_values, color="tab:green", label="Frontier models")
            legend_handles.append(scatter)
    else:
        scatter = ax.scatter(dates, values, color="0.5", label="Models")
        legend_handles.append(scatter)

    ax.set_title(plot_title or title)
    axis_label = y_axis_label or ylabel
    ax.set_ylabel(axis_label)
    ax.set_xlabel("Date")

    annotate_points(ax, dates, values, labels)
    add_reference_line(ax, dates, reference)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if probability_transform is not None:
        configure_probability_axis(ax, probability_transform, perc_ticks)
    ax.grid(True, linestyle="--", alpha=0.3)

    text_lines: list[str] = []
    projection_entries: list[dict[str, str]] = []

    frontier_regression = None
    if frontier_dates:
        if len(frontier_dates) >= 2:
            f_base, f_x_numeric, f_slope, f_intercept = fit_regression(frontier_dates, frontier_values)
            f_x_line = np.linspace(f_x_numeric.min(), f_x_numeric.max(), 100)
            f_y_line = f_slope * f_x_line + f_intercept
            ax.plot(
                mdates.num2date(f_x_line + f_base),
                f_y_line,
                color="tab:green",
                linewidth=1.5,
                linestyle="--",
            )
            frontier_pred = f_intercept + f_slope * f_x_numeric
            mean_frontier = np.mean(frontier_values)
            ss_tot = np.sum((np.array(frontier_values) - mean_frontier) ** 2)
            ss_res = np.sum((np.array(frontier_values) - frontier_pred) ** 2)
            r_squared = 1.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)
            text_lines.append(
                f"y = {f_intercept:.3f} + {f_slope:.3f} * delta_days (R² = {r_squared:.3f})"
            )
            frontier_regression = (f_base, f_x_numeric, f_slope, f_intercept)
        elif frontier_values:
            text_lines.append("Need ≥2 frontier points for regression")

    def project_lines(label: str, base_value, x_values, slope_value, intercept_value):
        lines: list[str] = []
        entries: list[dict[str, str]] = []
        if slope_value <= 0:
            if label == "frontier":
                lines.append("Trend non-increasing; extrapolation unavailable")
            return lines, entries
        x_current_value = x_values[-1]
        for target_prob in projection_targets or []:
            clamped_target = clamp_probability(target_prob)
            target_value = probability_transform(clamped_target)
            x_target_value = (target_value - intercept_value) / slope_value
            target_date = mdates.num2date(base_value + x_target_value).date()
            if label == "frontier":
                if x_target_value <= x_current_value:
                    lines.append(f"  {clamped_target * 100:.0f}% (reached): {target_date}")
                else:
                    lines.append(f"  {clamped_target * 100:.0f}%: {target_date}")
            entries.append(
                {
                    "metric": metric_key or title,
                    "percent": f"{clamped_target * 100:.0f}",
                    "date": target_date.isoformat(),
                    "line_type": label,
                }
            )
        return lines, entries

    if annotate_projection and probability_transform and probability_inverse and projection_targets:
        _, overall_entries = project_lines("overall", base, x_numeric, slope, intercept)
        projection_entries.extend(overall_entries)
        if frontier_regression is not None:
            frontier_lines, frontier_entries = project_lines("frontier", *frontier_regression)
            if frontier_lines:
                if text_lines:
                    text_lines.append("")
                text_lines.append("Extrapolated date for model win rate to reach:")
                text_lines.extend(frontier_lines)
            projection_entries.extend(frontier_entries)

    text_y = 0.97
    for index, line in enumerate(text_lines):
        ax.text(
            0.03,
            text_y - index * 0.075,
            line,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="best")

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return projection_entries


def plot_transforms(data):
    dates = data["dates"]
    labels = data["labels"]

    specs = [
        {
            "title": "Actual %",
            "metric": "actual_percent",
            "plot_title": "Win Rate (%) Over Time",
            "y_label": "Win Rate (%)",
            "values": data["actual_percent"],
            "ylabel": "%",
            "reference": REFERENCE_P * 100.0,
            "path": FIGURE_PATHS["actual_percent"],
            "limits": (0, 100),
            "transform": transform_actual,
            "inverse": inverse_actual,
            "annotate": True,
            "targets": [0.50, 0.75, 0.90, 0.95, 0.99],
            "ticks": range(0, 101, 20),
            "frontier_mask": data["frontier_mask"],
        },
        {
            "title": "Log %",
            "metric": "log_percent",
            "plot_title": "Log Win Rate (%) Over Time",
            "y_label": "Win Rate (%) (log scale)",
            "values": data["log_percent"],
            "ylabel": "log %",
            "reference": math.log(REFERENCE_P * 100.0),
            "path": FIGURE_PATHS["log_percent"],
            "limits": (
                transform_log_percent(0.10),
                transform_log_percent(0.90),
            ),
            "transform": transform_log_percent,
            "inverse": inverse_log_percent,
            "annotate": True,
            "targets": [0.50, 0.75, 0.90, 0.95, 0.99],
            "ticks": range(10, 91, 20),
            "frontier_mask": data["frontier_mask"],
        },
        {
            "title": "Odds",
            "metric": "odds",
            "plot_title": "Win Odds Over Time",
            "y_label": "Win Rate (%) (odds scale)",
            "values": data["odds"],
            "ylabel": "odds",
            "reference": REFERENCE_P / (1.0 - REFERENCE_P),
            "path": FIGURE_PATHS["odds"],
            "limits": (
                transform_odds(EPS_PROB),
                transform_odds(0.75),
            ),
            "transform": transform_odds,
            "inverse": inverse_odds,
            "annotate": True,
            "targets": [0.50, 0.75, 0.90, 0.95, 0.99],
            "ticks": [0, 10, 25, 50, 75],
            "frontier_mask": data["frontier_mask"],
        },
        {
            "title": "Logit",
            "metric": "logit",
            "plot_title": "Win Rate Logit Over Time",
            "y_label": "Win Rate (%) (logit scale)",
            "values": data["logit"],
            "ylabel": "logit",
            "reference": 0.0,
            "path": FIGURE_PATHS["logit"],
            "limits": (
                transform_logit(0.10),
                transform_logit(0.90),
            ),
            "transform": transform_logit,
            "inverse": inverse_logit,
            "annotate": True,
            "targets": [0.50, 0.75, 0.90, 0.95, 0.99],
            "ticks": range(10, 91, 20),
            "frontier_mask": data["frontier_mask"],
        },
    ]

    outputs = []
    projection_records: list[dict[str, str]] = []
    for spec in specs:
        entries = plot_metric(
            spec["title"],
            dates,
            spec["values"],
            labels,
            spec["ylabel"],
            spec["reference"],
            spec["path"],
            spec["limits"],
            probability_transform=spec["transform"],
            probability_inverse=spec["inverse"],
            annotate_projection=spec["annotate"],
            projection_targets=spec["targets"],
            perc_ticks=spec["ticks"],
            frontier_mask=spec["frontier_mask"],
            metric_key=spec["metric"],
            plot_title=spec["plot_title"],
            y_axis_label=spec["y_label"],
        )
        outputs.append(spec["path"])
        projection_records.extend(entries)
    return outputs, projection_records


def main() -> None:
    records = build_dataset()
    write_merged_csv(records)
    data = compute_transforms(records)
    figure_paths, projection_records = plot_transforms(data)
    write_projection_csv(projection_records)
    write_projection_markdown(projection_records)
    print(f"Merged CSV saved to {OUTPUT_CSV}")
    print(f"Projection CSV saved to {PROJECTION_CSV}")
    print(f"Projection markdown saved to {PROJECTION_MD}")
    for path in figure_paths:
        print(f"Figure saved to {path}")


if __name__ == "__main__":
    main()
