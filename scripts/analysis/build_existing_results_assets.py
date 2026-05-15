#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = REPO_ROOT / "samples" / "metrics" / "summary.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "existing_experiment_assets"


@dataclass(frozen=True)
class MethodSpec:
    label: str
    summary_prefix: str


COMPARISON_METHODS = [
    MethodSpec("Baseline", "sd-turbo-baseline/default"),
    MethodSpec("Draft", "sd-turbo-lora/draft/sdturbo_lora/default"),
    MethodSpec("DPO-latent", "sd-turbo-lora/dpo/online/latent"),
    MethodSpec("DPO-mae-latent", "sd-turbo-lora/dpo/online/mae_latent"),
    MethodSpec("DrPO-default", "sd-turbo-lora/drpo/online/default"),
    MethodSpec("VGG-Flow", "sd-turbo-lora/vggflow/pickscore/default"),
]

DRPO_ABLATIONS = [
    MethodSpec("Default", "sd-turbo-lora/drpo/online/default"),
    MethodSpec("Batch Gen 16", "sd-turbo-lora/drpo/online/batch_gen_ablation/gen16"),
    MethodSpec("Batch Gen 24", "sd-turbo-lora/drpo/online/batch_gen_ablation/gen24"),
    MethodSpec("Batch Gen 32", "sd-turbo-lora/drpo/online/batch_gen_ablation/gen32"),
    MethodSpec("Reward AES", "sd-turbo-lora/drpo/online/different_reward/aes"),
    MethodSpec("Reward CLIP", "sd-turbo-lora/drpo/online/different_reward/clip"),
    MethodSpec("Reward CLIP+Pick", "sd-turbo-lora/drpo/online/different_reward/clip-pickscore"),
    MethodSpec("Reward AES+Pick", "sd-turbo-lora/drpo/online/different_reward/aes-pickscore"),
    MethodSpec("Reward CLIP+AES+Pick", "sd-turbo-lora/drpo/online/different_reward/clip-aes-pickscore"),
]

DRPO_ABLATION_GROUPS = [
    (
        "batch_gen",
        "DrPO Batch-Gen Learning Curves on PickScore (step 0 = raw baseline)",
        [
            MethodSpec("Default", "sd-turbo-lora/drpo/online/default"),
            MethodSpec("Batch Gen 16", "sd-turbo-lora/drpo/online/batch_gen_ablation/gen16"),
            MethodSpec("Batch Gen 24", "sd-turbo-lora/drpo/online/batch_gen_ablation/gen24"),
            MethodSpec("Batch Gen 32", "sd-turbo-lora/drpo/online/batch_gen_ablation/gen32"),
        ],
    ),
    (
        "reward",
        "DrPO Reward-Ablation Learning Curves on PickScore (step 0 = raw baseline)",
        [
            MethodSpec("Default", "sd-turbo-lora/drpo/online/default"),
            MethodSpec("Reward AES", "sd-turbo-lora/drpo/online/different_reward/aes"),
            MethodSpec("Reward CLIP", "sd-turbo-lora/drpo/online/different_reward/clip"),
            MethodSpec("Reward CLIP+Pick", "sd-turbo-lora/drpo/online/different_reward/clip-pickscore"),
            MethodSpec("Reward AES+Pick", "sd-turbo-lora/drpo/online/different_reward/aes-pickscore"),
            MethodSpec("Reward CLIP+AES+Pick", "sd-turbo-lora/drpo/online/different_reward/clip-aes-pickscore"),
        ],
    ),
]

GRID_PROMPT_IDS = [0, 1, 4, 5, 8, 9]
MAX_PLOT_STEP = 1000


def parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def checkpoint_step(summary_path: str) -> int | None:
    match = re.search(r"/checkpoint-(\d+)/summary\.json$", summary_path)
    if match:
        return int(match.group(1))
    return None


def run_id(summary_path: str) -> str:
    parts = summary_path.split("/")
    if "final" in parts:
        idx = parts.index("final")
        return parts[idx - 1]
    for index, part in enumerate(parts):
        if part.startswith("checkpoint-"):
            return parts[index - 1]
    return "unknown"


def summary_to_manifest(summary_path: str) -> Path:
    return REPO_ROOT / "samples" / summary_path.replace("/summary.json", "/manifest.jsonl")


def shorten_checkpoint(summary_path: str) -> str:
    if summary_path.endswith("/summary.json"):
        return summary_path.rsplit("/", 2)[-2]
    return summary_path


def load_rows() -> list[dict[str, str]]:
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["_pickscore"] = parse_float(row["pickscore_mean"])
        row["_clip"] = parse_float(row["clip_mean"])
        row["_hpsv2"] = parse_float(row["hpsv2_mean"])
        row["_aes"] = parse_float(row["aes_mean"])
        row["_fid"] = parse_float(row["fid_vs_baseline"])
        row["_step"] = checkpoint_step(row["summary_path"])
        row["_run_id"] = run_id(row["summary_path"])
    return rows


def rows_for_prefix(rows: Iterable[dict[str, str]], prefix: str) -> list[dict[str, str]]:
    return [row for row in rows if row["summary_path"].startswith(prefix)]


def pick_best_row(rows: Iterable[dict[str, str]]) -> dict[str, str] | None:
    candidates = [row for row in rows if row["_pickscore"] is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["_pickscore"], -(row["_fid"] or float("inf"))))


def pick_latest_run_rows(rows: Iterable[dict[str, str]], prefix: str) -> list[dict[str, str]]:
    prefixed = rows_for_prefix(rows, prefix)
    if not prefixed:
        return []
    latest_run = max(row["_run_id"] for row in prefixed)
    return [row for row in prefixed if row["_run_id"] == latest_run]


def pick_representative_run_rows(rows: Iterable[dict[str, str]], prefix: str) -> list[dict[str, str]]:
    prefixed = rows_for_prefix(rows, prefix)
    if not prefixed:
        return []
    run_groups: dict[str, list[dict[str, str]]] = {}
    for row in prefixed:
        run_groups.setdefault(row["_run_id"], []).append(row)
    best_run = max(
        run_groups,
        key=lambda run: (
            max((item["_step"] or 0) for item in run_groups[run]),
            len(run_groups[run]),
            run,
        ),
    )
    return run_groups[best_run]


def pick_step_1000_or_latest_row(rows: Iterable[dict[str, str]], prefix: str) -> dict[str, str] | None:
    representative_rows = pick_representative_run_rows(rows, prefix)
    if not representative_rows:
        return None
    checkpoint_rows = [row for row in representative_rows if row["_step"] is not None and row["_step"] <= MAX_PLOT_STEP]
    if not checkpoint_rows:
        return None
    target = [row for row in checkpoint_rows if row["_step"] == MAX_PLOT_STEP]
    if target:
        return target[0]
    return max(checkpoint_rows, key=lambda row: row["_step"])


def pick_last_checkpoint_row(rows: Iterable[dict[str, str]], prefix: str) -> dict[str, str] | None:
    representative_rows = pick_representative_run_rows(rows, prefix)
    if not representative_rows:
        return None
    checkpoint_rows = [row for row in representative_rows if row["_step"] is not None]
    if checkpoint_rows:
        return max(checkpoint_rows, key=lambda row: (row["_step"], row["_pickscore"] or float("-inf")))
    final_rows = [row for row in representative_rows if row["summary_path"].endswith("/final/summary.json")]
    if final_rows:
        return final_rows[0]
    return None


def format_metric(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparison_tables(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    baseline_row = pick_best_row(rows_for_prefix(rows, COMPARISON_METHODS[0].summary_prefix))
    baseline_pickscore = baseline_row["_pickscore"] if baseline_row else None
    table_rows: list[dict[str, str]] = []
    md_rows: list[list[str]] = []
    for spec in COMPARISON_METHODS:
        selected = baseline_row if spec.label == "Baseline" else pick_step_1000_or_latest_row(rows, spec.summary_prefix)
        if not selected:
            continue
        delta = None
        if baseline_pickscore is not None and selected["_pickscore"] is not None:
            delta = selected["_pickscore"] - baseline_pickscore
        item = {
            "method": spec.label,
            "checkpoint": shorten_checkpoint(selected["summary_path"]) if spec.label == "Baseline" else f"step={selected['_step']}",
            "pickscore_mean": format_metric(selected["_pickscore"]),
            "delta_vs_baseline": format_metric(delta),
            "clip_mean": format_metric(selected["_clip"]),
            "hpsv2_mean": format_metric(selected["_hpsv2"]),
            "aes_mean": format_metric(selected["_aes"], 3),
            "fid_vs_baseline": format_metric(selected["_fid"], 3),
        }
        table_rows.append(item)
        md_rows.append([
            item["method"],
            item["checkpoint"],
            item["pickscore_mean"],
            item["delta_vs_baseline"],
            item["clip_mean"],
            item["hpsv2_mean"],
            item["aes_mean"],
            item["fid_vs_baseline"],
        ])
    write_csv(
        OUTPUT_DIR / "table_comparison_best.csv",
        [
            "method",
            "checkpoint",
            "pickscore_mean",
            "delta_vs_baseline",
            "clip_mean",
            "hpsv2_mean",
            "aes_mean",
            "fid_vs_baseline",
        ],
        table_rows,
    )
    write_markdown_table(
        OUTPUT_DIR / "table_comparison_best.md",
        ["Method", "Checkpoint", "PickScore", "Delta vs Baseline", "CLIP", "HPSv2", "AES", "FID"],
        md_rows,
    )
    return table_rows


def build_drpo_ablation_tables(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    table_rows: list[dict[str, str]] = []
    md_rows: list[list[str]] = []
    for spec in DRPO_ABLATIONS:
        selected = pick_step_1000_or_latest_row(rows, spec.summary_prefix)
        if not selected:
            continue
        item = {
            "setting": spec.label,
            "checkpoint": f"step={selected['_step']}",
            "pickscore_mean": format_metric(selected["_pickscore"]),
            "clip_mean": format_metric(selected["_clip"]),
            "hpsv2_mean": format_metric(selected["_hpsv2"]),
            "aes_mean": format_metric(selected["_aes"], 3),
            "fid_vs_baseline": format_metric(selected["_fid"], 3),
        }
        table_rows.append(item)
        md_rows.append([
            item["setting"],
            item["checkpoint"],
            item["pickscore_mean"],
            item["clip_mean"],
            item["hpsv2_mean"],
            item["aes_mean"],
            item["fid_vs_baseline"],
        ])
    write_csv(
        OUTPUT_DIR / "table_drpo_ablation_best.csv",
        ["setting", "checkpoint", "pickscore_mean", "clip_mean", "hpsv2_mean", "aes_mean", "fid_vs_baseline"],
        table_rows,
    )
    write_markdown_table(
        OUTPUT_DIR / "table_drpo_ablation_best.md",
        ["Setting", "Checkpoint", "PickScore", "CLIP", "HPSv2", "AES", "FID"],
        md_rows,
    )
    return table_rows


def svg_escape(text: str) -> str:
    return html.escape(text, quote=True)


def draw_line_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]], str]],
    horizontal_lines: list[tuple[str, float, str]] | None = None,
) -> None:
    horizontal_lines = horizontal_lines or []
    width = 1200
    height = 720
    margin_left = 95
    margin_right = 220
    margin_top = 70
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    xs = [x for _, points, _ in series for x, _ in points]
    ys = [y for _, points, _ in series for _, y in points]
    ys.extend(value for _, value, _ in horizontal_lines)
    if not xs or not ys:
        return
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1e-6
    y_pad = (y_max - y_min) * 0.12
    y_min -= y_pad
    y_max += y_pad

    def x_to_px(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(value: float) -> float:
        return margin_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcfa"/>',
        f'<text x="{margin_left}" y="36" font-size="28" font-family="monospace" fill="#222">{svg_escape(title)}</text>',
    ]

    for i in range(6):
        y_value = y_min + (y_max - y_min) * i / 5
        y_px = y_to_px(y_value)
        parts.append(f'<line x1="{margin_left}" y1="{y_px:.2f}" x2="{margin_left + plot_width}" y2="{y_px:.2f}" stroke="#e3e1dc" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y_px + 5:.2f}" text-anchor="end" font-size="14" font-family="monospace" fill="#555">{y_value:.4f}</text>')

    for i in range(6):
        x_value = x_min + (x_max - x_min) * i / 5
        x_px = x_to_px(x_value)
        parts.append(f'<line x1="{x_px:.2f}" y1="{margin_top}" x2="{x_px:.2f}" y2="{margin_top + plot_height}" stroke="#eceae5" stroke-width="1"/>')
        parts.append(f'<text x="{x_px:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="14" font-family="monospace" fill="#555">{int(round(x_value))}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#444" stroke-width="2"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#444" stroke-width="2"/>')
    parts.append(f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 20}" text-anchor="middle" font-size="18" font-family="monospace" fill="#333">{svg_escape(x_label)}</text>')
    parts.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" font-size="18" font-family="monospace" fill="#333" transform="rotate(-90 24 {margin_top + plot_height / 2:.2f})">{svg_escape(y_label)}</text>'
    )

    for label, value, color in horizontal_lines:
        y_px = y_to_px(value)
        parts.append(f'<line x1="{margin_left}" y1="{y_px:.2f}" x2="{margin_left + plot_width}" y2="{y_px:.2f}" stroke="{color}" stroke-width="2" stroke-dasharray="10,6"/>')
        parts.append(f'<text x="{margin_left + plot_width + 10}" y="{y_px + 5:.2f}" font-size="14" font-family="monospace" fill="{color}">{svg_escape(label)} {value:.4f}</text>')

    legend_y = margin_top + 20
    for index, (label, points, color) in enumerate(series):
        polyline = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x_to_px(x):.2f}" cy="{y_to_px(y):.2f}" r="3.5" fill="{color}"/>')
        ly = legend_y + index * 28
        parts.append(f'<line x1="{margin_left + plot_width + 10}" y1="{ly}" x2="{margin_left + plot_width + 38}" y2="{ly}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{margin_left + plot_width + 48}" y="{ly + 5}" font-size="15" font-family="monospace" fill="#333">{svg_escape(label)}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def draw_scatter_plot(path: Path, title: str, points: list[tuple[str, float, float, str]]) -> None:
    width = 1100
    height = 720
    margin_left = 100
    margin_right = 120
    margin_top = 70
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1
    x_pad = (x_max - x_min) * 0.12
    y_pad = (y_max - y_min) * 0.12
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def x_to_px(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(value: float) -> float:
        return margin_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcfa"/>',
        f'<text x="{margin_left}" y="36" font-size="28" font-family="monospace" fill="#222">{svg_escape(title)}</text>',
    ]

    for i in range(6):
        x_value = x_min + (x_max - x_min) * i / 5
        y_value = y_min + (y_max - y_min) * i / 5
        x_px = x_to_px(x_value)
        y_px = y_to_px(y_value)
        parts.append(f'<line x1="{x_px:.2f}" y1="{margin_top}" x2="{x_px:.2f}" y2="{margin_top + plot_height}" stroke="#eceae5" stroke-width="1"/>')
        parts.append(f'<line x1="{margin_left}" y1="{y_px:.2f}" x2="{margin_left + plot_width}" y2="{y_px:.2f}" stroke="#eceae5" stroke-width="1"/>')
        parts.append(f'<text x="{x_px:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="14" font-family="monospace" fill="#555">{x_value:.4f}</text>')
        parts.append(f'<text x="{margin_left - 10}" y="{y_px + 5:.2f}" text-anchor="end" font-size="14" font-family="monospace" fill="#555">{y_value:.1f}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#444" stroke-width="2"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#444" stroke-width="2"/>')
    parts.append(f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 20}" text-anchor="middle" font-size="18" font-family="monospace" fill="#333">PickScore mean</text>')
    parts.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" font-size="18" font-family="monospace" fill="#333" transform="rotate(-90 24 {margin_top + plot_height / 2:.2f})">FID vs baseline</text>'
    )

    for label, x_value, y_value, color in points:
        x_px = x_to_px(x_value)
        y_px = y_to_px(y_value)
        parts.append(f'<circle cx="{x_px:.2f}" cy="{y_px:.2f}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{x_px + 10:.2f}" y="{y_px - 10:.2f}" font-size="15" font-family="monospace" fill="#333">{svg_escape(label)}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_figures(rows: list[dict[str, str]]) -> None:
    colors = {
        "Draft": "#c44e52",
        "DPO-latent": "#4c72b0",
        "DPO-mae-latent": "#8172b2",
        "DrPO-default": "#55a868",
        "VGG-Flow": "#dd8452",
        "Baseline": "#222222",
    }
    baseline_row = pick_best_row(rows_for_prefix(rows, COMPARISON_METHODS[0].summary_prefix))
    baseline_pickscore = baseline_row["_pickscore"] if baseline_row else None

    learning_series: list[tuple[str, list[tuple[float, float]], str]] = []
    for spec in COMPARISON_METHODS[1:]:
        latest_rows = pick_representative_run_rows(rows, spec.summary_prefix)
        points = sorted(
            [
                (row["_step"], row["_pickscore"])
                for row in latest_rows
                if row["_step"] is not None and row["_pickscore"] is not None and row["_step"] <= MAX_PLOT_STEP
            ],
            key=lambda item: item[0],
        )
        if points:
            if baseline_pickscore is not None:
                points = [(0, baseline_pickscore), *points]
            learning_series.append((spec.label, points, colors[spec.label]))

    draw_line_chart(
        OUTPUT_DIR / "figure_comparison_learning_curves.svg",
        "Comparison Learning Curves on PickScore (step 0 = raw baseline)",
        "Checkpoint step",
        "PickScore mean",
        learning_series,
        horizontal_lines=[("Baseline", baseline_pickscore, colors["Baseline"])] if baseline_pickscore is not None else [],
    )

    drpo_colors = ["#55a868", "#4c72b0", "#dd8452", "#c44e52", "#8172b2", "#937860", "#64b5cd", "#da8bc3"]
    for group_name, group_title, specs in DRPO_ABLATION_GROUPS:
        drpo_series: list[tuple[str, list[tuple[float, float]], str]] = []
        for index, spec in enumerate(specs):
            run_rows = pick_representative_run_rows(rows, spec.summary_prefix)
            points = sorted(
                [
                    (row["_step"], row["_pickscore"])
                    for row in run_rows
                    if row["_step"] is not None and row["_pickscore"] is not None and row["_step"] <= MAX_PLOT_STEP
                ],
                key=lambda item: item[0],
            )
            if points and baseline_pickscore is not None:
                points = [(0, baseline_pickscore), *points]
            if points:
                drpo_series.append((spec.label, points, drpo_colors[index % len(drpo_colors)]))
        draw_line_chart(
            OUTPUT_DIR / f"figure_drpo_ablations_{group_name}_learning_curves.svg",
            group_title,
            "Checkpoint step",
            "PickScore mean",
            drpo_series,
            horizontal_lines=[("Baseline", baseline_pickscore, colors["Baseline"])] if baseline_pickscore is not None else [],
        )

    scatter_points: list[tuple[str, float, float, str]] = []
    for spec in COMPARISON_METHODS:
        selected = baseline_row if spec.label == "Baseline" else pick_step_1000_or_latest_row(rows, spec.summary_prefix)
        if not selected or selected["_pickscore"] is None or selected["_fid"] is None:
            continue
        scatter_points.append((spec.label, selected["_pickscore"], selected["_fid"], colors.get(spec.label, "#333333")))
    draw_scatter_plot(
        OUTPUT_DIR / "figure_comparison_pickscore_vs_fid.svg",
        "Fixed-Step Trade-off: PickScore vs FID",
        scatter_points,
    )


def load_manifest(manifest_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def build_image_grid(rows: list[dict[str, str]]) -> None:
    methods = [
        ("Baseline", REPO_ROOT / "samples" / "sd-turbo-baseline" / "default" / "manifest.jsonl"),
    ]

    manifest_specs = [
        ("Draft", "sd-turbo-lora/draft/sdturbo_lora/default"),
        ("DPO-latent", "sd-turbo-lora/dpo/online/latent"),
        ("DPO-mae-latent", "sd-turbo-lora/dpo/online/mae_latent"),
        ("DrPO-default", "sd-turbo-lora/drpo/online/default"),
        ("VGG-Flow", "sd-turbo-lora/vggflow/pickscore/default"),
    ]
    for label, prefix in manifest_specs:
        row = pick_last_checkpoint_row(rows, prefix)
        if row:
            methods.append((label, summary_to_manifest(row["summary_path"])))

    manifest_cache: dict[str, dict[int, dict[str, object]]] = {}
    for label, manifest in methods:
        entries = load_manifest(manifest)
        by_prompt = {
            int(item["prompt_id"]): item
            for item in entries
            if int(item["seed"]) == 42 and int(item["prompt_id"]) in GRID_PROMPT_IDS
        }
        manifest_cache[label] = by_prompt

    baseline_entries = manifest_cache["Baseline"]
    lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8' />",
        "<title>Existing Experiment Image Grid</title>",
        "<style>",
        "body { font-family: monospace; margin: 24px; background: #faf9f6; color: #222; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #d8d2c8; vertical-align: top; padding: 8px; }",
        "th { background: #efe8dc; position: sticky; top: 0; }",
        "img { width: 192px; height: 192px; object-fit: cover; display: block; background: #ddd; }",
        ".prompt { min-width: 280px; max-width: 280px; font-size: 13px; line-height: 1.35; }",
        ".caption { margin-top: 6px; font-size: 12px; color: #555; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Existing Experiment Image Grid</h1>",
        "<p>Seed fixed to 42. Methods use the best available checkpoint sampled in the current workspace for each family.</p>",
        "<table>",
        "<thead><tr><th>Prompt</th>" + "".join(f"<th>{html.escape(label)}</th>" for label, _ in methods) + "</tr></thead>",
        "<tbody>",
    ]
    for prompt_id in GRID_PROMPT_IDS:
        baseline_item = baseline_entries[prompt_id]
        prompt = str(baseline_item["prompt"])
        lines.append("<tr>")
        lines.append(f"<td class='prompt'><strong>prompt_id={prompt_id}</strong><br />{html.escape(prompt)}</td>")
        for label, _ in methods:
            item = manifest_cache[label].get(prompt_id)
            if not item:
                lines.append("<td>NA</td>")
                continue
            img_path = REPO_ROOT / str(item["image_path"])
            rel_path = Path(
                Path(
                    Path(
                        Path(
                            str(img_path.relative_to(REPO_ROOT))
                        )
                    )
                )
            )
            html_rel = Path(Path(rel_path).as_posix())
            image_href = Path(REPO_ROOT / "results" / "existing_experiment_assets").joinpath("dummy")
            relative = Path(img_path).relative_to(REPO_ROOT)
            rel_from_output = Path("../..") / relative
            lines.append(
                "<td>"
                f"<img src='{html.escape(rel_from_output.as_posix())}' alt='{html.escape(label)} prompt {prompt_id}' />"
                f"<div class='caption'>{html.escape(label)}</div>"
                "</td>"
            )
        lines.append("</tr>")
    lines.extend(["</tbody>", "</table>", "</body>", "</html>"])
    (OUTPUT_DIR / "image_grid_comparison.html").write_text("\n".join(lines), encoding="utf-8")


def build_readme(comparison_rows: list[dict[str, str]], ablation_rows: list[dict[str, str]]) -> None:
    lines = [
        "# Existing Experiment Assets",
        "",
        "Generated from `samples/metrics/summary.csv` and sampled manifests already present in the workspace.",
        "Tables and curves use a fixed comparison protocol: `step=0` is the raw SD-Turbo baseline, and learned methods are reported at `step=1000` when available, otherwise at their latest sampled checkpoint below 1000.",
        "",
        "## Files",
        "",
        "- `table_comparison_best.csv` / `table_comparison_best.md`: fixed-step comparison table using `step=1000` when available.",
        "- `table_drpo_ablation_best.csv` / `table_drpo_ablation_best.md`: fixed-step DrPO ablation table using `step=1000` when available.",
        "- `figure_comparison_learning_curves.svg`: PickScore-vs-checkpoint curves, all anchored at the same `step=0` raw baseline.",
        "- `figure_drpo_ablations_batch_gen_learning_curves.svg`: batch-size ablation curves with the same `step=0` baseline anchor.",
        "- `figure_drpo_ablations_reward_learning_curves.svg`: reward-choice ablation curves with the same `step=0` baseline anchor.",
        "- `figure_comparison_pickscore_vs_fid.svg`: fixed-step (`step=1000` when available) PickScore/FID trade-off scatter.",
        "- `image_grid_comparison.html`: prompt-aligned image grid for baseline, Draft, DPO variants, DrPO, and VGG-Flow.",
        "",
        "## Quick Notes",
        "",
    ]
    if comparison_rows:
        learned = [row for row in comparison_rows if row["method"] != "Baseline"]
        top = max(learned, key=lambda row: float(row["pickscore_mean"])) if learned else comparison_rows[0]
        lines.append(f"- Highest PickScore at the fixed comparison step is `{top['method']}` at `{top['pickscore_mean']}`.")
    if ablation_rows:
        top = max(ablation_rows, key=lambda row: float(row["pickscore_mean"]))
        lines.append(f"- Highest PickScore among completed DrPO ablations at the fixed step is `{top['setting']}` at `{top['pickscore_mean']}`.")
    lines.append("")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    comparison_rows = build_comparison_tables(rows)
    ablation_rows = build_drpo_ablation_tables(rows)
    build_figures(rows)
    build_image_grid(rows)
    build_readme(comparison_rows, ablation_rows)
    print(f"Wrote assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
