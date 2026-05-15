#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    left_name = rows[0]["comparison"].split(" vs ")[0]
    right_name = rows[0]["comparison"].split(" vs ")[1]
    valid = [r for r in rows if r["winner_method"] in {left_name, right_name, "tie"}]
    left_wins = sum(r["winner_method"] == left_name for r in valid)
    right_wins = sum(r["winner_method"] == right_name for r in valid)
    ties = sum(r["winner_method"] == "tie" for r in valid)
    n = len(valid)
    denom = max(1, n - ties)
    return {
        "comparison": rows[0]["comparison"],
        "left_name": left_name,
        "right_name": right_name,
        "num_pairs": len(rows),
        "num_valid": n,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": ties,
        "invalid": len(rows) - n,
        "left_win_rate_excluding_ties": left_wins / denom,
        "right_win_rate_excluding_ties": right_wins / denom,
        "left_preference_share_ties_half": (left_wins + 0.5 * ties) / max(1, n),
        "right_preference_share_ties_half": (right_wins + 0.5 * ties) / max(1, n),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sharded Qwen-VL win-rate judgments.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-prefix", default="draft_drpo_qwen3vl_aggregate")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*_judgments.jsonl"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in files:
        for row in load_jsonl(path):
            grouped[row["comparison"]].append(row)

    summaries = []
    for comparison, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: (int(r["prompt_id"]), int(r["seed"])))
        safe_name = comparison.replace(" ", "_").replace("/", "_")
        jsonl_path = args.input_dir / f"{args.output_prefix}_{safe_name}_judgments.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_csv(args.input_dir / f"{args.output_prefix}_{safe_name}_judgments.csv", rows)
        summary = summarize(rows)
        summary["judgment_file"] = str(jsonl_path)
        summaries.append(summary)

    output = {"comparisons": summaries}
    summary_path = args.input_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
