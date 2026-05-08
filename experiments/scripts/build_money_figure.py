#!/usr/bin/env python3
"""Build the thesis "money figure" — signal × corpus_axis_bin reliability map.

Layout (single PNG, two stacked panels sharing x-axis):

  Panel A (top, candidate-level scope):
    y-axis: candidate-level signals
      - mean_negative_log_probability
      - confidence_margin
      - logit_variance
      - entity_pair_cooccurrence_axis (the only candidate-level corpus signal)
    x-axis: corpus support bins (5-bin sensitivity primary; switch via --bin-field)
    cell value: paired win-rate (paired Δ ≠ 0 prompts only, ties excluded)
    color scale: diverging around 0.5 (RdBu_r) — red = hallucinated wins (>0.5),
                 blue = correct wins (<0.5)
    annotation: cell text shows "win 0.55 / n=812" etc.

  Panel B (bottom, prompt-level scope):
    y-axis: prompt-level signals
      - semantic_entropy_nli_likelihood (SE)
      - semantic_entropy_cluster_count
      - semantic_energy_cluster_uncertainty
      - semantic_energy_sample_energy
      - semantic_energy_boltzmann_diagnostic
      - entity_frequency_axis (largely prompt-level by data; 86% identical)
    x-axis: same corpus support bins
    cell value: bin mean (z-scored within signal)
    color scale: viridis (sequential)
    annotation: cell text shows "μ=0.66 (n=2751)" or similar

Both panels share x-tick labels (corpus support bin order).

Usage:
  uv sync --group figures
  uv run python experiments/scripts/build_money_figure.py \
      --features /mnt/data/.../features.parquet \
      --out thesis/figures/money_figure.png \
      --bin-field corpus_axis_bin_5

Output:
  PNG file at the given path. Companion JSON sidecar (path + ".data.json") with
  the underlying matrices for caption-table cross-reference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CANDIDATE_LEVEL_SIGNALS = [
    ("mean_negative_log_probability", "NLL (mean)"),
    ("confidence_margin", "confidence margin"),
    ("logit_variance", "logit variance"),
    ("entity_pair_cooccurrence_axis", "pair co-occurrence axis"),
]
PROMPT_LEVEL_SIGNALS = [
    ("semantic_entropy_nli_likelihood", "Semantic Entropy"),
    ("semantic_entropy_cluster_count", "SE cluster count"),
    ("semantic_energy_cluster_uncertainty", "Semantic Energy"),
    ("semantic_energy_sample_energy", "Energy (sample mean)"),
    ("semantic_energy_boltzmann_diagnostic", "Boltzmann diagnostic"),
    ("entity_frequency_axis", "entity frequency axis"),
]
BIN_ORDER = {
    "corpus_axis_bin": ("low_support", "medium_support", "high_support"),
    "corpus_axis_bin_5": (
        "very_low_support",
        "low_support",
        "mid_support",
        "high_support",
        "very_high_support",
    ),
    "corpus_axis_bin_10": tuple(f"decile_{i:02d}_{i+10:02d}" for i in range(0, 100, 10)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="features.parquet path")
    parser.add_argument(
        "--bin-field",
        default="corpus_axis_bin_5",
        choices=list(BIN_ORDER),
        help="Corpus support bin column to put on x-axis",
    )
    parser.add_argument(
        "--out",
        default="thesis/figures/money_figure.png",
        help="Output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--width", type=float, default=12.0, help="Figure width (inches)")
    parser.add_argument(
        "--height",
        type=float,
        default=10.0,
        help="Figure height (inches); panels stack vertically",
    )
    return parser.parse_args()


def _features(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("features")
    return payload if isinstance(payload, dict) else {}


def _is_hallucination(row: dict[str, Any]) -> bool:
    raw = row.get("is_hallucination")
    if isinstance(raw, bool):
        return raw
    correct = row.get("is_correct")
    if isinstance(correct, bool):
        return not correct
    return bool(raw)


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _bin_value(row: dict[str, Any], field: str) -> str | None:
    value = _features(row).get(field)
    if value in (None, ""):
        return None
    return str(value)


def compute_candidate_winrate(
    rows: list[dict[str, Any]], signal: str, bin_field: str, bins: tuple[str, ...]
) -> dict[str, dict[str, float | int]]:
    """Per-bin paired win-rate for a candidate-level signal.

    win = hallucinated candidate has *higher* score than correct candidate within
    the same prompt. ties excluded from denominator.
    """
    result: dict[str, dict[str, float | int]] = {}
    for bin_id in bins:
        by_pid: dict[str, dict[str, float]] = defaultdict(dict)
        for row in rows:
            if _bin_value(row, bin_field) != bin_id:
                continue
            score = _coerce_float(_features(row).get(signal))
            if score is None:
                continue
            slot = "hall" if _is_hallucination(row) else "ok"
            by_pid[row["prompt_id"]][slot] = score
        wins = ties = total_nontied = total = 0
        for pair in by_pid.values():
            if "hall" not in pair or "ok" not in pair:
                continue
            total += 1
            delta = pair["hall"] - pair["ok"]
            if delta == 0:
                ties += 1
                continue
            total_nontied += 1
            if delta > 0:
                wins += 1
        win_rate = wins / total_nontied if total_nontied else None
        result[bin_id] = {
            "win_rate": win_rate,
            "n_pairs": total,
            "n_nontied": total_nontied,
            "n_wins": wins,
            "n_ties": ties,
        }
    return result


def compute_prompt_bin_mean(
    rows: list[dict[str, Any]], signal: str, bin_field: str, bins: tuple[str, ...]
) -> dict[str, dict[str, float | int]]:
    """Per-bin prompt-level mean for a prompt-level signal.

    Collapses each prompt to a single value (first row's signal — they are
    broadcast or equivalent at prompt scope).
    """
    seen: set[str] = set()
    by_bin: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        pid = row["prompt_id"]
        if pid in seen:
            continue
        b = _bin_value(row, bin_field)
        v = _coerce_float(_features(row).get(signal))
        if b is None or v is None:
            continue
        seen.add(pid)
        by_bin[b].append(v)
    result: dict[str, dict[str, float | int]] = {}
    for bin_id in bins:
        values = by_bin.get(bin_id, [])
        if not values:
            result[bin_id] = {"mean": None, "n": 0}
            continue
        n = len(values)
        mean = sum(values) / n
        var = sum((x - mean) ** 2 for x in values) / n
        result[bin_id] = {"mean": mean, "std": math.sqrt(var), "n": n}
    return result


def zscore_within_signal(matrix: list[list[float | None]]) -> list[list[float | None]]:
    """Z-score each row independently so different-scale signals share a colormap."""
    out: list[list[float | None]] = []
    for row in matrix:
        finite = [v for v in row if v is not None and math.isfinite(v)]
        if len(finite) < 2:
            out.append([None] * len(row))
            continue
        mean = sum(finite) / len(finite)
        var = sum((v - mean) ** 2 for v in finite) / len(finite)
        std = math.sqrt(var) or 1.0
        out.append([None if v is None else (v - mean) / std for v in row])
    return out


def main() -> int:
    args = parse_args()
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc}. Run `uv sync --group figures`.", file=sys.stderr)
        return 2

    features_path = Path(args.features)
    if not features_path.exists():
        print(f"features parquet not found: {features_path}", file=sys.stderr)
        return 2

    table = pq.read_table(
        features_path,
        columns=["prompt_id", "candidate_label", "is_hallucination", "is_correct", "features"],
    )
    rows = table.to_pylist()
    bin_field = args.bin_field
    bins = BIN_ORDER[bin_field]

    # ---- Panel A: candidate-level signals × bins (paired win-rate) ----
    candidate_matrix = []
    candidate_n = []
    for sig, _ in CANDIDATE_LEVEL_SIGNALS:
        per_bin = compute_candidate_winrate(rows, sig, bin_field, bins)
        candidate_matrix.append([per_bin[b]["win_rate"] for b in bins])
        candidate_n.append([per_bin[b]["n_nontied"] for b in bins])

    # ---- Panel B: prompt-level signals × bins (bin mean, then z-scored per signal) ----
    prompt_matrix_raw = []
    prompt_n = []
    for sig, _ in PROMPT_LEVEL_SIGNALS:
        per_bin = compute_prompt_bin_mean(rows, sig, bin_field, bins)
        prompt_matrix_raw.append([per_bin[b]["mean"] for b in bins])
        prompt_n.append([per_bin[b]["n"] for b in bins])
    prompt_matrix_z = zscore_within_signal(prompt_matrix_raw)

    # ---- Plot ----
    panel_a_height = 0.35
    fig = plt.figure(figsize=(args.width, args.height))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[len(CANDIDATE_LEVEL_SIGNALS), len(PROMPT_LEVEL_SIGNALS)],
        width_ratios=[1, 0.025],
        hspace=0.45,
        wspace=0.04,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    cax_a = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 0])
    cax_b = fig.add_subplot(gs[1, 1])

    # --- Panel A heatmap ---
    arr_a = np.array(
        [[v if v is not None else np.nan for v in row] for row in candidate_matrix],
        dtype=float,
    )
    im_a = ax_a.imshow(arr_a, cmap="coolwarm", vmin=0.3, vmax=0.7, aspect="auto")
    ax_a.set_yticks(range(len(CANDIDATE_LEVEL_SIGNALS)))
    ax_a.set_yticklabels([label for _, label in CANDIDATE_LEVEL_SIGNALS])
    ax_a.set_xticks(range(len(bins)))
    ax_a.set_xticklabels([b.replace("_", " ") for b in bins], rotation=20, ha="right")
    ax_a.set_title(
        "Panel A: candidate-level signals — paired win-rate (hallucinated vs correct, ties excluded)",
        fontsize=11,
        loc="left",
    )
    for i, row in enumerate(candidate_matrix):
        for j, v in enumerate(row):
            n = candidate_n[i][j]
            if v is None:
                ax_a.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                ax_a.text(
                    j,
                    i,
                    f"{v:.2f}\nn={n}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white" if abs(v - 0.5) > 0.12 else "black",
                )
    plt.colorbar(im_a, cax=cax_a).set_label("paired win-rate", fontsize=8)

    # --- Panel B heatmap ---
    arr_b = np.array(
        [[v if v is not None else np.nan for v in row] for row in prompt_matrix_z],
        dtype=float,
    )
    im_b = ax_b.imshow(arr_b, cmap="cividis", aspect="auto")
    ax_b.set_yticks(range(len(PROMPT_LEVEL_SIGNALS)))
    ax_b.set_yticklabels([label for _, label in PROMPT_LEVEL_SIGNALS])
    ax_b.set_xticks(range(len(bins)))
    ax_b.set_xticklabels([b.replace("_", " ") for b in bins], rotation=20, ha="right")
    ax_b.set_title(
        "Panel B: prompt-level signals — bin mean (z-scored within signal so different-scale signals share the colormap)",
        fontsize=11,
        loc="left",
    )
    for i, row in enumerate(prompt_matrix_raw):
        for j, raw_v in enumerate(row):
            n = prompt_n[i][j]
            if raw_v is None:
                ax_b.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                ax_b.text(
                    j,
                    i,
                    f"μ={raw_v:.2f}\nn={n}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                )
    plt.colorbar(im_b, cax=cax_b).set_label("z-score (per signal)", fontsize=8)

    fig.suptitle(
        f"Money Figure — Signal × Corpus Support Reliability Map ({bin_field})",
        fontsize=13,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Also dump the underlying data as a sidecar JSON so the LaTeX caption /
    # tables can cite exact numbers without re-running this script.
    sidecar = {
        "bin_field": bin_field,
        "bins": list(bins),
        "candidate_signals": [
            {
                "feature_key": sig,
                "label": label,
                "win_rate": candidate_matrix[i],
                "n_nontied": candidate_n[i],
            }
            for i, (sig, label) in enumerate(CANDIDATE_LEVEL_SIGNALS)
        ],
        "prompt_signals": [
            {
                "feature_key": sig,
                "label": label,
                "bin_mean": prompt_matrix_raw[i],
                "bin_mean_zscore": prompt_matrix_z[i],
                "n_prompts": prompt_n[i],
            }
            for i, (sig, label) in enumerate(PROMPT_LEVEL_SIGNALS)
        ],
        "notes": [
            "Panel A cells: paired win-rate of hallucinated vs correct candidate within the same prompt, ties excluded.",
            "Panel B cells: prompt-level bin mean z-scored within each signal row to share a single colormap.",
            "SE / Energy / energy_diagnostic / entity_frequency_axis are prompt-level (broadcast); their candidate-row paired win-rate is structurally undefined and reported instead via Panel B.",
        ],
    }
    sidecar_path = out_path.with_suffix(out_path.suffix + ".data.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out_path), "sidecar": str(sidecar_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
