"""Build thesis/results_macros.tex from Phase 3 (SE 5-dataset) run artifacts.

Produces 30 \\providecommand macros that thesis/main.tex actually cites:

[from results/fusion.generation_level/summary.json]
- HeadlineSEAuroc, HeadlineEnergyAuroc
- HeadlineCorpusLiftRF, HeadlineCorpusLiftGBM

[from results/robustness.generation_level/corpus_bin_reliability.json]
- HeadlineEntityPairSEDelta, HeadlineEntityFreqSEDelta
- HeadlineEntityPairEnergyDelta, HeadlineEntityFreqEnergyDelta
- HeadlineEntityPairSEHigh, HeadlineEntityPairSELow
- HeadlineEntityPairEnergyHigh, HeadlineEntityPairEnergyLow
- HeadlineQaBridgeNllDelta

[from results/review_ablations.json]
- HeadlinePairSEDeltaCIlow/high, HeadlineFreqSEDeltaCIlow/high
- HeadlinePairMinusFreqSECIlow/high, HeadlinePairMinusFreqEnergyCIlow/high
- HeadlinePairMinusFreqSEPositiveFrac, HeadlinePairMinusFreqEnergyPositiveFrac
- HeadlineFusionLiftCIlow/high
- HeadlineSVAMPExclSERatio, HeadlineSVAMPExclEnergyRatio

[derived]
- HeadlineEntityPairOverEntityFreqRatio (= EntityPairSEDelta / EntityFreqSEDelta)
- HeadlineRatioRangeLow, HeadlineRatioRangeHigh (= min / max of SVAMP-incl vs SVAMP-excl ratios)

Usage:
  uv run python experiments/scripts/build_results_macros.py \\
    --run-dir /path/to/run/qwen \\
    --out thesis/results_macros.tex
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text())


def _delta_from_bins(bin_records: list[dict]) -> tuple[float | None, float | None, float | None]:
    """Return (delta = max - min, max_auroc, min_auroc) over a method's per-decile bins."""
    aurocs = [b["auroc"] for b in bin_records if b.get("auroc") is not None]
    if len(aurocs) < 2:
        return None, None, None
    hi, lo = max(aurocs), min(aurocs)
    return hi - lo, hi, lo


def _method_lookup(corpus_bin: dict, axis: str, method: str) -> list[dict]:
    """Return per-decile bin records for (axis, method)."""
    return corpus_bin.get(axis, {}).get(method, {}).get("bins", [])


def _ratio(num: float | None, denom: float | None) -> float | None:
    if num is None or denom is None or denom == 0:
        return None
    return num / denom


def _fmt3(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _fmt2(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _fmt_pct(x: float | None) -> str:
    """Bootstrap positive frac → percent string."""
    return "n/a" if x is None else f"{x * 100:.1f}"


def build_macros(run_dir: Path) -> dict[str, Any]:
    results = run_dir / "results"
    fusion_summary = _load(results / "fusion.generation_level/summary.json")
    corpus_bin = _load(results / "robustness.generation_level/corpus_bin_reliability.json")
    review = _load(results / "review_ablations.json")

    # --- Fusion single + lift ---
    methods = fusion_summary["methods"]
    se = methods.get("SE-only", {}).get("auroc")
    energy = methods.get("Energy-only", {}).get("auroc")

    def lift(model_short: str) -> float | None:
        with_c = methods.get(f"{model_short} (with corpus)", {}).get("auroc")
        no_c = methods.get(f"{model_short} (no corpus)", {}).get("auroc")
        if with_c is None or no_c is None:
            return None
        return with_c - no_c

    rf_lift = lift("random forest")
    gbm_lift = lift("gradient boosting")

    # --- Decile range Δ + endpoint values ---
    pair_se_delta, pair_se_high, pair_se_low = _delta_from_bins(
        _method_lookup(corpus_bin, "entity_pair_cooccurrence_axis_bin_10", "SE-only"))
    freq_se_delta, _, _ = _delta_from_bins(
        _method_lookup(corpus_bin, "entity_frequency_axis_bin_10", "SE-only"))
    pair_en_delta, pair_en_high, pair_en_low = _delta_from_bins(
        _method_lookup(corpus_bin, "entity_pair_cooccurrence_axis_bin_10", "Energy-only"))
    freq_en_delta, _, _ = _delta_from_bins(
        _method_lookup(corpus_bin, "entity_frequency_axis_bin_10", "Energy-only"))
    qa_nll_delta, _, _ = _delta_from_bins(
        _method_lookup(corpus_bin, "qa_bridge_axis_bin_10", "logit-diagnostic-only"))

    ratio_full = _ratio(pair_se_delta, freq_se_delta)

    # --- Bootstrap CI (review_ablations.py) ---
    bs_se = review.get("bootstrap_se", {})
    bs_en = review.get("bootstrap_energy", {})
    pair_se_ci = bs_se.get("delta_a_ci") or (None, None)
    freq_se_ci = bs_se.get("delta_b_ci") or (None, None)
    pair_minus_freq_se_ci = bs_se.get("diff_ci") or (None, None)
    pair_minus_freq_en_ci = bs_en.get("diff_ci") or (None, None)
    pair_minus_freq_se_pos = bs_se.get("diff_positive_frac")
    pair_minus_freq_en_pos = bs_en.get("diff_positive_frac")
    fusion_lift_ci = review.get("fusion_lift_gbm", {}).get("lift_ci") or (None, None)

    svamp = review.get("svamp_excluded", {})
    svamp_se_ratio = svamp.get("se_ratio")
    svamp_en_ratio = svamp.get("energy_ratio")

    # Ratio range = min/max of (full sample ratio, SVAMP-excl ratio)
    ratios = [r for r in (ratio_full, svamp_se_ratio) if r is not None]
    ratio_low = min(ratios) if ratios else None
    ratio_high = max(ratios) if ratios else None

    return {
        # fusion single + lift
        "HeadlineSEAuroc": _fmt3(se),
        "HeadlineEnergyAuroc": _fmt3(energy),
        "HeadlineCorpusLiftRF": _fmt3(rf_lift),
        "HeadlineCorpusLiftGBM": _fmt3(gbm_lift),
        # decile range Δ
        "HeadlineEntityPairSEDelta": _fmt3(pair_se_delta),
        "HeadlineEntityFreqSEDelta": _fmt3(freq_se_delta),
        "HeadlineEntityPairEnergyDelta": _fmt3(pair_en_delta),
        "HeadlineEntityFreqEnergyDelta": _fmt3(freq_en_delta),
        "HeadlineEntityPairSEHigh": _fmt3(pair_se_high),
        "HeadlineEntityPairSELow": _fmt3(pair_se_low),
        "HeadlineEntityPairEnergyHigh": _fmt3(pair_en_high),
        "HeadlineEntityPairEnergyLow": _fmt3(pair_en_low),
        "HeadlineQaBridgeNllDelta": _fmt3(qa_nll_delta),
        # bootstrap CI
        "HeadlinePairSEDeltaCIlow": _fmt3(pair_se_ci[0]),
        "HeadlinePairSEDeltaCIhigh": _fmt3(pair_se_ci[1]),
        "HeadlineFreqSEDeltaCIlow": _fmt3(freq_se_ci[0]),
        "HeadlineFreqSEDeltaCIhigh": _fmt3(freq_se_ci[1]),
        "HeadlinePairMinusFreqSECIlow": _fmt3(pair_minus_freq_se_ci[0]),
        "HeadlinePairMinusFreqSECIhigh": _fmt3(pair_minus_freq_se_ci[1]),
        "HeadlinePairMinusFreqEnergyCIlow": _fmt3(pair_minus_freq_en_ci[0]),
        "HeadlinePairMinusFreqEnergyCIhigh": _fmt3(pair_minus_freq_en_ci[1]),
        "HeadlinePairMinusFreqSEPositiveFrac": _fmt_pct(pair_minus_freq_se_pos),
        "HeadlinePairMinusFreqEnergyPositiveFrac": _fmt_pct(pair_minus_freq_en_pos),
        "HeadlineFusionLiftCIlow": _fmt3(fusion_lift_ci[0]),
        "HeadlineFusionLiftCIhigh": _fmt3(fusion_lift_ci[1]),
        # SVAMP sensitivity + ratio
        "HeadlineSVAMPExclSERatio": _fmt2(svamp_se_ratio),
        "HeadlineSVAMPExclEnergyRatio": _fmt2(svamp_en_ratio),
        "HeadlineEntityPairOverEntityFreqRatio": _fmt2(ratio_full),
        "HeadlineRatioRangeLow": _fmt2(ratio_low),
        "HeadlineRatioRangeHigh": _fmt2(ratio_high),
    }


HEADER = """% Auto-generated by experiments/scripts/build_results_macros.py.
% DO NOT hand-edit. Re-run after each pipeline pass to refresh.
%
% Run dir: {run_dir}
% Built at: {built_at}
%
% These commands hold the headline numbers cited by thesis/main.tex.
"""


def emit_tex(macros: dict[str, str], run_dir: Path) -> str:
    body = HEADER.format(
        run_dir=run_dir,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    for name, value in macros.items():
        body += f"\\providecommand{{\\{name}}}{{{value}}}\n"
    return body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Per-model run dir (e.g. <RUN>/qwen) — must contain results/{fusion.generation_level,robustness.generation_level,review_ablations.json}.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("thesis/results_macros.tex"),
        help="Output .tex path (default: thesis/results_macros.tex).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    macros = build_macros(args.run_dir)
    text = emit_tex(macros, args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {len(macros)} macros → {args.out}")
    # also print n/a count for visibility
    na_count = sum(1 for v in macros.values() if v == "n/a")
    if na_count:
        print(f"WARNING: {na_count} macros are 'n/a' (missing input columns / methods)")
        for k, v in macros.items():
            if v == "n/a":
                print(f"  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
