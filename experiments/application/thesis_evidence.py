"""Application-layer thesis evidence export and validation services."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE_ORDER = [
    "SE-only",
    "Energy-only",
    "corpus-risk-only",
    "fixed linear 0.1/0.9",
    "fixed linear 0.5/0.5",
    "fixed linear 0.9/0.1",
    "hard cascade",
    "learned fusion without corpus",
    "learned fusion with corpus",
]

SOURCE_RELATIVE_PATHS = {
    "type_analysis": "type_analysis/summary.json",
    "fusion": "fusion/summary.json",
    "fusion_report": "fusion/report.md",
    "robustness": "robustness/summary.json",
}

FORBIDDEN_TEXT = [
    "0.736",
    "0.643",
    "0.613",
    "+0.029",
    "SE-gated Cascade",
    "SE-gated cascade",
    "Qwen2.5",
    "RTX 5090",
    "hallucination_lfe",
    "Energy AUROC 0.736",
    "xticklabels={SE-only, Energy-only, Cascade}",
    "old coverage-adaptive baseline",
    "Energy(q)",
    "upstream",
]

REQUIRED_TEXT = [
    "Corpus-Grounded Selective Fusion Detector",
    "full logits",
    "semantic_energy_proxy",
    "uv run",
    "thesis_evidence_table.tex",
    "thesis_evidence_summary.json",
    "learned fusion with corpus",
    "learned fusion without corpus",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def fmt_metric(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def display_status(value: Any) -> str:
    return latex_escape(value or "")


def display_reason(value: Any) -> str:
    if value == "full_logits_required":
        return "full logits"
    return latex_escape(value or "")


def collect_baselines(fusion: dict[str, Any]) -> list[dict[str, Any]]:
    by_name = {entry["method_name"]: entry for entry in fusion.get("baselines", [])}
    rows: list[dict[str, Any]] = []
    for name in BASELINE_ORDER:
        entry = by_name.get(name)
        if not entry:
            continue
        aggregate = entry.get("aggregate") or {}
        rows.append(
            {
                "method_name": name,
                "status": entry.get("status"),
                "auroc": aggregate.get("auroc"),
                "auprc": aggregate.get("auprc"),
                "f1": aggregate.get("f1"),
                "unavailable_reason": entry.get("unavailable_reason"),
                "full_logits_required": bool(entry.get("full_logits_required")),
                "rerun_required": bool(entry.get("rerun_required")),
            }
        )
    return rows


def collect_energy_status(type_summary: dict[str, Any]) -> dict[str, Any]:
    storage_report = type_summary.get("features_storage", {}).get("report", {})
    source_energy = (
        storage_report.get("source_artifacts", {})
        .get("energy", {})
        .get("storage", {})
        .get("report", {})
    )
    return {
        "status": source_energy.get("status"),
        "full_logits_required": source_energy.get("full_logits_required"),
        "rerun_required": source_energy.get("rerun_required"),
        "true_boltzmann_available": source_energy.get("true_boltzmann_available"),
        "message": source_energy.get("message"),
        "rerun_instructions": source_energy.get("rerun_instructions"),
        "formula_manifest_ref": source_energy.get("formula_manifest_ref"),
        "dataset_manifest_ref": source_energy.get("dataset_manifest_ref"),
    }


def collect_robustness(robustness: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = robustness.get("bootstrap", {}).get("comparisons", [])
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        if comparison.get("candidate_method") != "learned fusion with corpus":
            continue
        for metric in comparison.get("metrics", []):
            rows.append(
                {
                    "candidate_method": comparison.get("candidate_method"),
                    "reference_method": comparison.get("reference_method"),
                    "metric": metric.get("metric"),
                    "observed_delta": metric.get("observed_delta"),
                    "ci_95_lower": metric.get("ci_95_lower"),
                    "ci_95_upper": metric.get("ci_95_upper"),
                    "ci_crosses_zero": metric.get("ci_crosses_zero"),
                    "statistically_significant": metric.get("statistically_significant"),
                    "claim_text": metric.get("claim_text"),
                }
            )
    return rows


def write_latex_table(path: Path, baselines: list[dict[str, Any]], energy_status: dict[str, Any]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{현재 산출물 기반 탐지 기준선과 Energy 재실행 상태}",
        r"\label{tab:current_thesis_evidence}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"방법 & 상태 & AUROC & AUPRC & F1 & 재실행 사유 \\",
        r"\midrule",
    ]
    row_suffix = r"\\"
    for row in baselines:
        reason = row["unavailable_reason"] or ""
        line = "{} & {} & {} & {} & {} & {} ".format(
            latex_escape(row["method_name"]),
            display_status(row["status"]),
            fmt_metric(row["auroc"]),
            fmt_metric(row["auprc"]),
            fmt_metric(row["f1"]),
            display_reason(reason),
        )
        lines.append(line + row_suffix)
    lines.append(r"\midrule")
    energy_line = "{} & {} & {} & {} & {} & {} ".format(
        latex_escape("true Boltzmann Energy availability"),
        display_status("unavailable" if not energy_status.get("true_boltzmann_available") else "available"),
        "--",
        "--",
        "--",
        display_reason(energy_status.get("status") or ""),
    )
    lines.extend(
        [
            energy_line + row_suffix,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(results_dir: Path, table_path: Path) -> dict[str, Any]:
    source_paths = {key: str(results_dir / rel) for key, rel in SOURCE_RELATIVE_PATHS.items()}
    type_summary = load_json(Path(source_paths["type_analysis"]))
    fusion = load_json(Path(source_paths["fusion"]))
    robustness = load_json(Path(source_paths["robustness"]))

    baselines = collect_baselines(fusion)
    energy_status = collect_energy_status(type_summary)
    robustness_rows = collect_robustness(robustness)
    storage_report = type_summary.get("features_storage", {}).get("report", {})

    learned_with = next(row for row in baselines if row["method_name"] == "learned fusion with corpus")
    learned_without = next(row for row in baselines if row["method_name"] == "learned fusion without corpus")
    se_only = next(row for row in baselines if row["method_name"] == "SE-only")
    corpus_only = next(row for row in baselines if row["method_name"] == "corpus-risk-only")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact_paths": source_paths,
        "table_path": str(table_path),
        "method_name": fusion.get("method_name"),
        "row_count": fusion.get("row_count"),
        "datasets": fusion.get("datasets", []),
        "label_counts": storage_report.get("label_counts", {}),
        "label_presence_explanations": storage_report.get("label_presence_explanations", {}),
        "baselines": baselines,
        "headline_metrics": {
            "se_only_auroc": se_only["auroc"],
            "corpus_risk_only_auroc": corpus_only["auroc"],
            "learned_fusion_without_corpus_auroc": learned_without["auroc"],
            "learned_fusion_with_corpus_auroc": learned_with["auroc"],
            "learned_fusion_with_minus_without_corpus_auroc": learned_with["auroc"] - learned_without["auroc"],
            "learned_fusion_with_minus_without_corpus_auprc": learned_with["auprc"] - learned_without["auprc"],
        },
        "energy_status": energy_status,
        "robustness_comparisons": robustness_rows,
        "evidence_notes": [
            "experiments/literature/evidence_notes/semantic_energy_single_cluster.md",
            "experiments/literature/evidence_notes/quco_vs_probe.md",
        ],
        "thesis_safe_claims": [
            "Current learned fusion with corpus underperforms learned fusion without corpus on observed AUROC.",
            "Energy-dependent baselines are unavailable until row-level full logits are preserved by a repo-owned live generation run.",
            "semantic_energy_proxy is diagnostic metadata only and is not thesis-valid evidence.",
            "Future execution should use uv-managed commands and preserve row-level full logits.",
        ],
    }


def export_thesis_evidence(results_dir: Path, table_path: Path) -> dict[str, Any]:
    summary = build_summary(results_dir, table_path)
    write_latex_table(table_path, summary["baselines"], summary["energy_status"])
    summary_path = table_path.with_name("thesis_evidence_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "table_path": str(table_path),
        "summary_path": str(summary_path),
    }


def collect_tex_sources(thesis_path: Path) -> dict[Path, str]:
    sources: dict[Path, str] = {thesis_path: thesis_path.read_text(encoding="utf-8")}
    thesis_root = thesis_path.parent
    for tex_path in sorted(thesis_root.rglob("*.tex")):
        if tex_path == thesis_path:
            continue
        sources[tex_path] = tex_path.read_text(encoding="utf-8")
    return sources


def validate_thesis_evidence_links(thesis_path: Path, notes_dir: Path, summary_path: Path) -> list[str]:
    problems: list[str] = []
    tex_sources = collect_tex_sources(thesis_path)
    thesis = tex_sources[thesis_path]
    all_tex = "\n".join(tex_sources.values())
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for forbidden in FORBIDDEN_TEXT:
        for tex_path, text in tex_sources.items():
            if forbidden in text:
                problems.append(f"unsupported stale claim remains in {tex_path}: {forbidden}")

    for required in REQUIRED_TEXT:
        if required not in all_tex:
            problems.append(f"required evidence wording missing from thesis sources: {required}")

    if "Energy-only & unavailable" not in summary_path.with_name("thesis_evidence_table.tex").read_text(encoding="utf-8"):
        problems.append("exported thesis table must keep Energy-only unavailable")

    for note in summary.get("evidence_notes", []):
        note_path = Path(note)
        if not note_path.exists():
            alt_path = notes_dir / note_path.name
            if not alt_path.exists():
                problems.append(f"summary references missing evidence note: {note}")

    energy_status = summary.get("energy_status", {})
    if energy_status.get("true_boltzmann_available") is not False:
        problems.append("summary must keep true_boltzmann_available=false for current evidence")
    if energy_status.get("status") != "full_logits_required":
        problems.append("summary must record Energy status as full_logits_required")

    headline = summary.get("headline_metrics", {})
    with_corpus = headline.get("learned_fusion_with_corpus_auroc")
    without_corpus = headline.get("learned_fusion_without_corpus_auroc")
    if not isinstance(with_corpus, (int, float)) or not isinstance(without_corpus, (int, float)):
        problems.append("summary missing learned-fusion AUROC values")
    elif with_corpus >= without_corpus:
        problems.append("current summary no longer supports the underperformance caveat")

    if "semantic_energy_single_cluster.md" not in thesis or "quco_vs_probe.md" not in thesis:
        problems.append("thesis must name both evidence note files used for citation guardrails")

    return problems
