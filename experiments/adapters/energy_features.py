"""Manifest-backed semantic energy extraction with rerun-required default behavior."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.adapters.corpus_features import infer_label, infer_sample_id, load_json, write_json
from experiments.domain import EnergyComputationKind, FeatureProvenance, FeatureRole

TRUE_BOLTZMANN_RERUN_INSTRUCTIONS = (
    "True Boltzmann semantic energy is unavailable from the current upstream artifacts. "
    "Re-run upstream generation/export with row-level raw per-token full logits or equivalent "
    "full-logit/logsumexp artifacts, then regenerate experiments/results/energy_features.parquet."
)
FORMULA_MANIFEST_REF = "experiments/literature/formula_notes.md"


@dataclass(frozen=True)
class EnergyArtifact:
    """Normalized upstream row-result artifact descriptor for energy extraction."""

    artifact_id: str
    artifact_kind: str
    dataset_id: str
    dataset_name: str
    split_id: str
    absolute_path: str
    sample_count: int
    timestamp: str | None
    has_logits: bool
    has_full_logits: bool


class EnergyFeatureUnavailableError(RuntimeError):
    """Raised when true Boltzmann energy cannot be computed from current artifacts."""

    def __init__(self, report: dict[str, Any]) -> None:
        message = (
            "full-logits-required: "
            + str(report.get("rerun_instructions", TRUE_BOLTZMANN_RERUN_INSTRUCTIONS))
            + " Missing capability on selected artifacts: "
            + repr(report.get("selected_artifacts", []))
        )
        super().__init__(message)
        self.report = report


class EnergyManifestCatalog:
    """Loads upstream artifacts and selects the preferred row-level source per dataset."""

    def __init__(self, manifest_dir: Path) -> None:
        self.manifest_dir = manifest_dir
        self.manifest_path = manifest_dir / "upstream_artifacts_manifest.json"
        payload = load_json(self.manifest_path)
        self.source_root = str(payload.get("source_root", ""))
        self.artifacts = tuple(self._normalize_artifact(entry) for entry in payload.get("artifacts", []))

    def _normalize_artifact(self, entry: dict[str, Any]) -> EnergyArtifact:
        dataset = entry.get("dataset") or {}
        metadata = entry.get("metadata") or {}
        prompt_model_metadata = entry.get("prompt_model_metadata") or {}
        raw_metadata_config = metadata.get("config")
        metadata_config = raw_metadata_config if isinstance(raw_metadata_config, dict) else {}
        split_id = (
            dataset.get("registry_split_id")
            or prompt_model_metadata.get("dataset_split")
            or metadata_config.get("dataset_split")
            or metadata_config.get("split")
            or "unknown_split"
        )
        availability = entry.get("availability") or {}
        return EnergyArtifact(
            artifact_id=str(entry.get("artifact_id", "unknown_artifact")),
            artifact_kind=str(entry.get("artifact_kind", "unknown_kind")),
            dataset_id=str(dataset.get("dataset_id", "unknown")),
            dataset_name=str(dataset.get("dataset_name", dataset.get("source_dataset_value", "unknown"))),
            split_id=str(split_id),
            absolute_path=str(entry.get("absolute_path", "")),
            sample_count=int(entry.get("sample_count", 0) or 0),
            timestamp=(metadata.get("timestamp") or prompt_model_metadata.get("timestamp")),
            has_logits=bool(availability.get("has_logits", False)),
            has_full_logits=bool(availability.get("has_full_logits", False)),
        )

    def preferred_row_artifacts(self) -> tuple[EnergyArtifact, ...]:
        grouped: dict[str, list[EnergyArtifact]] = defaultdict(list)
        for artifact in self.artifacts:
            if artifact.artifact_kind not in {"row_results", "row_results_with_corpus"}:
                continue
            if not artifact.absolute_path or artifact.dataset_id == "unknown":
                continue
            grouped[artifact.dataset_id].append(artifact)

        selected: list[EnergyArtifact] = []
        for _dataset_id, candidates in grouped.items():
            candidates.sort(
                key=lambda artifact: (
                    0 if artifact.has_full_logits else 1,
                    0 if artifact.has_logits else 1,
                    -artifact.sample_count,
                    artifact.timestamp or "",
                    artifact.artifact_id,
                )
            )
            selected.append(candidates[0])
        selected.sort(key=lambda artifact: (artifact.dataset_name, artifact.split_id, artifact.artifact_id))
        return tuple(selected)


class EnergyFeatureAdapter:
    """Builds true-Boltzmann rows when available, otherwise blocks by default."""

    def __init__(self, manifest_dir: Path) -> None:
        self.catalog = EnergyManifestCatalog(manifest_dir)

    def build_feature_rows(
        self,
        *,
        require_true_boltzmann: bool = False,
        allow_proxy_diagnostic: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        artifacts = self.catalog.preferred_row_artifacts()
        unavailable = [artifact for artifact in artifacts if not artifact.has_full_logits]
        rerun_report = self.build_rerun_required_report(
            artifacts=artifacts,
            unavailable=unavailable,
            allow_proxy_diagnostic=allow_proxy_diagnostic,
            require_true_boltzmann=require_true_boltzmann,
        )
        if unavailable and not allow_proxy_diagnostic:
            raise EnergyFeatureUnavailableError(rerun_report)
        if unavailable and require_true_boltzmann:
            raise EnergyFeatureUnavailableError(rerun_report)

        run_id = f"energy-features-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        rows: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        energy_kind_counts = {
            EnergyComputationKind.TRUE_BOLTZMANN.value: 0,
            EnergyComputationKind.PROXY_SELECTED_LOGIT.value: 0,
        }

        for artifact in artifacts:
            path = Path(artifact.absolute_path)
            if not path.exists():
                skipped.append({"artifact_id": artifact.artifact_id, "reason": "missing_artifact_path", "path": artifact.absolute_path})
                continue
            payload = load_json(path)
            samples = payload.get("samples")
            if not isinstance(samples, list):
                skipped.append({"artifact_id": artifact.artifact_id, "reason": "missing_samples_array", "path": artifact.absolute_path})
                continue
            for index, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    continue
                row = self._build_row(
                    run_id=run_id,
                    artifact=artifact,
                    sample=sample,
                    sample_index=index,
                    diagnostic_only=allow_proxy_diagnostic,
                )
                energy_kind_counts[row["features"]["energy_kind"]] += 1
                rows.append(row)

        report = {
            "run_id": run_id,
            "row_count": len(rows),
            "selected_artifact_count": len(artifacts),
            "selected_artifacts": [serialize_artifact(artifact) for artifact in artifacts],
            "energy_kind_counts": energy_kind_counts,
            "true_boltzmann_available": not unavailable,
            "rerun_required_for_true_boltzmann": bool(unavailable),
            "rerun_instructions": TRUE_BOLTZMANN_RERUN_INSTRUCTIONS,
            "diagnostic_only": bool(unavailable and allow_proxy_diagnostic),
            "not_for_thesis_claims": bool(unavailable),
            "not_for_validation": bool(unavailable),
            "artifact_status": "diagnostic_proxy_only" if unavailable else "true_boltzmann_ready",
            "skipped_artifacts": skipped,
        }
        return rows, report

    def build_rerun_required_report(
        self,
        *,
        artifacts: tuple[EnergyArtifact, ...] | None = None,
        unavailable: list[EnergyArtifact] | None = None,
        allow_proxy_diagnostic: bool = False,
        require_true_boltzmann: bool = False,
    ) -> dict[str, Any]:
        selected_artifacts = artifacts if artifacts is not None else self.catalog.preferred_row_artifacts()
        missing = unavailable if unavailable is not None else [artifact for artifact in selected_artifacts if not artifact.has_full_logits]
        return {
            "report_type": "energy_feature_rerun_required",
            "status": "full_logits_required",
            "full_logits_required": True,
            "rerun_required": True,
            "allow_proxy_diagnostic": allow_proxy_diagnostic,
            "require_true_boltzmann": require_true_boltzmann,
            "diagnostic_only": False,
            "not_for_thesis_claims": True,
            "not_for_validation": True,
            "true_boltzmann_available": not bool(missing),
            "selected_artifact_count": len(selected_artifacts),
            "selected_artifacts": [serialize_artifact(artifact) for artifact in selected_artifacts],
            "missing_full_logits_artifacts": [serialize_artifact(artifact) for artifact in missing],
            "rerun_instructions": TRUE_BOLTZMANN_RERUN_INSTRUCTIONS,
            "formula_manifest_ref": FORMULA_MANIFEST_REF,
            "dataset_manifest_ref": str(self.catalog.manifest_path),
            "message": (
                "Current upstream artifacts do not expose row-level full logits, so paper-faithful Ma/Boltzmann "
                "Semantic Energy cannot be computed or validated for thesis/fusion claims."
            ),
        }

    def _build_row(
        self,
        *,
        run_id: str,
        artifact: EnergyArtifact,
        sample: dict[str, Any],
        sample_index: int,
        diagnostic_only: bool,
    ) -> dict[str, Any]:
        semantic_entropy = float(sample.get("semantic_entropy", 0.0) or 0.0)
        label = infer_label(sample.get("is_hallucination", 0), semantic_entropy)
        sample_id = infer_sample_id(artifact.dataset_id, sample, sample_index)
        cluster_count = int(sample.get("num_clusters", 0) or 0)
        semantic_energy = float(sample.get("semantic_energy", 0.0) or 0.0)

        if artifact.has_full_logits:
            energy_kind = EnergyComputationKind.TRUE_BOLTZMANN
            semantic_energy_boltzmann = semantic_energy
            semantic_energy_proxy = None
            provenance_note = (
                "Selected artifact claims full logits availability, so the upstream semantic_energy scalar "
                "is treated as a true_boltzmann export from that source artifact."
            )
        else:
            energy_kind = EnergyComputationKind.PROXY_SELECTED_LOGIT
            semantic_energy_boltzmann = None
            semantic_energy_proxy = semantic_energy
            provenance_note = (
                "Diagnostic-only proxy row derived from upstream scalar semantic_energy. This is not paper-faithful "
                "Boltzmann Semantic Energy and must not be used for thesis claims or default experiment validation."
            )

        features = {
            "semantic_entropy": semantic_entropy,
            "cluster_count": cluster_count,
            "semantic_energy_boltzmann": semantic_energy_boltzmann,
            "semantic_energy_proxy": semantic_energy_proxy,
            "energy_kind": energy_kind.value,
            "logit_variance": None,
            "confidence_margin": None,
            "diagnostic_only": diagnostic_only,
            "not_for_thesis_claims": diagnostic_only,
            "not_for_validation": diagnostic_only,
        }
        energy_provenance = {
            "source_field": "sample.semantic_energy",
            "source_artifact_has_logits": artifact.has_logits,
            "source_artifact_has_full_logits": artifact.has_full_logits,
            "requires_rerun_for_true_boltzmann": not artifact.has_full_logits,
            "rerun_required_for_true_boltzmann": not artifact.has_full_logits,
            "diagnostic_only": diagnostic_only,
            "not_for_thesis_claims": diagnostic_only,
            "not_for_validation": diagnostic_only,
            "rerun_instructions": TRUE_BOLTZMANN_RERUN_INSTRUCTIONS,
            "note": provenance_note,
        }
        return {
            "run_id": run_id,
            "dataset": artifact.dataset_name,
            "dataset_id": artifact.dataset_id,
            "split_id": artifact.split_id,
            "sample_id": sample_id,
            "label": label.value,
            "artifact_id": artifact.artifact_id,
            "source_artifact_path": artifact.absolute_path,
            "diagnostic_only": diagnostic_only,
            "not_for_thesis_claims": diagnostic_only,
            "not_for_validation": diagnostic_only,
            "features": features,
            "energy_provenance": energy_provenance,
            "feature_provenance": [
                serialize_provenance(entry)
                for entry in build_provenance_entries(
                    source_artifact_path=artifact.absolute_path,
                    energy_kind=energy_kind,
                    has_full_logits=artifact.has_full_logits,
                    diagnostic_only=diagnostic_only,
                )
            ],
            "formula_manifest_ref": FORMULA_MANIFEST_REF,
            "dataset_manifest_ref": str(self.catalog.manifest_path),
        }


def build_provenance_entries(
    *,
    source_artifact_path: str,
    energy_kind: EnergyComputationKind,
    has_full_logits: bool,
    diagnostic_only: bool,
) -> tuple[FeatureProvenance, ...]:
    energy_source = "sample.semantic_energy"
    energy_role = FeatureRole.TRAINABLE if has_full_logits else FeatureRole.ANALYSIS_ONLY
    proxy_role = FeatureRole.ANALYSIS_ONLY if diagnostic_only else FeatureRole.TRAINABLE if not has_full_logits else FeatureRole.ANALYSIS_ONLY
    return (
        FeatureProvenance(
            feature_name="label",
            role=FeatureRole.LABEL_ONLY,
            source="sample.is_hallucination + semantic_entropy thresholds",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=True,
            trainable=False,
            note="Operational label only. Never use as a trainable feature.",
        ),
        FeatureProvenance(
            feature_name="semantic_entropy",
            role=FeatureRole.TRAINABLE,
            source="sample.semantic_entropy",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="cluster_count",
            role=FeatureRole.TRAINABLE,
            source="sample.num_clusters",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="energy_kind",
            role=FeatureRole.ANALYSIS_ONLY,
            source=f"artifact availability branch -> {energy_kind.value}",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=False,
            note="Branch marker only. This row must not overclaim true Boltzmann support.",
        ),
        FeatureProvenance(
            feature_name="semantic_energy_boltzmann",
            role=energy_role,
            source=energy_source if has_full_logits else "unavailable_without_full_logits",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=has_full_logits,
            note=(
                "True Boltzmann field is available only when row-level full logits exist."
                if has_full_logits
                else TRUE_BOLTZMANN_RERUN_INSTRUCTIONS
            ),
        ),
        FeatureProvenance(
            feature_name="semantic_energy_proxy",
            role=proxy_role,
            source=energy_source,
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=False if diagnostic_only else not has_full_logits,
            note=(
                "Diagnostic-only proxy-selected-logit export. Not for thesis claims or default validation."
                if diagnostic_only
                else "Unused when true Boltzmann is available."
                if has_full_logits
                else "Proxy-selected-logit style energy derived from the upstream scalar semantic_energy export."
            ),
        ),
        FeatureProvenance(
            feature_name="logit_variance",
            role=FeatureRole.ANALYSIS_ONLY,
            source="unavailable_without_row_level_logits",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=False,
            note="Left null unless future upstream artifacts expose row-level logits suitable for derivation.",
        ),
        FeatureProvenance(
            feature_name="confidence_margin",
            role=FeatureRole.ANALYSIS_ONLY,
            source="unavailable_without_full_logit_arrays",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=False,
            note="Left null unless future upstream artifacts expose full token logit arrays.",
        ),
    )


def serialize_provenance(entry: FeatureProvenance) -> dict[str, Any]:
    payload = asdict(entry)
    payload["role"] = entry.role.value
    return payload


def serialize_artifact(artifact: EnergyArtifact) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "dataset_id": artifact.dataset_id,
        "dataset_name": artifact.dataset_name,
        "split_id": artifact.split_id,
        "has_logits": artifact.has_logits,
        "has_full_logits": artifact.has_full_logits,
        "path": artifact.absolute_path,
    }


def rerun_report_path_for(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.rerun_required.json")


def write_rerun_required_report(out_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    materialized_path = rerun_report_path_for(out_path)
    payload_summary = {
        "requested_out_path": str(out_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "storage_kind": "rerun_required_report",
        "materialized_path": str(materialized_path),
        "artifact_status": "rerun_required",
        "report": report,
    }
    write_json(materialized_path, report)
    write_json(out_path.with_suffix(out_path.suffix + ".storage.json"), payload_summary)
    return payload_summary
