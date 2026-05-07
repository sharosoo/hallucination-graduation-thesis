"""Architecture validation logic for the experiments package."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


REQUIRED_DIRECTORIES = ("domain", "ports", "adapters", "application", "scripts")

REQUIRED_MODULES = {
    "domain": ("__init__.py", "labels.py", "records.py", "features.py", "metrics.py", "manifests.py"),
    "ports": (
        "__init__.py",
        "dataset_loader.py",
        "corpus_stats.py",
        "feature_extractor.py",
        "fusion_strategy.py",
        "evaluator.py",
        "artifact_store.py",
    ),
    "adapters": ("__init__.py",),
    "application": ("__init__.py", "architecture_validation.py"),
    "scripts": ("__init__.py", "validate_architecture.py"),
}

REQUIRED_FROZEN_DATACLASSES = {
    "QuestionExample",
    "ModelResponse",
    "CorrectnessJudgment",
    "SemanticEntropyResult",
    "EnergyResult",
    "CorpusStats",
    "FeatureVector",
    "ExperimentManifest",
    "MetricResult",
}

REQUIRED_PORTS = {
    "DatasetLoaderPort",
    "CorpusStatsPort",
    "FeatureExtractorPort",
    "FusionStrategyPort",
    "EvaluatorPort",
    "ArtifactStorePort",
}

EXPECTED_TYPE_LABELS = [
    "NORMAL",
    "HIGH_DIVERSITY",
    "LOW_DIVERSITY",
    "AMBIGUOUS_INCORRECT",
]

SCRIPT_LOGIC_NAME_HINTS = {
    "train_model",
    "fit_model",
    "compute_features",
    "run_fusion",
    "semantic_entropy",
    "score_batch",
}

SCRIPT_LOGIC_CALL_HINTS = {"fit", "predict", "score_batch"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to the experiments package root",
    )
    parser.add_argument(
        "--scripts-path",
        default=None,
        help="Optional override path for scripts-only guard validation",
    )
    return parser.parse_args(argv[1:])


def read_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def is_dataclass_decorator(decorator: ast.expr) -> bool:
    if isinstance(decorator, ast.Name):
        return decorator.id == "dataclass"
    if isinstance(decorator, ast.Call):
        if isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass":
            return True
    return False


def is_frozen_dataclass(class_def: ast.ClassDef) -> bool:
    for decorator in class_def.decorator_list:
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass":
                for keyword in decorator.keywords:
                    if keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant):
                        return bool(keyword.value.value)
        if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
            return False
    return False


def class_fields_are_annotated(class_def: ast.ClassDef) -> bool:
    for statement in class_def.body:
        if isinstance(statement, ast.Assign):
            return False
    return True


def collect_classes(package_root: Path) -> dict[str, ast.ClassDef]:
    classes: dict[str, ast.ClassDef] = {}
    for directory in ("domain", "ports"):
        for module_path in (package_root / directory).glob("*.py"):
            tree = read_module(module_path)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes[node.name] = node
    return classes


def validate_structure(package_root: Path) -> list[str]:
    problems: list[str] = []
    if package_root.name != "experiments":
        problems.append(f"expected experiments package root, got: {package_root}")

    init_file = package_root / "__init__.py"
    if not init_file.exists():
        problems.append(f"missing package file: {init_file}")

    for directory in REQUIRED_DIRECTORIES:
        path = package_root / directory
        if not path.is_dir():
            problems.append(f"missing required directory: {path}")
            continue
        for module_name in REQUIRED_MODULES[directory]:
            module_path = path / module_name
            if not module_path.exists():
                problems.append(f"missing required module: {module_path}")
    return problems


def validate_domain_and_ports(package_root: Path) -> list[str]:
    problems: list[str] = []
    classes = collect_classes(package_root)

    for class_name in REQUIRED_FROZEN_DATACLASSES:
        class_def = classes.get(class_name)
        if class_def is None:
            problems.append(f"missing domain dataclass: {class_name}")
            continue
        if not is_frozen_dataclass(class_def):
            problems.append(f"domain dataclass must be frozen: {class_name}")
        if not class_fields_are_annotated(class_def):
            problems.append(f"domain dataclass has untyped field assignment: {class_name}")

    type_label = classes.get("TypeLabel")
    if type_label is None:
        problems.append("missing TypeLabel enum")
    else:
        values: list[str] = []
        for statement in type_label.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name) and isinstance(statement.value, ast.Constant):
                    values.append(str(statement.value.value))
        if values != EXPECTED_TYPE_LABELS:
            problems.append(f"TypeLabel values mismatch: expected {EXPECTED_TYPE_LABELS}, got {values}")

    feature_vector = classes.get("FeatureVector")
    if feature_vector is None:
        problems.append("missing FeatureVector dataclass")
    else:
        field_names = [
            statement.target.id
            for statement in feature_vector.body
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
        ]
        if "label" not in field_names:
            problems.append("FeatureVector must expose row-level operational label field")
        if "se_bin" not in field_names:
            problems.append("FeatureVector must expose SE-bin analysis metadata field")
        if "provenance" not in field_names:
            problems.append("FeatureVector must expose feature provenance field")

    corpus_stats = classes.get("CorpusStats")
    if corpus_stats is None:
        problems.append("missing CorpusStats dataclass")
    else:
        field_names = [
            statement.target.id
            for statement in corpus_stats.body
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
        ]
        for required_field in (
            "entity_frequencies",
            "entity_pair_cooccurrence",
            "low_frequency_entity_flags",
            "coverage_score",
            "corpus_source",
            "corpus_provenance",
        ):
            if required_field not in field_names:
                problems.append(f"CorpusStats missing field: {required_field}")

    energy_result = classes.get("EnergyResult")
    if energy_result is None:
        problems.append("missing EnergyResult dataclass")
    else:
        field_names = [
            statement.target.id
            for statement in energy_result.body
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
        ]
        for required_field in ("energy_kind", "selected_token_logit"):
            if required_field not in field_names:
                problems.append(f"EnergyResult missing field: {required_field}")

    for class_name in REQUIRED_PORTS:
        class_def = classes.get(class_name)
        if class_def is None:
            problems.append(f"missing abstract port: {class_name}")
            continue
        has_abstract_method = False
        for statement in class_def.body:
            if isinstance(statement, ast.FunctionDef):
                for decorator in statement.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                        has_abstract_method = True
        if not has_abstract_method:
            problems.append(f"port must define abstract methods: {class_name}")
    return problems


def validate_scripts(scripts_path: Path) -> list[str]:
    problems: list[str] = []
    if not scripts_path.is_dir():
        return [f"scripts path is not a directory: {scripts_path}"]

    for script_path in scripts_path.glob("*.py"):
        tree = read_module(script_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if any(is_dataclass_decorator(decorator) for decorator in node.decorator_list):
                    problems.append(
                        f"script layer must not define domain dataclasses: {script_path.name}:{node.name}"
                    )
            if isinstance(node, ast.FunctionDef) and node.name in SCRIPT_LOGIC_NAME_HINTS:
                problems.append(
                    f"script layer appears to contain business/model logic: {script_path.name}:{node.name}"
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in SCRIPT_LOGIC_CALL_HINTS:
                    problems.append(
                        f"script layer appears to execute model/business logic directly: {script_path.name}:{node.func.attr}()"
                    )
    return problems


def validate(package_root: Path, scripts_path: Path) -> list[str]:
    problems: list[str] = []
    problems.extend(validate_structure(package_root))
    if any(message.startswith("missing required") or message.startswith("missing package") for message in problems):
        return problems
    problems.extend(validate_domain_and_ports(package_root))
    problems.extend(validate_scripts(scripts_path))
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv
    args = parse_args(argv)
    package_root = Path(args.root).resolve()
    scripts_path = Path(args.scripts_path).resolve() if args.scripts_path else package_root / "scripts"

    problems = validate(package_root, scripts_path)
    if problems:
        print("Architecture validation failed.", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    print(f"Architecture validation passed for {package_root}")
    print(f"Scripts guard validated: {scripts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
