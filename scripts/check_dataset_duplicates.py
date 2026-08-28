"""Audit exact, visual, and textual duplicates in the training dataset.

This script never removes or modifies dataset documents. It writes only the
requested duplicate report unless --no-write is supplied.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_expansion import (  # noqa: E402
    CLASS_NAMES,
    METADATA_FIELDS,
    SUPPORTED_EXTENSIONS,
    FingerprintIndex,
    atomic_write_csv,
    build_fingerprint_record,
    project_path,
    read_csv_rows,
    relative_project_path,
    validate_training_path,
)


METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
SOURCE_TRACKING_PATH = PROJECT_ROOT / "data" / "dataset_sources_extra.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "dataset_expansion" / "duplicate_report.csv"
REPORT_FIELDS = (
    "candidate_path",
    "label",
    "source",
    "decision",
    "reason",
    "similar_to",
    "similarity_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only duplicate audit for data/raw.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        help="Optional additional candidate directory to compare without adding files.",
    )
    parser.add_argument("--image-similarity-threshold", type=float, default=0.94)
    parser.add_argument("--text-similarity-threshold", type=float, default=0.95)
    parser.add_argument("--no-write", action="store_true", help="Print results without writing the report.")
    return parser.parse_args()


def source_metadata() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    sources: dict[str, str] = {}
    parent_by_id: dict[str, str] = {}
    path_by_id: dict[str, str] = {}
    if not SOURCE_TRACKING_PATH.exists():
        return sources, parent_by_id, path_by_id
    rows = read_csv_rows(SOURCE_TRACKING_PATH)
    for row in rows:
        document_id = row.get("id", "")
        sources[document_id] = row.get("source_name", "dataset_sources_extra")
        path_by_id[document_id] = row.get("raw_path", "")
        if row.get("is_augmented", "").casefold() in {"true", "1", "yes"}:
            parent_by_id[document_id] = row.get("original_id", "")
    return sources, parent_by_id, path_by_id


def split_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        path = SPLITS_DIR / f"{split_name}.csv"
        if not path.exists():
            continue
        for row in read_csv_rows(path, ("id",)):
            document_id = row["id"]
            if document_id in mapping:
                mapping[document_id] = f"{mapping[document_id]}+{split_name}"
            else:
                mapping[document_id] = split_name
    return mapping


def report_row(
    *,
    candidate_path: str,
    label: str,
    source: str,
    decision: str,
    reason: str,
    similar_to: str = "",
    similarity_score: float | str = "",
) -> dict[str, str]:
    return {
        "candidate_path": candidate_path,
        "label": label,
        "source": source,
        "decision": decision,
        "reason": reason,
        "similar_to": similar_to,
        "similarity_score": (
            f"{similarity_score:.6f}" if isinstance(similarity_score, float) else str(similarity_score)
        ),
    }


def infer_candidate_label(path: Path, root: Path) -> str:
    parts = {part.casefold() for part in path.relative_to(root).parts}
    labels = [label for label in CLASS_NAMES if label in parts or path.stem.casefold().startswith(label)]
    return labels[0] if len(labels) == 1 else "unknown"


def preload_holdout_guards(index: FingerprintIndex, report: list[dict[str, str]]) -> None:
    for holdout_name in ("external_test", "external_robus_test"):
        holdout_root = PROJECT_ROOT / "data" / holdout_name
        if not holdout_root.exists():
            continue
        for label in CLASS_NAMES:
            label_root = holdout_root / label
            if not label_root.exists():
                continue
            for path in sorted(label_root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                try:
                    index.add(
                        build_fingerprint_record(
                            key=f"holdout:{path}",
                            raw_path=path,
                            label=label,
                            source=f"{holdout_name}_guard",
                        )
                    )
                except Exception as error:
                    report.append(
                        report_row(
                            candidate_path=str(path),
                            label=label,
                            source=f"{holdout_name}_guard",
                            decision="error",
                            reason=f"holdout_fingerprint_error: {error}",
                        )
                    )


def main() -> None:
    args = parse_args()
    metadata_rows = read_csv_rows(METADATA_PATH, METADATA_FIELDS)
    sources, parent_by_id, path_by_id = source_metadata()
    split_by_id = split_map()
    index = FingerprintIndex(
        image_threshold=args.image_similarity_threshold,
        text_threshold=args.text_similarity_threshold,
    )
    report: list[dict[str, str]] = []
    decisions = Counter()
    reasons = Counter()
    preload_holdout_guards(index, report)
    tracked_raw_paths: set[Path] = set()

    print(f"Auditing {len(metadata_rows)} metadata documents...")
    for row in metadata_rows:
        document_id = row["id"]
        raw_path = project_path(PROJECT_ROOT, row["raw_path"])
        image_path = project_path(PROJECT_ROOT, row["image_path"])
        text_path = project_path(PROJECT_ROOT, row["text_path"])
        tracked_raw_paths.add(raw_path.resolve())
        validate_training_path(PROJECT_ROOT, raw_path)
        if not raw_path.exists():
            item = report_row(
                candidate_path=str(raw_path),
                label=row["label"],
                source=sources.get(document_id, "existing_dataset"),
                decision="error",
                reason="missing_raw_file",
            )
            report.append(item)
            decisions[item["decision"]] += 1
            reasons[item["reason"]] += 1
            continue

        try:
            fingerprint = build_fingerprint_record(
                key=document_id,
                raw_path=raw_path,
                image_path=image_path,
                text_path=text_path,
                label=row["label"],
                source=sources.get(document_id, "existing_dataset"),
                group_id=parent_by_id.get(document_id, document_id),
            )
            ignored = set()
            parent_id = parent_by_id.get(document_id)
            if parent_id:
                ignored.add(parent_id)
                parent_path = path_by_id.get(parent_id)
                if parent_path:
                    ignored.add(parent_path)
            match = index.find_duplicate(fingerprint, ignore=ignored)
            if match:
                item = report_row(
                    candidate_path=row["raw_path"],
                    label=row["label"],
                    source=fingerprint.source,
                    decision="duplicate",
                    reason=match.reason,
                    similar_to=match.similar_to,
                    similarity_score=match.similarity_score,
                )
            else:
                reason = "controlled_augmentation" if parent_id else "unique"
                item = report_row(
                    candidate_path=row["raw_path"],
                    label=row["label"],
                    source=fingerprint.source,
                    decision="keep",
                    reason=reason,
                    similar_to=parent_id or "",
                )
            index.add(fingerprint)
        except Exception as error:
            item = report_row(
                candidate_path=row["raw_path"],
                label=row["label"],
                source=sources.get(document_id, "existing_dataset"),
                decision="error",
                reason=f"fingerprint_error: {error}",
            )
        report.append(item)
        decisions[item["decision"]] += 1
        reasons[item["reason"]] += 1

    raw_root = PROJECT_ROOT / "data" / "raw"
    for label in CLASS_NAMES:
        label_root = raw_root / label
        if not label_root.exists():
            continue
        for path in sorted(label_root.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() not in SUPPORTED_EXTENSIONS
                or path.resolve() in tracked_raw_paths
            ):
                continue
            try:
                fingerprint = build_fingerprint_record(
                    key=f"untracked:{path}",
                    raw_path=path,
                    label=label,
                    source="untracked_data_raw",
                )
                match = index.find_duplicate(fingerprint)
                if match:
                    item = report_row(
                        candidate_path=relative_project_path(PROJECT_ROOT, path),
                        label=label,
                        source="untracked_data_raw",
                        decision="duplicate",
                        reason=match.reason,
                        similar_to=match.similar_to,
                        similarity_score=match.similarity_score,
                    )
                else:
                    item = report_row(
                        candidate_path=relative_project_path(PROJECT_ROOT, path),
                        label=label,
                        source="untracked_data_raw",
                        decision="untracked",
                        reason="raw_file_missing_from_metadata",
                    )
                index.add(fingerprint)
            except Exception as error:
                item = report_row(
                    candidate_path=str(path),
                    label=label,
                    source="untracked_data_raw",
                    decision="error",
                    reason=f"fingerprint_error: {error}",
                )
            report.append(item)
            decisions[item["decision"]] += 1
            reasons[item["reason"]] += 1

    for child, parent in parent_by_id.items():
        child_split = split_by_id.get(child)
        parent_split = split_by_id.get(parent)
        if child_split and parent_split and child_split != parent_split:
            row = next((item for item in metadata_rows if item["id"] == child), None)
            item = report_row(
                candidate_path=(row or {}).get("raw_path", child),
                label=(row or {}).get("label", "unknown"),
                source=sources.get(child, "dataset_sources_extra"),
                decision="split_leak",
                reason=f"augmentation_in_{child_split}_parent_in_{parent_split}",
                similar_to=parent,
                similarity_score=1.0,
            )
            report.append(item)
            decisions[item["decision"]] += 1
            reasons[item["reason"]] += 1

    if args.candidate_root:
        candidate_root = args.candidate_root.resolve()
        validate_training_path(PROJECT_ROOT, candidate_root)
        if not candidate_root.exists():
            raise FileNotFoundError(f"Candidate root does not exist: {candidate_root}")
        for path in sorted(candidate_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            label = infer_candidate_label(path, candidate_root)
            try:
                fingerprint = build_fingerprint_record(
                    key=f"candidate:{path}",
                    raw_path=path,
                    label=label,
                    source="candidate_root",
                )
                match = index.find_duplicate(fingerprint)
                if match:
                    item = report_row(
                        candidate_path=str(path),
                        label=label,
                        source="candidate_root",
                        decision="duplicate",
                        reason=match.reason,
                        similar_to=match.similar_to,
                        similarity_score=match.similarity_score,
                    )
                else:
                    item = report_row(
                        candidate_path=str(path),
                        label=label,
                        source="candidate_root",
                        decision="keep",
                        reason="unique_candidate",
                    )
                index.add(fingerprint)
            except Exception as error:
                item = report_row(
                    candidate_path=str(path),
                    label=label,
                    source="candidate_root",
                    decision="error",
                    reason=f"fingerprint_error: {error}",
                )
            report.append(item)
            decisions[item["decision"]] += 1
            reasons[item["reason"]] += 1

    if not args.no_write:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        atomic_write_csv(output, report, REPORT_FIELDS)
        print(f"Report: {output}")

    print("\nDUPLICATE AUDIT SUMMARY")
    print("=" * 64)
    for decision, count in sorted(decisions.items()):
        print(f"{decision}: {count}")
    print(f"Augmentation split leaks: {decisions['split_leak']}")
    duplicate_reasons = {reason: count for reason, count in reasons.items() if "duplicate" in reason or "identical" in reason}
    for reason, count in sorted(duplicate_reasons.items()):
        print(f"{reason}: {count}")


if __name__ == "__main__":
    main()
