"""Validate private poker competition submissions without loading full gameplay tables.

IDs are opaque strings used only for matching. Large hand/seat tables are scanned
lazily with streaming execution; only submitted evidence is materialized.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

EVIDENCE_COLUMNS = [f"evidence_hand_{index}" for index in range(1, 6)]
SUBMISSION_COLUMNS = ["pair_id", "risk_score", "predicted_behavior", *EVIDENCE_COLUMNS]
BEHAVIORS = {
    "none", "directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"
}
NO_EVIDENCE = "NO_EVIDENCE"


@dataclass
class ValidationIssue:
    code: str
    message: str
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": [asdict(error) for error in self.errors],
                "stats": self.stats}

    def add(self, code: str, message: str, rows: pl.DataFrame | None = None) -> None:
        examples = [] if rows is None else rows.head(5).to_dicts()
        self.errors.append(ValidationIssue(code, message, examples))


def _scan(path: str | Path, string_columns: list[str]) -> pl.LazyFrame:
    """Preserve CSV ID text, including leading zeros, before type inference."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pl.scan_parquet(path)
    else:
        frame = pl.scan_csv(
            path, schema_overrides={name: pl.String for name in string_columns},
            infer_schema_length=1000,
        )
    available = frame.collect_schema().names()
    return frame.with_columns(
        pl.col(name).cast(pl.String) for name in string_columns if name in available
    )


def _collect(frame: pl.LazyFrame) -> pl.DataFrame:
    return frame.collect(engine="streaming")


def _check_columns(
    frame: pl.LazyFrame, required: list[str], name: str, report: ValidationReport,
    *, exact: bool = False,
) -> bool:
    actual = frame.collect_schema().names()
    if (actual != required if exact else not set(required).issubset(actual)):
        report.add("schema", f"{name} columns must be "
                   f"{'exactly, in this order' if exact else 'inclusive of'} {required}; got {actual}.")
        return False
    return True


def validate_submission(
    submission_path: str | Path,
    evaluation_pairs_path: str | Path,
    hands_path: str | Path,
    seats_path: str | Path,
    sample_submission_path: str | Path | None = None,
    evaluation_phase: str = "evaluation",
) -> ValidationReport:
    """Return schema, score, pair-coverage and evidence-membership errors.

    Repeated NO_EVIDENCE is allowed. Any other hand ID can occur at most once per
    pair, must exist in the evaluation phase, and must contain both pair players.
    Source files may be CSV (including .csv.gz) or Parquet.
    """
    report = ValidationReport()
    try:
        _validate(
            submission_path, evaluation_pairs_path, hands_path, seats_path,
            sample_submission_path, evaluation_phase, report,
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        report.add("read_error", f"Unable to read or process an input file: {exc}")
    return report


def _validate(
    submission_path: str | Path,
    evaluation_pairs_path: str | Path,
    hands_path: str | Path,
    seats_path: str | Path,
    sample_submission_path: str | Path | None,
    evaluation_phase: str,
    report: ValidationReport,
) -> None:
    submitted = _scan(submission_path, SUBMISSION_COLUMNS)
    if not _check_columns(submitted, SUBMISSION_COLUMNS, "Submission", report, exact=True):
        return
    if sample_submission_path is not None:
        sample = _scan(sample_submission_path, SUBMISSION_COLUMNS)
        if not _check_columns(sample, SUBMISSION_COLUMNS, "Sample submission", report, exact=True):
            return
    pairs_lazy = _scan(evaluation_pairs_path, ["pair_id", "player_1", "player_2"])
    if not _check_columns(pairs_lazy, ["pair_id", "player_1", "player_2"], "Evaluation pairs", report):
        return
    submission = _collect(submitted)
    pairs = _collect(pairs_lazy.select("pair_id", "player_1", "player_2"))
    report.stats.update(submission_rows=submission.height, evaluation_pairs=pairs.height)

    empty = submission.filter(pl.any_horizontal(
        pl.col(name).is_null() | (pl.col(name).str.strip_chars() == "")
        for name in SUBMISSION_COLUMNS
    ))
    if empty.height:
        report.add("empty_cell", f"{empty.height} rows contain null, empty or whitespace-only cells.",
                   empty.select("pair_id"))
    invalid_pairs = pairs.filter(pl.any_horizontal(
        pl.col(name).is_null() | (pl.col(name).str.strip_chars() == "")
        for name in ("pair_id", "player_1", "player_2")
    ))
    duplicate_pairs = pairs.group_by("pair_id").len().filter(pl.col("len") != 1)
    if invalid_pairs.height or duplicate_pairs.height:
        report.add("source_pairs", "Evaluation-pair source has empty IDs or duplicate pair IDs.")
        return

    duplicates = submission.group_by("pair_id").len().filter(pl.col("len") != 1)
    if duplicates.height:
        report.add("duplicate_pair", "Submission repeats pair IDs.", duplicates)
    missing = pairs.select("pair_id").join(submission.select("pair_id"), on="pair_id", how="anti")
    unexpected = submission.select("pair_id").unique().join(
        pairs.select("pair_id"), on="pair_id", how="anti")
    if missing.height:
        report.add("missing_pair", f"Submission is missing {missing.height} evaluation pairs.", missing)
    if unexpected.height:
        report.add("unexpected_pair", "Submission contains IDs absent from evaluation pairs.", unexpected)

    score = pl.col("risk_score").cast(pl.Float64, strict=False)
    invalid_score = submission.filter(
        ~((score.is_finite()) & score.is_between(0, 1, closed="both")).fill_null(False)
    )
    if invalid_score.height:
        report.add("risk_score", "risk_score must be numeric, finite and within [0, 1].",
                   invalid_score.select("pair_id", "risk_score"))
    invalid_behavior = submission.filter(~pl.col("predicted_behavior").is_in(BEHAVIORS).fill_null(False))
    if invalid_behavior.height:
        report.add("behavior", f"predicted_behavior must belong to {sorted(BEHAVIORS)}.",
                   invalid_behavior.select("pair_id", "predicted_behavior"))

    evidence = submission.select("pair_id", *EVIDENCE_COLUMNS).unpivot(
        on=EVIDENCE_COLUMNS, index="pair_id", variable_name="evidence_column", value_name="hand_id"
    ).filter(pl.col("hand_id").is_not_null() & (pl.col("hand_id") != NO_EVIDENCE)
             & (pl.col("hand_id").str.strip_chars() != ""))
    duplicate_evidence = evidence.group_by("pair_id", "hand_id").len().filter(pl.col("len") > 1)
    if duplicate_evidence.height:
        report.add("duplicate_evidence", "Actual evidence hand IDs must be unique within each pair.",
                   duplicate_evidence)
    report.stats.update(evidence_cells=evidence.height,
                        distinct_evidence_hands=evidence.get_column("hand_id").n_unique())
    # Structural errors make further large-table work unnecessary/ambiguous.
    if report.errors or evidence.is_empty():
        return

    hands = _scan(hands_path, ["hand_id", "phase"])
    seats = _scan(seats_path, ["hand_id", "player_id"])
    if not _check_columns(hands, ["hand_id", "phase"], "Hands", report):
        return
    if not _check_columns(seats, ["hand_id", "player_id"], "Seats", report):
        return
    evidence_ids = evidence.select("hand_id").unique()
    matched_hands = _collect(hands.select("hand_id", "phase").join(
        evidence_ids.lazy(), on="hand_id", how="semi"))
    duplicate_hands = matched_hands.group_by("hand_id").len().filter(pl.col("len") > 1)
    if duplicate_hands.height:
        report.add("source_hands", "Hands source repeats an evidence hand ID.", duplicate_hands)
        return
    phase_checks = evidence.join(matched_hands, on="hand_id", how="left")
    invalid_phase = phase_checks.filter(~(pl.col("phase") == evaluation_phase).fill_null(False))
    if invalid_phase.height:
        report.add("evidence_phase", f"Evidence must reference existing hands in phase {evaluation_phase!r}.",
                   invalid_phase.select("pair_id", "hand_id", "phase"))

    joined = evidence.select("pair_id", "hand_id").join(pairs, on="pair_id", how="left")
    requirements = pl.concat([
        joined.select("pair_id", "hand_id", pl.col(player).alias("player_id"))
        for player in ("player_1", "player_2")
    ]).unique()
    keys = ["hand_id", "player_id"]
    # Filter the large seats source down to required keys before materializing.
    matched_seats = _collect(seats.select(keys).join(
        requirements.select(keys).unique().lazy(), on=keys, how="semi").unique())
    absent_seats = requirements.join(matched_seats, on=keys, how="anti")
    if absent_seats.height:
        report.add("evidence_membership", "Each evidence hand must seat both players in its evaluation pair.",
                   absent_seats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--evaluation-pairs", type=Path)
    parser.add_argument("--hands", type=Path)
    parser.add_argument("--seats", type=Path)
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--evaluation-phase", default="evaluation")
    parser.add_argument("--report", type=Path, help="Optional private JSON report path")
    args = parser.parse_args(argv)
    sample = args.sample_submission
    if sample is None and (args.data_dir / "sample_submission.csv").is_file():
        sample = args.data_dir / "sample_submission.csv"
    def gameplay_path(name: str) -> Path:
        parquet = args.data_dir / f"{name}.parquet"
        csv = args.data_dir / f"{name}.csv"
        return parquet if parquet.is_file() or not csv.is_file() else csv

    report = validate_submission(
        args.submission, args.evaluation_pairs or args.data_dir / "evaluation_pairs.csv",
        args.hands or gameplay_path("hands"), args.seats or gameplay_path("seats"),
        sample, args.evaluation_phase,
    )
    payload = json.dumps(report.to_dict(), indent=2, allow_nan=False)
    print(payload)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
