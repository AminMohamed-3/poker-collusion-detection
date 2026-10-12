import csv
import json

import polars as pl
import pytest

from src.validation import SUBMISSION_COLUMNS, main, validate_submission


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.fixture
def dataset(tmp_path):
    write_csv(tmp_path / "evaluation_pairs.csv", ["pair_id", "player_1", "player_2"], [
        ["001", "01", "02"], ["002", "02", "03"]])
    write_csv(tmp_path / "hands.csv", ["hand_id", "table_id", "phase"], [
        ["0010", "t1", "evaluation"], ["h2", "t1", "evaluation"],
        ["dev", "t1", "development"], ["not_shared", "t2", "evaluation"]])
    write_csv(tmp_path / "seats.csv", ["hand_id", "player_id", "unused"], [
        ["0010", "01", "x"], ["0010", "02", "x"], ["0010", "03", "x"],
        ["h2", "01", "x"], ["h2", "02", "x"], ["h2", "03", "x"],
        ["dev", "01", "x"], ["dev", "02", "x"],
        ["not_shared", "01", "x"], ["not_shared", "03", "x"]])
    rows = [
        ["001", "0.85", "soft_play", "0010", "h2", "NO_EVIDENCE", "NO_EVIDENCE", "NO_EVIDENCE"],
        ["002", "0", "none", "NO_EVIDENCE", "NO_EVIDENCE", "NO_EVIDENCE", "NO_EVIDENCE", "NO_EVIDENCE"],
    ]
    write_csv(tmp_path / "sample_submission.csv", SUBMISSION_COLUMNS, rows)
    return tmp_path, rows


def validate(dataset, rows=None, columns=SUBMISSION_COLUMNS):
    directory, default_rows = dataset
    write_csv(directory / "submission.csv", columns, default_rows if rows is None else rows)
    return validate_submission(
        directory / "submission.csv", directory / "evaluation_pairs.csv",
        directory / "hands.csv", directory / "seats.csv", directory / "sample_submission.csv")


def codes(report):
    return {issue.code for issue in report.errors}


def test_valid_preserves_opaque_leading_zero_ids_and_repeated_sentinel(dataset):
    report = validate(dataset)
    assert report.valid, report.to_dict()
    assert report.stats == {"submission_rows": 2, "evaluation_pairs": 2,
                            "evidence_cells": 2, "distinct_evidence_hands": 2}


def test_all_no_evidence_is_valid_without_scanning_gameplay(dataset):
    directory, rows = dataset
    rows[0][3:] = ["NO_EVIDENCE"] * 5
    (directory / "hands.csv").unlink()
    (directory / "seats.csv").unlink()
    assert validate(dataset).valid


def test_rejects_development_evidence_even_with_both_players_present(dataset):
    _, rows = dataset
    rows[0][3] = "dev"
    report = validate(dataset)
    assert "evidence_phase" in codes(report)
    assert "evidence_membership" not in codes(report)


def test_rejects_evaluation_hand_missing_one_pair_player(dataset):
    _, rows = dataset
    rows[0][3] = "not_shared"
    report = validate(dataset)
    assert "evidence_membership" in codes(report)
    assert "evidence_phase" not in codes(report)


def test_rejects_unknown_evidence_hand(dataset):
    _, rows = dataset
    rows[0][3] = "does_not_exist"
    assert "evidence_phase" in codes(validate(dataset))


def test_rejects_actual_hand_repeated_within_pair(dataset):
    _, rows = dataset
    rows[0][4] = "0010"
    assert "duplicate_evidence" in codes(validate(dataset))


def test_same_shared_hand_can_be_evidence_for_different_pairs(dataset):
    _, rows = dataset
    rows[1][3] = "0010"
    assert validate(dataset).valid


def test_rejects_missing_pair(dataset):
    assert "missing_pair" in codes(validate(dataset, dataset[1][:1]))


def test_rejects_duplicate_and_missing_pair_even_when_row_count_matches(dataset):
    _, rows = dataset
    report = validate(dataset, [rows[0], rows[0]])
    assert {"duplicate_pair", "missing_pair"}.issubset(codes(report))


def test_rejects_unexpected_pair(dataset):
    _, rows = dataset
    rows[1][0] = "unknown"
    assert {"missing_pair", "unexpected_pair"}.issubset(codes(validate(dataset)))


@pytest.mark.parametrize("value", ["NaN", "inf", "-inf", "1.0001", "-0.01", "not-a-number"])
def test_rejects_nonfinite_out_of_range_or_nonnumeric_scores(dataset, value):
    _, rows = dataset
    rows[0][1] = value
    report = validate(dataset)
    assert "risk_score" in codes(report)
    json.dumps(report.to_dict(), allow_nan=False)  # diagnostics must remain valid JSON


@pytest.mark.parametrize("value", ["0", "1"])
def test_accepts_closed_score_boundaries(dataset, value):
    _, rows = dataset
    rows[0][1] = value
    assert validate(dataset).valid


@pytest.mark.parametrize("column", range(len(SUBMISSION_COLUMNS)))
@pytest.mark.parametrize("empty", ["", "   "])
def test_rejects_any_empty_cell(dataset, column, empty):
    _, rows = dataset
    rows[0][column] = empty
    assert "empty_cell" in codes(validate(dataset))


def test_rejects_unknown_behavior(dataset):
    _, rows = dataset
    rows[0][2] = "collusion"
    assert "behavior" in codes(validate(dataset))


def test_rejects_wrong_column_order_and_extra_columns(dataset):
    _, rows = dataset
    swapped = [row[1:2] + row[0:1] + row[2:] for row in rows]
    columns = SUBMISSION_COLUMNS[1:2] + SUBMISSION_COLUMNS[0:1] + SUBMISSION_COLUMNS[2:]
    assert "schema" in codes(validate(dataset, swapped, columns))
    assert "schema" in codes(validate(dataset, [row + ["x"] for row in rows],
                                      SUBMISSION_COLUMNS + ["extra"]))


def test_parquet_sources_and_cli_report(dataset, capsys):
    directory, _ = dataset
    assert validate(dataset).valid
    for name in ("evaluation_pairs", "hands", "seats"):
        pl.read_csv(directory / f"{name}.csv", infer_schema=False).write_parquet(directory / f"{name}.parquet")
    report_path = directory / "private-report.json"
    result = main([
        str(directory / "submission.csv"), "--data-dir", str(directory),
        "--evaluation-pairs", str(directory / "evaluation_pairs.parquet"),
        "--hands", str(directory / "hands.parquet"), "--seats", str(directory / "seats.parquet"),
        "--report", str(report_path),
    ])
    assert result == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    assert json.loads(report_path.read_text())["valid"]


def test_cli_returns_nonzero_for_invalid_submission(dataset, capsys):
    directory, rows = dataset
    rows[0][3] = "dev"
    validate(dataset)
    assert main([str(directory / "submission.csv"), "--data-dir", str(directory)]) == 1
    assert not json.loads(capsys.readouterr().out)["valid"]


def test_missing_input_is_reported_not_a_traceback(tmp_path):
    report = validate_submission(*(tmp_path / name for name in ["sub.csv", "pairs.csv", "hands.csv", "seats.csv"]))
    assert "read_error" in codes(report)
