"""Tests for pure presentation helpers used by the feedback Reports app."""

from __future__ import annotations

import csv
import io
import json
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone

import pytest

from tools.feedback_reports import ui


def _business_summary(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "answer_count": 5,
        "unknown_answer_count": 0,
    }
    row.update(overrides)
    return row


def _ingestion_summary(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "recorded_request_count": 4,
        "client_rejection_count": 0,
        "service_failure_count": 0,
        "event_id_conflict_request_count": 0,
        "duplicate_event_count": 0,
        "idempotent_duplicate_event_count": 0,
        "unclassified_duplicate_event_count": 0,
        "conflicting_event_count": 0,
        "conflict_audit_event_count": 0,
        "event_id_reuse_count": 0,
        "classified_event_count": 5,
        "known_event_result_count": 5,
        "end_to_end_coverage_available": False,
    }
    row.update(overrides)
    return row


def _registry_summary(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "registered_release_count": 1,
        "registered_question_count": 60,
        "registered_choice_count": 180,
        "registry_available": True,
        "raw_event_count": 8,
        "authoritative_event_count": 8,
        "excluded_event_count": 0,
        "missing_release_event_count": 0,
        "unknown_release_event_count": 0,
        "question_not_in_release_event_count": 0,
        "raw_answer_count": 5,
        "authoritative_answer_count": 5,
        "unresolved_answer_count": 0,
        "invalid_selected_letter_answer_count": 0,
        "selected_candidate_mismatch_answer_count": 0,
        "unmatched_comment_count": 0,
        "unmatched_proposal_count": 0,
        "client_context_mismatch_event_count": 0,
        "client_correctness_mismatch_answer_count": 0,
        "registry_match_rate": 1.0,
        "answer_resolution_rate": 1.0,
    }
    row.update(overrides)
    return row


def test_data_quality_classifies_registry_exclusions_and_disagreements() -> None:
    signals = ui.build_data_quality_signals(
        _business_summary(unknown_answer_count=1),
        registry_summary=_registry_summary(
            raw_event_count=12,
            authoritative_event_count=8,
            excluded_event_count=4,
            missing_release_event_count=1,
            unknown_release_event_count=1,
            question_not_in_release_event_count=2,
            raw_answer_count=7,
            authoritative_answer_count=5,
            unresolved_answer_count=2,
            invalid_selected_letter_answer_count=1,
            selected_candidate_mismatch_answer_count=1,
            unmatched_comment_count=1,
            unmatched_proposal_count=1,
            client_context_mismatch_event_count=2,
            client_correctness_mismatch_answer_count=1,
            registry_match_rate=0.6667,
            answer_resolution_rate=0.7143,
        ),
        ingestion_summary=_ingestion_summary(),
    )

    by_code = {signal.code: signal for signal in signals}
    assert by_code["unresolved_question_registry_events"].severity == "warning"
    assert "1 missing release" in by_code["unresolved_question_registry_events"].body
    assert by_code["unresolved_registered_answers"].severity == "error"
    assert by_code["client_registry_context_mismatch"].severity == "warning"
    assert by_code["client_registry_correctness_mismatch"].severity == "error"
    assert by_code["unmatched_feedback_footprints"].severity == "warning"
    assert "question_registry_unavailable" not in by_code


def test_data_quality_signals_classify_observed_feedback_problems() -> None:
    signals = ui.build_data_quality_signals(
        _business_summary(unknown_answer_count=2),
        ingestion_summary=_ingestion_summary(
            client_rejection_count=1,
            service_failure_count=1,
            duplicate_event_count=1,
            idempotent_duplicate_event_count=1,
            event_id_reuse_count=1,
        ),
    )

    by_code = {signal.code: signal for signal in signals}
    assert by_code["unknown_answer_correctness"].severity == "warning"
    assert "2 of 5" in by_code["unknown_answer_correctness"].body
    assert "40.0%" in by_code["unknown_answer_correctness"].body
    assert by_code["recorded_client_rejections"].severity == "warning"
    assert "25.0%" in by_code["recorded_client_rejections"].body
    assert by_code["recorded_service_failures"].severity == "error"
    assert "25.0%" in by_code["recorded_service_failures"].body
    assert by_code["verified_idempotent_duplicates"].severity == "info"
    assert (
        "matched the stored logical content"
        in by_code["verified_idempotent_duplicates"].body
    )
    assert by_code["ingestion_coverage_incomplete"].severity == "warning"
    assert not any(signal.severity == "success" for signal in signals)


def test_data_quality_separates_conflicts_legacy_and_ordinary_rejections() -> None:
    signals = ui.build_data_quality_signals(
        _business_summary(),
        ingestion_summary=_ingestion_summary(
            client_rejection_count=2,
            event_id_conflict_request_count=1,
            duplicate_event_count=3,
            idempotent_duplicate_event_count=2,
            unclassified_duplicate_event_count=1,
            conflicting_event_count=1,
            conflict_audit_event_count=1,
            event_id_reuse_count=4,
        ),
    )

    by_code = {signal.code: signal for signal in signals}
    conflict = by_code["conflicting_event_ids"]
    assert conflict.severity == "error"
    assert "1 of 3 classified event-ID reuse" in conflict.body
    assert "33.3%" in conflict.body
    assert "1 correlated private-audit row" in conflict.body
    assert "stored first write was preserved" in conflict.body
    assert "before event storage" not in conflict.body

    ordinary = by_code["recorded_client_rejections"]
    assert ordinary.severity == "warning"
    assert "1 of 4" in ordinary.body
    assert "25.0%" in ordinary.body
    assert "before event storage" in ordinary.body
    assert "reported separately" in ordinary.body

    legacy = by_code["unclassified_duplicate_event_ids"]
    assert legacy.severity == "warning"
    assert "neither verified idempotent retries nor confirmed conflicts" in legacy.body
    assert by_code["verified_idempotent_duplicates"].severity == "info"


def test_data_quality_signals_handle_zero_denominators_without_false_health() -> None:
    signals = ui.build_data_quality_signals(
        _business_summary(answer_count=0),
        ingestion_summary=_ingestion_summary(
            recorded_request_count=0,
            classified_event_count=0,
            known_event_result_count=0,
        ),
    )

    assert [signal.code for signal in signals] == [
        "no_recorded_ingestion_outcomes",
        "ingestion_coverage_incomplete",
    ]
    assert all(signal.severity != "success" for signal in signals)
    assert "N/A" in signals[0].body


def test_data_quality_signals_report_unavailable_ingestion_without_green_status() -> (
    None
):
    signals = ui.build_data_quality_signals(
        _business_summary(),
        ingestion_unavailable_reason="Content filters exclude ingestion observability.",
    )

    assert signals == (
        ui.DataQualitySignal(
            code="question_registry_unavailable",
            severity="warning",
            title="Question registry unavailable",
            body=(
                "No server registry evidence is available for this snapshot. "
                "Correctness and release/family/question-type dimensions must not "
                "fall back to the uploaded client payload."
            ),
        ),
        ui.DataQualitySignal(
            code="ingestion_unavailable",
            severity="warning",
            title="Ingestion quality unavailable",
            body=(
                "Content filters exclude ingestion observability. No healthy ingestion "
                "conclusion can be drawn from the business event counts alone."
            ),
        ),
    )
    assert all(signal.severity != "success" for signal in signals)


def test_data_quality_signals_use_safe_default_for_missing_ingestion() -> None:
    signals = ui.build_data_quality_signals(_business_summary())

    assert [signal.code for signal in signals] == [
        "question_registry_unavailable",
        "ingestion_unavailable",
    ]
    assert "No ingestion observability snapshot is available" in signals[1].body


def test_data_quality_signals_require_boolean_registry_evidence() -> None:
    with pytest.raises(TypeError, match="question_registry_available"):
        ui.build_data_quality_signals(
            _business_summary(),
            question_registry_available=1,  # type: ignore[arg-type]
        )


def test_data_quality_signals_never_emit_green_health_for_non_atomic_snapshots() -> (
    None
):
    complete = ui.build_data_quality_signals(
        _business_summary(),
        ingestion_summary=_ingestion_summary(
            end_to_end_coverage_available=True,
        ),
        question_registry_available=True,
    )
    incomplete = ui.build_data_quality_signals(
        _business_summary(),
        ingestion_summary=_ingestion_summary(),
    )

    assert [signal.code for signal in complete] == ["no_observed_quality_issues"]
    assert complete[0].severity == "info"
    assert "not an overall health guarantee" in complete[0].body
    assert [signal.code for signal in incomplete] == [
        "question_registry_unavailable",
        "ingestion_coverage_incomplete",
    ]
    assert all(signal.severity != "success" for signal in incomplete)


def test_data_quality_signals_do_not_classify_release_identifiers() -> None:
    baseline = ui.build_data_quality_signals(
        _business_summary(),
        ingestion_summary=_ingestion_summary(),
    )
    with_unrecognized_release = ui.build_data_quality_signals(
        {**_business_summary(), "release_id": "release_not_in_any_local_manifest"},
        ingestion_summary=_ingestion_summary(),
    )

    assert with_unrecognized_release == baseline
    assert all(
        "release_not_in_any_local_manifest" not in signal.body for signal in baseline
    )


@pytest.mark.parametrize(
    ("business", "ingestion", "reason", "message"),
    [
        ({"answer_count": 1}, None, None, "missing unknown_answer_count"),
        (_business_summary(answer_count=True), None, None, "non-negative integer"),
        (
            _business_summary(answer_count=0, unknown_answer_count=1),
            None,
            None,
            "cannot exceed",
        ),
        (
            _business_summary(),
            _ingestion_summary(recorded_request_count=0, client_rejection_count=1),
            None,
            "cannot exceed recorded requests",
        ),
        (
            _business_summary(),
            _ingestion_summary(
                known_event_result_count=0,
                duplicate_event_count=1,
                idempotent_duplicate_event_count=1,
                event_id_reuse_count=1,
            ),
            None,
            "cannot exceed its denominator",
        ),
        (
            _business_summary(),
            _ingestion_summary(event_id_conflict_request_count=1),
            None,
            "cannot exceed client rejections",
        ),
        (
            _business_summary(),
            _ingestion_summary(duplicate_event_count=1),
            None,
            "classifications must add up",
        ),
        (
            _business_summary(),
            _ingestion_summary(event_id_reuse_count=1),
            None,
            "reuse counts must add up",
        ),
        (
            _business_summary(),
            _ingestion_summary(
                duplicate_event_count=1,
                idempotent_duplicate_event_count=1,
                event_id_reuse_count=1,
                classified_event_count=0,
            ),
            None,
            "cannot exceed its denominator",
        ),
        (
            _business_summary(),
            _ingestion_summary(conflicting_event_count=1, event_id_reuse_count=1),
            None,
            "inconsistent with conflict requests",
        ),
        (
            _business_summary(),
            _ingestion_summary(conflict_audit_event_count=1),
            None,
            "audit does not match classified conflicts",
        ),
        (
            _business_summary(),
            _ingestion_summary(end_to_end_coverage_available=0),
            None,
            "must be a boolean",
        ),
        (_business_summary(), None, " reason with spaces ", "trimmed non-empty"),
    ],
)
def test_data_quality_signals_reject_invalid_summary_facts(
    business: object,
    ingestion: object,
    reason: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ui.build_data_quality_signals(
            business,  # type: ignore[arg-type]
            ingestion_summary=ingestion,  # type: ignore[arg-type]
            ingestion_unavailable_reason=reason,  # type: ignore[arg-type]
        )


def test_data_quality_signals_reject_ambiguous_ingestion_inputs() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ui.build_data_quality_signals(
            _business_summary(),
            ingestion_summary=_ingestion_summary(),
            ingestion_unavailable_reason="Ingestion query failed.",
        )


def test_build_global_filters_trims_scalars_and_uses_half_open_utc_dates() -> None:
    filters = ui.build_global_filters(
        release_id=" release_123 ",
        family="bigram_language_model",
        question_type=" architecture_only ",
        question_id="q_123",
        session_id=" anon_session_123 ",
        attempt_id="attempt_123",
        start_date=date(2026, 7, 11),
        end_date=date(2026, 7, 12),
    )

    assert filters == {
        "release_id": "release_123",
        "family": "bigram_language_model",
        "question_type": "architecture_only",
        "question_id": "q_123",
        "session_id": "anon_session_123",
        "attempt_id": "attempt_123",
        "from": "2026-07-11T00:00:00Z",
        "to": "2026-07-13T00:00:00Z",
    }
    assert all(isinstance(value, str) for value in filters.values())
    assert ui.build_global_filters(release_id=" ") == {}


def test_build_global_filters_allows_one_day_and_open_date_bounds() -> None:
    assert ui.build_global_filters(
        start_date=date(2026, 7, 12),
        end_date=date(2026, 7, 12),
    ) == {
        "from": "2026-07-12T00:00:00Z",
        "to": "2026-07-13T00:00:00Z",
    }
    assert ui.build_global_filters(start_date=date(2026, 7, 12)) == {
        "from": "2026-07-12T00:00:00Z"
    }
    assert ui.build_global_filters(end_date=date(2026, 7, 12)) == {
        "to": "2026-07-13T00:00:00Z"
    }


def test_build_global_filters_rejects_reversed_or_ambiguous_dates() -> None:
    with pytest.raises(ValueError, match="must not be after"):
        ui.build_global_filters(
            start_date=date(2026, 7, 13),
            end_date=date(2026, 7, 12),
        )
    with pytest.raises(TypeError, match="must be a date"):
        ui.build_global_filters(start_date=datetime(2026, 7, 12, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="exclusive end"):
        ui.build_global_filters(end_date=date.max)


def test_percentage_and_optional_kpi_keep_none_distinct_from_zero() -> None:
    assert ui.format_percentage(None) == "—"
    assert ui.format_percentage(0) == "0.0%"
    assert ui.format_percentage(0.625) == "62.5%"
    assert ui.format_percentage(1, decimals=0) == "100%"
    assert ui.format_optional_kpi(None) == "—"
    assert ui.format_optional_kpi(0) == "0"
    assert ui.format_optional_kpi(12_345) == "12,345"


def test_table_rows_are_flattened_without_mutating_input() -> None:
    rows = [
        {
            "session_id": "anon_1",
            "release_ids": ["release_1", "release_2"],
            "families": [],
            "question_types": None,
            "accuracy": 0.75,
            "proposal_usage_rate": 0.5,
            "nested": {"keep": [1, 2]},
        }
    ]
    original = deepcopy(rows)

    formatted = ui.format_report_table_rows(
        "feedback_report_sessions",
        rows,
    )

    assert formatted == [
        {
            "session_id": "anon_1",
            "release_ids": "release_1, release_2",
            "families": "—",
            "question_types": "—",
            "accuracy": "75.0%",
            "proposal_usage_rate": "50.0%",
            "nested": {"keep": [1, 2]},
        }
    ]
    formatted[0]["nested"]["keep"].append(3)
    assert rows == original


def test_format_report_table_rows_preserves_missing_accuracy() -> None:
    assert ui.format_report_table_rows(
        "feedback_report_summary",
        [{"accuracy": None, "answer_count": 0}],
    ) == [{"accuracy": "—", "answer_count": 0}]

    assert ui.format_report_table_rows(
        "feedback_report_ingestion_summary",
        [
            {
                "request_failure_rate": 0.25,
                "duplicate_event_rate": None,
                "event_id_reuse_rate": 0.5,
                "classified_conflicting_event_rate": 1.0,
            }
        ],
    ) == [
        {
            "request_failure_rate": "25.0%",
            "duplicate_event_rate": "—",
            "event_id_reuse_rate": "50.0%",
            "classified_conflicting_event_rate": "100.0%",
        }
    ]


def test_surprise_table_keeps_missing_distinct_and_formats_all_ratios() -> None:
    row = {
        "rating_count": 0,
        "surprised_count": 0,
        "not_surprised_count": 0,
        "rating_coverage_rate": 0.0,
        "observed_surprise_rate": None,
        "posterior_mean": 0.5,
    }
    assert ui.format_report_table_rows("feedback_report_surprise_questions", [row]) == [
        {
            "rating_count": 0,
            "surprised_count": 0,
            "not_surprised_count": 0,
            "rating_coverage_rate": "0.0%",
            "observed_surprise_rate": "—",
            "posterior_mean": "50.0%",
        }
    ]


def test_proposal_json_display_is_canonical_bounded_and_plain_text() -> None:
    long_value = "x" * 1_200
    rows = [
        {
            "event_id": "evt_proposal",
            "setting_json": (
                '{ "z": 1, "markup": "<script>alert(\\"not executed\\")</script>" }'
            ),
            "inherited_from_json": json.dumps({"notes": long_value}),
        }
    ]
    original = deepcopy(rows)

    formatted = ui.format_report_table_rows("feedback_report_proposals", rows)

    assert formatted[0]["setting_json"] == (
        '{"markup":"<script>alert(\\"not executed\\")</script>","z":1}'
    )
    assert isinstance(formatted[0]["inherited_from_json"], str)
    assert len(formatted[0]["inherited_from_json"]) == 1_000
    assert formatted[0]["inherited_from_json"].endswith("…")
    assert rows == original


@pytest.mark.parametrize(
    "value",
    [
        {"not": "text"},
        "[]",
        '{"duplicate":1,"duplicate":2}',
        '{"constant":NaN}',
    ],
)
def test_proposal_json_display_rejects_non_object_or_non_strict_text(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="JSON display"):
        ui.format_report_table_rows(
            "feedback_report_proposals",
            [{"setting_json": value}],
        )


def test_csv_download_is_utf8_bom_and_delegates_formula_protection() -> None:
    rows = [
        {
            "event_id": "evt_1",
            "occurred_at": "2026-07-12T08:00:00Z",
            "received_at": "2026-07-12T08:00:01Z",
            "session_id": "anon_1",
            "attempt_id": "attempt_1",
            "question_id": "q_1",
            "question_version": "qv1_1",
            "release_id": "release_1",
            "family": "bigram_language_model",
            "question_type": "architecture_only",
            "comment_text": '=HYPERLINK("https://example.invalid")',
            "category": "suggestion",
        }
    ]

    encoded = ui.report_csv_download_bytes("feedback_report_comments", rows)

    assert encoded.startswith(b"\xef\xbb\xbf")
    decoded = encoded.decode("utf-8-sig")
    parsed = list(csv.DictReader(io.StringIO(decoded)))
    assert parsed[0]["comment_text"].startswith("'=HYPERLINK")
    assert parsed[0]["question_id"] == "q_1"


def test_proposal_csv_keeps_full_json_and_protects_formula_strings() -> None:
    long_value = "x" * 1_200
    row = {
        "event_id": "evt_proposal",
        "occurred_at": "2026-07-12T08:00:00Z",
        "received_at": "2026-07-12T08:00:01Z",
        "session_id": "anon_1",
        "attempt_id": "attempt_1",
        "question_id": "q_1",
        "question_version": f"qv1_{'b' * 64}",
        "release_id": f"release_{'a' * 64}",
        "family": "bigram_language_model",
        "dataset_id": "bg_1",
        "question_type": "architecture_only",
        "setting_status": "proposed",
        "label": '=HYPERLINK("https://example.invalid")',
        "setting_json": ('{"notes":"' + long_value + '","optimizer":"Adam"}'),
        "inherited_from_json": '{"choice":"A"}',
        "n_seeds": 3,
        "base_seed": 42,
        "error_type": None,
    }

    encoded = ui.report_csv_download_bytes("feedback_report_proposals", [row])

    parsed = list(csv.DictReader(io.StringIO(encoded.decode("utf-8-sig"))))
    assert parsed[0]["label"].startswith("'=HYPERLINK")
    assert parsed[0]["setting_json"] == (
        '{"notes":"' + long_value + '","optimizer":"Adam"}'
    )
    assert parsed[0]["inherited_from_json"] == '{"choice":"A"}'


def test_surprise_csv_uses_exact_safe_columns_without_answer_or_gt_fields() -> None:
    row = {
        "question_id": "q_1",
        "question_version": f"qv1_{'b' * 64}",
        "release_id": f"release_{'a' * 64}",
        "family": "bigram_lm",
        "dataset_id": "bg_1",
        "question_type": "mixed",
        "answered_attempt_count": 3,
        "rating_count": 2,
        "surprised_count": 1,
        "not_surprised_count": 1,
        "rating_coverage_rate": 0.6667,
        "observed_surprise_rate": 0.5,
        "posterior_mean": 0.5,
        "first_rating_at": "2026-07-12T08:00:00Z",
        "last_rating_at": "2026-07-12T08:01:00Z",
    }

    encoded = ui.report_csv_download_bytes("feedback_report_surprise_questions", [row])
    reader = csv.DictReader(io.StringIO(encoded.decode("utf-8-sig")))
    parsed = list(reader)
    assert reader.fieldnames == list(row)
    assert parsed[0]["observed_surprise_rate"] == "0.5"
    assert not ({"correct_letter", "ground_truth", "prior_components"} & set(row))


def test_csv_filename_is_deterministic_and_normalized_to_utc() -> None:
    china_time = datetime(
        2026,
        7,
        12,
        16,
        30,
        45,
        999_999,
        tzinfo=timezone(timedelta(hours=8)),
    )

    assert (
        ui.report_csv_filename(
            "feedback_report_questions",
            generated_at=china_time,
        )
        == "architectureiq-report-questions-20260712T083045Z.csv"
    )
    assert (
        ui.report_csv_filename(
            "feedback_report_answers",
            generated_at=china_time,
        )
        == "architectureiq-report-answers-20260712T083045Z.csv"
    )
    assert (
        ui.report_csv_filename(
            "feedback_report_proposals",
            generated_at=china_time,
        )
        == "architectureiq-report-proposals-20260712T083045Z.csv"
    )
    assert (
        ui.report_csv_filename(
            "feedback_report_surprise_questions",
            generated_at=china_time,
        )
        == "architectureiq-report-surprise-questions-20260712T083045Z.csv"
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        ui.report_csv_filename(
            "feedback_report_questions",
            generated_at=datetime(2026, 7, 12),
        )


@pytest.mark.parametrize(
    ("total", "row_count", "expected"),
    [(0, 0, False), (2, 2, False), (3, 2, True)],
)
def test_report_page_is_truncated(
    total: int,
    row_count: int,
    expected: bool,
) -> None:
    assert (
        ui.report_page_is_truncated(
            total=total,
            row_count=row_count,
        )
        is expected
    )


def test_report_page_is_truncated_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ui.report_page_is_truncated(total=1, row_count=2)
    with pytest.raises(ValueError, match="non-negative integer"):
        ui.report_page_is_truncated(total=True, row_count=0)


def test_ui_helpers_accept_only_allowlisted_report_view_keys() -> None:
    with pytest.raises(ValueError, match="unsupported report view"):
        ui.format_report_table_rows("feedback_session_summary", [])
    with pytest.raises(ValueError, match="unsupported report view"):
        ui.report_csv_filename(
            "feedback_session_summary",
            generated_at=datetime.now(timezone.utc),
        )
