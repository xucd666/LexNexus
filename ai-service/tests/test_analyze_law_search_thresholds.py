import json
from hashlib import sha256

import pytest
from scripts.analyze_law_search_thresholds import (
    ScoredCase,
    ThresholdOutcome,
    compare_thresholds,
    format_table,
    load_scored_cases,
)


def test_thresholds_distinguish_acceptance_from_correct_retrieval():
    cases = [
        ScoredCase("low-correct", "in_scope", 0.51, 1),
        ScoredCase("boundary-ranked-second", "in_scope", 0.60, 2),
        ScoredCase("high-wrong", "in_scope", 0.80, None),
        ScoredCase("unrelated", "unrelated", 0.60, None),
        ScoredCase("outside", "out_of_scope", 0.62, None),
        ScoredCase("outside-empty", "out_of_scope", None, None),
    ]

    outcome = compare_thresholds(cases, [0.60, 0.61, 0.80])

    # 等于阈值的候选会放行。高分漏召回即使放行，也不能计作命中。
    assert outcome == [
        ThresholdOutcome(None, 0, 1, 1, 1, 2),
        ThresholdOutcome(0.60, 1, 1, 1, 0, 1),
        ThresholdOutcome(0.61, 2, 0, 1, 0, 0),
        ThresholdOutcome(0.80, 2, 0, 0, 0, 0),
    ]
    assert "| 0.6 | 1/3 (33.3%) | 1/1 (100.0%) | 1/2 (50.0%) | 0/3 (0.0%) | 1/3 (33.3%) |" in (
        format_table(cases, outcome)
    )


@pytest.mark.parametrize("threshold", [-1.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="阈值"):
        compare_thresholds([], [threshold])


@pytest.fixture
def recorded():
    label = {
        "case_id": "in-01",
        "question": "已标注问题",
        "category": "in_scope",
        "expected_article_nos": ["第五百七十七条"],
        "annotation_reason": "标注理由",
    }
    dataset = json.dumps([label], ensure_ascii=False).encode()
    corpus = b"corpus bytes used for a fingerprint check"
    report = {
        "dataset_sha256": sha256(dataset).hexdigest(),
        "corpus": {"source_sha256": sha256(corpus).hexdigest()},
        "top_k": 3,
        "cases": [
            label
            | {
                "top_3": [
                    {"article_no": "第五百八十二条", "score": 0.51},
                    {"article_no": "第五百七十七条", "score": 0.83},
                ],
                "first_relevant_rank": 2,
            }
        ],
    }
    return report, dataset, corpus


def test_reuses_report_order_instead_of_replacing_top_one_with_maximum(recorded):
    cases = load_scored_cases(*recorded)

    assert cases == [ScoredCase("in-01", "in_scope", 0.51, 2)]
    assert compare_thresholds(cases, [0.52])[-1].in_scope_rejected == 1


def test_rejects_report_for_a_different_dataset(recorded):
    report, dataset, corpus = recorded
    with pytest.raises(ValueError, match="题集摘要"):
        load_scored_cases(report, dataset + b" ", corpus)


def test_rejects_report_for_a_different_corpus(recorded):
    report, dataset, corpus = recorded
    with pytest.raises(ValueError, match="原文摘要"):
        load_scored_cases(report, dataset, corpus + b" ")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"category": "unrelated"}, "原始标注"),
        ({"expected_article_nos": ["第五百八十二条"]}, "原始标注"),
        ({"first_relevant_rank": 1}, "命中排名"),
        ({"top_3": [{"article_no": "第五百七十七条", "score": float("nan")}]}, "有限数值"),
    ],
)
def test_rejects_mismatched_or_invalid_record_fields(recorded, mutation, message):
    report, dataset, corpus = recorded
    report["cases"][0].update(mutation)

    with pytest.raises(ValueError, match=message):
        load_scored_cases(report, dataset, corpus)


def test_rejects_missing_report_rows(recorded):
    report, dataset, corpus = recorded
    report["cases"].clear()

    with pytest.raises(ValueError, match="缺少题集"):
        load_scored_cases(report, dataset, corpus)
