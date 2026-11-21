"""用保存的评测结果模拟 Top-1 阈值，保持原有检索排序。"""

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from scripts.eval_law_search import CORPUS, DEFAULT_CASES, TOP_K, Category

DEFAULT_THRESHOLDS = (0.40, 0.50, 0.52, 0.55, 0.60, 0.65, 0.68, 0.69, 0.75)


@dataclass(frozen=True)
class ScoredCase:
    case_id: str
    category: Category
    top_score: float | None
    first_relevant_rank: int | None


@dataclass(frozen=True)
class ThresholdOutcome:
    threshold: float | None
    in_scope_rejected: int
    unrelated_accepted: int
    out_of_scope_accepted: int
    kept_hit_at_1: int
    kept_hit_at_3: int


def load_scored_cases(report: dict, dataset: bytes, corpus: bytes) -> list[ScoredCase]:
    """核对报告来源和逐题标注，从保存的原始排名提取分数。"""
    if report["dataset_sha256"] != sha256(dataset).hexdigest():
        raise ValueError("报告中的题集摘要与当前固定题集不一致")
    if report["corpus"]["source_sha256"] != sha256(corpus).hexdigest() or report["top_k"] != TOP_K:
        raise ValueError("报告的原文摘要或 Top-K 与评测基线不一致")

    labels = {row["case_id"]: row for row in json.loads(dataset)}
    cases = []
    for row in report["cases"]:
        case_id = row["case_id"]
        label = labels.pop(case_id, None)
        if label is None or any(
            row[field] != label[field]
            for field in ("question", "category", "expected_article_nos", "annotation_reason")
        ):
            raise ValueError(f"{case_id} 缺少匹配的原始标注或重复出现")
        hits = row["top_3"]
        if not isinstance(hits, list) or len(hits) > TOP_K:
            raise ValueError(f"{case_id} 的命中列表无效")
        for hit in hits:
            article = hit["article_no"]
            if not isinstance(article, str) or not article.strip():
                raise ValueError(f"{case_id} 的命中缺少法条号")
            score = hit["score"]
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError(f"{case_id} 的相似度须为有限数值")

        # 重新对照标注计算命中位置。阈值只决定是否保留这组候选，原排名始终不变。
        rank = next(
            (i for i, hit in enumerate(hits, start=1) if hit["article_no"] in label["expected_article_nos"]),
            None,
        )
        if row["first_relevant_rank"] != rank:
            raise ValueError(f"{case_id} 的命中排名与原始候选不一致")
        cases.append(ScoredCase(case_id, label["category"], hits[0]["score"] if hits else None, rank))
    if labels:
        raise ValueError(f"报告缺少题集中的 {len(labels)} 条题目")
    return cases


def compare_thresholds(cases: list[ScoredCase], thresholds: Sequence[float]) -> list[ThresholdOutcome]:
    """模拟保留整组候选的开关。边界分数按原精度使用大于等于比较。"""
    if any(not -1 <= threshold <= 1 for threshold in thresholds):
        raise ValueError("候选余弦阈值须为 -1 至 1 之间的有限值")
    in_scope_total = sum(case.category == "in_scope" for case in cases)
    outcomes = []
    for threshold in (None, *thresholds):
        # 空返回没有候选。库内题可能因分数不足被拒绝，也可能放行一组未命中法条的候选。
        accepted = [
            case
            for case in cases
            if case.top_score is not None and (threshold is None or case.top_score >= threshold)
        ]
        accepted_in_scope = [case for case in accepted if case.category == "in_scope"]
        outcomes.append(
            ThresholdOutcome(
                threshold=threshold,
                in_scope_rejected=in_scope_total - len(accepted_in_scope),
                unrelated_accepted=sum(case.category == "unrelated" for case in accepted),
                out_of_scope_accepted=sum(case.category == "out_of_scope" for case in accepted),
                kept_hit_at_1=sum(case.first_relevant_rank == 1 for case in accepted_in_scope),
                kept_hit_at_3=sum(
                    case.first_relevant_rank is not None and case.first_relevant_rank <= TOP_K
                    for case in accepted_in_scope
                ),
            )
        )
    return outcomes


def format_table(cases: list[ScoredCase], outcomes: list[ThresholdOutcome]) -> str:
    """按各组原始题数给出误判率，方便在同一分母下比较。"""
    totals = Counter(case.category for case in cases)
    lines = [
        "| 阈值 | 库内拒绝 | 完全无关接受 | 库外法律接受 | 保留 Hit@1 | 保留 Hit@3 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in outcomes:
        cells = ["无" if row.threshold is None else str(row.threshold)]
        for count, category in (
            (row.in_scope_rejected, "in_scope"),
            (row.unrelated_accepted, "unrelated"),
            (row.out_of_scope_accepted, "out_of_scope"),
            (row.kept_hit_at_1, "in_scope"),
            (row.kept_hit_at_3, "in_scope"),
        ):
            total = totals[category]
            cells.append(f"{count}/{total} ({count / total:.1%})" if total else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线比较固定 Top-1 阈值")
    parser.add_argument("--report", type=Path, required=True, help="E-04-A 真实评测输出 JSON")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS, help="候选阈值")
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.expanduser().read_text(encoding="utf-8-sig"))
        cases = load_scored_cases(report, DEFAULT_CASES.read_bytes(), CORPUS.with_suffix(".txt").read_bytes())
        outcomes = compare_thresholds(cases, args.thresholds)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(format_table(cases, outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
