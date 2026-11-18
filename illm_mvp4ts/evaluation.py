"""Table 1 metrics. ACC/professionalism require explicit human annotations."""
import argparse
from pathlib import Path
import re

import numpy as np

from .common import read_json, write_json
from .knowledge import DenseEncoder, normalize


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant = set(relevant_ids)
    return next((1. / rank for rank, item in enumerate(ranked_ids, 1) if item in relevant), 0.)


def rouge_l(answer, reference, beta=1.):
    # English words and individual CJK characters; tokenizer choice is part of the protocol.
    tokenize = lambda text: re.findall(r"[\u4e00-\u9fff]|[\w]+", text.casefold())
    a, b = tokenize(answer), tokenize(reference)
    if not a or not b:
        return 0.
    row = [0] * (len(b) + 1)
    for x in a:
        previous = row
        row = [0]
        for j, y in enumerate(b, 1):
            row.append(previous[j - 1] + 1 if x == y else max(previous[j], row[-1]))
    precision, recall = row[-1] / len(a), row[-1] / len(b)
    denominator = recall + beta ** 2 * precision
    return (1 + beta ** 2) * precision * recall / denominator if denominator else 0.


def coverage(concerns, induced, threshold=.8):
    if not -1 <= threshold <= 1:
        raise ValueError("Cosine threshold must be in [-1,1].")
    return float(((normalize(concerns) @ normalize(induced).T).max(axis=1) >= threshold).mean())


def consistency(claims, evidence):
    # Average of each claim's best support; not a single document maximum for the whole report.
    return float((normalize(claims) @ normalize(evidence).T).max(axis=1).mean())


def evaluate_case(report, annotations, encoder, threshold=.8):
    explanation = report["explanation"]
    scores = {}
    if "relevant_ids" in annotations:
        rr = []
        for query_index, relevant in annotations["relevant_ids"].items():
            rr.append(reciprocal_rank([r["id"] for r in report["rankings"][int(query_index)]], relevant))
        if rr:
            scores["MRR_at_retrieval_depth"] = float(np.mean(rr))
    if "reference_answer" in annotations:
        scores["ROUGE_L"] = rouge_l(explanation["report"], annotations["reference_answer"])
    if annotations.get("historical_questions"):
        scores["Coverage"] = coverage(encoder.encode(annotations["historical_questions"]),
                                       encoder.encode(explanation["induced_questions"]), threshold)
    # Only evidence explicitly validated by an annotator is eligible for this metric.
    if "validated_evidence_ids" in annotations:
        valid_ids = set(annotations["validated_evidence_ids"])
        available = {item["id"] for item in report["evidence"]}
        if not valid_ids or not valid_ids <= available:
            raise ValueError("Validated evidence IDs must name a nonempty subset of retrieved evidence.")
        evidence = [x["text"] for x in report["evidence"] if x["id"] in valid_ids]
        scores["Consistency"] = consistency(encoder.encode([x["text"] for x in explanation["claims"]]),
                                             encoder.encode(evidence))
    if "correct" in annotations:
        if not isinstance(annotations["correct"], bool):
            raise ValueError("Expert correctness annotation must be boolean.")
        scores["ACC"] = float(annotations["correct"])
    if "professionalism_scores" in annotations:
        values = annotations["professionalism_scores"]
        if not values or any(not isinstance(v, (int, float)) or not 1 <= v <= 5 for v in values):
            raise ValueError("This release uses an explicit 1-5 expert professionalism rubric.")
        scores["Professionalism_1_to_5"] = float(np.mean(values))
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of {report, annotations}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-revision", default=None)
    parser.add_argument("--threshold", type=float, default=.8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file.")
    encoder = DenseEncoder(args.embedding_model, args.embedding_revision)
    cases = read_json(args.cases)
    if not isinstance(cases, list) or not cases:
        parser.error("Expected a nonempty list of annotated evaluation cases.")
    rows = [evaluate_case(read_json(args.cases.parent / case["report"]), case["annotations"], encoder, args.threshold)
            for case in cases]
    keys = sorted({key for row in rows for key in row})
    aggregate = {key: {"mean": float(np.mean([row[key] for row in rows if key in row])),
                       "annotated_cases": sum(key in row for row in rows)} for key in keys}
    write_json(args.output, {"per_case": rows, "aggregate": aggregate, "coverage_threshold": args.threshold,
                            "embedding_model": args.embedding_model, "embedding_revision": args.embedding_revision,
                            "note": "Missing annotations are omitted, never replaced by inferred scores."})


if __name__ == "__main__":
    main()
