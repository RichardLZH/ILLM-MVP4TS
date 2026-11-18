"""Local knowledge fragments and dense cosine retrieval for Stage II."""
from pathlib import Path
import re

import numpy as np


def chunks(directory, size=512, overlap=32):
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < chunk size.")
    root = Path(directory)
    fragments = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            # Prefer a nearby paragraph/sentence boundary within the character budget.
            if end < len(text):
                for delimiter in ("\n\n", ". ", "。", "\n"):
                    boundary = text.rfind(delimiter, start + size // 2, end)
                    if boundary >= 0:
                        end = boundary + len(delimiter)
                        break
            content = text[start:end].strip()
            if content:
                fragments.append({"id": f"K{len(fragments) + 1}", "source": path.relative_to(root).as_posix(),
                                  "start": start, "end": end, "text": content})
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    if not fragments:
        raise ValueError("Knowledge directory contains no nonempty .txt/.md documents.")
    return fragments


def normalize(vectors):
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite embedding matrix.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero embedding cannot define cosine similarity.")
    return values / norms


def topk_union(query_vectors, document_vectors, fragments, k=5):
    if k < 1 or not len(fragments):
        raise ValueError("Positive retrieval depth and nonempty corpus required.")
    q, d = normalize(query_vectors), normalize(document_vectors)
    if len(d) != len(fragments) or q.shape[1] != d.shape[1]:
        raise ValueError("Embedding/corpus shapes differ.")
    scores = q @ d.T
    rankings, selected = [], {}
    for row in scores:
        indices = np.argsort(-row, kind="stable")[:min(k, len(row))]
        ranking = [{"id": fragments[i]["id"], "score": float(row[i])} for i in indices]
        rankings.append(ranking)
        for i in indices:
            selected[int(i)] = max(float(row[i]), selected.get(int(i), -float("inf")))
    evidence = [{**fragments[i], "score": score} for i, score in
                sorted(selected.items(), key=lambda item: (-item[1], item[0]))]
    return evidence, rankings


class DenseEncoder:
    def __init__(self, name, revision=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, revision=revision)

    def encode(self, texts):
        return normalize(self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False))


def keyword_evidence(fragments, terms, limit=20):
    """DA-Agent keyword tool; separate from Stage II dense retrieval."""
    scores = []
    for i, fragment in enumerate(fragments):
        text = fragment["text"].casefold()
        score = sum(len(re.findall(re.escape(term.casefold()), text)) for term in terms if term)
        if score:
            scores.append((score, i))
    return [fragments[i] for _, i in sorted(scores, key=lambda pair: (-pair[0], pair[1]))[:limit]]
