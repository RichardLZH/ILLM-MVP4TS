"""Equation (1): domain, relationship, trend and statistical views, in order."""
import numpy as np

from .common import validate_graph


def multi_view(window, labels, metadata, graph, target, horizon=1):
    x = np.asarray(window, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(labels) or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Prompt construction needs a finite history window [time,variables].")
    if target not in labels:
        raise ValueError("Target is not a process variable.")
    adjacency = validate_graph(graph, labels)
    definitions = metadata.get("variables", {})
    domain = str(metadata.get("description", "User-supplied industrial time series."))
    domain += f" Forecast {target} for the next {horizon} step(s) using {len(x)} historical observations."
    for name in labels:
        if name in definitions:
            domain += f" {name}: {definitions[name]}."
    relations = [f"{labels[cause]} -> {labels[effect]}"
                 for effect, cause in zip(*np.where(adjacency == 1))]
    relation = ("Domain-refined predictive dependencies: " if graph.get("refined") else
                "Initial statistical predictive dependencies (no domain refinement): ")
    relation += "; ".join(relations) if relations else "No edges retained."
    relation += " These dependencies alone do not prove physical causation."
    time = np.arange(len(x), dtype=float)
    time -= time.mean()
    slopes = (time[:, None] * (x - x.mean(axis=0))).sum(axis=0) / (time @ time)
    # Equal temporal segments are an explicit implementation choice, not future observations.
    segments = np.array_split(x, min(3, len(x)))
    trend, stats = [], []
    for i, name in enumerate(labels):
        levels = ",".join(f"{segment[:, i].mean():.4g}" for segment in segments)
        trend.append(f"{name}: slope/step={slopes[i]:.4g}, segment means=[{levels}]")
        values = x[:, i]
        stats.append(f"{name}: min={values.min():.4g}, max={values.max():.4g}, "
                     f"mean={values.mean():.4g}, std={values.std():.4g}, range={np.ptp(values):.4g}")
    return {"domain": domain, "relationship": relation,
            "trend": "; ".join(trend), "statistics": "; ".join(stats)}


def render(views):
    return "\n".join(f"{name.upper()} VIEW: {views[name]}"
                     for name in ("domain", "relationship", "trend", "statistics"))
