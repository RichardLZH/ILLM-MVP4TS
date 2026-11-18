"""Train/evaluate the paper-oriented model; persist exact inputs and configuration."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import file_hash, new_output, read_json, validate_graph, write_json
from .data import Windows, forecast_frame
from .model import ForecastModel, load_backbone
from .prompts import multi_view, render


def metrics(predictions, targets):
    p, y = np.asarray(predictions, dtype=float).ravel(), np.asarray(targets, dtype=float).ravel()
    if p.shape != y.shape or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Metrics require equally shaped finite predictions and targets.")
    error = p - y
    denominator = float(((y - y.mean()) ** 2).sum())
    return {"MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt((error ** 2).mean())),
            "R2": 1 - float((error ** 2).sum()) / denominator if denominator else None,
            "HR_percent": float((np.abs(error) < .04).mean() * 100),
            "HR_absolute_tolerance": .04, "scale": "input target physical units", "n": len(y)}


def tokenize(tokenizer, texts, limit, device):
    encoded = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    if encoded["input_ids"].shape[1] > limit:
        raise ValueError(f"Multi-view prompt exceeds {limit} tokens; increase --max-prompt-tokens within backbone capacity.")
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def validate_training_graph(graph, path, labels, frame, timestamp):
    validate_graph(graph, labels)
    if graph.get("data_sha256") != file_hash(path):
        raise ValueError("Graph must be generated from this data file; source hash differs.")
    if graph.get("train_fraction") != .7:
        raise ValueError("This training entry requires a graph generated on the first 70% only.")
    expected = str(frame[timestamp].iloc[int(len(frame) * .7) - 1])
    if graph.get("raw_train_last_timestamp") != expected:
        raise ValueError("Graph training boundary does not match forecasting split.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--timestamp", default="date")
    parser.add_argument("--target", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a trusted checkpoint from this entry; no training")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        options = read_json(preliminary.config)
        tunable = {"backbone", "revision", "seq_len", "horizon", "heads", "lora_rank", "learning_rate",
                   "batch_size", "epochs", "max_prompt_tokens", "seed"}
        if set(options) - tunable:
            parser.error("Unknown config options: " + str(sorted(set(options) - tunable)))
        parser.set_defaults(**options)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.learning_rate <= 0:
        parser.error("Batch size, epochs and learning rate must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame, labels, values = forecast_frame(args.data, args.timestamp)
    if args.target not in labels:
        parser.error("Target variable does not exist in data.")
    graph, metadata = read_json(args.graph), read_json(args.metadata)
    validate_training_graph(graph, args.data, labels, frame, args.timestamp)
    target_index = labels.index(args.target)
    datasets = {split: Windows(values, target_index, args.seq_len, args.horizon, split)
                for split in ("train", "val", "test")}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", drop_last=False)
               for split, dataset in datasets.items()}
    output = new_output(args.output)
    backbone, tokenizer = load_backbone(args.backbone, args.revision, args.local_files_only)
    model = ForecastModel(backbone, args.seq_len, len(labels), target_index, args.horizon,
                          args.heads, args.lora_rank).to(args.device)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    identity = {"labels": labels, "target": args.target, "data_sha256": file_hash(args.data),
                "graph_sha256": file_hash(args.graph), "metadata_sha256": file_hash(args.metadata),
                "seq_len": args.seq_len, "horizon": args.horizon, "heads": args.heads,
                "lora_rank": args.lora_rank, "backbone": args.backbone, "revision": args.revision,
                "resolved_backbone_commit": getattr(backbone.config, "_commit_hash", None)}
    write_json(output / "configuration.json", {**config, **identity, "adapter_modules": model.adapter_modules,
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "split_windows": {name: len(data) for name, data in datasets.items()},
               "implementation": "paper-oriented release; not a historical experiment"})
    write_json(output / "graph_used.json", graph)
    write_json(output / "metadata_used.json", metadata)

    def step(history):
        texts = [render(multi_view(x.numpy(), labels, metadata, graph, args.target, args.horizon)) for x in history]
        ids, mask = tokenize(tokenizer, texts, args.max_prompt_tokens, args.device)
        return model(history.to(args.device), ids, mask)

    def evaluate(split):
        model.eval()
        preds, truths, origins = [], [], []
        with torch.no_grad():
            for history, target, origin in loaders[split]:
                preds.append(step(history).cpu().numpy())
                truths.append(target.numpy())
                origins.extend(origin.tolist())
        return np.concatenate(preds), np.concatenate(truths), origins

    checkpoint = args.checkpoint or output / "best.pt"
    if not args.checkpoint:
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
        best_loss, history_log = float("inf"), []
        for epoch in range(args.epochs):
            model.train()
            total, count = 0., 0
            for history, target, _ in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                prediction = step(history)
                loss = (prediction - target.to(args.device)).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss.")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(history)
                count += len(history)
            p, y, _ = evaluate("val")
            val_loss = float(np.mean((p - y) ** 2))
            history_log.append({"epoch": epoch + 1, "train_mse": total / count, "val_mse": val_loss})
            print(json.dumps(history_log[-1]), flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                # Save all state for exact reload, including a locally initialized test backbone.
                torch.save({"model": model.state_dict(), "identity": identity}, checkpoint)
            write_json(output / "training_history.json", history_log)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["identity"] != identity:
        raise ValueError("Checkpoint identity differs from data, graph, metadata or model configuration.")
    model.load_state_dict(saved["model"])
    p, y, origins = evaluate("test")
    write_json(output / "metrics.json", metrics(p, y))
    predictions = []
    for pred, truth, origin in zip(p, y, origins):
        predictions.append({"history_end": str(frame[args.timestamp].iloc[origin - 1]),
                            "forecast_timestamps": [str(t) for t in frame[args.timestamp].iloc[origin:origin + args.horizon]],
                            "prediction": pred.tolist(), "target": truth.tolist(), "target_name": args.target})
    write_json(output / "predictions.json", predictions)
    first = origins[0]
    views = multi_view(values[first - args.seq_len:first], labels, metadata, graph, args.target, args.horizon)
    # No held-out ground truth is passed to the explanation stage.
    write_json(output / "explanation_context.json", {"views": views, "target": args.target,
               "history_end": str(frame[args.timestamp].iloc[first - 1]), "prediction": p[0].tolist(),
               "forecast_timestamps": predictions[0]["forecast_timestamps"]})
    print(f"Saved predictions, checkpoint and physical-scale metrics to {output}")


if __name__ == "__main__":
    main()
