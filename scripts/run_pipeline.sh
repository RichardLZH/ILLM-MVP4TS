#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATA:?Set DATA to an absolute CSV or XLSX path}"
: "${TARGET:?Set TARGET to the exact target column name}"
: "${METADATA:?Set METADATA to the metadata JSON path}"
: "${CORPUS:?Set CORPUS to the authorized knowledge document directory}"
: "${QUESTION:?Set QUESTION to the operator question}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-outputs/run}"
TIMESTAMP_COLUMN="${TIMESTAMP_COLUMN:-date}"
"$PYTHON_BIN" -m illm_mvp4ts.graph --data "$DATA" --timestamp "$TIMESTAMP_COLUMN" \
  --metadata "$METADATA" --corpus "$CORPUS" --output "$RUN_DIR/graph"
"$PYTHON_BIN" -m illm_mvp4ts.train --config configs/paper.json --data "$DATA" \
  --timestamp "$TIMESTAMP_COLUMN" --target "$TARGET" --metadata "$METADATA" \
  --graph "$RUN_DIR/graph/graph.json" --output "$RUN_DIR/forecast"
"$PYTHON_BIN" -m illm_mvp4ts.rag --corpus "$CORPUS" \
  --context "$RUN_DIR/forecast/explanation_context.json" --question "$QUESTION" \
  --output "$RUN_DIR/explanation"
