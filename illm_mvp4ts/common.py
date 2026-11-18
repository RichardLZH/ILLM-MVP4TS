"""Validated inputs and explicit, environment-configured text completions."""
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_graph(graph, labels):
    if graph.get("labels") != list(labels):
        raise ValueError("Graph variable names/order differ from forecasting data.")
    if graph.get("orientation") != "row=effect,column=cause":
        raise ValueError("Graph must declare orientation row=effect,column=cause.")
    matrix = np.asarray(graph["adjacency"])
    if matrix.shape != (len(labels), len(labels)) or not np.isin(matrix, [0, 1]).all():
        raise ValueError("Expected a binary square adjacency matrix.")
    if np.any(np.diag(matrix)):
        raise ValueError("Self edges are not supported.")
    return matrix.astype(int)


def json_object(answer):
    text = answer.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


class ChatClient:
    """OpenAI-compatible /chat/completions; no service is called at import time."""

    def __init__(self):
        self.model = os.environ.get("ILLM_CHAT_MODEL", "")
        self.base_url = os.environ.get("ILLM_CHAT_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("ILLM_CHAT_API_KEY", "")
        if not self.model or not self.base_url:
            raise ValueError("Set ILLM_CHAT_MODEL and ILLM_CHAT_BASE_URL explicitly.")

    def inquire_LLMs(self, prompt, system_prompt, temperature=0.2):
        payload = {"model": self.model, "messages": [
            {"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": int(os.getenv("ILLM_MAX_TOKENS", "4096"))}
        thinking = os.getenv("ILLM_ENABLE_THINKING")
        if thinking is not None:
            if thinking.lower() not in {"true", "false"}:
                raise ValueError("ILLM_ENABLE_THINKING must be true or false when specified.")
            payload["chat_template_kwargs"] = {"enable_thinking": thinking.lower() == "true"}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.base_url + "/chat/completions",
                                         json.dumps(payload).encode(), headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        choice = result["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("Completion was truncated; increase ILLM_MAX_TOKENS.")
        answer = choice["message"].get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Empty completion.")
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).replace("</think>", "").strip()
        if not answer:
            raise ValueError("Completion contains no final answer.")
        return answer

    def object(self, prompt, system_prompt="Return only the requested JSON object."):
        return json_object(self.inquire_LLMs(prompt, system_prompt))
