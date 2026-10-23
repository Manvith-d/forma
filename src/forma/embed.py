"""bge-small-en-v1.5 embeddings (384 dims) through fastembed's local ONNX runtime: free, offline, CPU."""

from __future__ import annotations

import os
from functools import cache

import numpy as np

DIM = 384
MODEL = "BAAI/bge-small-en-v1.5"


@cache
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(MODEL, cache_dir=os.environ.get("FORMA_MODEL_CACHE"))


def _unit(vectors) -> np.ndarray:
    v = np.array(list(vectors), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed_passages(texts: list[str]) -> np.ndarray:
    # One process: ONNX threads already use every core, and a worker per core runs out of memory in a 4 GB container
    return _unit(_model().passage_embed(texts, batch_size=32))


def embed_query(text: str) -> np.ndarray:
    return _unit(_model().query_embed([text]))[0]  # adds bge's retrieval instruction to the query
