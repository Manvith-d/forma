"""Settings from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data: Path  # raw downloads and the DuckDB warehouse
    warehouse: Path
    home: Path  # sessions and runtime state
    llm: str  # ollama:<model> | none
    ollama_url: str


def settings() -> Settings:
    data = Path(os.environ.get("FORMA_DATA", ROOT / "data"))
    home = Path(os.environ.get("FORMA_HOME", ROOT / "runtime"))
    data.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    return Settings(data=data, warehouse=Path(os.environ.get("FORMA_WAREHOUSE", data / "forma.duckdb")), home=home,
                    llm=os.environ.get("FORMA_LLM", "ollama:qwen3:8b"),
                    ollama_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))


@cache
def manifest() -> dict:
    return yaml.safe_load((ROOT / "sources.yaml").read_text())
