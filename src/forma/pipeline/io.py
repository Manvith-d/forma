"""Bulk-load Python rows into DuckDB through its JSON reader (far faster than row inserts)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import duckdb


def put(con: duckdb.DuckDBPyConnection, table: str, rows: list[dict]) -> int:
    """Create `table` from a list of dicts (types inferred by DuckDB's JSON reader)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(rows, f)
    try:
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_json('{f.name}', format='array', "
                    "maximum_object_size=67108864, sample_size=-1)")
    finally:
        Path(f.name).unlink()
    return len(rows)
