"""The pipeline: bronze -> silver -> gold -> quality, into one DuckDB warehouse.

    forma pipeline            build (downloads only what is missing)
    forma pipeline --refresh  re-download every source

Each run is recorded in ops.runs (step timings and row counts) and ops.checks (quality results).
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime

import duckdb

from ..config import manifest, settings
from ..foods import ALIASES
from . import bronze, gold, quality, silver
from .io import put


def records(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    rel = con.sql(sql)
    names = [c[0] for c in rel.description]
    return [dict(zip(names, row, strict=True)) for row in rel.fetchall()]


def build(refresh: bool = False, log=print) -> dict:
    s, m = settings(), manifest()
    run_id, started = uuid.uuid4().hex[:12], datetime.now(UTC)
    con = duckdb.connect(str(s.warehouse))
    for schema in ("bronze", "silver", "gold", "ops"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    steps: list[dict] = []

    def step(name: str, fn):
        t = time.perf_counter()
        counts = fn()
        steps.append({"step": name, "seconds": round(time.perf_counter() - t, 1), "counts": counts})
        log(f"  {name:<28} {steps[-1]['seconds']:>6.1f}s  {counts}")

    raw = s.data / "raw"
    log(f"pipeline run {run_id}")
    step("bronze: download sources", lambda: {"files": put(con, "bronze.files", bronze.fetch_all(s, m, refresh))})

    def silver_docs():
        docs, sections = silver.documents(m, raw)
        return {"documents": put(con, "silver.document", docs), "sections": put(con, "silver.section", sections)}

    def silver_fdc():
        silver.load_fdc(con, raw)
        return {"foods": con.sql("SELECT count(*) FROM silver.food").fetchone()[0],
                "nutrient_values": con.sql("SELECT count(*) FROM silver.food_nutrient").fetchone()[0]}

    def silver_rest():
        meals, ings = silver.recipes(raw)
        return {"exercises": put(con, "silver.exercise", silver.exercises(raw)),
                "label_sections": put(con, "silver.drug_label", silver.drug_labels(m, raw)),
                "recipes": put(con, "silver.recipe", meals), "recipe_lines": put(con, "silver.recipe_ingredient", ings)}

    step("silver: documents", silver_docs)
    step("silver: USDA FoodData Central", silver_fdc)
    step("silver: exercises, labels, recipes", silver_rest)

    index_box = {}

    def gold_foods():
        index_box["index"] = gold.foods(con)
        return {"foods": len(index_box["index"].foods), "portions": con.sql("SELECT count(*) FROM gold.portion").fetchone()[0]}

    def gold_exercises():
        gold.exercises(con, records(con, "SELECT * FROM silver.exercise"))
        return {"exercises": con.sql("SELECT count(*) FROM gold.exercise").fetchone()[0]}

    def gold_rules():
        gold.drugs(con)
        missing = gold.rules(con)
        for line in missing:
            log(f"    evidence missing: {line}")
        return {"rules": con.sql("SELECT count(*) FROM gold.rule").fetchone()[0],
                "evidence": con.sql("SELECT count(*) FROM gold.rule_evidence").fetchone()[0], "missing": len(missing)}

    step("gold: foods and portions", gold_foods)
    step("gold: exercise library", gold_exercises)
    step("gold: chunks and embeddings", lambda: {"chunks": gold.chunks(con)})
    step("gold: drugs and rulebook", gold_rules)
    step("gold: recipes and nutrition", lambda: gold.recipes(con, index_box["index"]))

    checks = quality.run(con, {"sources": len(m["documents"]) + len(m["datasets"]),
                               "licenses": ", ".join(f"'{x}'" for x in m["licenses_allowed"]),
                               "medications": sum(len(v) for v in m["medications"].values()),
                               "aliases": "[" + ", ".join("'" + v.replace("'", "''") + "'" for v in sorted(set(ALIASES.values()))) + "]"})
    for c in checks:
        log(f"  {'PASS' if c['passed'] else 'FAIL'}  [{c['layer']}] {c['check']}: {c['detail']}")
    ok = all(c["passed"] for c in checks)
    con.execute("CREATE TABLE IF NOT EXISTS ops.runs (run_id VARCHAR, started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, "
                "passed BOOLEAN, steps JSON)")
    con.execute("CREATE TABLE IF NOT EXISTS ops.checks (run_id VARCHAR, layer VARCHAR, check_name VARCHAR, passed BOOLEAN, "
                "detail VARCHAR)")
    con.execute("INSERT INTO ops.runs VALUES (?, ?, ?, ?, ?)", [run_id, started, datetime.now(UTC), ok, json.dumps(steps)])
    con.executemany("INSERT INTO ops.checks VALUES (?, ?, ?, ?, ?)",
                    [[run_id, c["layer"], c["check"], c["passed"], c["detail"]] for c in checks])
    con.close()
    return {"run_id": run_id, "passed": ok, "steps": steps, "checks": checks}
