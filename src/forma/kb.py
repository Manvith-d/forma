"""The agents' read-only view of the gold layer: retrieval, foods, exercises, drugs, rules and recipes."""

from __future__ import annotations

import json
import threading
from functools import cached_property

import duckdb
from rapidfuzz import fuzz, process

from .config import settings
from .embed import DIM, embed_query
from .foods import Food, FoodIndex
from .models import Citation


class KB:
    def __init__(self, path: str | None = None):
        self.con = duckdb.connect(str(path or settings().warehouse), read_only=True)
        self.lock = threading.Lock()

    def q(self, sql: str, params: list | None = None) -> list[dict]:
        with self.lock:
            rel = self.con.execute(sql, params or [])
            names = [c[0] for c in rel.description]
            return [dict(zip(names, r, strict=True)) for r in rel.fetchall()]

    # ---- retrieval ------------------------------------------------------------------------
    @cached_property
    def documents(self) -> dict[str, dict]:
        docs = {d["doc_id"]: d for d in self.q("SELECT * FROM silver.document")}
        for d in self.q("SELECT generic FROM gold.drug"):  # brand lists include homeopathics; they only help resolve names
            docs[f"label:{d['generic']}"] = {"doc_id": f"label:{d['generic']}", "title": f"FDA label: {d['generic']}",
                                             "publisher": "U.S. Food and Drug Administration",
                                             "url": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={d['generic']}",
                                             "topic": "medical", "license": "public-domain"}
        return docs

    def search(self, query: str, k: int = 5, topic: str | None = None, doc_prefix: str | None = None) -> list[Citation]:
        where, params = [], [embed_query(query).tolist()]
        if topic:
            where.append("d.topic = ?")
            params.append(topic)
        if doc_prefix:
            where.append("c.doc_id LIKE ?")
            params.append(doc_prefix + "%")
        rows = self.q(f"""
            SELECT c.doc_id, c.heading, c.page, c.text, array_cosine_similarity(c.embedding, ?::FLOAT[{DIM}]) AS score
            FROM gold.chunk c LEFT JOIN silver.document d USING (doc_id)
            {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY score DESC LIMIT {int(k)}""", params)
        return [self.cite(r["doc_id"], r["text"], r["heading"], r["page"]) for r in rows]

    def cite(self, doc_id: str, snippet: str, heading: str | None = None, page: int | None = None) -> Citation:
        d = self.documents.get(doc_id, {"title": doc_id, "url": None})
        location = f"p. {page}" if page else heading
        return Citation(source_id=doc_id, title=d["title"], location=location, url=d.get("url"), snippet=snippet[:600])

    # ---- rules and drugs ------------------------------------------------------------------
    @cached_property
    def rules(self) -> list[dict]:
        evidence: dict[str, list[Citation]] = {}
        for e in self.q("SELECT * FROM gold.rule_evidence"):
            evidence.setdefault(e["rule_id"], []).append(self.cite(e["source_id"], e["snippet"], e["location"], e["page"]))
        return [{**r, "applies": json.loads(r["applies"]), "effects": json.loads(r["effects"]),
                 "citations": evidence.get(r["rule_id"], [])} for r in self.q("SELECT * FROM gold.rule")]

    @cached_property
    def drugs(self) -> list[dict]:
        return self.q("SELECT * FROM gold.drug")

    def resolve_drug(self, name: str) -> dict | None:
        """Generic or brand name (any case, small typos) -> the screened drug."""
        names = {}
        for d in self.drugs:
            names[d["generic"].lower()] = d
            names[d["generic"].split()[0].lower()] = d
            for b in d["brand_names"]:
                names[b.lower()] = d
        key = name.lower().strip()
        if key in names:
            return names[key]
        hit = process.extractOne(key, list(names), scorer=fuzz.ratio, score_cutoff=86)
        return names[hit[0]] if hit else None

    # ---- foods, exercises, recipes --------------------------------------------------------
    @cached_property
    def foods(self) -> FoodIndex:
        portions: dict[int, dict[str, float]] = {}
        for p in self.q("SELECT * FROM gold.portion"):
            portions.setdefault(p["fdc_id"], {})[p["unit"]] = p["grams"]
        keys = ["kcal", "protein_g", "fat_g", "carb_g", "fiber_g", "sugar_g", "sat_fat_g", "sodium_mg", "potassium_mg",
                "calcium_mg", "iron_mg", "vitamin_k_ug", "vitamin_b12_ug", "vitamin_d_ug", "alcohol_g", "caffeine_mg"]
        return FoodIndex([Food(r["fdc_id"], r["name"], {k: r[k] for k in keys}, r["category"] or "", set(r["tags"]),
                               portions.get(r["fdc_id"], {})) for r in self.q("SELECT * FROM gold.food")])

    @cached_property
    def exercises(self) -> list[dict]:
        return self.q("SELECT * FROM gold.exercise")

    @cached_property
    def recipes(self) -> list[dict]:
        ings: dict[str, list[dict]] = {}
        for i in self.q("SELECT * FROM gold.recipe_ingredient ORDER BY recipe_id, position"):
            ings.setdefault(i["recipe_id"], []).append(i)
        return [{**r, "ingredients": ings.get(r["recipe_id"], [])} for r in self.q("SELECT * FROM gold.recipe WHERE usable")]

    def stats(self) -> dict:
        run = self.q("SELECT run_id, CAST(finished_at AS VARCHAR) finished_at, passed FROM ops.runs ORDER BY finished_at DESC LIMIT 1")
        return {"documents": len([d for d in self.documents if not d.startswith("label:")]), "labels": len(self.drugs),
                "chunks": self.q("SELECT count(*) n FROM gold.chunk")[0]["n"], "foods": len(self.foods.foods),
                "exercises": len(self.exercises), "recipes": len(self.recipes), "rules": len(self.rules),
                "checks_passed": self.q("SELECT count(*) FILTER (WHERE passed) p, count(*) n FROM ops.checks WHERE run_id = ?",
                                        [run[0]["run_id"]])[0] if run else None,
                "pipeline_run": run[0]["run_id"] if run else None}
