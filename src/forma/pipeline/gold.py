"""Gold: the curated, agent-ready layer.

    gold.food / gold.portion     foods with per-100 g nutrients, safety and allergen tags, portion weights
    gold.exercise                exercises with movement pattern, equipment and contraindication tags
    gold.chunk                   cited passages from every document and FDA label, with bge-small embeddings
    gold.drug                    screened medications: class, generic and brand names, label version
    gold.rule / rule_evidence    the rulebook, each rule with the verbatim passage that supports it
    gold.recipe / recipe_ingredient   recipes mapped to USDA foods with grams and nutrition per serving
"""

from __future__ import annotations

import hashlib
import json
import re

import duckdb
import yaml

from ..config import ROOT
from ..embed import DIM, embed_passages
from ..foods import Food, FoodIndex, food_tags
from .io import put

NUTRIENTS = {"kcal": (1008, 2047, 2048), "protein_g": (1003,), "fat_g": (1004,), "carb_g": (1005,), "fiber_g": (1079,),
             "sugar_g": (2000, 1063), "sat_fat_g": (1258,), "sodium_mg": (1093,), "potassium_mg": (1092,),
             "calcium_mg": (1087,), "iron_mg": (1089,), "vitamin_k_ug": (1185,), "vitamin_b12_ug": (1178,),
             "vitamin_d_ug": (1114,), "alcohol_g": (1018,), "caffeine_mg": (1057,)}
# ---- foods ------------------------------------------------------------------------------
def foods(con: duckdb.DuckDBPyConnection) -> FoodIndex:
    cols = ", ".join(
        f"coalesce({', '.join(f'max(amount) FILTER (WHERE nutrient_id = {i})' for i in ids)}) AS {k}"
        for k, ids in NUTRIENTS.items())
    rows = con.sql(f"""
        SELECT f.fdc_id, f.description, f.data_type, c.description AS category, {cols}
        FROM silver.food f JOIN silver.food_nutrient USING (fdc_id) LEFT JOIN silver.food_category c ON c.id = f.category_id
        GROUP BY ALL""").fetchall()
    names = ["fdc_id", "name", "data_type", "category", *NUTRIENTS]
    portions: dict[int, dict[str, float]] = {}
    for fdc_id, desc, modifier, unit, amount, grams in con.sql("""
            SELECT p.fdc_id, p.portion_description, p.modifier, u.name, coalesce(p.amount, 1), p.gram_weight
            FROM silver.food_portion p LEFT JOIN silver.measure_unit u ON u.id = p.unit_id ORDER BY p.fdc_id""").fetchall():
        label = " ".join(x for x in (unit if unit not in (None, "undetermined") else "", modifier or "", desc or "") if x).lower()
        key = next((k for k, words in (("cup", ("cup",)), ("tbsp", ("tbsp", "tablespoon")), ("tsp", ("tsp", "teaspoon")),
                                       ("large", ("large",)), ("medium", ("medium",)), ("small", ("small",)),
                                       ("clove", ("clove",)), ("slice", ("slice",)), ("stalk", ("stalk",)),
                                       ("whole", ("whole", "fruit", "each", "egg", "breast", "thigh")))
                    if any(w in label for w in words)), None)
        if key and amount:
            portions.setdefault(fdc_id, {}).setdefault(key, grams / amount)
    legacy = {d["name"].lower(): d for d in (dict(zip(names, r, strict=True)) for r in rows) if d["data_type"] == "sr_legacy_food"}
    out = []
    for r in rows:
        d = dict(zip(names, r, strict=True))
        if d["data_type"] != "sr_legacy_food" and (old := legacy.get(d["name"].lower())):
            d |= {k: old[k] for k in NUTRIENTS if d[k] is None}  # Foundation analyses skip some nutrients (berries: fiber)
        if d["kcal"] is None and d["protein_g"] is not None:  # some Foundation foods report only macros
            d["kcal"] = 4 * (d["protein_g"] or 0) + 4 * (d["carb_g"] or 0) + 9 * (d["fat_g"] or 0)
        if d["kcal"] is None:
            continue
        per100 = {k: float(d[k] or 0.0) for k in NUTRIENTS}
        tags = food_tags(d["name"], d["category"], per100)
        for k in ("carb_g", "fat_g", "protein_g"):  # "by difference" can dip just below zero for meats and fish
            if -1.0 < per100[k] < 0:
                per100[k] = 0.0
        out.append(Food(d["fdc_id"], d["name"], per100, d["category"] or "", tags, portions.get(d["fdc_id"], {})))
    con.execute("CREATE OR REPLACE TABLE gold.food (fdc_id INTEGER, name VARCHAR, data_type VARCHAR, category VARCHAR, "
                + ", ".join(f"{k} DOUBLE" for k in NUTRIENTS) + ", tags VARCHAR[])")
    meta = {r[0]: (r[2], r[3]) for r in rows}
    con.executemany(f"INSERT INTO gold.food VALUES ({', '.join('?' * (5 + len(NUTRIENTS)))})",
                    [[f.fdc_id, f.name, *meta[f.fdc_id], *f.per100.values(), sorted(f.tags)] for f in out])
    con.execute("CREATE OR REPLACE TABLE gold.portion (fdc_id INTEGER, unit VARCHAR, grams DOUBLE)")
    con.executemany("INSERT INTO gold.portion VALUES (?, ?, ?)",
                    [[f.fdc_id, k, g] for f in out for k, g in f.portions.items()])
    return FoodIndex(out)


# ---- exercises ----------------------------------------------------------------------------
PATTERNS = [  # first match wins; tested on the lowercased name
    ("hinge", r"deadlift|good morning|hip thrust|glute bridge|kettlebell swing|romanian|pull through|hyperextension"),
    ("isolation", r"curl|extension|kickback|fly\b|flye|raise|shrug|crunch|pullover"),
    ("lunge", r"lunge|split squat|step[- ]?up|bulgarian"),
    ("squat", r"squat|leg press|hack"),
    ("pull_vertical", r"pull-?up|chin-?up|pulldown|pull down"),
    ("pull_horizontal", r"\brow\b|rows\b|rowing|face pull|inverted"),
    ("push_vertical", r"overhead|shoulder press|military|arnold|push press|handstand"),
    ("push_horizontal", r"bench press|push-?up|chest press|floor press|dip"),
    ("carry", r"carry|farmer|suitcase"),
]


def exercise_tags(e: dict) -> tuple[str, list[str]]:
    name, cat = e["name"].lower(), (e["category"] or "").lower()
    pattern = next((p for p, rx in PATTERNS if re.search(rx, name)), None)
    if pattern is None:
        pattern = {"cardio": "cardio", "stretching": "mobility", "plyometrics": "plyometric"}.get(cat) or (
            "core" if "abdominals" in e["primary_muscles"] else "isolation")
    tags = set()
    if cat == "plyometrics" or re.search(r"jump|bound|hop|skip|burpee|box ", name):
        tags |= {"plyometric", "high_impact"}
    if re.search(r"sprint|dash|shuttle", name):
        tags |= {"sprint", "high_impact"}
    if re.search(r"negative|eccentric|nordic", name):
        tags.add("heavy_eccentric")
    if re.search(r"full squat|pistol|sissy|deep|ass to grass|atg", name):
        tags.add("deep_knee_flexion")
    if e["equipment"] == "barbell" and pattern in ("squat", "hinge"):
        tags |= {"heavy_spinal_load", "heavy"}
    if re.search(r"good morning|jefferson|weighted crunch|weighted sit|toe touch|stiff[- ]leg", name):
        tags.add("loaded_flexion")
    if re.search(r"plank|hold|wall sit|isometric|static", name):
        tags.add("isometric_hold")
    if cat in ("olympic weightlifting", "powerlifting", "strongman"):
        tags |= {"max_effort", "heavy", "fall_risk"}
    if re.search(r"lying|supine|bench press|crunch|floor press|glute bridge|sit-?up", name):
        tags.add("supine_late")
    if cat == "cardio" and re.search(r"run|jog|treadmill|jump rope|rope jump|stair", name):
        tags.add("high_impact")
    return pattern, sorted(tags)


def exercises(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    con.execute("""CREATE OR REPLACE TABLE gold.exercise (exercise_id VARCHAR, name VARCHAR, pattern VARCHAR, category VARCHAR,
        level VARCHAR, equipment VARCHAR, mechanic VARCHAR, primary_muscles VARCHAR[], secondary_muscles VARCHAR[],
        tags VARCHAR[], instructions VARCHAR[], image VARCHAR)""")
    con.executemany("INSERT INTO gold.exercise VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
        [e["exercise_id"], e["name"], *exercise_tags(e)[:1], e["category"], e["level"], e["equipment"], e["mechanic"],
         e["primary_muscles"], e["secondary_muscles"], exercise_tags(e)[1], e["instructions"], e["image"]] for e in rows])


# ---- knowledge chunks ----------------------------------------------------------------------
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•(\[])")


def chunk(text: str, size: int = 900) -> list[str]:
    out, cur = [], ""
    for s in _SENT.split(text):
        if cur and len(cur) + len(s) > size:
            out.append(cur.strip())
            cur = cur.rsplit(". ", 1)[-1] + " " if len(cur) > 200 else ""  # carry one sentence of overlap
        cur += s + " "
    if cur.strip():
        out.append(cur.strip())
    return [c for c in out if len(c) > 60]


def chunks(con: duckdb.DuckDBPyConnection) -> int:
    rows = []
    for doc_id, heading, page, text in con.sql("SELECT doc_id, heading, page, text FROM silver.section ORDER BY doc_id, seq").fetchall():
        rows += [(doc_id, heading, page, c) for c in chunk(text)]
    for generic, cls, section, text in con.sql(
            "SELECT generic, drug_class, section, text FROM silver.drug_label ORDER BY generic, section").fetchall():
        rows += [(f"label:{generic}", section.replace("_", " "), None, c) for c in chunk(text)]
    # Embeddings are cached by text hash, so a rebuild only embeds passages that changed. Vectors stay inside DuckDB.
    con.execute(f"CREATE TABLE IF NOT EXISTS ops.embedding_cache (sha1 VARCHAR PRIMARY KEY, embedding FLOAT[{DIM}])")
    meta = [{"chunk_id": i, "doc_id": r[0], "heading": r[1], "page": r[2], "text": r[3],
             "sha1": hashlib.sha1(r[3].encode()).hexdigest()} for i, r in enumerate(rows)]
    put(con, "ops.chunk_meta", meta)
    todo = con.sql("SELECT DISTINCT m.sha1, m.text FROM ops.chunk_meta m ANTI JOIN ops.embedding_cache c USING (sha1)").fetchall()
    if todo:
        vectors = embed_passages([text for _, text in todo])
        put(con, "ops.new_embeddings", [{"sha1": k, "embedding": v.tolist()} for (k, _), v in zip(todo, vectors, strict=True)])
        con.execute(f"INSERT INTO ops.embedding_cache SELECT sha1, embedding::FLOAT[{DIM}] FROM ops.new_embeddings")
        con.execute("DROP TABLE ops.new_embeddings")
    con.execute("""CREATE OR REPLACE TABLE gold.chunk AS
        SELECT m.chunk_id::INTEGER chunk_id, m.doc_id, m.heading, m.page::INTEGER page, m.text, c.embedding
        FROM ops.chunk_meta m JOIN ops.embedding_cache c USING (sha1) ORDER BY m.chunk_id""")
    con.execute("DROP TABLE ops.chunk_meta")
    return len(rows)


# ---- drugs and the rulebook ------------------------------------------------------------------
def drugs(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""CREATE OR REPLACE TABLE gold.drug AS
        SELECT generic, any_value(drug_class) drug_class, any_value(brand_names) brand_names, any_value(set_id) set_id,
               any_value(effective_time) effective_time FROM silver.drug_label GROUP BY generic""")


def rules(con: duckdb.DuckDBPyConnection) -> list[dict]:
    book = yaml.safe_load((ROOT / "knowledge" / "rules.yaml").read_text())
    docs = {}
    for doc_id, heading, page, text in con.sql("SELECT doc_id, heading, page, text FROM silver.section").fetchall():
        docs.setdefault(doc_id, []).append((heading, page, text))
    for generic, section, text in con.sql("SELECT generic, section, text FROM silver.drug_label").fetchall():
        docs.setdefault(f"label:{generic}", []).append((section.replace("_", " "), None, text))
    rule_rows, evidence_rows, missing = [], [], []
    for kind, items in (("medication", book["medications"]), ("condition", book["conditions"]), ("standard", book["standards"])):
        for r in items:
            applies = {k: r[k] for k in ("drug_class", "generics", "condition") if k in r}
            rule_rows.append([r["id"], kind, json.dumps(applies), json.dumps(r.get("effects", {})),
                              r.get("message") or r.get("value")])
            for ev in r["evidence"] if isinstance(r["evidence"], list) else [r["evidence"]]:
                source = ev.get("doc") or f"label:{ev['label']}"
                hit = None
                for heading, page, text in docs.get(source, []):
                    m = re.search(re.escape(ev["pattern"]), text, re.IGNORECASE)
                    if m:
                        start = max(text.rfind(". ", 0, m.start()) + 2, m.start() - 220, 0)
                        end = text.find(". ", m.end())
                        end = len(text) if end == -1 else min(end + 1, m.end() + 220)
                        hit = (heading, page, text[start:end].strip())
                        break
                if hit:
                    evidence_rows.append([r["id"], source, hit[0], hit[1], hit[2]])
                else:
                    missing.append(f"{r['id']}: '{ev['pattern']}' not found in {source}")
    con.execute("CREATE OR REPLACE TABLE gold.rule (rule_id VARCHAR, kind VARCHAR, applies JSON, effects JSON, message VARCHAR)")
    con.executemany("INSERT INTO gold.rule VALUES (?, ?, ?, ?, ?)", rule_rows)
    con.execute("CREATE OR REPLACE TABLE gold.rule_evidence (rule_id VARCHAR, source_id VARCHAR, location VARCHAR, page INTEGER, "
                "snippet VARCHAR)")
    con.executemany("INSERT INTO gold.rule_evidence VALUES (?, ?, ?, ?, ?)", evidence_rows)
    return missing


# ---- recipes ---------------------------------------------------------------------------------
def recipes(con: duckdb.DuckDBPyConnection, index: FoodIndex) -> dict:
    meals = con.sql("SELECT recipe_id, name, category, area, tags, thumb, source_url FROM silver.recipe").fetchall()
    lines: dict[str, list] = {}
    for rid, pos, ing, measure in con.sql(
            "SELECT recipe_id, position, ingredient, measure FROM silver.recipe_ingredient ORDER BY recipe_id, position").fetchall():
        lines.setdefault(rid, []).append((pos, ing, measure))
    recipe_rows, ing_rows, stats = [], [], {"recipes": len(meals), "lines": 0, "mapped": 0, "kept": 0}
    for rid, name, category, area, tags, thumb, url in meals:
        totals: dict[str, float] = {}
        resolved, food_tags = 0, set()
        for pos, ing, measure in lines.get(rid, []):
            stats["lines"] += 1
            food, score = index.match(ing)
            grams = index.grams(food, measure, ing) if food else None
            if food and grams is not None:
                resolved += 1
                stats["mapped"] += 1
                food_tags |= food.tags
                for k, v in index.nutrition(food, grams).items():
                    totals[k] = totals.get(k, 0.0) + v
            ing_rows.append([rid, pos, ing, measure, food.fdc_id if food else None, food.name if food else None,
                             round(score, 1), grams])
        n = len(lines.get(rid, []))
        coverage = resolved / n if n else 0.0
        servings = max(2, min(8, round(totals.get("kcal", 0) / 550))) if totals else None
        keep = coverage >= 0.85 and category != "Dessert" and totals.get("kcal", 0) > 150
        stats["kept"] += keep
        recipe_rows.append([rid, name, category, area, tags, thumb, url, round(coverage, 3), servings, keep, sorted(food_tags),
                            *[round(totals.get(k, 0.0) / servings, 2) if servings else None for k in NUTRIENTS]])
    con.execute("CREATE OR REPLACE TABLE gold.recipe (recipe_id VARCHAR, name VARCHAR, category VARCHAR, area VARCHAR, "
                "tags VARCHAR[], thumb VARCHAR, source_url VARCHAR, coverage DOUBLE, servings INTEGER, usable BOOLEAN, "
                "food_tags VARCHAR[], " + ", ".join(f"{k}_per_serving DOUBLE" for k in NUTRIENTS) + ")")
    con.executemany(f"INSERT INTO gold.recipe VALUES ({', '.join('?' * (11 + len(NUTRIENTS)))})", recipe_rows)
    con.execute("CREATE OR REPLACE TABLE gold.recipe_ingredient (recipe_id VARCHAR, position INTEGER, ingredient VARCHAR, "
                "measure VARCHAR, fdc_id INTEGER, food VARCHAR, match_score DOUBLE, grams DOUBLE)")
    con.executemany("INSERT INTO gold.recipe_ingredient VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ing_rows)
    return stats
