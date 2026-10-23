"""Silver: parse every raw source into clean, typed tables. Nothing is interpreted yet.

    silver.document   one row per document with its citation
    silver.section    the documents' text, by section (papers, fact sheets) or by page (PDFs)
    silver.food / food_nutrient / food_portion / nutrient   USDA FoodData Central (Foundation + SR Legacy)
    silver.exercise   Free Exercise DB
    silver.drug_label one row per FDA label section for each screened medication
    silver.recipe / recipe_ingredient   TheMealDB
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import duckdb
from bs4 import BeautifulSoup
from pypdf import PdfReader

LABEL_SECTIONS = ("boxed_warning", "warnings_and_cautions", "warnings", "precautions", "drug_interactions",
                  "dosage_and_administration", "information_for_patients", "patient_medication_information",
                  "spl_patient_package_insert", "adverse_reactions", "contraindications", "clinical_pharmacology",
                  "use_in_specific_populations", "pregnancy")
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate PDF line breaks
    return _WS.sub(" ", text).strip()


# ---- documents ---------------------------------------------------------------------------
def _pdf(path: Path) -> list[dict]:
    return [{"heading": f"page {i}", "page": i, "text": clean(p.extract_text() or "")}
            for i, p in enumerate(PdfReader(path).pages, start=1)]


def _jats(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    out = []
    abstract = root.find(".//front//abstract")
    if abstract is not None:
        out.append({"heading": "Abstract", "page": None, "text": clean(" ".join(abstract.itertext()))})

    def walk(sec, trail):
        title = clean(sec.findtext("title") or "")
        path_ = [*trail, title] if title else trail
        paras = [clean(" ".join(p.itertext())) for p in sec.findall("p")]
        if paras:
            out.append({"heading": " > ".join(path_), "page": None, "text": " ".join(paras)})
        for child in sec.findall("sec"):
            walk(child, path_)

    body = root.find("body")
    for sec in body.findall("sec") if body is not None else []:
        walk(sec, [])
    return out


def _html(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    main = soup.select_one("article") or soup.select_one("main") or soup.body
    out, heading, buf, skip = [], "Introduction", [], False
    for el in main.find_all(["h2", "h3", "p", "li"]):
        if el.name in ("h2", "h3"):
            if buf and not skip:
                out.append({"heading": heading, "page": None, "text": clean(" ".join(buf))})
            heading, buf = clean(el.get_text(" ")), []
            skip = heading.lower() in ("references", "disclaimer", "table of contents")
        elif not skip and not el.find_parent("li"):  # nested items are already in their parent's text
            buf.append(el.get_text(" "))
    if buf and not skip:
        out.append({"heading": heading, "page": None, "text": clean(" ".join(buf))})
    return [s for s in out if len(s["text"]) > 40]


def documents(manifest: dict, raw: Path) -> tuple[list[dict], list[dict]]:
    docs, sections = [], []
    for d in manifest["documents"]:
        base = raw / d["id"]
        if d["kind"] == "europepmc":
            m = json.loads((base / "meta.json").read_text())["resultList"]["result"][0]
            authors = m.get("authorString", "")
            first = authors.split(",")[0] + (" et al." if "," in authors else "")
            journal = m.get("journalInfo", {}).get("journal", {}).get("title", "")
            docs.append({"doc_id": d["id"], "title": clean(m["title"]).rstrip("."), "publisher": journal,
                         "year": int(m.get("pubYear") or 0) or None, "license": d["license"], "topic": d["topic"],
                         "url": f"https://doi.org/{m['doi']}" if m.get("doi") else f"https://europepmc.org/article/PMC/{d['pmcid']}",
                         "citation": f"{first} ({m.get('pubYear')}). {clean(m['title']).rstrip('.')}. {journal}."})
            parts = _jats(base / "fulltext.xml")
        else:
            year = int(re.search(r"(20\d\d)", d["title"]).group(1)) if re.search(r"(20\d\d)", d["title"]) else None
            docs.append({"doc_id": d["id"], "title": d["title"], "publisher": d["publisher"], "year": year,
                         "license": d["license"], "topic": d["topic"], "url": d["url"],
                         "citation": f"{d['publisher']}. {d['title']}."})
            parts = _pdf(base / "document.pdf") if d["kind"] == "pdf" else _html(base / "page.html")
        sections += [{"doc_id": d["id"], "seq": i, **p} for i, p in enumerate(parts) if p["text"]]
    return docs, sections


# ---- USDA FoodData Central -----------------------------------------------------------------
def load_fdc(con: duckdb.DuckDBPyConnection, raw: Path) -> None:
    dirs = {}
    for ds in ("fdc-foundation", "fdc-sr-legacy"):
        z = next((raw / ds).glob("*.zip"))
        out = raw / ds / "extracted"
        if not out.exists():
            zipfile.ZipFile(z).extractall(out)
        dirs[ds] = next(p for p in out.iterdir() if p.is_dir())
    f, s = dirs["fdc-foundation"], dirs["fdc-sr-legacy"]

    def csv(path: Path) -> str:
        return f"read_csv('{path}', header=true, all_varchar=true)"

    con.execute(f"""
        CREATE OR REPLACE TABLE silver.food AS
        SELECT CAST(fdc_id AS INTEGER) fdc_id, data_type, description, CAST(food_category_id AS INTEGER) category_id
        FROM {csv(f / 'food.csv')} WHERE data_type = 'foundation_food'
        UNION ALL
        SELECT CAST(fdc_id AS INTEGER), data_type, description, CAST(food_category_id AS INTEGER)
        FROM {csv(s / 'food.csv')} WHERE data_type = 'sr_legacy_food'""")
    con.execute(f"CREATE OR REPLACE TABLE silver.food_category AS SELECT CAST(id AS INTEGER) id, description "
                f"FROM {csv(s / 'food_category.csv')}")
    con.execute(f"""CREATE OR REPLACE TABLE silver.nutrient AS
        SELECT CAST(id AS INTEGER) id, name, unit_name FROM {csv(f / 'nutrient.csv')}""")
    con.execute(f"""
        CREATE OR REPLACE TABLE silver.food_nutrient AS
        SELECT CAST(fdc_id AS INTEGER) fdc_id, CAST(nutrient_id AS INTEGER) nutrient_id, CAST(amount AS DOUBLE) amount
        FROM (SELECT * FROM {csv(f / 'food_nutrient.csv')} UNION ALL BY NAME SELECT * FROM {csv(s / 'food_nutrient.csv')})
        WHERE CAST(fdc_id AS INTEGER) IN (SELECT fdc_id FROM silver.food) AND amount IS NOT NULL""")
    con.execute(f"""CREATE OR REPLACE TABLE silver.measure_unit AS
        SELECT CAST(id AS INTEGER) id, name FROM {csv(s / 'measure_unit.csv')}""")
    con.execute(f"""
        CREATE OR REPLACE TABLE silver.food_portion AS
        SELECT CAST(fdc_id AS INTEGER) fdc_id, TRY_CAST(amount AS DOUBLE) amount, CAST(measure_unit_id AS INTEGER) unit_id,
               portion_description, modifier, CAST(gram_weight AS DOUBLE) gram_weight
        FROM (SELECT * FROM {csv(f / 'food_portion.csv')} UNION ALL BY NAME SELECT * FROM {csv(s / 'food_portion.csv')})
        WHERE CAST(fdc_id AS INTEGER) IN (SELECT fdc_id FROM silver.food) AND TRY_CAST(gram_weight AS DOUBLE) > 0""")


# ---- exercises, drug labels, recipes ---------------------------------------------------------
def exercises(raw: Path) -> list[dict]:
    rows = json.loads((raw / "exercises" / "exercises.json").read_text())
    return [{"exercise_id": r["id"], "name": r["name"], "force": r.get("force"), "level": r.get("level"),
             "mechanic": r.get("mechanic"), "equipment": r.get("equipment") or "body only",
             "primary_muscles": r.get("primaryMuscles") or [], "secondary_muscles": r.get("secondaryMuscles") or [],
             "category": r.get("category"), "instructions": [clean(i) for i in r.get("instructions") or []],
             "image": r["images"][0] if r.get("images") else None} for r in rows]


def drug_labels(manifest: dict, raw: Path) -> list[dict]:
    rows = []
    for cls, generics in manifest["medications"].items():
        for g in generics:
            results = json.loads((raw / "openfda-labels" / f"{g.replace(' ', '_')}.json").read_text())["results"]
            # The most complete label (most of the sections we screen), newest first on ties
            label = max(results, key=lambda r: (sum(k in r for k in LABEL_SECTIONS), r.get("effective_time", "")))
            counted = json.loads((raw / "openfda-labels" / "brands" / f"{g.replace(' ', '_')}.json").read_text())
            brands = sorted({b["term"].title() for b in counted.get("results", [])}
                            | {b.title() for r in results for b in r.get("openfda", {}).get("brand_name", [])})
            for section in LABEL_SECTIONS:
                for text in label.get(section, []):
                    rows.append({"generic": g, "drug_class": cls, "brand_names": brands, "set_id": label.get("set_id"),
                                 "effective_time": label.get("effective_time"), "section": section, "text": clean(text)})
    return rows


def recipes(raw: Path) -> tuple[list[dict], list[dict]]:
    meals, ingredients = {}, []
    for f in sorted((raw / "themealdb").glob("*.json")):
        for m in json.loads(f.read_text()).get("meals") or []:
            if m["idMeal"] in meals:
                continue
            meals[m["idMeal"]] = {"recipe_id": m["idMeal"], "name": m["strMeal"].strip(), "category": m.get("strCategory"),
                                  "area": m.get("strArea"), "tags": [t for t in (m.get("strTags") or "").split(",") if t],
                                  "instructions": clean(m.get("strInstructions") or ""), "thumb": m.get("strMealThumb"),
                                  "source_url": m.get("strSource") or f"https://www.themealdb.com/meal/{m['idMeal']}"}
            for i in range(1, 21):
                name, measure = (m.get(f"strIngredient{i}") or "").strip(), (m.get(f"strMeasure{i}") or "").strip()
                if name:
                    ingredients.append({"recipe_id": m["idMeal"], "position": i, "ingredient": name, "measure": measure})
    return list(meals.values()), ingredients
