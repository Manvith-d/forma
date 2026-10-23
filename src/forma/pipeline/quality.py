"""Data quality checks, run after every pipeline build. A failed check fails the build.

Each check is (layer, name, SQL returning one row: passed boolean, detail text).
"""

from __future__ import annotations

CHECKS = [
    ("bronze", "every manifest source downloaded",
     "SELECT count(DISTINCT source_id) = {sources}, count(DISTINCT source_id) || ' of {sources} sources' FROM bronze.files"),
    ("bronze", "only allowlisted licenses",
     "SELECT bool_and(license IN ({licenses})), string_agg(DISTINCT license, ', ') FROM bronze.files"),
    ("silver", "every document parsed into text",
     "SELECT min(chars) > 2000, 'shortest document: ' || min(chars) || ' chars' FROM "
     "(SELECT doc_id, sum(length(text)) chars FROM silver.section GROUP BY doc_id)"),
    ("silver", "USDA foods loaded",
     "SELECT count(*) > 8000, count(*) || ' foods' FROM silver.food"),
    ("silver", "every screened medication has a label",
     "SELECT count(DISTINCT generic) = {medications}, count(DISTINCT generic) || ' of {medications} medications' FROM silver.drug_label"),
    ("gold", "food energy agrees with its macronutrients",
     "SELECT avg(CASE WHEN abs(4*protein_g + 4*carb_g + 9*fat_g + 7*alcohol_g - kcal) <= greatest(0.2*kcal, 15) THEN 1 ELSE 0 END) >= 0.9, "
     "round(100*avg(CASE WHEN abs(4*protein_g + 4*carb_g + 9*fat_g + 7*alcohol_g - kcal) <= greatest(0.2*kcal, 15) THEN 1 ELSE 0 END), 1) "
     "|| '% of foods within 20%' FROM gold.food WHERE kcal > 0"),
    ("gold", "no negative nutrient values",
     "SELECT count(*) = 0, count(*) || ' foods with a negative value' FROM gold.food "
     "WHERE least(kcal, protein_g, fat_g, carb_g, sodium_mg, potassium_mg) < 0"),
    ("gold", "ingredient aliases resolve to USDA foods",
     "SELECT count(*) FILTER (WHERE f.name IS NULL) = 0, count(*) FILTER (WHERE f.name IS NULL) || ' of ' || count(*) "
     "|| ' alias targets missing' FROM (SELECT unnest({aliases}) AS target) a LEFT JOIN gold.food f ON f.name = a.target"),
    ("gold", "plant foods people cook with report fiber",
     "SELECT count(*) FILTER (WHERE f.fiber_g = 0) = 0, count(*) FILTER (WHERE f.fiber_g = 0) || ' of ' || count(*) "
     "|| ' fruit, vegetable, legume and grain alias targets without fiber' "
     "FROM (SELECT DISTINCT unnest({aliases}) AS target) a JOIN gold.food f ON f.name = a.target "
     "WHERE f.category SIMILAR TO '.*(Fruits|Vegetables|Legume|Cereal Grains).*'"),
    ("gold", "exercise library is usable",
     "SELECT count(*) >= 800 AND count(*) FILTER (WHERE len(instructions) = 0) < 20, "
     "count(*) || ' exercises, ' || count(DISTINCT pattern) || ' movement patterns' FROM gold.exercise"),
    ("gold", "every chunk embedded and normalized",
     "SELECT count(*) FILTER (WHERE embedding IS NULL OR abs(list_dot_product(embedding, embedding) - 1) > 0.01) = 0, "
     "count(*) || ' chunks' FROM gold.chunk"),
    ("gold", "every rule backed by evidence",
     "SELECT count(*) = 0, CASE WHEN count(*) = 0 THEN 'all rules have a verbatim source passage' "
     "ELSE string_agg(rule_id, ', ') END FROM gold.rule r WHERE NOT EXISTS "
     "(SELECT 1 FROM gold.rule_evidence e WHERE e.rule_id = r.rule_id)"),
    ("gold", "usable recipes with resolved nutrition",
     "SELECT count(*) FILTER (WHERE usable) >= 250, count(*) FILTER (WHERE usable) || ' of ' || count(*) "
     "|| ' recipes have at least 85% of ingredients resolved to USDA foods and grams' FROM gold.recipe"),
]


def run(con, context: dict) -> list[dict]:
    results = []
    for layer, name, sql in CHECKS:
        try:
            passed, detail = con.sql(sql.format(**context)).fetchone()
        except Exception as e:  # noqa: BLE001 - a check that cannot run is a failed check
            passed, detail = False, f"error: {e}"
        results.append({"layer": layer, "check": name, "passed": bool(passed), "detail": str(detail)})
    return results
