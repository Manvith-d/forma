"""Medical safety: turn conditions, medications and symptoms into constraints, then check plans against them.

The medical agent never diagnoses and never touches medication. It applies the rulebook (each rule
backed by a verbatim FDA-label or guideline passage), decides whether a clinician should clear the
person first, and vetoes any exercise, intensity or food that breaks a constraint.
"""

from __future__ import annotations

import re

from .kb import KB
from .models import Constraints, Medical, Profile

CONDITIONS = {  # canonical id -> words people use
    "hypertension": ("high blood pressure", "hypertension", "hypertensive", "for blood pressure", "for my blood pressure",
                     "blood pressure medication", "blood pressure meds", "high bp"),
    "type2_diabetes": ("type 2 diabetes", "type ii diabetes", "t2d", "prediabetes", "diabetic"),
    "type1_diabetes": ("type 1 diabetes", "type i diabetes", "t1d"),
    "cardiovascular_disease": ("heart disease", "coronary", "heart attack", "stent", "heart failure", "arrhythmia", "afib",
                               "atrial fibrillation", "angina", "stroke", "bypass"),
    "kidney_disease": ("kidney disease", "ckd", "renal", "dialysis"),
    "knee_osteoarthritis": ("knee osteoarthritis", "knee arthritis", "arthritic knee", "osteoarthritis"),
    "knee_pain": ("knee pain", "bad knee", "knee injury", "knee surgery", "hurt my knee", "injured my knee", "knee hurts", "knees hurt", "sore knee", "knee aches", "knees ache"),
    "knee_ligament_tear": ("acl", "mcl", "pcl", "meniscus", "knee ligament", "torn ligament in my knee"),
    "acute_injury": ("torn", "tear", "ruptur", "fractur", "broke", "broken", "sprain", "strain", "dislocat", "injur",
                     "hurt my", "pulled a", "pulled my", "surgery", "post-op", "achilles", "rotator cuff"),
    "low_back_pain": ("back pain", "bad back", "herniated", "slipped disc", "sciatica", "lower back"),
    "osteoporosis": ("osteoporosis", "osteopenia", "low bone density"),
    "pregnancy": ("pregnant", "pregnancy", "expecting"),
    "eating_disorder_history": ("eating disorder", "anorexia", "bulimia", "binge eating"),
    "asthma": ("asthma",),
}
SYMPTOMS = ("chest pain", "chest pressure", "chest tightness", "faint", "passed out", "blackout", "dizzy", "dizziness",
            "short of breath", "shortness of breath", "breathless", "palpitations", "racing heart", "swollen ankles")
INTENSITY_ORDER = {"light": 0, "moderate": 1, "vigorous": 2}


def detect_conditions(text: str) -> list[str]:
    low = re.sub(r"\b(?:not|never been|no longer) pregnant\b|\bno pregnancy\b|\b(?:no|not|never|without|never had|haven'?t had)\s+"
                 r"(?:any\s+|recent\s+|major\s+)?(?:injur\w*|surger(?:y|ies)|fractures?|broken bones?|tears?)\b", "", text.lower())
    found = {cid for cid, words in CONDITIONS.items() if any(re.search(rf"\b{re.escape(w)}", low) for w in words)}
    if "knee_ligament_tear" in found:
        found.discard("acute_injury")  # the knee rule already covers the tear
    return sorted(found)


def detect_symptoms(text: str) -> list[str]:
    low = text.lower()
    negated = any(n in low for n in ("no chest", "never", "no symptoms", "none of", "no dizziness", "nope"))
    return [] if negated else sorted({s for s in SYMPTOMS if s in low})


def constraints(kb: KB, profile: Profile, medical: Medical) -> Constraints:
    c = Constraints()
    conditions = set(medical.conditions)
    if medical.symptoms:
        conditions.add("exertional_symptoms")
    if medical.pregnant:
        conditions.add("pregnancy")
    if profile.age and profile.age >= 65:
        conditions.add("older_adult")
    classes = {kb.resolve_drug(m)["drug_class"] for m in medical.medications if kb.resolve_drug(m)}
    generics = {kb.resolve_drug(m)["generic"] for m in medical.medications if kb.resolve_drug(m)}
    for rule in kb.rules:
        a = rule["applies"]
        if rule["kind"] == "medication":
            wanted = a["drug_class"] if isinstance(a["drug_class"], list) else [a["drug_class"]]
            hit = bool(classes & set(wanted)) and (not a.get("generics") or bool(generics & set(a["generics"])))
        elif rule["kind"] == "condition":
            wanted = a["condition"] if isinstance(a["condition"], list) else [a["condition"]]
            hit = bool(conditions & set(wanted))
        else:
            continue
        if hit:
            apply(c, rule)
    return c


def apply(c: Constraints, rule: dict) -> None:
    e = rule["effects"]
    c.avoid_food_tags |= set(e.get("avoid_food_tags", []))
    c.avoid_exercise_tags |= set(e.get("avoid_exercise_tags", []))
    if e.get("steady_nutrient"):
        c.steady_nutrients.add(e["steady_nutrient"])
    if e.get("spacing"):
        c.spacing.append({**e["spacing"], "rule_id": rule["rule_id"], "message": rule["message"]})
    if "min_carb_g" in e:
        c.min_carb_g = max(c.min_carb_g or 0, e["min_carb_g"])
    if "max_deficit_pct" in e:
        c.max_deficit_pct = min(c.max_deficit_pct or 100, e["max_deficit_pct"])
    if "protein_cap_g_per_kg" in e:
        c.protein_cap_g_per_kg = min(c.protein_cap_g_per_kg or 9, e["protein_cap_g_per_kg"])
    c.no_deficit |= bool(e.get("no_deficit"))
    c.hydration |= bool(e.get("hydration"))
    c.add_balance |= bool(e.get("add_balance"))
    if e.get("intensity") == "rpe":
        c.intensity = "rpe"
    if e.get("max_intensity") and INTENSITY_ORDER[e["max_intensity"]] < INTENSITY_ORDER[c.max_intensity]:
        c.max_intensity = e["max_intensity"]
    c.pre_exercise += [p for p in e.get("pre_exercise", []) if p not in c.pre_exercise]
    order = ["none", "vigorous", "required"]
    if e.get("clearance") and order.index(e["clearance"]) > order.index(c.clearance):
        c.clearance = e["clearance"]
    c.notes.append({"rule_id": rule["rule_id"], "message": rule["message"], "citations": rule["citations"]})


def review_plan(c: Constraints, plan: dict) -> list[str]:
    """Violations in a training plan (an empty list means it passes)."""
    problems = []
    for day in plan["template"]:
        for ex in day["exercises"]:
            bad = set(ex["tags"]) & c.avoid_exercise_tags
            if bad:
                problems.append(f"{day['name']}: {ex['name']} has {', '.join(sorted(bad))}")
        if INTENSITY_ORDER[day["intensity"]] > INTENSITY_ORDER[c.max_intensity]:
            problems.append(f"{day['name']}: {day['intensity']} intensity exceeds the {c.max_intensity} limit")
    return problems


def review_meals(c: Constraints, meals: list[dict], allergies: list[str], diet: list[str]) -> list[str]:
    problems = []
    blocked = c.avoid_food_tags | set(allergies) | ({"meat", "fish", "shellfish"} if "vegetarian" in diet or "vegan" in diet else set())
    if "vegan" in diet:
        blocked |= {"dairy", "egg"}
    if "pescatarian" in diet:
        blocked |= {"meat"}
    for m in meals:
        for ing in m["ingredients"]:
            bad = set(ing.get("tags", [])) & blocked
            if bad:
                problems.append(f"{m['title']}: {ing['name']} is {', '.join(sorted(bad))}")
    return problems
