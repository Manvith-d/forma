"""Fast checks with no warehouse: parsing, food tags, energy targets, planning and the rulebook's effects."""

import itertools

import pytest

from forma import planner
from forma.foods import food_tags
from forma.models import Constraints, Profile
from forma.nutrition import bmr, targets
from forma.safety import apply, detect_conditions, detect_symptoms, review_plan
from forma.understand import parse_medical, parse_profile


@pytest.mark.parametrize("text, expected", [
    ("I'm 29, female, 165 cm and 72 kg. I want to lose about 6 kg in 3 months. Gym 3 times a week for 45 minutes.",
     {"age": 29, "sex": "female", "height_cm": 165, "weight_kg": 72, "target_change_kg": -6, "weeks": 13, "days_per_week": 3,
      "minutes_per_session": 45, "goal": "lose_fat"}),
    ("hey! 35M, 6ft 1, 210 lbs. trying to drop 25 pounds, four days a week, ~50 min",
     {"age": 35, "sex": "male", "height_cm": 185, "weight_kg": 95.3, "days_per_week": 4, "minutes_per_session": 50}),
    ("I want to build muscle, 4 days a week for an hour", {"days_per_week": 4, "minutes_per_session": 60, "weeks": None}),
    ("28 year old, I can lift 3x/week", {"age": 28, "days_per_week": 3, "weeks": None}),  # "year old" is not a timeline
    ("I've been lifting for two years, 5 days a week", {"weeks": None, "experience": "intermediate"}),
    ("Female, 42, lose ~10 lbs over 12 weeks, 3 times weekly", {"age": 42, "weeks": 12, "days_per_week": 3}),
    ("38, female, half marathon in 16 weeks", {"age": 38, "goal": "endurance", "weeks": 16}),
    ("I want to stay fit, I'm lactose intolerant", {"goal": "general_fitness", "allergies": ["dairy"]}),
    ("Age is 25, Sex is male and weight is 79 kgs. Height is 5'9 in feet", {"age": 25, "sex": "male", "weight_kg": 79.0, "height_cm": 175}),
    ("I can train for 3 days in a week and each session can be 1 hour long and my age is 25 years",
     {"days_per_week": 3, "minutes_per_session": 60, "age": 25, "weeks": None}),  # not a 1-week or 25-year timeline
    ("age :25 years", {"age": 25, "weeks": None}), ("I'm 25", {"age": 25}), ("twenty-five years old", {"age": 25}),
    ("M, 28", {"age": 28, "sex": "male"}), ("Sex: F", {"sex": "female"}),
    ("I'm 30, male, my wife wants to join but her knees hurt", {"sex": "male"}),
    ("5.9 ft", {"height_cm": 175}), ("5ft9", {"height_cm": 175}), ("five foot nine", {"height_cm": 175}), ("height is 175", {"height_cm": 175}),
    ("79 kilograms", {"weight_kg": 79.0}), ("12 stone", {"weight_kg": 76.2}),
    ("Mon, Wed and Fri", {"days_per_week": 3}), ("3-4 times a week", {"days_per_week": 3}), ("thrice a week", {"days_per_week": 3}),
    ("weekends only", {"days_per_week": 2}), ("an hour and a half", {"minutes_per_session": 90}), ("1hr", {"minutes_per_session": 60}),
    ("get lean", {"goal": "lose_fat"}), ("I want to bulk", {"goal": "build_muscle"}),
    ("I've been lifting for 10 years", {"age": None, "weeks": None}),
])
def test_parse_profile(text, expected):
    got = parse_profile(text)
    for k, v in expected.items():
        assert (got.get(k) == pytest.approx(v, abs=1) if isinstance(v, float) else got.get(k) == v), (k, got)


def test_parse_medical_resolves_brands_and_negations():
    drugs = {"lipitor": "atorvastatin", "synthroid": "levothyroxine"}
    got = parse_medical("I take Lipitor and Synthroid, my knee hurts. No chest pain or dizziness.",
                        lambda w: {"generic": drugs[w]} if w in drugs else None)
    assert sorted(got["medications"]) == ["atorvastatin", "levothyroxine"]
    assert got["conditions"] == ["knee_pain"] and got["symptoms"] == [] and not got["none"]
    assert parse_medical("I'm allergic to peanuts. No medications.", lambda w: None)["allergies"] == ["peanut"]
    assert detect_symptoms("I get chest tightness on stairs") == ["chest tightness"]
    assert detect_conditions("on atenolol for blood pressure") == ["hypertension"]


@pytest.mark.parametrize("text, expected", [
    ("I have a torn ACL and I haven't got a surgery for it yet.", ["knee_ligament_tear"]),  # a tear, not knee pain
    ("Injury : Torn Achilles", ["acute_injury"]), ("I broke my wrist last month", ["acute_injury"]),
    ("sprained my ankle", ["acute_injury"]), ("I hurt my knee yesterday", ["acute_injury", "knee_pain"]),
    ("no injuries, no surgery", []), ("I'm not pregnant", []), ("I work at Oracle", []), ("adrenal fatigue", []),
])
def test_detect_conditions(text, expected):
    assert detect_conditions(text) == expected


@pytest.mark.parametrize("name, category, per100, has, lacks", [
    ("Peanut butter, smooth style", "Legumes", {}, {"peanut"}, {"dairy"}),
    ("Beverages, almond milk, unsweetened", "Beverages", {}, {"tree_nut"}, {"dairy"}),
    ("Eggplant, raw", "Vegetables", {}, set(), {"egg"}),
    ("Rice flour, white", "Cereal Grains", {}, set(), {"gluten"}),
    ("Yogurt, Greek, plain, nonfat", "Dairy", {"calcium_mg": 111, "kcal": 61}, {"dairy", "calcium_rich"}, set()),
    ("Garlic, raw", "Vegetables", {"calcium_mg": 181, "kcal": 149}, {"calcium_rich"}, set()),  # by density; meals skip garnishes
    ("Nuts, almonds", "Nuts", {"calcium_mg": 269, "kcal": 579}, {"tree_nut"}, {"calcium_rich"}),
    ("Grapefruit, raw, pink", "Fruits", {}, {"grapefruit"}, set()),
    ("Cheese, cheddar", "Dairy", {}, {"dairy", "tyramine"}, set()),
])
def test_food_tags(name, category, per100, has, lacks):
    tags = food_tags(name, category, per100)
    assert has <= tags and not lacks & tags, tags


def profile(**kw) -> Profile:
    base = {"age": 30, "sex": "female", "height_cm": 165, "weight_kg": 70, "goal": "lose_fat", "target_change_kg": -10,
            "weeks": 8, "days_per_week": 3, "minutes_per_session": 45, "modalities": ["strength"], "equipment": "full_gym",
            "experience": "beginner", "activity_level": "sedentary"}
    return Profile(**base | kw)


def test_targets_cap_the_rate_and_respect_the_floor():
    p = profile()
    t = targets(p, Constraints(), {d: [] for d in planner.WEEKDAYS}, p.weeks)
    assert abs(t["rate_kg_per_week"]) <= 0.01 * p.weight_kg  # 10 kg in 8 weeks is too fast
    assert t["weeks_needed"] > p.weeks and t["notes"]
    assert all(d["kcal"] >= max(bmr(p), 1200) - 10 for d in t["days"].values())
    assert targets(p, Constraints(no_deficit=True), {d: [] for d in planner.WEEKDAYS}, 8)["daily_adjust_kcal"] == 0
    assert targets(p, Constraints(protein_cap_g_per_kg=0.8), {d: [] for d in planner.WEEKDAYS}, 8)["protein_g_per_kg"] == 0.8


def library(tagged: dict[str, list[str]] | None = None) -> list[dict]:
    names = {n for staples in planner.STAPLES.values() for n in staples}
    return [{"exercise_id": n, "name": n, "equipment": "barbell", "level": "beginner", "tags": (tagged or {}).get(n, []),
             "primary_muscles": [], "instructions": ["Do it."], "image": None} for n in sorted(names)]


def test_plan_skips_unsafe_exercises_and_meets_pag():
    c = Constraints(avoid_exercise_tags={"heavy_spinal_load"})
    plan = planner.plan(library({"Barbell Squat": ["heavy_spinal_load"]}), profile(), c, 12)
    names = [e["name"] for d in plan["template"] for e in d["exercises"]]
    assert "Barbell Squat" not in names and "Leg Press" in names
    assert plan["aerobic_minutes"] >= 150 and plan["strength_days"] >= 2 and not review_plan(c, plan)
    assert [w["phase"] for w in plan["weekly"]][3] == "Deload"


def test_plan_stays_light_until_cleared_and_runners_get_strength():
    c = Constraints(max_intensity="light", clearance="required")
    plan = planner.plan(library(), profile(modalities=["running", "strength"]), c, 12)
    assert all(d["intensity"] == "light" for d in plan["template"] if d["type"] != "rest") and not review_plan(c, plan)
    assert all((d["cardio"] or {}).get("modality") in (None, "walking") for d in plan["template"] if d["type"] == "rest")
    week = planner.plan(library(), profile(days_per_week=5, modalities=["strength", "swimming"]), Constraints(), 12)["template"]
    lifting = [planner.WEEKDAYS.index(d["day"]) for d in week if d["type"] == "strength"]
    assert len(lifting) == 3 and all(b - a >= 2 for a, b in itertools.pairwise(lifting))  # 48 h apart
    runner = planner.plan(library(), profile(goal="endurance", modalities=["running"], days_per_week=4), Constraints(), 16)
    assert runner["strength_days"] == 2 and sum(d["type"] == "cardio" for d in runner["template"]) == 4


def test_rules_combine_to_the_strictest_limit():
    c = Constraints()
    apply(c, {"rule_id": "a", "message": "", "citations": [], "effects": {"clearance": "vigorous", "max_intensity": "moderate"}})
    apply(c, {"rule_id": "b", "message": "", "citations": [], "effects": {"clearance": "required", "max_intensity": "light",
                                                                          "avoid_food_tags": ["grapefruit"]}})
    apply(c, {"rule_id": "c", "message": "", "citations": [], "effects": {"clearance": "none", "max_intensity": "vigorous"}})
    assert (c.clearance, c.max_intensity, c.avoid_food_tags) == ("required", "light", {"grapefruit"})
    assert [n["rule_id"] for n in c.notes] == ["a", "b", "c"]


def test_a_refused_page_falls_back_to_the_archive(monkeypatch):
    import httpx

    from forma.pipeline import bronze

    monkeypatch.setattr(bronze.time, "sleep", lambda s: None)
    page = "https://ods.od.nih.gov/factsheets/VitaminK-HealthProfessional/"

    def web(r):
        if r.url.host == "ods.od.nih.gov" or "/20991231id_/" in r.url.path:  # refused, and the newest capture is a 403
            return httpx.Response(403)
        if r.url.path == "/cdx/search/cdx":  # the index lists the newest capture that returned 200
            return httpx.Response(200, json=[["urlkey", "timestamp", "original"], ["x", "20261015065630", page]])
        return httpx.Response(200, content=b"ok")

    with httpx.Client(transport=httpx.MockTransport(web)) as client:
        r = bronze._page(client, page)
    assert r.content == b"ok" and str(r.url) == f"{bronze.WAYBACK}/20261015065630id_/{page}"
