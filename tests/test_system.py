"""End to end on the built warehouse (run `forma pipeline` first): the agent graph, the chef and the web API."""

import asyncio

import pytest

from forma.config import settings

if not settings().warehouse.exists():
    pytest.skip("no warehouse: run `forma pipeline` first", allow_module_level=True)

from fastapi.testclient import TestClient

from forma import chef
from forma.agents import Coach
from forma.api import app
from forma.kb import KB
from forma.llm import LLM
from forma.models import Constraints, Profile

TURNS = ["I'm 34, male, 5'10\" and 92 kg. I want to lose 12 kg in 4 months. I can lift 4 times a week for an hour. Beginner.",
         "I have high blood pressure and take lisinopril and Lipitor. My knee hurts sometimes.",
         "eggs, spinach, chicken breast, brown rice, broccoli, greek yogurt, oats, blueberries, garlic, grapefruit, salmon"]


@pytest.fixture(scope="module")
def coach(tmp_path_factory):
    c = Coach(KB(), LLM("none"))
    c.dir = tmp_path_factory.mktemp("sessions")
    return c


def converse(coach, s, text):
    async def go():
        return [e async for e in coach.send(s, text)]
    return asyncio.run(go())


def test_full_conversation(coach):
    s = coach.new()
    events = [e for t in TURNS for e in converse(coach, s, t)]
    agents = {e["agent"] for e in events if e["type"] == "agent"}
    assert {"Intake", "Medical", "Trainer", "Nutritionist", "Chef", "Coach"} <= agents
    assert {("Nutritionist", "Chef")} <= {(e["source"], e["target"]) for e in events if e["type"] == "handoff"}
    assert {"statin-grapefruit", "raas-potassium", "hypertension", "knee-osteoarthritis"} == {n["rule_id"] for n in s.constraints.notes}
    assert all(n["citations"] for n in s.constraints.notes)
    assert "grapefruit" in s.meals["excluded_for_safety"] and s.review["passed"]
    assert s.messages[-1]["kind"] == "summary" and s.messages[-1]["text"]
    before = {m["meal"]: m["title"] for m in s.meals["meals"]}
    tomorrow = converse(coach, s, "What can I cook tomorrow?")
    assert any(e["type"] == "message" and e["message"]["agent"] == "Chef" for e in tomorrow)
    assert {m["meal"]: m["title"] for m in s.meals["meals"]} != before  # another day leads with other mains
    safety = converse(coach, s, "Is this safe with my medication?")  # a question, not a new history: no re-plan
    assert not {e["agent"] for e in safety if e["type"] == "agent"} & {"Trainer", "Nutritionist", "Chef"}
    assert s.messages[-1]["kind"] == "answer" and "grapefruit" in s.messages[-1]["text"]


def test_a_bare_number_answers_the_age_question(coach):
    s = coach.new()
    converse(coach, s, "Male, 5'9, 79 kg. I want to build muscle, gym 4 days a week for an hour.")
    assert s.profile.missing() == ["age"]
    converse(coach, s, "25")
    assert s.profile.age == 25 and s.stage == "medical"


PROFILE = "I'm 30, male, 180 cm, 80 kg, want to lose fat, 3 days a week, 45 minutes"


def test_injuries_including_one_given_at_the_pantry_step(coach):
    s = coach.new()
    for text in [PROFILE, "I have a torn ACL and I haven't got a surgery for it yet.", "Injury : Torn Achilles"]:
        converse(coach, s, text)
    assert s.stage == "pantry" and not s.pantry  # the Achilles went to Medical, not into the pantry
    converse(coach, s, "Skip for now")
    assert {"knee-ligament-tear", "injury-clearance"} <= {n["rule_id"] for n in s.constraints.notes}
    assert s.constraints.clearance == "required" and s.meals["totals"]["kcal"] > 1200  # "skip" cooks from common staples


def test_an_unrecognised_pantry_is_asked_again(coach):
    s = coach.new()
    for text in [PROFILE, "none", "asdf qwerty"]:
        converse(coach, s, text)
    assert s.stage == "pantry" and s.plan is None
    converse(coach, s, "eggs bread milk bananas")
    assert len(s.pantry) == 4 and len(s.meals["meals"]) == 4 and s.meals["totals"]["kcal"] > 1200


def test_chef_closes_protein_gaps_and_flags_medication_timing():
    kb = KB()
    p = Profile(sex="female", weight_kg=72)
    c = Constraints(spacing=[{"tags": ["calcium_rich"], "hours": 4, "rule_id": "thyroid-spacing", "message": "Space it."}])
    day = {"kcal": 1450, "protein_g": 132}
    out = chef.plan_day(kb.foods, kb.recipes, ["greek yogurt", "eggs", "spinach", "chicken thighs", "quinoa", "zucchini", "oats",
                                               "raspberries", "almonds", "garlic", "chickpeas"], day, c, p)
    assert 85 <= out["fit_pct"] <= 110 and out["totals"]["protein_g"] >= 115
    timing = [m for m in out["meals"] if m.get("timing")]
    assert timing and all("greek yogurt" in t for m in timing for t in m["timing"])  # not the 10 g of garlic


def test_api_streams_a_turn(monkeypatch, tmp_path):
    monkeypatch.setenv("FORMA_LLM", "none")
    monkeypatch.setenv("FORMA_HOME", str(tmp_path))
    with TestClient(app) as client:
        assert client.get("/api/health").json()["knowledge"]["foods"] > 8000
        sid = client.post("/api/sessions").json()["id"]
        with client.websocket_connect(f"/api/sessions/{sid}/ws") as ws:
            ws.send_json({"text": TURNS[0]})
            kinds = []
            while (e := ws.receive_json())["type"] != "state":
                kinds.append(e["type"])
        assert {"route", "agent", "handoff", "message"} <= set(kinds)
        assert e["session"]["stage"] == "medical" and e["session"]["profile"]["weight_kg"] == 92
        assert client.get(f"/api/sessions/{sid}").json()["profile"]["age"] == 34
        assert client.get("/api/sessions/nope").status_code == 404


def test_chef_keeps_sauces_small_and_sodium_under_the_limit():
    kb = KB()
    pantry = ["eggs", "spinach", "chicken breast", "brown rice", "broccoli", "greek yogurt", "lentils", "soy sauce"]
    out = chef.plan_day(kb.foods, kb.recipes, pantry, {"kcal": 1870, "protein_g": 160}, Constraints(), Profile(weight_kg=92))
    sauce = [li["grams"] for m in out["meals"] for li in m["ingredients"] if li["name"] == "soy sauce"]
    assert sauce and max(sauce) <= 20  # the protein top-up once mistook soy sauce for a lean protein
    assert out["totals"]["sodium_mg"] <= chef.SODIUM_MAX
