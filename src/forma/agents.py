"""Forma's agent team, as one LangGraph graph run once per user message.

    Coach         routes each message, answers questions from the knowledge base, writes the summary
    Intake        builds your profile: goal, timeline, body metrics, training days, minutes, equipment
    Medical       screens conditions, medications and symptoms; sets constraints; reviews every plan
    Trainer       builds the periodized training plan within those constraints
    Nutritionist  sets calories and macros for each day from the plan and a sustainable rate of change
    Chef          turns your pantry into today's meals, with recipes and exact nutrition

    START -> coach -> intake | medical | pantry | answer
             intake/medical/pantry -> (profile complete?) -> constraints -> trainer -> nutritionist
                                       -> chef -> safety_review -(violations)-> trainer/chef (max 2 rounds)
                                                               -> summary -> END

Every step streams events (agent status, hand-offs between agents, tool calls, tokens) through
LangGraph's custom stream, which is what the live dashboard draws.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import chef, nutrition, planner, safety
from .config import settings
from .foods import food_tags
from .kb import KB
from .llm import LLM
from .models import REQUIRED, Citation, Constraints, Medical, Profile
from .understand import NO_WORDS, parse_medical, parse_profile, understand_medical, understand_profile

AGENTS = ["Coach", "Intake", "Medical", "Trainer", "Nutritionist", "Chef"]
FIELD_PHRASES = {"age": "your age", "sex": "your sex (the energy equations differ)", "height_cm": "your height (cm, or feet and inches)",
                 "weight_kg": "your weight (kg or lbs)", "goal": "your main goal", "days_per_week": "how many days a week you can train",
                 "minutes_per_session": "how long each session can be"}
BAD_VALUE = {"age": "an age of {}", "height_cm": "a height of {} cm", "weight_kg": "a weight of {} kg", "weeks": "a timeline of {} weeks",
             "days_per_week": "{} training days a week", "minutes_per_session": "{}-minute sessions"}
BARE_FIELDS = ("age", "height_cm", "days_per_week", "minutes_per_session")  # a lone number can answer one of these; weight needs a unit
BARE = re.compile(r"\s*(\d{1,3})\s*(?:years?(?: old)?|cm|days?(?: a week)?|times|x|min(?:ute)?s?)?\s*[.!]?\s*", re.IGNORECASE)
# "chicken breast", "brown rice", "peanut butter": a modifier and its food stay one item when a pantry list has no commas
FOOD_MODIFIERS = {"brown", "white", "greek", "sweet", "whole", "skim", "olive", "peanut", "bell", "green", "red", "ground", "cottage",
                  "soy", "almond", "coconut", "sour", "cream", "baby", "cherry", "rolled", "frozen", "canned"}
FOOD_HEADS = {"breast", "breasts", "thigh", "thighs", "wings", "sauce", "butter", "fillet", "fillets", "mince"}
PANTRY_FILLER = {"some", "a", "an", "the", "in", "my", "of", "with", "few", "bit", "lots", "lot", "fridge", "freezer", "pantry", "got"}
GREETING = ("Hi, I'm your Forma coach. I lead a small team of specialist agents: Intake, Medical, Trainer, Nutritionist "
            "and Chef. Everything we recommend comes from public health guidelines, FDA drug labels, USDA food data and "
            "peer-reviewed research, and every recommendation shows its source.\n\nTo start, tell me about your goal and "
            "your timeline, what kind of training you can do and how often, and your age, sex, height and weight.")
MEDICAL_ASK = ("Before anyone writes a plan, I check a few things, the way a certified trainer screens a new client:\n"
               "- Any medical conditions? (for example high blood pressure, diabetes, heart, kidney or joint problems)\n"
               "- Any medications you take regularly?\n"
               "- Any injuries, or chest pain, dizziness or unusual breathlessness when you exert yourself?\n"
               "- Are you pregnant? (if it applies)\n\nIf nothing applies, just say \"none\".")
PANTRY_ASK = ("Last thing. What's in your fridge and pantry right now? List whatever you have (for example: eggs, "
              "spinach, chicken, rice, Greek yogurt, oats, bananas) and I'll turn it into today's meals. "
              "Or say \"skip\" and I'll plan around common staples.")
SAMPLE_PANTRY = ["eggs", "spinach", "chicken breast", "brown rice", "broccoli", "greek yogurt", "oats", "blueberries",
                 "garlic", "soy sauce", "sweet potatoes", "onion", "bell pepper", "salmon", "lemon", "bread", "tofu", "lentils"]


class Session(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    messages: list[dict] = Field(default_factory=list)
    stage: Literal["intake", "medical", "pantry", "ready"] = "intake"
    profile: Profile = Field(default_factory=Profile)
    medical: Medical = Field(default_factory=Medical)
    pantry: list[str] | None = None
    constraints: Constraints | None = None
    plan: dict | None = None
    targets: dict | None = None
    meals: dict | None = None
    review: dict | None = None
    citations: list[Citation] = Field(default_factory=list)
    excluded_exercises: list[str] = Field(default_factory=list)  # swapped out by you, or vetoed by the Medical agent
    excluded_foods: list[str] = Field(default_factory=list)
    activity: list[dict] = Field(default_factory=list)  # the recent agent log, so a reloaded page can show it
    today: str = Field(default_factory=lambda: planner.WEEKDAYS[datetime.now(UTC).weekday()])

    def say(self, agent: str, text: str, quick: list[str] | None = None, kind: str = "message") -> dict:
        m = {"id": uuid.uuid4().hex[:8], "role": "agent", "agent": agent, "text": text, "quick": quick or [], "kind": kind}
        self.messages.append(m)
        return m

    def cite(self, cs: list[Citation]) -> list[int]:
        """Add citations (deduplicated) and return their 1-based numbers."""
        nums = []
        for c in cs:
            key = (c.source_id, c.snippet[:80])
            idx = next((i for i, x in enumerate(self.citations) if (x.source_id, x.snippet[:80]) == key), None)
            if idx is None:
                self.citations.append(c)
                idx = len(self.citations) - 1
            nums.append(idx + 1)
        return nums


class State(TypedDict, total=False):
    session: Session
    text: str
    route: str
    rounds: int
    violations: list[str]


def emit(kind: str, **data: Any) -> None:
    try:
        get_stream_writer()({"type": kind, "at": round(time.time(), 3), **data})
    except RuntimeError:  # not inside a streaming run (tests calling nodes directly)
        pass


def working(agent: str, summary: str) -> None:
    emit("agent", agent=agent, status="working", summary=summary)


def done(agent: str, summary: str, **extra: Any) -> None:
    emit("agent", agent=agent, status="done", summary=summary, **extra)


def handoff(src: str, dst: str, text: str) -> None:
    emit("handoff", source=src, target=dst, text=text)


async def speak(session: Session, agent: str, text: str, quick: list[str] | None = None) -> dict:
    m = session.say(agent, text, quick)
    emit("message", message=m)
    return m


def list_phrase(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def split_foods(phrase: str) -> list[str]:
    """"eggs bread milk bananas" -> four items; "chicken breast brown rice" -> two.
    ponytail: a word-list heuristic; an unlisted two-word food ("mung beans") splits in two."""
    words, out = phrase.split(), []
    for w in words:
        if out and (out[-1].split()[-1] in FOOD_MODIFIERS or w in FOOD_HEADS):
            out[-1] += " " + w
        else:
            out.append(w)
    return out


def merge_medical(m: Medical, got: dict) -> None:
    m.conditions = sorted(set(m.conditions) | set(got["conditions"]))
    m.medications = sorted(set(m.medications) | set(got["medications"]))
    m.unrecognized_medications = sorted(set(m.unrecognized_medications) | set(got.get("unrecognized", [])))
    m.symptoms = sorted(set(m.symptoms) | set(got["symptoms"]))
    if got.get("pregnant"):
        m.pregnant = True


def medical_found(m: Medical) -> list[str]:
    return [*(x.replace("_", " ") for x in m.conditions), *m.medications, *m.symptoms]


def build(kb: KB, llm: LLM):
    # ---- routing ---------------------------------------------------------------------------
    async def coach(state: State) -> dict:
        s, text = state["session"], state["text"].strip()
        s.messages.append({"id": uuid.uuid4().hex[:8], "role": "user", "agent": None, "text": text, "quick": [], "kind": "message"})
        low = text.lower()
        named_drug = any(kb.resolve_drug(w) for w in re.findall(r"[a-z]{5,}", low))
        health_news = bool(safety.detect_conditions(text) or safety.detect_symptoms(text))  # "I hurt my knee", "torn Achilles"
        new_health = bool(set(safety.detect_conditions(text)) - set(s.medical.conditions)
                          or set(safety.detect_symptoms(text)) - set(s.medical.symptoms))

        def food_list() -> bool:  # "eggs, chicken breast, rice": most parts are foods
            parts = [p for p in re.split(r",|\n|;|\band\b", low) if p.strip()]
            return len(parts) >= 2 and sum(bool(kb.foods.match(p)[0]) for p in parts) * 2 >= len(parts)
        if s.stage == "intake":
            route = "intake"
        elif s.stage == "medical":  # a correction ("actually I'm 32") goes back to Intake; anything else is screening
            correction = set(parse_profile(text)) & {*REQUIRED, "weeks", "target_change_kg"}  # not allergies: Medical takes those
            route = "intake" if correction and not (named_drug or health_news or NO_WORDS.search(text)) else "medical"
        elif s.stage == "pantry":
            route = "medical" if named_drug or health_news else "pantry"
        elif s.plan and (re.search(r"\b(swap|replace|instead of|don'?t like|hate|can'?t do)\b", low) or any(
                e["name"].lower() == low for d in s.plan["template"] for e in d["exercises"])):
            route = "swap"
        elif new_health:
            route = "medical"  # a new injury or condition: screen it and rebuild the plan
        elif re.search(r"\b(fridge|pantry|i have|cook|recipe|meals?|ingredients|dinner|lunch|breakfast|snack|tomorrow)\b", low) or food_list():
            route = "pantry"
        elif named_drug:
            route = "medical"  # a medication by name: screen it
        elif re.search(r"\b(weigh|now \d|days a week|times a week|minutes|new plan|change my plan|update)\b", low):
            route = "intake"
        elif s.plan and low.endswith("?"):
            route = "answer"  # "is this safe with my medication?" asks about the plan; it doesn't change the history
        elif re.search(r"\b(started|taking|prescribed|diagnosed|medication|pregnan\w*|doctor)\b", low):
            route = "medical"
        else:
            route = "answer"
        emit("route", agent="Coach", route=route)
        return {"route": route, "rounds": 0}

    # ---- intake -----------------------------------------------------------------------------
    async def intake(state: State) -> dict:
        s, text = state["session"], state["text"]
        working("Intake", "Reading your goals and stats")
        got = await understand_profile(llm, text)
        numeric = [f for f in s.profile.missing() if f in BARE_FIELDS]
        if not got and len(numeric) == 1 and (m := BARE.fullmatch(text)):
            got[numeric[0]] = int(m.group(1))  # a bare "25" answers the one number Intake asked for
        bad = []
        for k in list(got):  # keep every value that makes sense; name the ones that don't
            try:
                Profile(**{k: got[k]})
            except ValueError:
                bad.append(BAD_VALUE.get(k, k.replace("_", " ") + " {}").format(got.pop(k)))
        merged = s.profile.model_dump()
        for k, v in got.items():
            if k in merged and v not in (None, [], ""):
                merged[k] = v
        s.profile = Profile(**merged)
        if s.stage == "intake":  # conditions or medications mentioned up front go straight to Medical's notes
            merge_medical(s.medical, parse_medical(text, kb.resolve_drug))
        missing = s.profile.missing()
        done("Intake", f"Recorded {', '.join(sorted(got)) or 'nothing new'}", profile=s.profile.model_dump())
        doubt = f"I left out {list_phrase(bad)}, since that doesn't look right. " if bad else ""
        if missing:
            quick = ["Lose fat", "Build muscle", "Train for a race", "Get generally fitter"] if "goal" in missing else []
            lead = "Thanks! To build your plan I still need" if got else "Sorry, I didn't catch that. I still need"
            await speak(s, "Intake", f"{doubt}{lead} {list_phrase([FIELD_PHRASES[m] for m in missing])}.", quick)
            return {"route": "wait"}
        if s.stage == "intake":
            p = s.profile
            s.stage = "medical"
            if doubt:
                await speak(s, "Intake", doubt.strip())
            handoff("Intake", "Medical", f"Profile complete: {p.age}, {p.sex}, {p.height_cm:.0f} cm, {p.weight_kg:.0f} kg, "
                                         f"goal {p.goal.replace('_', ' ')}, {p.days_per_week} days a week. Please screen.")
            noted = medical_found(s.medical)
            ask = (f"You mentioned {list_phrase(noted)}; I've noted that. Anything else?\n\n" if noted else "") + MEDICAL_ASK
            await speak(s, "Medical", ask, ["None of these apply", "I have a condition or take medication"])
            return {"route": "wait"}
        if s.stage in ("medical", "pantry"):  # a correction mid-way: confirm it, then carry on with the step we were on
            changed = ", ".join(k.replace("_cm", "").replace("_kg", "").replace("_", " ") for k in sorted(got)) or "nothing"
            await speak(s, "Intake", f"{doubt}Updated: {changed}.")
            await speak(s, *(("Medical", "Now, any conditions, medications or injuries? Say \"none\" if nothing applies.")
                             if s.stage == "medical" else ("Chef", "Now, what's in your fridge and pantry? Or say \"skip\".")))
            return {"route": "wait"}
        if doubt:
            await speak(s, "Intake", doubt.strip())
        return {"route": "plan"}

    # ---- medical screening -------------------------------------------------------------------
    async def medical(state: State) -> dict:
        s = state["session"]
        working("Medical", "Screening conditions, medications and symptoms")
        got = await understand_medical(llm, state["text"], kb.resolve_drug)
        m = s.medical
        s.profile.allergies = sorted(set(s.profile.allergies) | set(got.get("allergies", [])))
        if not got["none"] and not (got["conditions"] or got["medications"] or got["symptoms"] or got["unrecognized"]
                                    or got.get("allergies")) and s.stage == "medical":
            await speak(s, "Medical", "Could you name the condition or medication? For example: \"type 2 diabetes, metformin\".")
            return {"route": "wait"}
        merge_medical(m, got)
        m.screened = True
        emit("tool", agent="Medical", tool="FDA label lookup",
             detail=", ".join(m.medications) or "no medications", count=len(m.medications))
        found = [*medical_found(m), *(f"{a.replace('_', ' ')} allergy" for a in got.get("allergies", []))]
        done("Medical", f"Screened: {', '.join(found) or 'nothing to flag'}", medical=m.model_dump())
        if s.stage in ("medical", "pantry"):  # from the pantry step: a late injury or medication, then the pantry again
            first = s.stage == "medical"
            s.stage = "pantry"
            note = (f"Noted: {list_phrase(found)}. I'll set the safety limits for the team when we plan." if found
                    else "Nothing to flag. Thanks.")
            if m.unrecognized_medications:
                note += (f" I don't have FDA label data for {list_phrase(m.unrecognized_medications)}, so please ask your "
                         "pharmacist whether it affects exercise or diet.")
            await speak(s, "Medical", note)
            handoff("Medical", "Chef", "Screening done. Ask about the pantry.")
            await speak(s, "Chef", PANTRY_ASK if first else "Got it. Now, what's in your fridge and pantry? Or say \"skip\".",
                        ["Skip for now", "Use a sample pantry"])
            return {"route": "wait"}
        return {"route": "plan"}

    # ---- pantry --------------------------------------------------------------------------------
    async def pantry(state: State) -> dict:
        s, text = state["session"], state["text"]
        working("Chef", "Reading your pantry")
        low = text.lower().strip()
        if "tomorrow" in low:
            s.today = planner.WEEKDAYS[(planner.WEEKDAYS.index(s.today) + 1) % 7]
        is_list = "," in low or "\n" in low or re.match(r"^(i have|i've got|we have|in my|my pantry|there'?s)", low) or s.stage == "pantry"
        skipped = bool(re.match(r"^(skip|no|none|nothing|later)\b", low))
        if "sample" in low or skipped:
            items = SAMPLE_PANTRY  # "skip" plans around common staples, as the question promised
        elif not is_list and s.pantry is not None:
            items = s.pantry  # a question about meals: keep the pantry, re-plan the meals
        else:
            body = re.sub(r"^(i have|i've got|we have|there'?s|in my (fridge|pantry)( is| are)?|my pantry has)\s*[:,-]?\s*", "", low)
            items = [i.strip(" .") for i in re.split(r",|\n|;|\band\b|\bplus\b", body) if 1 < len(i.strip(" .")) < 40]
            if len(items) == 1:  # "eggs bread milk bananas": no commas
                items = [w for w in split_foods(items[0]) if w not in PANTRY_FILLER]
        matched = [i for i in items if kb.foods.match(i)[0]]
        emit("tool", agent="Chef", tool="USDA FoodData Central", detail=f"matched {len(matched)} of {len(items)} items",
             count=len(matched))
        if not matched:
            done("Chef", "No foods recognised")
            await speak(s, "Chef", f"I couldn't find {list_phrase(items[:3] or [repr(text)])} in the USDA food data. Could you list "
                                   "a few ingredients, like \"eggs, rice, spinach\"? Or say \"skip\" and I'll use common staples.",
                        ["Skip for now", "Use a sample pantry"])
            return {"route": "wait"}
        s.pantry = items
        done("Chef", "No pantry; using common staples" if skipped else f"{len(items)} pantry items")
        s.stage = "ready"
        return {"route": "plan" if s.plan is None else "meals"}

    async def swap(state: State) -> dict:
        s, low = state["session"], state["text"].lower()
        names = [e["name"] for d in s.plan["template"] for e in d["exercises"]]
        hit = next((n for n in names if n.lower() in low), None)
        if not hit:
            await speak(s, "Trainer", "Which exercise would you like to swap?", sorted(set(names))[:8])
            return {"route": "wait"}
        s.excluded_exercises = sorted(set(s.excluded_exercises) | {hit})
        handoff("Coach", "Trainer", f"Swap out {hit}.")
        return {"route": "plan"}

    # ---- planning --------------------------------------------------------------------------------
    async def constraints(state: State) -> dict:
        s = state["session"]
        working("Medical", "Applying the rulebook to your answers")
        emit("tool", agent="Medical", tool="Rulebook", detail=f"{len(kb.rules)} evidence-backed rules", count=len(kb.rules))
        c = safety.constraints(kb, s.profile, s.medical)
        s.constraints = c
        for n in c.notes:
            n["refs"] = s.cite(n["citations"])
        ex_tags = ", ".join(sorted(c.avoid_exercise_tags)) or "none"
        handoff("Medical", "Trainer", f"Intensity cap: {c.max_intensity}; prescribe by {'RPE' if c.intensity == 'rpe' else 'RPE or heart rate'}; "
                                      f"exclude exercises tagged: {ex_tags}.")
        handoff("Medical", "Nutritionist", "No calorie deficit." if c.no_deficit else
                f"Deficit cap {c.max_deficit_pct or 25}% of maintenance" + (f"; at least {c.min_carb_g:.0f} g carbohydrate a day" if c.min_carb_g else "")
                + (f"; protein at most {c.protein_cap_g_per_kg} g/kg" if c.protein_cap_g_per_kg else "") + ".")
        handoff("Medical", "Chef", "Exclude foods tagged: " + (", ".join(sorted(c.avoid_food_tags | set(s.profile.allergies))) or "none")
                + (f"; keep {', '.join(x.replace('_ug', '').replace('_mg', '').replace('_', ' ') for x in c.steady_nutrients)} steady" if c.steady_nutrients else "") + ".")
        done("Medical", f"{len(c.notes)} rules apply" + (f"; clearance needed ({c.clearance})" if c.clearance != "none" else ""),
             constraints=c.model_dump(mode="json"))
        return {}

    async def trainer(state: State) -> dict:
        s = state["session"]
        working("Trainer", "Building the periodized plan")
        c = s.constraints
        weeks = s.profile.weeks or 12
        library = [e for e in kb.exercises if e["name"] not in s.excluded_exercises]
        p = planner.plan(library, s.profile, c, weeks)
        emit("tool", agent="Trainer", tool="Exercise library", detail=f"{len(kb.exercises)} exercises filtered by equipment, level and safety tags",
             count=len(kb.exercises))
        cites = kb.search("physical activity guidelines aerobic minutes per week muscle strengthening", 2, doc_prefix="pag")
        cites += kb.search("progressive overload repetitions load progression", 1, doc_prefix="rt-progressive")
        p["refs"] = s.cite(cites)
        s.plan = p
        done("Trainer", f"{p['strength_days']} strength days, {p['aerobic_minutes']} aerobic min/week, {weeks} weeks",
             plan=p)
        return {}

    async def nutritionist(state: State) -> dict:
        s = state["session"]
        working("Nutritionist", "Calculating calories and macros for each day")
        sessions = {}
        for d in s.plan["template"]:
            ss = []
            cardio = d["cardio"] or {}
            if d["type"] == "strength":
                ss.append({"modality": "strength", "minutes": d["minutes"] - (cardio.get("minutes") or 0), "intensity": d["intensity"]})
            if cardio:
                ss.append({"modality": cardio["modality"], "minutes": cardio["minutes"],
                           "intensity": "vigorous" if cardio["rpe"].startswith("7") else "moderate"})
            sessions[d["day"]] = ss
        t = nutrition.targets(s.profile, s.constraints, sessions, s.profile.weeks or 12)
        cites = [*kb.search("rate of weight loss per week percent body weight lean mass", 1, doc_prefix="issn-weight"),
                 *kb.search("protein intake grams per kilogram per day exercise", 1, doc_prefix="issn-protein"),
                 *kb.search("saturated fat percent of daily calories", 1, doc_prefix="dga")]
        t["refs"] = s.cite(cites)
        s.targets = t
        today = t["days"][s.today]
        emit("tool", agent="Nutritionist", tool="Mifflin-St Jeor + METs", detail=f"BMR {t['bmr']} kcal, maintenance ~{t['maintenance_avg_kcal']} kcal")
        handoff("Nutritionist", "Chef", f"Today ({s.today}): {today['kcal']} kcal, {today['protein_g']} g protein, "
                                        f"{today['carb_g']} g carbs, {today['fat_g']} g fat.")
        done("Nutritionist", f"Today {today['kcal']} kcal; {t['rate_kg_per_week']:+.2f} kg/week", targets=t)
        return {}

    async def chef_meals(state: State) -> dict:
        s = state["session"]
        working("Chef", f"Turning your pantry into {s.today}'s meals")
        today = s.targets["days"][s.today]
        pantry_items = [i for i in (s.pantry or []) if i not in s.excluded_foods]
        # Another day's meals: start breakfast and lunch from different mains than last time
        prior = {m["ingredients"][0]["fdc_id"] for m in (s.meals or {}).get("meals", [])[:2] if m["ingredients"]} \
            if state.get("route") == "meals" else set()
        out = chef.plan_day(kb.foods, kb.recipes, pantry_items, today, s.constraints, s.profile, prior)
        emit("tool", agent="Chef", tool="Recipe corpus + USDA", detail=f"{len(kb.recipes)} recipes, {len(out['meals'])} meals sized to targets")

        async def write(meal: dict) -> None:
            meal["steps"] = chef.template_steps(meal) if meal["source"] == "composed" else []
            meal["title"] = meal["title"] or title_for(meal)
            if not llm.available:
                return
            got = await llm.json(
                "You are a chef writing a short recipe for a home cook. Use only the listed ingredients and amounts, "
                "keep every ingredient, add nothing else except water, salt and pepper. Write 3-6 short steps.",
                json.dumps({"meal": meal["meal"], "technique": meal["technique"], "dish": meal["title"],
                            "ingredients": [f"{li['grams']} g {li['name']}" for li in meal["ingredients"]],
                            "draft_steps": meal["steps"]}),
                {"type": "object", "properties": {"title": {"type": "string"},
                                                  "steps": {"type": "array", "items": {"type": "string"}}},
                 "required": ["title", "steps"]})
            blocked = chef.blocked_tags(s.constraints, s.profile)
            if got and 2 <= len(got["steps"]) <= 8 and chef.ingredients_mentioned(got["steps"], meal) and not (
                    food_tags(" ".join(got["steps"]), "", {}) & blocked):  # the model may not slip in a blocked food
                meal["steps"], meal["title"], meal["written_by"] = got["steps"], got["title"][:70], "llm"
            else:
                meal["written_by"] = "template"

        await asyncio.gather(*(write(m) for m in out["meals"]))
        s.meals = out
        done("Chef", f"{len(out['meals'])} meals, {out['totals']['kcal']:.0f} kcal ({out['fit_pct']}% of target)", meals=out)
        return {}

    async def safety_review(state: State) -> dict:
        s = state["session"]
        working("Medical", "Reviewing the plan and meals against the constraints")
        problems = safety.review_plan(s.constraints, s.plan) + safety.review_meals(
            s.constraints, s.meals["meals"], s.profile.allergies, s.profile.diet)
        rounds = state.get("rounds", 0)
        s.review = {"passed": not problems, "problems": problems, "rounds": rounds + 1}
        if problems and rounds < 2:
            for d in s.plan["template"]:
                for e in d["exercises"]:
                    if set(e["tags"]) & s.constraints.avoid_exercise_tags:
                        s.excluded_exercises.append(e["name"])
            for m in s.meals["meals"]:
                s.excluded_foods += [li["name"] for li in m["ingredients"] if any(li["name"] in p for p in problems)]
            handoff("Medical", "Trainer", "Revise: " + "; ".join(problems[:3]))
            done("Medical", f"Sent back {len(problems)} issue(s)")
            return {"violations": problems, "rounds": rounds + 1}
        done("Medical", "Approved: every exercise and food passes" if not problems else f"{len(problems)} issue(s) remain",
             review=s.review)
        return {"violations": []}

    async def summary(state: State) -> dict:
        s = state["session"]
        if state.get("route") == "meals":  # a new pantry or another day: the meals, not the whole plan again
            day, m = s.targets["days"][s.today], s.meals
            await speak(s, "Chef", "\n".join([f"{s.today}: about {day['kcal']} kcal and {day['protein_g']} g protein.", *(
                f"- {x['meal']}: {x['title']} ({round(x['total']['kcal'])} kcal, {round(x['total']['protein_g'])} g protein)"
                for x in m["meals"]), "Recipes and timing notes are in the Meals tab."]), ["What can I cook tomorrow?"])
            done("Chef", f"{s.today}'s meals delivered")
            return {}
        working("Coach", "Writing your summary")
        p, t, c, meals = s.profile, s.targets, s.constraints, s.meals
        today = t["days"][s.today]
        tpl = s.plan["template"]
        facts = {
            "goal": p.goal.replace("_", " "), "weeks_you_asked_for": p.weeks, "sustainable_weeks_needed": t["weeks_needed"],
            "sustainable_note": t["notes"], "rate_kg_per_week": t["rate_kg_per_week"],
            "strength_days": [d["day"] for d in tpl if d["type"] == "strength"],
            "cardio_days": [f"{d['day']} ({d['name']})" for d in tpl if d["type"] == "cardio"],
            "easy_cardio_on_rest_days": [f"{d['day']} ({d['name']}, {d['cardio']['minutes']} min)" for d in tpl
                                         if d["type"] == "rest" and d["cardio"]],
            "training_days_per_week": sum(d["type"] in ("strength", "cardio") for d in tpl),
            "aerobic_minutes_per_week": s.plan["aerobic_minutes"], "today": s.today,
            "today_calorie_target": today["kcal"], "today_protein_target_g": today["protein_g"],
            "meals_planned_for_today": [m["title"] for m in meals["meals"]],
            "planned_meals_protein": f"{round(meals['totals']['protein_g'])} g of the {today['protein_g']} g target",
            "medical_notes": [n["message"] for n in c.notes], "clearance": c.clearance,
            "pre_exercise_checks": c.pre_exercise, "protein_gap_note": [n for n in meals["notes"] if "short" in n],
        }
        facts = {k: v for k, v in facts.items() if v not in ([], None, "")}  # an empty list reads as "none" to the model
        msg = {"id": uuid.uuid4().hex[:8], "role": "agent", "agent": "Coach", "text": "", "kind": "summary",
               "quick": ["Why this many calories?", "Swap an exercise", "What can I cook tomorrow?", "Is this safe with my medication?"]}
        emit("message_start", message=msg)
        text = ""
        async for tok in llm.stream(
                "You are Forma's head coach. Write a warm, specific summary of the plan your team built, in at most 160 "
                "words, second person, plain text with short paragraphs and no markdown headings. Use ONLY the facts given "
                "and copy numbers exactly; do not count or calculate anything yourself. Mention each medical note in one short "
                "sentence. The meals are a plan for today, not something already eaten. If clearance is 'required', say first "
                "and clearly to see a clinician before starting. If sustainable_weeks_needed is set, explain the goal takes "
                "that long at a sustainable rate. End by inviting questions.", json.dumps(facts)):
            text += tok
            emit("token", id=msg["id"], text=tok)
        if not text.strip():
            text = fallback_summary(s, facts)
            emit("token", id=msg["id"], text=text)
        msg["text"] = text.strip()
        s.messages.append(msg)
        emit("message_end", message=msg)
        done("Coach", "Plan delivered")
        return {}

    # ---- questions -----------------------------------------------------------------------------
    async def answer(state: State) -> dict:
        s, q = state["session"], state["text"]
        working("Coach", "Searching the knowledge base")
        passages = kb.search(q, 5)
        refs = s.cite(passages)
        emit("tool", agent="Coach", tool="Vector search", detail=f"{len(passages)} passages from {len({p.source_id for p in passages})} sources",
             count=len(passages))
        context = "\n\n".join(f"[{n}] {p.title} ({p.location or ''}): {p.snippet}" for n, p in zip(refs, passages, strict=True))
        plan_ctx = {}
        if s.targets:
            plan_ctx = {"how_todays_calories_were_set": s.targets["days"][s.today]["explain"],
                        "rate_kg_per_week": s.targets["rate_kg_per_week"], "why_that_rate": "0.5-1.0% of body weight per week, ISSN",
                        "protein": f"{s.targets['protein_g_per_kg']} g/kg of {s.targets['protein_basis_kg']} kg",
                        "today_calories": s.targets["days"][s.today]["kcal"], "bmr": s.targets["bmr"],
                        "maintenance_avg_kcal": s.targets["maintenance_avg_kcal"], "daily_adjust_kcal": s.targets["daily_adjust_kcal"],
                        "protein_g_per_kg": s.targets["protein_g_per_kg"], "medications": s.medical.medications,
                        "conditions": s.medical.conditions, "week": [f"{d['day']}: {d['name']}" for d in s.plan["template"]],
                        "medical_notes": [n["message"] for n in s.constraints.notes], "clearance": s.constraints.clearance}
        msg = {"id": uuid.uuid4().hex[:8], "role": "agent", "agent": "Coach", "text": "", "kind": "answer", "quick": [], "refs": refs}
        emit("message_start", message=msg)
        text = ""
        async for tok in llm.stream(
                "You are Forma's head coach, answering with the evidence given. Answer in at most 140 words, speaking to "
                "the person as 'you'. When the "
                "question is about the person's plan, explain it from the plan details first, copying numbers exactly. Cite "
                "evidence with its bracketed number exactly as given, like [3], only for claims it supports. Never advise "
                "changing a medication; for medical decisions, point to the person's clinician. If the evidence does not "
                "cover something, say so.",
                f"Question: {q}\n\nPlan details: {json.dumps(plan_ctx)}\n\nEvidence:\n{context}"):
            text += tok
            emit("token", id=msg["id"], text=tok)
        if not text.strip():
            text = rules_answer(s, q, refs, passages)
            emit("token", id=msg["id"], text=text)
        msg["text"] = re.sub(r"\[n(\d+)\]", r"[\1]", text.strip())  # normalise "[n3]" to "[3]"
        msg["refs"] = sorted({int(x) for x in re.findall(r"\[(\d+)\]", msg["text"])} & set(refs))
        s.messages.append(msg)
        emit("message_end", message=msg)
        done("Coach", f"Answered with {len(msg['refs'])} cited sources")
        return {}

    def after(state: State) -> str:
        return state["route"]

    g = StateGraph(State)
    for name, fn in (("coach", coach), ("intake", intake), ("medical", medical), ("pantry", pantry), ("answer", answer), ("swap", swap),
                     ("constraints", constraints), ("trainer", trainer), ("nutritionist", nutritionist), ("chef", chef_meals),
                     ("safety_review", safety_review), ("summary", summary)):
        g.add_node(name, fn)
    g.add_edge(START, "coach")
    g.add_conditional_edges("coach", after, ["intake", "medical", "pantry", "answer", "swap"])
    g.add_conditional_edges("swap", lambda st: "trainer" if st["route"] == "plan" else END, ["trainer", END])
    for n in ("intake", "medical"):
        g.add_conditional_edges(n, lambda st: "constraints" if st["route"] == "plan" else END, ["constraints", END])
    g.add_conditional_edges("pantry", lambda st: {"plan": "constraints", "meals": "chef"}.get(st["route"], END),
                            ["constraints", "chef", END])
    g.add_edge("constraints", "trainer")
    g.add_edge("trainer", "nutritionist")
    g.add_edge("nutritionist", "chef")
    g.add_edge("chef", "safety_review")
    g.add_conditional_edges("safety_review", lambda st: "trainer" if st.get("violations") else "summary", ["trainer", "summary"])
    g.add_edge("summary", END)
    g.add_edge("answer", END)
    return g.compile()


def rules_answer(s: Session, q: str, refs: list[int], passages: list) -> str:
    """Without the model: answer from the plan's own numbers and rules, then quote the two closest passages."""
    low, lead = q.lower(), []
    if s.targets:
        t, day = s.targets, s.targets["days"][s.today]
        if re.search(r"calorie|kcal|eat|energy|food", low):
            lead.append(f"Today's {day['kcal']} kcal: {day['explain']}.")
        if re.search(r"calorie|kcal|fast|rate|weeks|long|lose", low) and t["rate_kg_per_week"]:
            lead.append(f"That sets a change of {abs(t['rate_kg_per_week'])} kg a week, inside the 0.5-1.0% of body weight a "
                        "week that keeps muscle (ISSN)." + (f" {t['notes'][0]}" if t["notes"] else ""))
        if "protein" in low:
            lead.append(f"Protein: {day['protein_g']} g a day, {t['protein_g_per_kg']} g per kg of {t['protein_basis_kg']} kg.")
    if s.constraints and re.search(r"safe|medic|drug|pill|condition|risk|doctor", low):
        lead += [f"- {n['message']} " + "".join(f"[{r}]" for r in n.get("refs", [])) for n in s.constraints.notes] or [
            "Nothing in your answers changes the plan: no rule in the rulebook applies."]
        lead.append("Forma never changes medication; ask your clinician or pharmacist about anything not covered here.")
    quotes = [f"[{n}] {p.snippet[:220]}..." for n, p in list(zip(refs, passages, strict=True))[: 2 if lead else 3]]
    return "\n".join([*lead, "What the sources say:" if lead else "Here is what the sources say:", *quotes])


def title_for(meal: dict) -> str:
    n = meal["names"]
    main = n.get("protein") or n.get("morning_protein") or n.get("fruit") or "Snack"
    side = n.get("veg") or n.get("fruit") or n.get("starch")
    dish = {"stir_fry": "stir-fry", "tray_bake": "tray bake", "pasta": "pasta", "simmer": "stew", "skillet": "skillet",
            "bowl": "bowl", "scramble": "scramble", "oats": "oats", "yogurt_bowl": "bowl", "snack": "snack",
            "boiled_eggs": "boiled eggs"}.get(meal["technique"], "")
    if main.lower() in dish:
        return dish.capitalize()
    return " ".join(f"{main.capitalize()}{' & ' + side if side and side != main else ''} {dish}".split())


def fallback_summary(s: Session, f: dict) -> str:
    lines = []
    if f.get("clearance") == "required":
        lines.append("First: please see a clinician before starting. Until then the plan stays at light activity.")
    week = [f"strength on {', '.join(f['strength_days'])}" if f.get("strength_days") else "", *f.get("cardio_days", []),
            *f.get("easy_cardio_on_rest_days", [])]
    lines.append(f"Your week: {'; '.join(w for w in week if w)}. That's {f['aerobic_minutes_per_week']} aerobic minutes.")
    lines += f.get("sustainable_note", [])
    meals = f.get("meals_planned_for_today", [])
    lines.append(f"Today ({f['today']}) aim for about {f['today_calorie_target']} kcal and {f['today_protein_target_g']} g protein"
                 + (": " + ", ".join(meals) if meals else "") + ".")
    lines += f.get("medical_notes", [])[:3]
    lines.append("Ask me anything about the plan.")
    return "\n\n".join(lines)


ACTIVITY = {"agent", "handoff", "tool", "route"}
ACTIVITY_KEYS = ("type", "at", "agent", "status", "summary", "source", "target", "text", "tool", "detail", "route")


class Coach:
    """Holds sessions and runs the graph for each message, streaming events."""

    def __init__(self, kb: KB | None = None, llm: LLM | None = None):
        self.kb, self.llm = kb or KB(), llm or LLM()
        self.graph = build(self.kb, self.llm)
        self.sessions: dict[str, Session] = {}
        self.dir = settings().home / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)

    def new(self) -> Session:
        s = Session()
        s.say("Coach", GREETING, ["I want to lose fat", "I want to build muscle", "I'm training for a race"])
        self.sessions[s.id] = s
        self.save(s)
        return s

    def get(self, sid: str) -> Session | None:
        if sid in self.sessions:
            return self.sessions[sid]
        path = self.dir / f"{sid}.json"
        if path.exists():
            self.sessions[sid] = Session.model_validate_json(path.read_text())
            return self.sessions[sid]
        return None

    def save(self, s: Session) -> None:
        (self.dir / f"{s.id}.json").write_text(s.model_dump_json())

    async def send(self, s: Session, text: str):
        """Yield every event of one turn, then the final session state."""
        async for chunk in self.graph.astream({"session": s, "text": text}, stream_mode="custom"):
            if chunk["type"] in ACTIVITY:
                s.activity = [*s.activity, {k: chunk[k] for k in ACTIVITY_KEYS if k in chunk}][-80:]
            yield chunk
        self.save(s)
        yield {"type": "state", "session": view(s, self)}


def view(s: Session, coach: Coach) -> dict:
    return {**s.model_dump(mode="json"), "llm": {"model": coach.llm.model, "available": coach.llm.available}}
