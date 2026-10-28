"""Understanding what people type: deterministic parsers for numbers and keywords, the local LLM for the rest.

Numbers (age, height, weight, timeline, frequency) are parsed with rules, because a small model will
happily read "4 months" as 4 weeks. The LLM fills categorical fields the rules miss; when both speak,
the rules win. Everything here returns plain dicts of profile/medical fields.
"""

from __future__ import annotations

import itertools
import re

from .llm import LLM
from .safety import detect_conditions, detect_symptoms

MODALITY_WORDS = {
    "strength": ("lift", "weights", "gym", "strength", "resistance", "dumbbell", "barbell", "kettlebell"),
    "running": ("run", "jog", "5k", "10k", "marathon"), "walking": ("walk", "hike"), "cycling": ("cycl", "bike", "spin"),
    "swimming": ("swim", "pool", "laps"), "rowing": ("row",), "yoga": ("yoga", "pilates"), "hiit": ("hiit", "interval", "crossfit"),
}
ALLERGY_WORDS = {"dairy": ("dairy", "lactose", "milk"), "egg": ("egg",), "peanut": ("peanut",), "tree_nut": ("tree nut", "almond", "walnut", "cashew"),
                 "soy": ("soy",), "gluten": ("gluten", "celiac", "coeliac", "wheat"), "fish": ("fish",), "shellfish": ("shellfish", "shrimp", "prawn"),
                 "sesame": ("sesame",)}
NO_WORDS = re.compile(r"^\s*(no|nope|none|nothing|nah|n/a|not really|all good|i'?m healthy|healthy)\b", re.IGNORECASE)


UNITS = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"])}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
WEEKDAY = r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tues?|wed|thu(?:rs?)?|fri)s?\b"  # not "sat"/"sun"
# "3 days a week", "3 days in a week", "3-4 times per week", "3x/week", "4 sessions weekly"
FREQ = (r"\b([1-7])\s*(?:(?:-|to)\s*[1-7]\s*)?(?:\w+\s+)?(?:x|times|days|sessions|workouts)\s*"
        r"(?:(?:a|per|each|every|/|in\s+a|in\s+the)?\s*(?:week|wk)\b|weekly)|\b(?:once|twice|thrice)\s+(?:a|per|in\s+a)\s+week\b")


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _digits(t: str) -> str:
    """Number words to digits, so every rule below reads "twenty-five" and "five foot nine" like "25" and "5 foot 9"."""
    t = re.sub(rf"\b({'|'.join(TENS)})(?:[\s-]+({'|'.join(list(UNITS)[1:10])}))?\b",
               lambda m: str(TENS[m.group(1)] + UNITS.get(m.group(2) or "zero")), t)
    return re.sub(rf"\b({'|'.join(UNITS)})\b", lambda m: str(UNITS[m.group(1)]), t)


def parse_profile(text: str) -> dict:
    t = _digits(text.lower())
    out: dict = {}
    m = re.search(r"(?<!for )(?<!in )(?<!over )(?<!past )(?<!last )\b(\d{2})\s*(?:years?(?:\s*(?:old|of age))?|yrs?|yo|y/o)\b"
                  r"|\baged?\s*(?:is|of|:|=|-)?\s*(\d{2})\b"
                  r"|\b(?:i'?m|i am)\s+(?:a\s+)?(\d{2})(?=\s*(?:[,.!]|$|and\b|[mf]\b|male|female|man\b|woman|guy|girl))"
                  r"|\b(\d{2})\s*/?\s*[mf]\b|\b[mf]\s*[,/]\s*(\d{2})\b"
                  r"|\b(?:male|female|man|woman)\s*,\s*(\d{2})\b|(?:^|[.!]\s*)(\d{2})\s*,\s*(?:male|female|man|woman)\b", t)
    if m:
        out["age"] = int(next(g for g in m.groups() if g))
    age_span = m.span() if m else (0, 0)
    if re.search(r"\b(female|woman|girl|lady)\b|\b\d{2}\s*/?\s*f\b|\bf\s*[,/]\s*\d{2}\b|\b(?:sex|gender)\s*(?:is|:|-)?\s*f\b", t):
        out["sex"] = "female"
    elif re.search(r"\b(male|man|guy|boy)\b|\b\d{2}\s*/?\s*m\b|\bm\s*[,/]\s*\d{2}\b|\b(?:sex|gender)\s*(?:is|:|-)?\s*m\b", t):
        out["sex"] = "male"
    if (m := re.search(r"\b([4-7])[.,](\d{1,2})\s*(?:ft|feet|foot)\b", t)):  # "5.9 ft" means 5'9"; "5.75 ft" means 5¾ feet
        frac = m.group(2)
        inches = int(frac) if int(frac) < 12 else round(float("0." + frac) * 12)
        out["height_cm"] = round((int(m.group(1)) * 12 + inches) * 2.54)
    elif m := re.search(r"\b([4-7])\s*(?:'|’|ft(?![a-z])|feet|foot)\s*(\d{1,2})?", t):
        out["height_cm"] = round((int(m.group(1)) * 12 + int(m.group(2) or 0)) * 2.54)
    elif m := re.search(r"\b(1[2-9]\d|2[0-2]\d)\s*(?:cm|centimet)", t):
        out["height_cm"] = int(m.group(1))
    elif m := re.search(r"\b([12][.,]\d{1,2})\s*m(?:eters?|etres?)?\b", t):
        out["height_cm"] = round(_num(m.group(1)) * 100)
    elif m := re.search(r"\bheight\s*(?:is|:|of|=|-)?\s*(1[2-9]\d|2[0-2]\d)\b", t):
        out["height_cm"] = int(m.group(1))
    for m in re.finditer(r"\b(\d{1,3}(?:[.,]\d)?)\s*(kg|kgs|kilos?|kilograms?|lbs?|pounds|stone)\b", t):
        kg = _num(m.group(1)) * {"l": 0.4536, "p": 0.4536, "s": 6.35}.get(m.group(2)[0], 1)
        before = t[max(0, m.start() - 25):m.start()]
        if re.search(r"(lose|drop|shed|cut|gain|put on|add)\s*(?:about|around|like|roughly)?\s*$", before):
            out["target_change_kg"] = round(-kg if re.search(r"(lose|drop|shed|cut)", before) else kg, 1)
        elif 35 <= kg <= 300 and "weight_kg" not in out:
            out["weight_kg"] = round(kg, 1)
    # A timeline, once the age ("25 years"), frequencies ("3 days in a week") and history ("lifting for 2 years") are out of the way
    timeline = t[:age_span[0]] + " " + t[age_span[1]:]
    timeline = re.sub(FREQ + r"|\bper week\b|\b(?:been|for the (?:past|last))\b[^.,]*", "", timeline)
    if m := re.search(r"\b(\d{1,2}|a)\s*(weeks?|months?|years?)\b(?![\s-]*(?:old|ago|of age))", timeline):
        n = 1 if m.group(1) == "a" else int(m.group(1))
        out["weeks"] = round(n * {"w": 1, "m": 4.35, "y": 52}[m.group(2)[0]])
    if m := re.search(FREQ, t):
        out["days_per_week"] = int(m.group(1)) if m.group(1) else {"o": 1, "tw": 2, "th": 3}[
            next(k for k in ("o", "tw", "th") if m.group(0).startswith(k))]
    elif len(days := {d[:3] for d in re.findall(WEEKDAY, t)}) >= 2:  # "Mon, Wed and Fri"
        out["days_per_week"] = len(days)
    elif re.search(r"\b(?:every )?weekdays?\b|\bmonday to friday\b", t):
        out["days_per_week"] = 5
    elif re.search(r"\bweekends?\b", t):
        out["days_per_week"] = 2
    elif "every day" in t or "daily" in t:
        out["days_per_week"] = 6
    if m := re.search(r"\b([1-3])\s*(?:hours?|hrs?)\s*(?:and\s*)?(\d{2})\s*(?:mins?|minutes?)\b", t):
        out["minutes_per_session"] = int(m.group(1)) * 60 + int(m.group(2))
    elif m := re.search(r"\b(\d{2,3})\s*(?:-|to)?\s*(?:\d{2,3}\s*)?(?:mins?|minutes?)\b", t):
        out["minutes_per_session"] = int(m.group(1))
    elif re.search(r"\b(?:an?|1)\s*(?:hour|hr)s?\s*and\s*a\s*half\b|\b1[.,]5\s*(?:hours?|hrs?)\b", t):
        out["minutes_per_session"] = 90
    elif re.search(r"\bhalf an? hour\b|\bhalf hour\b", t):
        out["minutes_per_session"] = 30
    elif m := re.search(r"\b(an|1|2)[\s-]*(?:hours?|hrs?)\b", t):
        out["minutes_per_session"] = 120 if m.group(1) == "2" else 60
    if re.search(r"\b(lose|drop|shed|cut|slim|tone up|lean out)\b.*\b(weight|fat|kg|lbs?|pounds|belly)|\bweight loss\b|\bfat loss\b"
                 r"|\b(get lean|leaner|lean out|tone up|get toned|slim down|shred)\b", t):
        out["goal"] = "lose_fat"
    elif re.search(r"\b(build|gain|grow|bulk|put on)\b.*\b(muscle|mass|size|strength)|\bget (stronger|bigger|jacked)\b"
                   r"|\bbulk(?:ing)?\b", t):
        out["goal"] = "build_muscle"
    elif re.search(r"\b(marathon|half marathon|5k|10k|endurance|stamina|triathlon|run faster)\b", t):
        out["goal"] = "endurance"
    elif re.search(r"\b(healthier|get fit|fitter|stronger|in shape|general fitness|stay active|feel better|more energy|"
                   r"(?:stay|keep) (?:fit|healthy|in shape))\b", t):
        out["goal"] = "general_fitness"
    if "goal" not in out and out.get("target_change_kg"):
        out["goal"] = "lose_fat" if out["target_change_kg"] < 0 else "build_muscle"
    mods = sorted({m for m, words in MODALITY_WORDS.items() if any(w in t for w in words)})
    if mods:
        out["modalities"] = mods
    if re.search(r"\b(gym|machines|barbell)\b", t):
        out["equipment"] = "full_gym"
    elif re.search(r"\b(dumbbells?|kettlebells?|home gym|bands)\b", t):
        out["equipment"] = "home_dumbbells"
    elif re.search(r"\b(no equipment|bodyweight|body weight|at home|no gym)\b", t):
        out["equipment"] = "bodyweight"
    if re.search(r"\b(beginner|never|new to|just starting|first time|out of shape|haven'?t (?:worked out|trained))", t):
        out["experience"] = "beginner"
    elif re.search(r"\b(advanced|years of (?:lifting|training)|compete|competitive)\b", t):
        out["experience"] = "advanced"
    elif re.search(r"\b(intermediate|a year of (?:lifting|training)|couple of years|some experience|"
                   r"(?:lifting|training) for (?:two|three|2|3|a few) years)\b", t):
        out["experience"] = "intermediate"
    if re.search(r"\b(desk job|office job|sit all day|sedentary|work from home)\b", t):
        out["activity_level"] = "sedentary"
    elif re.search(r"\b(on my feet|nurse|retail|walk a lot|active job|construction|warehouse)\b", t):
        out["activity_level"] = "moderate"
    diet = [d for d in ("vegan", "vegetarian", "pescatarian") if d in t]
    if diet:
        out["diet"] = diet[:1]
    allergies = sorted({a for a, words in ALLERGY_WORDS.items() if any(
        re.search(rf"(allerg\w*|intoleran\w*|can'?t (?:eat|have)|no|avoid|free)\W+(?:\w+\W+){{0,3}}{w}", t)
        or re.search(rf"{w}\w*[- ](?:free|allerg\w*|intoleran\w*)", t) for w in words)})
    if allergies:
        out["allergies"] = allergies
    return out


def parse_medical(text: str, resolve_drug) -> dict:
    out = {"conditions": detect_conditions(text), "symptoms": detect_symptoms(text), "medications": [], "unrecognized": []}
    low = text.lower()
    words = re.findall(r"[a-z][a-z\-]{3,}", low)
    for w in dict.fromkeys(words + [" ".join(p) for p in itertools.pairwise(words)]):
        d = resolve_drug(w)
        if d and d["generic"] not in out["medications"]:
            out["medications"].append(d["generic"])
    out["pregnant"] = True if "pregnan" in low and not re.search(r"not pregnant|no pregnan", low) else None
    said_none = NO_WORDS.search(text) or re.search(r"\bno (?:medications?|meds|conditions|health (?:issues|problems))\b", low)
    out["none"] = bool(said_none) and not (out["conditions"] or out["medications"] or out["symptoms"])
    out["allergies"] = parse_profile(text).get("allergies", [])  # people often mention allergies here
    return out


PROFILE_SCHEMA = {"type": "object", "properties": {
    "goal": {"type": ["string", "null"], "enum": ["lose_fat", "build_muscle", "endurance", "general_fitness", None]},
    "sex": {"type": ["string", "null"], "enum": ["male", "female", None]},
    "modalities": {"type": "array", "items": {"type": "string", "enum": list(MODALITY_WORDS)}},
    "equipment": {"type": ["string", "null"], "enum": ["full_gym", "home_dumbbells", "bodyweight", None]},
    "experience": {"type": ["string", "null"], "enum": ["beginner", "intermediate", "advanced", None]},
    "name": {"type": ["string", "null"]}}, "required": ["goal", "modalities"]}
MEDICAL_SCHEMA = {"type": "object", "properties": {
    "medications": {"type": "array", "items": {"type": "string"}, "description": "medication names exactly as written"},
    "conditions": {"type": "array", "items": {"type": "string"}, "description": "medical conditions or injuries as written"}},
    "required": ["medications", "conditions"]}


async def understand_profile(llm: LLM, text: str) -> dict:
    rules = parse_profile(text)
    if llm.available:
        got = await llm.json("Extract the person's fitness profile. Use null for anything the message does not state. Set goal "
                             "to lose_fat only if they mention losing weight or fat, build_muscle only for muscle or size, "
                             "endurance only for races or endurance; otherwise general_fitness.", text, PROFILE_SCHEMA)
        for k, v in (got or {}).items():
            if v not in (None, [], "") and k not in rules:
                rules[k] = v
    return rules


async def understand_medical(llm: LLM, text: str, resolve_drug) -> dict:
    out = parse_medical(text, resolve_drug)
    if llm.available and not out["none"]:
        got = await llm.json("List every medication and medical condition the person mentions. Do not add any they do not mention.",
                             text, MEDICAL_SCHEMA) or {}
        for name in got.get("medications", []):
            d = resolve_drug(name)
            if d and d["generic"] not in out["medications"]:
                out["medications"].append(d["generic"])
            elif not d and name.lower() in text.lower() and name not in out["unrecognized"]:
                out["unrecognized"].append(name)  # mentioned, but not a medication Forma screens: flag it for a clinician
        for cond in got.get("conditions", []):
            out["conditions"] = sorted(set(out["conditions"]) | set(detect_conditions(cond)))
    return out
