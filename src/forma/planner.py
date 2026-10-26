"""The training planner: a periodized week-by-week plan from your days, minutes, equipment, goal and constraints.

Strength days follow full-body, upper/lower or push/pull/legs templates depending on frequency. Each slot
is a movement pattern filled from the exercise library, skipping anything the medical agent has ruled
out. Aerobic work is sized to the PAG target of 150-300 moderate minutes a week. Progression is
gradual: reps first, then load, volume rising every few weeks, with a lighter deload week every fourth.
Intensity is prescribed by perceived exertion (RPE, 0-10) and the talk test, so it works on beta-blockers too.
"""

from __future__ import annotations

from .models import Constraints, Profile

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SPREAD = {1: [0], 2: [0, 3], 3: [0, 2, 4], 4: [0, 1, 3, 4], 5: [0, 1, 2, 4, 5], 6: [0, 1, 2, 3, 4, 5], 7: list(range(7))}
CARDIO_DAYS = [5, 1, 3, 6, 2, 0, 4]  # Sat, Tue, Thu, Sun...: the days between lifting days first
EQUIPMENT = {
    "full_gym": None,  # anything
    "home_dumbbells": {"dumbbell", "body only", "kettlebells", "bands", "exercise ball", "medicine ball", "foam roll"},
    "bodyweight": {"body only"},
}
STAPLES = {  # most to least equipment; the first one allowed wins
    "squat": ["Barbell Squat", "Leg Press", "Goblet Squat", "Dumbbell Squat", "Bodyweight Squat"],
    "hinge": ["Romanian Deadlift", "Stiff-Legged Dumbbell Deadlift", "Kettlebell One-Legged Deadlift", "Butt Lift (Bridge)",
              "Single Leg Glute Bridge"],
    "lunge": ["Split Squat with Dumbbells", "Dumbbell Step Ups", "Dumbbell Lunges", "Bodyweight Walking Lunge",
              "Step-up with Knee Raise"],
    "push_horizontal": ["Barbell Bench Press - Medium Grip", "Dumbbell Bench Press", "Machine Bench Press", "Pushups",
                        "Incline Push-Up"],
    "push_vertical": ["Dumbbell Shoulder Press", "Standing Military Press", "Seated Cable Shoulder Press",
                      "Two-Arm Kettlebell Military Press", "Shoulder Press - With Bands", "Incline Push-Up"],
    "pull_horizontal": ["Seated Cable Rows", "One-Arm Dumbbell Row", "Bent Over Two-Dumbbell Row", "Inverted Row", "Superman"],
    "pull_vertical": ["Wide-Grip Lat Pulldown", "Close-Grip Front Lat Pulldown", "Band Assisted Pull-Up", "Chin-Up",
                      "One-Arm Dumbbell Row", "Superman"],
    "core": ["Dead Bug", "Pallof Press", "Plank", "Side Bridge", "Reverse Crunch"],
    "balance": ["Kettlebell One-Legged Deadlift", "Step-up with Knee Raise", "Single Leg Glute Bridge"],
}
TEMPLATES = {
    "full_a": ("Full body A", ["squat", "push_horizontal", "pull_horizontal", "hinge", "core"]),
    "full_b": ("Full body B", ["lunge", "push_vertical", "pull_vertical", "hinge", "core"]),
    "full_c": ("Full body C", ["squat", "push_vertical", "pull_horizontal", "lunge", "core"]),
    "upper_a": ("Upper body A", ["push_horizontal", "pull_horizontal", "push_vertical", "pull_vertical", "core"]),
    "lower_a": ("Lower body A", ["squat", "hinge", "lunge", "core"]),
    "upper_b": ("Upper body B", ["push_vertical", "pull_vertical", "push_horizontal", "pull_horizontal", "core"]),
    "lower_b": ("Lower body B", ["hinge", "lunge", "squat", "core"]),
}
SPLITS = {1: ["full_a"], 2: ["full_a", "full_b"], 3: ["full_a", "full_b", "full_c"],
          4: ["upper_a", "lower_a", "upper_b", "lower_b"], 5: ["upper_a", "lower_a", "full_c", "upper_b", "lower_b"]}
CARDIO = {
    "running": "Run at a conversational pace (you can talk, not sing). Beginners alternate 2 min running with 1 min walking.",
    "walking": "Brisk walk: breathing harder, but you can still hold a conversation.",
    "cycling": "Steady ride at a pace you could keep up for an hour.",
    "swimming": "Continuous easy laps in any stroke; rest at the wall when you need to and keep moving.",
    "rowing": "Steady rowing at an even stroke rate, driving through the legs.",
    "elliptical": "Steady elliptical at a moderate resistance.",
    "hiit": "Intervals: 30-60 s hard (RPE 8) then 90 s easy, repeated.",
    "yoga": "Flow-style yoga focused on mobility and breathing.",
}
CARDIO_MODALITIES = set(CARDIO) - {"yoga"}


def _allowed(ex: dict, p: Profile, c: Constraints) -> bool:
    eq = EQUIPMENT[p.equipment or "bodyweight"]
    levels = {"beginner": {"beginner"}, "intermediate": {"beginner", "intermediate"}}.get(p.experience or "beginner",
                                                                                         {"beginner", "intermediate", "expert"})
    return (eq is None or ex["equipment"] in eq) and ex["level"] in levels and not set(ex["tags"]) & c.avoid_exercise_tags


def pick(library: dict[str, dict], pattern: str, p: Profile, c: Constraints, used: set[str]) -> dict | None:
    for name in STAPLES[pattern]:
        ex = library.get(name)
        if ex and name not in used and _allowed(ex, p, c):
            return ex
    for name in STAPLES[pattern]:  # repeat a staple rather than leave the slot empty
        ex = library.get(name)
        if ex and _allowed(ex, p, c):
            return ex
    return None


def prescription(p: Profile, c: Constraints) -> tuple[int, str, str]:
    sets, reps, rpe = {"build_muscle": (3, "8-12", "7-8"), "endurance": (2, "12-20", "6-7")}.get(p.goal or "", (3, "10-15", "6-8"))
    if (p.experience or "beginner") == "beginner":
        sets = min(sets, 2)
    if c.max_intensity != "vigorous" or c.clearance != "none":
        rpe = "3-4" if c.max_intensity == "light" else "6-7"
    return sets, reps, rpe


def _slot(ex: dict, pattern: str, sets: int, reps: str, rpe: str) -> dict:
    core = pattern == "core"
    return {"exercise_id": ex["exercise_id"], "name": ex["name"], "pattern": pattern, "sets": 2 if core else sets,
            "reps": "30-45 s" if core else reps, "rpe": rpe, "equipment": ex["equipment"], "tags": ex["tags"],
            "muscles": ex["primary_muscles"], "how": ex["instructions"][:3], "image": ex["image"]}


def plan(library_rows: list[dict], p: Profile, c: Constraints, weeks: int) -> dict:
    library = {e["name"]: e for e in library_rows}
    days = p.days_per_week or 3
    minutes = p.minutes_per_session or 45
    cardio_modes = [m for m in p.modalities if m in CARDIO_MODALITIES] or ["walking"]
    wants_strength = "strength" in p.modalities or p.goal in ("build_muscle", "lose_fat", "general_fitness") or not p.modalities
    vigorous_ok = c.max_intensity == "vigorous" and c.clearance == "none"
    easy, easy_rpe = ("light", "3-4") if c.max_intensity == "light" else ("moderate", "5-6")  # light only until cleared
    strength_days = 0 if not wants_strength else min(days, {"build_muscle": 5, "endurance": 2}.get(p.goal or "", 3))
    if p.goal == "build_muscle" and days >= 2:
        strength_days = min(days, 5)
    split = SPLITS[max(1, min(strength_days, 5))] if strength_days else []
    sets, reps, rpe = prescription(p, c)
    used: set[str] = set()
    template = []
    # Strength days are spread out first (about 48 h between full-body sessions); cardio fills the days between
    lifting = SPREAD[strength_days] if strength_days else []
    slots = sorted(lifting + [d for d in CARDIO_DAYS if d not in lifting][:days - strength_days]) if lifting else SPREAD[min(days, 7)]
    cardio_slots = [d for d in slots if d not in lifting]
    for wd in slots:
        if wd in lifting:
            name, patterns = TEMPLATES[split[lifting.index(wd) % len(split)]]
            if c.add_balance:
                patterns = [*patterns[:-1], "balance", patterns[-1]]
            budget = max(2, (minutes - 10) // (sets * 3))  # ~3 min a set with rest, after a 10-min warm-up
            exercises = []
            for pat in patterns[:budget]:
                ex = pick(library, pat, p, c, used)
                if ex:
                    used.add(ex["name"])
                    exercises.append(_slot(ex, pat, sets, reps, rpe))
            spare = minutes - 10 - sum(e["sets"] * 3 for e in exercises)
            finisher = {"modality": cardio_modes[0], "minutes": spare, "rpe": easy_rpe, "how": CARDIO[cardio_modes[0]]} if spare >= 10 else None
            template.append({"day": WEEKDAYS[wd], "name": name, "type": "strength", "minutes": minutes,
                             "intensity": "vigorous" if rpe.startswith("7") and vigorous_ok else easy,
                             "warmup": "5 min easy cardio, then 2 light sets of the first exercise", "exercises": exercises,
                             "cardio": finisher})
        else:
            mode = cardio_modes[cardio_slots.index(wd) % len(cardio_modes)]
            hard = (vigorous_ok and p.goal in ("endurance", "lose_fat") and wd == slots[-1] and mode != "walking"
                    and "sprint" not in c.avoid_exercise_tags)  # intervals, but not "walking intervals"
            template.append({"day": WEEKDAYS[wd], "name": f"{mode.capitalize()} {'intervals' if hard else 'endurance'}",
                             "type": "cardio", "minutes": minutes, "intensity": "vigorous" if hard else easy,
                             "warmup": "5 min easy", "exercises": [],
                             "cardio": {"modality": mode, "minutes": minutes - 5, "rpe": "7-8" if hard else easy_rpe,
                                        "how": CARDIO["hiit"] if hard else CARDIO[mode]}})
    for wd in set(range(7)) - set(slots):
        template.append({"day": WEEKDAYS[wd], "name": "Rest or easy walk", "type": "rest", "minutes": 0, "intensity": "light",
                         "warmup": "", "exercises": [], "cardio": None})
    template.sort(key=lambda d: WEEKDAYS.index(d["day"]))

    if sum(d["type"] == "strength" for d in template) < 2:  # PAG: muscle-strengthening on 2+ days, so runners get a short block
        for d in [d for d in template if d["type"] == "cardio"][::2][:2]:
            d["exercises"] = [_slot(ex, pat, 2, "10-12", rpe) for pat in ("squat", "hinge", "core")
                              if (ex := pick(library, pat, p, c, used)) and not used.add(ex["name"])]
            d["name"] += " + strength"
            d["cardio"]["minutes"] = max(15, d["cardio"]["minutes"] - 15)

    def aerobic(day):  # moderate-equivalent minutes (vigorous counts double, PAG)
        cm = (day["cardio"] or {}).get("minutes", 0)
        return cm * (2 if day["intensity"] == "vigorous" and day["type"] == "cardio" else 1)

    aerobic_min = sum(aerobic(d) for d in template)
    walk = None
    if aerobic_min < 150:
        rest_days = [d for d in template if d["type"] == "rest"] or template
        per_day = -(-(150 - aerobic_min) // len(rest_days))
        walk = {"minutes_per_day": int(per_day), "days": [d["day"] for d in rest_days]}
        liked = next((m for m in cardio_modes if m not in ("walking", "hiit")), None) if easy != "light" else None  # every other day
        for i, d in enumerate(rest_days):
            mode = liked if liked and i % 2 == 0 and d["type"] == "rest" else "walking"
            d["cardio"] = {"modality": mode, "minutes": int(per_day), "rpe": "3-4" if easy == "light" else "4-5" if mode == "walking"
                           else "5-6", "how": CARDIO[mode]}
            d["name"] = (f"Easy {mode}" if mode != "walking" else "Brisk walk") if d["type"] == "rest" else d["name"]
        aerobic_min = sum(aerobic(d) for d in template)
    weekly = []
    for w in range(1, weeks + 1):
        deload = w % 4 == 0
        block = (w - 1) // 4
        weekly.append({"week": w, "phase": "Deload" if deload else ("Foundation" if w <= 2 else f"Build {block + 1}"),
                       "sets_change": -1 if deload else min(block, 2),
                       "cardio_minutes_pct": 60 if deload else min(100 + 10 * (w - 1 - block), 150),
                       "focus": "Lighter week: fewer sets, same technique, recover." if deload else
                                ("Learn the movements; leave 3-4 reps in reserve." if w <= 2 else
                                 "Add a rep each session until the top of the range, then add a little load.")})
    strength_count = sum(bool(d["exercises"]) for d in template)
    return {"weeks": weeks, "template": template, "weekly": weekly, "aerobic_minutes": int(aerobic_min),
            "strength_days": strength_count, "intensity_mode": "RPE and talk test",
            "pag": {"aerobic_ok": aerobic_min >= 150, "strength_ok": strength_count >= 2 or not wants_strength,
                    "balance": c.add_balance}, "walk_added": walk}
