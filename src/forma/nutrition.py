"""Energy and macronutrient targets: a sustainable rate of change, calories per day, protein, fat and carbohydrate.

    BMR      Mifflin-St Jeor: 10*kg + 6.25*cm - 5*age + 5 (men) or - 161 (women)
    TDEE     BMR * lifestyle factor (outside training) + the net energy of that day's planned session (METs)
    Change   at most 0.5-1.0% of body weight per week when losing (ISSN), a small surplus when building muscle
    Floors   intake never below BMR, nor 1,200 / 1,500 kcal (women / men); deficit at most 25% of maintenance
    Protein  1.6 g/kg, 2.0 g/kg in a deficit (ISSN: 1.4-2.0 minimum, more when restricting energy)
    Fat      at least 25% of energy; saturated fat under 10% (DGA)
"""

from __future__ import annotations

import math

from .models import Constraints, Profile

LIFESTYLE = {"sedentary": 1.2, "light": 1.3, "moderate": 1.4, "very": 1.5}
KCAL_PER_KG = 7700  # energy in a kilogram of body-weight change, the usual planning figure
METS = {  # net METs above rest, by modality and intensity (moderate, vigorous)
    "strength": (3.5, 5.0), "running": (7.0, 9.8), "walking": (3.5, 4.3), "cycling": (6.8, 8.0), "swimming": (5.8, 8.3),
    "rowing": (6.0, 8.5), "yoga": (2.5, 3.0), "hiit": (6.0, 8.0), "elliptical": (5.0, 7.0), "mobility": (2.3, 2.5),
}


def bmr(p: Profile) -> float:
    return 10 * p.weight_kg + 6.25 * p.height_cm - 5 * p.age + (5 if p.sex == "male" else -161)


def session_kcal(p: Profile, modality: str, minutes: int, intensity: str) -> float:
    mets = METS.get(modality, METS["strength"])[1 if intensity == "vigorous" else 0]
    return (mets - 1) * p.weight_kg * minutes / 60  # net of the resting energy already in BMR


def targets(p: Profile, c: Constraints, sessions_by_day: dict[str, list[dict]], weeks: int) -> dict:
    """Daily calories and macros for every weekday, plus the sustainable timeline."""
    base = bmr(p) * LIFESTYLE.get(p.activity_level or "light", 1.3)
    floor = max(bmr(p), 1500 if p.sex == "male" else 1200)
    day_tdee = {d: base + sum(session_kcal(p, s["modality"], s["minutes"], s["intensity"]) for s in ss)
                for d, ss in sessions_by_day.items()}
    weekly_tdee = sum(day_tdee.values()) / 7
    notes, change_kg, rate = [], p.target_change_kg, 0.0
    goal = p.goal
    if c.no_deficit and goal == "lose_fat":
        goal = "general_fitness"
        notes.append("No calorie deficit: your medical answers rule one out, so the targets are set for maintenance.")
    if goal == "lose_fat":
        change_kg = -abs(change_kg or 0.05 * p.weight_kg)
        wanted = abs(change_kg) / max(weeks, 1)
        rate = min(max(wanted, 0.005 * p.weight_kg), 0.01 * p.weight_kg)  # 0.5-1.0% of body weight per week
        max_pct = min(25, c.max_deficit_pct or 25) / 100
        deficit = min(rate * KCAL_PER_KG / 7, max_pct * weekly_tdee, weekly_tdee - floor)
        rate = max(deficit, 0) * 7 / KCAL_PER_KG
        weeks_needed = math.ceil(abs(change_kg) / rate) if rate > 0 else None
        if weeks_needed and weeks_needed > weeks:
            notes.append(f"Losing {abs(change_kg):.1f} kg in {weeks} weeks would need more than 1% of body weight a week. "
                         f"At a sustainable {rate:.2f} kg a week it takes about {weeks_needed} weeks.")
        adjust = -deficit
    elif goal == "build_muscle":
        adjust, rate = 0.08 * weekly_tdee, 0.0025 * p.weight_kg  # ~8% surplus, ~0.25% of body weight a week
        weeks_needed = None
    else:
        adjust, weeks_needed = 0.0, None
    protein_per_kg = 2.0 if adjust < 0 else 1.6
    if c.protein_cap_g_per_kg:
        protein_per_kg = min(protein_per_kg, c.protein_cap_g_per_kg)
    # In a deficit, protein is scaled to the goal weight: per-kg targets on total weight overshoot for larger bodies
    basis_kg = p.weight_kg + change_kg if goal == "lose_fat" and change_kg else p.weight_kg
    protein = round(protein_per_kg * basis_kg)
    days = {}
    for d, tdee in day_tdee.items():
        kcal = max(floor, tdee + adjust)
        fat = max(0.25 * kcal / 9, 0.6 * p.weight_kg)
        carbs = max((kcal - protein * 4 - fat * 9) / 4, c.min_carb_g or 0)
        kcal = protein * 4 + fat * 9 + carbs * 4
        explain = (f"BMR {bmr(p):.0f} kcal (Mifflin-St Jeor) x {LIFESTYLE.get(p.activity_level or 'light', 1.3)} lifestyle "
                   f"= {base:.0f}; + {tdee - base:.0f} for today's training = {tdee:.0f} maintenance; "
                   + (f"{adjust:+.0f} {'deficit' if adjust < 0 else 'surplus'}" if adjust else "no adjustment")
                   + (f" (floor {floor:.0f})" if tdee + adjust < floor else "") + f" = {round(kcal / 10) * 10:.0f} kcal")
        days[d] = {"kcal": round(kcal / 10) * 10, "protein_g": protein, "fat_g": round(fat), "carb_g": round(carbs), "explain": explain,
                   "sat_fat_max_g": round(0.10 * kcal / 9), "maintenance_kcal": round(tdee / 10) * 10,
                   "training": bool(sessions_by_day.get(d))}
    return {"goal": goal, "bmr": round(bmr(p)), "maintenance_avg_kcal": round(weekly_tdee / 10) * 10,
            "daily_adjust_kcal": round(adjust), "rate_kg_per_week": round(-rate if goal == "lose_fat" else rate, 2),
            "change_kg": change_kg, "weeks_needed": weeks_needed, "protein_g_per_kg": protein_per_kg,
            "floor_kcal": round(floor), "days": days, "notes": notes,
            "protein_basis_kg": round(basis_kg, 1),
            "hydration": "Drink to thirst and replace sweat losses around training, more in heat." if c.hydration else None}
