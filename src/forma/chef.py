"""The pantry chef: turn what is in your fridge into a day of meals that hits the day's targets.

For each meal it first looks for a recipe in the corpus that your pantry mostly covers, scaled to the
meal's calories. Otherwise it composes one from your pantry by role (protein, carbohydrate, vegetable,
fat, flavour), sizes the portions to the meal's protein and calorie targets, and computes the nutrition
from USDA data. The LLM only writes the title and steps; every number comes from the food data, and
nothing the medical agent has ruled out (or you are allergic to) can enter a meal.
"""

from __future__ import annotations

from collections import defaultdict

from .foods import Food, FoodIndex, core_tokens
from .models import Constraints, Profile

SPLIT = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.30, "Snack": 0.10}
STAPLES = {"Salt, table", "Spices, pepper, black", "Oil, olive, salad or cooking", "Oil, canola", "Beverages, water, tap, drinking"}
def blocked_tags(c: Constraints, p: Profile) -> set[str]:
    blocked = set(c.avoid_food_tags) | set(p.allergies)
    if {"vegetarian", "vegan"} & set(p.diet):
        blocked |= {"meat", "fish", "shellfish"}
    if "vegan" in p.diet:
        blocked |= {"dairy", "egg"}
    if "pescatarian" in p.diet:
        blocked.add("meat")
    return blocked


GROUPS = [  # (group, name keywords); first match wins, on the USDA name or what you typed
    ("aromatic", ("garlic", "onion", "shallot", "ginger", "scallion", "chili", "chilli", "jalapeno", "lemon", "lime", "parsley",
                  "cilantro", "coriander", "basil", "mint", "dill", "thyme", "rosemary", "chive", "oregano")),
    ("sauce", ("soy sauce", "fish sauce", "oyster sauce", "vinegar", "mustard", "catsup", "ketchup", "salsa", "tomato products",
               "pesto", "curry", "hot sauce", "worcestershire", "honey", "mayonnaise", "tahini")),
    ("cheese", ("cheese, cheddar", "cheese, parmesan", "cheese, mozzarella", "cheese, feta", "cheese, swiss", "cheese, goat")),
    ("morning_protein", ("egg", "yogurt", "cottage", "kefir", "milk")),
    ("protein", ("chicken", "turkey", "beef", "pork", "lamb", "fish", "salmon", "tuna", "cod", "shrimp", "crustaceans", "tofu",
                 "tempeh", "lentil", "chickpea", "beans, black", "beans, kidney", "beans, white", "edamame")),
    ("starch", ("rice", "quinoa", "pasta", "spaghetti", "noodle", "bread", "tortilla", "oats", "couscous", "potato", "barley",
                "bulgur", "pita")),
    ("fat", ("oil", "butter", "avocado", "nuts", "almond", "walnut", "peanut", "seeds")),
]
CAPS = {"morning_protein": 250, "protein": 200, "starch": 250, "veg": 200, "fruit": 150, "fat": 15, "cheese": 30}


def group(label: str, f: Food) -> str | None:
    text = f"{label} {f.name}".lower()
    for g, words in GROUPS:
        if any(w in text for w in words):
            if g == "morning_protein" and f.per100["protein_g"] < 3 or g == "protein" and f.per100["protein_g"] < 6:
                return "liquid" if f.per100["kcal"] < 80 else None
            return g
    if f.category.startswith("Spices"):
        return "aromatic"
    if "Fruits" in f.category:
        return "fruit"
    if "Vegetables" in f.category:
        return "veg"
    return None


def pantry(index: FoodIndex, items: list[str]) -> tuple[list[tuple[str, Food]], list[str]]:
    found, missing = [], []
    for item in items:
        food, _ = index.match(item)
        (found.append((item, food)) if food else missing.append(item))
    return found, missing


def _line(food: Food, grams: float, label: str, index: FoodIndex) -> dict:
    return {"name": label, "food": food.name, "fdc_id": food.fdc_id, "grams": round(grams), "tags": sorted(food.tags),
            "nutrition": {k: round(v, 1) for k, v in index.nutrition(food, grams).items()}}


def _total(lines: list[dict]) -> dict:
    keys = ("kcal", "protein_g", "carb_g", "fat_g", "fiber_g", "sodium_mg", "vitamin_k_ug", "potassium_mg", "calcium_mg")
    return {k: round(sum(li["nutrition"].get(k, 0) for li in lines), 1) for k in keys}


def from_corpus(recipes: list[dict], have: set[int], blocked: set[str], meal: str, kcal: float, protein_g: float,
                index: FoodIndex) -> dict | None:
    """A corpus recipe your pantry covers: every substantial ingredient on hand, at least 70% of the rest."""
    best, best_cov = None, 0.0
    for r in recipes:
        if set(r["food_tags"]) & blocked or (meal == "Breakfast") != (r["category"] == "Breakfast") or not r["servings"]:
            continue
        needed = [i for i in r["ingredients"] if i["food"] not in STAPLES and not (i["grams"] is not None and i["grams"] < 3)]
        if len(needed) < 3 or (r["protein_g_per_serving"] or 0) < 15:
            continue
        on_hand = [i for i in needed if i["fdc_id"] in have]
        substantial = [i for i in needed if (i["grams"] or 999) / r["servings"] >= 25]
        cov = len(on_hand) / len(needed)
        if all(i["fdc_id"] in have for i in substantial) and cov > best_cov:
            best, best_cov = r, cov
    if not best or best_cov < 0.7:
        return None
    scale = min(2.0, max(0.5, kcal / max(best["kcal_per_serving"], 1)))
    by_id = {f.fdc_id: f for f in index.foods}
    lines = [_line(by_id[i["fdc_id"]], i["grams"] / best["servings"] * scale, i["ingredient"], index)
             for i in best["ingredients"] if i["fdc_id"] and i["grams"]]
    return {"title": best["name"], "source": "recipe", "source_url": best["source_url"], "image": best["thumb"],
            "coverage": round(best_cov, 2), "servings": round(scale, 2), "ingredients": lines,
            "missing": [i["ingredient"] for i in best["ingredients"] if i["fdc_id"] not in have and i["food"] not in STAPLES],
            "technique": "recipe", "names": {}, "steps": None}


SEASONING = {"Salt, table", "Spices, pepper, black"}
REUSABLE = {"fat", "aromatic", "sauce", "liquid"}


def compose(found: list[tuple[str, Food]], blocked: set[str], meal: str, kcal: float, protein_g: float,
            index: FoodIndex, used: set[int], avoid_technique: str | None = None) -> dict | None:
    pool: dict[str, list[tuple[str, Food]]] = {}
    for label, f in found:
        g = group(label, f)
        if g and not f.tags & blocked and f.name not in SEASONING:
            pool.setdefault(g, []).append((label, f))

    def take(g: str, prefer=(), topping: bool = False, savory: bool = False):
        fresh = (lambda lf: False) if g in REUSABLE else (lambda lf: lf[1].fdc_id in used)
        options = [lf for lf in pool.get(g, []) if not (topping and lf[1].name.startswith(("Oil", "Butter")))
                   and not (savory and "oat" in lf[1].name.lower())]
        options = sorted(options, key=lambda lf: (fresh(lf), not any(w in lf[1].name.lower() for w in prefer)))
        return options[0] if options else None

    parts: dict[str, tuple[str, Food, float]] = {}

    def add(g: str, item, grams: float):
        if item and grams >= 3:
            parts[g] = (item[0], item[1], min(grams, CAPS.get(g, 30)))

    def kcal_so_far() -> float:
        return sum(index.nutrition(f, gr)["kcal"] for _, f, gr in parts.values())

    def protein_portion(item, target, cap):  # enough for the protein target, but no more than half the meal's energy
        f = item[1]
        return min(target / max(f.per100["protein_g"], 1) * 100, 0.5 * kcal / max(f.per100["kcal"], 1) * 100, cap)

    def fill(g: str, item):
        left = kcal - kcal_so_far()
        if item and left > 60:
            add(g, item, left / max(item[1].per100["kcal"], 1) * 100)

    if meal == "Snack":
        prot = take("morning_protein", ("yogurt", "cottage", "egg"))
        fruit, nuts = take("fruit"), take("fat", ("almond", "walnut", "peanut", "seed"), topping=True)
        if not (prot or fruit or nuts):
            return None
        add("morning_protein", prot, protein_portion(prot, protein_g, 200) if prot else 0)
        add("fruit", fruit, 100)
        if not prot and nuts:
            add("fat", nuts, 20)
        if prot and "egg" in prot[1].tags:
            parts["morning_protein"] = (prot[0], prot[1], max(50, round(parts["morning_protein"][2] / 50) * 50))
        technique = "boiled_eggs" if prot and "egg" in prot[1].tags else "snack"
    elif meal == "Breakfast":
        prot, fruit = take("morning_protein", ("egg", "yogurt", "cottage")), take("fruit")
        oats = take("starch", ("oats",))
        if prot and "egg" in prot[1].tags:
            add("morning_protein", prot, max(100, round(protein_portion(prot, protein_g, 150) / 50) * 50))  # whole eggs
            add("veg", take("veg", ("spinach", "pepper", "tomato", "mushroom")), 100)
            add("fat", take("fat", ("oil", "butter")), 5)
            side = take("starch", ("bread", "potato", "tortilla"))
            if side and "oat" not in side[1].name.lower():
                fill("starch", side)
            technique = "scramble"
        elif prot or oats:
            if prot:
                add("morning_protein", prot, protein_portion(prot, protein_g, 250))
            add("fruit", fruit, 100)
            add("fat", take("fat", ("almond", "walnut", "seed", "peanut"), topping=True), 15)
            if not prot:
                add("liquid", take("liquid", ("soy", "almond", "oat")), 200)
            fill("starch", oats)
            technique = "oats" if "starch" in parts else "yogurt_bowl"
        else:
            return None
    else:
        prot = take("protein", ("chicken", "tofu", "salmon", "fish", "turkey", "beef", "lentil", "chickpea")) or next(
            (lf for lf in pool.get("morning_protein", []) if "egg" in lf[1].tags), None)  # eggs make a supper when nothing else does
        if not prot:
            return None
        legume = any(w in prot[1].name.lower() for w in ("lentil", "chickpea", "beans"))
        add("protein", prot, protein_portion(prot, protein_g, 220))
        add("veg", take("veg", ("broccoli", "pepper", "spinach", "green bean", "zucchini", "carrot", "tomato")), 150)
        add("aromatic", take("aromatic", ("garlic", "onion", "ginger")), 10)
        add("sauce", take("sauce", ("tomato",) if legume else ("soy", "tomato", "mustard")), 15)
        add("fat", take("fat", ("oil",)), 10 if prot[1].per100["fat_g"] < 8 else 5)  # less oil with fatty cuts
        starch = take("starch", ("rice", "potato", "pasta", "quinoa", "noodle", "bread"), savory=True)  # reuse beats none
        if meal == "Dinner" and kcal - kcal_so_far() > 250 if starch else kcal - kcal_so_far() > 100:
            add("cheese", take("cheese"), 20)  # only when the budget leaves room after the starch
        fill("starch", starch)
        name = parts["starch"][1].name.lower() if "starch" in parts else ""
        technique = ("simmer" if legume else "stir_fry" if "rice" in name or "noodle" in name else "tray_bake" if "potato" in name
                     else "pasta" if "pasta" in name or "spaghetti" in name else "skillet")
        if technique == avoid_technique:
            technique = "tray_bake" if "potato" in name and technique != "tray_bake" else "skillet" if technique != "skillet" else "bowl"
    order = ["morning_protein", "protein", "veg", "fruit", "starch", "liquid", "aromatic", "sauce", "fat", "cheese"]
    lines = [(g, _line(f, max(5, round(gr / 5) * 5), label, index)) for g in order if g in parts
             for label, f, gr in [parts[g]]]
    for g, li in lines:
        if g not in REUSABLE:
            used.add(li["fdc_id"])
    return {"title": None, "source": "composed", "source_url": None, "image": None, "coverage": 1.0, "servings": 1,
            "ingredients": [li for _, li in lines], "missing": [], "technique": technique,
            "names": {g: li["name"].lower() for g, li in lines}, "steps": None}


def template_steps(recipe: dict) -> list[str]:
    """Plain, correct steps for a composed meal; the LLM may rewrite them more warmly."""
    names = recipe["names"]
    n = defaultdict(str, names)  # every technique's steps are built; only the chosen one is returned
    t = recipe["technique"]

    def opt(key, text):
        return text.format_map(n) if key in names else ""

    steps = {
        "scramble": [f"Whisk the {names.get('morning_protein', 'eggs')} with a pinch of salt and pepper.",
                     opt("veg", "Soften the {veg} in " + ("the {fat}" if "fat" in names else "a little oil") + " over medium heat, 3-4 minutes."),
                     "Pour in the eggs and stir gently over low heat until just set.",
                     opt("starch", "Serve with the {starch}" + (", toasted." if "bread" in n["starch"] else
                                                               ", boiled until tender, then crisped in the pan." if "potato" in n["starch"] else "."))],
        "oats": [f"Simmer the {names.get('starch', 'oats')} with " + (f"the {n['liquid']}" if "liquid" in names else "water") + " for 5 minutes.",
                 opt("morning_protein", "Off the heat, stir in the {morning_protein}."),
                 opt("fruit", "Top with the {fruit}") + opt("fat", " and the {fat}") + "."],
        "yogurt_bowl": [opt("morning_protein", "Spoon the {morning_protein} into a bowl."),
                        opt("fruit", "Top with the {fruit}") + opt("fat", " and a sprinkle of the {fat}") + "."],
        "snack": [opt("morning_protein", "Serve the {morning_protein}") + opt("fruit", " with the {fruit}") + opt("fat", " and the {fat}") + "."
                  if "morning_protein" in names else opt("fruit", "Have the {fruit}") + opt("fat", " with the {fat}") + "."],
        "boiled_eggs": [f"Boil the {n['morning_protein']} for 9 minutes, cool in cold water, and peel.",
                        opt("fruit", "Eat with the {fruit}.")],
        "stir_fry": [opt("starch", "Cook the {starch} according to the packet."),
                     f"Cut the {n['protein']}" + opt("veg", " and {veg}") + " into bite-size pieces" + opt("aromatic", "; mince the {aromatic}") + ".",
                     f"Sear the {n['protein']} in " + ("the {fat}".format_map(n) if "fat" in names else "a little oil") + " over high heat until browned and cooked through.",
                     opt("veg", "Add " + opt("aromatic", "the {aromatic} and ") + "the {veg} and stir-fry 3-4 minutes")
                     + opt("sauce", ", then toss everything with the {sauce}") + ("." if "veg" in names else ""),
                     opt("starch", "Serve over the {starch}.")],
        "tray_bake": ["Heat the oven to 200 C (400 F).",
                      opt("starch", "Cut the {starch}") + opt("veg", (" and " if "starch" in names else "Cut the ") + "{veg}") + " into even chunks"
                      + " and toss with " + ("the {fat}".format_map(n) if "fat" in names else "a little oil") + opt("aromatic", " and the {aromatic}") + ", salt and pepper.",
                      f"Roast 15 minutes, add the {n['protein']}, and roast 15-20 minutes more until cooked through.",
                      opt("sauce", "Finish with the {sauce}") + opt("cheese", (" and " if "sauce" in names else "Finish with ") + "the {cheese}") + "."],
        "pasta": [opt("starch", "Boil the {starch} in salted water until al dente."),
                  f"Brown the {n['protein']} in " + ("the {fat}".format_map(n) if "fat" in names else "a little oil") + opt("aromatic", " with the {aromatic}") + ".",
                  opt("veg", "Add the {veg}") + opt("sauce", " and the {sauce}") + (" and simmer 5 minutes." if "veg" in names or "sauce" in n else ""),
                  opt("starch", "Toss with the drained {starch}") + opt("cheese", " and top with the {cheese}") + "."],
        "skillet": [f"Season the {n['protein']} with salt and pepper.",
                    "Cook it in " + ("the {fat}".format_map(n) if "fat" in names else "a little oil") + opt("aromatic", " with the {aromatic}") + " over medium-high heat until cooked through.",
                    opt("veg", "Add the {veg} for the last 4-5 minutes") + opt("sauce", " with the {sauce}") + ("." if "veg" in names else ""),
                    ("Serve with the {starch}".format_map(n) + opt("cheese", " and the {cheese}") + "." if "starch" in names
                     else opt("cheese", "Serve topped with the {cheese}."))],
        "simmer": [f"Rinse the {n['protein']}" + opt("aromatic", " and mince the {aromatic}") + ".",
                   "Soften " + opt("aromatic", "the {aromatic} ") + "in " + ("the {fat}".format_map(n) if "fat" in names else "a little oil")
                   + f", then add the {n['protein']}" + opt("sauce", ", the {sauce}") + " and water to cover by 3 cm.",
                   "Simmer 20-25 minutes (lentils) or 10 minutes (canned beans) until tender; season with salt and pepper.",
                   opt("veg", "Stir in the {veg} for the last 3-4 minutes."), opt("starch", "Serve with the {starch}.")],
        "bowl": [opt("starch", "Prepare the {starch}."),
                 f"Cook the {n['protein']} in " + ("the {fat}".format_map(n) if "fat" in names else "a little oil") + opt("aromatic", " with the {aromatic}") + " until done.",
                 "Pile the {protein}".format_map(n) + opt("veg", ", {veg}") + opt("starch", " and {starch}") + " into a bowl" + opt("sauce", " and finish with the {sauce}") + "."],
    }.get(t, [])
    return [s for s in (" ".join(x.split()) for x in steps) if len(s) > 2 and s != "."]


ASSUMED = ["olive oil", "salt", "black pepper"]
PROTEIN_IDEAS = ["Greek yogurt", "eggs", "chicken breast", "tofu", "canned tuna", "lentils", "cottage cheese"]
LEAN = 0.15  # g protein per kcal: chicken, fish, tofu, Greek yogurt; not eggs, beans or cheese
SODIUM_MAX = 2300  # mg a day, the DGA limit from age 14


def trim_sodium(meals: list[dict], index: FoodIndex) -> bool:
    """Keep the day under the sodium limit by using less of the salty condiments (soy sauce, salt, stock), saltiest first."""
    over = sum(m["total"]["sodium_mg"] for m in meals) - SODIUM_MAX
    salty = sorted(((m, li) for m in meals for li in m["ingredients"] if index.by_name[li["food"]].per100["sodium_mg"] >= 1000),
                   key=lambda ml: -ml[1]["nutrition"]["sodium_mg"])
    for m, li in salty:
        if over <= 0:
            break
        per_g = index.by_name[li["food"]].per100["sodium_mg"] / 100
        keep = max(0, int((li["grams"] - over / per_g) // 5 * 5))  # round down: rounding up can undo the cut
        over -= (li["grams"] - keep) * per_g
        if keep < 5:
            m["ingredients"].remove(li)
            m["names"] = {g: n for g, n in m["names"].items() if n != li["name"].lower()}
        else:
            m["ingredients"][m["ingredients"].index(li)] = _line(index.by_name[li["food"]], keep, li["name"], index)
        m["total"] = _total(m["ingredients"])
    return bool(salty)


def rebalance(meals: list[dict], gap: float, index: FoodIndex) -> float:
    """Close a protein gap at the same calories: more of each meal's leanest protein, less of its starch."""
    for m in meals:
        if gap <= 5 or m["source"] != "composed" or "starch" not in m["names"]:
            continue
        main = m["names"].get("protein") or m["names"].get("morning_protein")  # the meal's protein food, never a sauce
        lean = next((li for li in m["ingredients"] if li["name"].lower() == main), None)
        starch = next(li for li in m["ingredients"] if li["name"].lower() == m["names"]["starch"])
        if lean is None:
            continue
        f, s = index.by_name[lean["food"]], index.by_name[starch["food"]]
        if lean is starch or f.per100["protein_g"] / max(f.per100["kcal"], 1) < LEAN:
            continue
        net = f.per100["protein_g"] - s.per100["protein_g"] * f.per100["kcal"] / s.per100["kcal"]  # per 100 g swapped in
        extra = min(100, 300 - lean["grams"], gap / max(net, 1) * 100, starch["grams"] * s.per100["kcal"] / f.per100["kcal"])
        extra = round(extra / 10) * 10
        if extra < 20:
            continue
        cut = extra * f.per100["kcal"] / s.per100["kcal"]
        m["ingredients"][m["ingredients"].index(lean)] = _line(f, lean["grams"] + extra, lean["name"], index)
        if starch["grams"] - cut < 15:  # nothing worth serving is left
            m["ingredients"].remove(starch)
            del m["names"]["starch"]
            m["technique"] = "yogurt_bowl" if m["technique"] == "oats" else m["technique"]
        else:
            m["ingredients"][m["ingredients"].index(starch)] = _line(s, round((starch["grams"] - cut) / 5) * 5, starch["name"], index)
        m["total"] = _total(m["ingredients"])
        gap -= net * extra / 100
    return gap


def plan_day(index: FoodIndex, recipes: list[dict], items: list[str], day: dict, c: Constraints, p: Profile,
             prior: set[int] = frozenset()) -> dict:
    found, unmatched = pantry(index, items)
    have_names = {f.name for _, f in found}
    found += [(s, f) for s, (f, _) in ((s, index.match(s)) for s in ASSUMED) if f and f.name not in have_names]
    blocked = blocked_tags(c, p)
    unsafe = [label for label, f in found if f.tags & blocked]
    have = {f.fdc_id for _, f in found}
    used: set[int] = set(prior)  # foods already served count as used, so another day leads with other mains
    meals = []
    for meal, share in SPLIT.items():
        kcal, protein = day["kcal"] * share, day["protein_g"] * share
        r = from_corpus(recipes, have, blocked, meal, kcal, protein, index) if meal in ("Lunch", "Dinner") else None
        if r and any(m["title"] == r["title"] for m in meals):
            r = None
        last = next((m["technique"] for m in reversed(meals) if m["meal"] in ("Lunch", "Dinner")), None)
        r = r or compose(found, blocked, meal, kcal, protein, index, used, last if meal == "Dinner" else None)
        if not r:
            continue
        r.update(meal=meal, target_kcal=round(kcal), total=_total(r["ingredients"]))
        meals.append(r)
    totals = _total([li for m in meals for li in m["ingredients"]])
    notes = [s["message"] for s in c.spacing]
    if "vitamin_k_ug" in c.steady_nutrients:
        notes.append(f"Today's vitamin K: about {totals['vitamin_k_ug']:.0f} µg. Aim to keep a similar amount every day.")
    gap = day["protein_g"] - totals["protein_g"]
    if gap > 5:
        gap = rebalance(meals, gap, index)
        totals = _total([li for m in meals for li in m["ingredients"]])
    if totals["sodium_mg"] > SODIUM_MAX and trim_sodium(meals, index):  # last, after any portion changes
        totals = _total([li for m in meals for li in m["ingredients"]])
        notes.append(f"Salty condiments are kept light so the day stays under {SODIUM_MAX:,} mg of sodium (DGA).")
    for sp in c.spacing:  # a timing line on each meal holding a food the medication must be kept apart from
        for m in meals:
            hits = [f"{li['name']} ({t.replace('_', '-')})" for li in m["ingredients"] if li["grams"] >= 25  # a portion, not a garnish
                    for t in sp["tags"] if t in li["tags"]]
            if hits:
                m.setdefault("timing", []).append(f"Eat at least {sp['hours']:g} h apart from your medication: {', '.join(hits)}.")
    ideas = [x for x in PROTEIN_IDEAS if (f := index.match(x)[0]) and f.fdc_id not in have and not f.tags & blocked][:4]
    if gap > 15 and ideas:
        notes.append(f"These meals are about {gap:.0f} g short of today's protein target. Adding "
                     f"{', '.join(ideas[:-1]) + ' or ' if len(ideas) > 1 else ''}{ideas[-1]} to your pantry would close the gap.")
    return {"meals": meals, "totals": totals, "unmatched": unmatched, "excluded_for_safety": unsafe, "notes": notes,
            "assumed_staples": ASSUMED,
            "fit_pct": round(100 * totals["kcal"] / day["kcal"]) if day["kcal"] else None}


def describe(meal: dict) -> str:
    return ", ".join(f"{li['grams']} g {li['name']}" for li in meal["ingredients"])


def ingredients_mentioned(steps: list[str], meal: dict) -> bool:
    text = " ".join(steps).lower()
    return all(core_tokens(li["name"]) & set(text.replace(",", " ").split()) or li["grams"] < 10 for li in meal["ingredients"])
