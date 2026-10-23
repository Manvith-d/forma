"""Foods: map an ingredient name to a USDA FoodData Central food, and a measure ("2 tbs", "400g") to grams.

Used by the pipeline (recipe nutrition) and by the chef agent (whatever is in your pantry).
Common ingredients map through a hand-checked alias table to a specific FDC description; anything
else goes through fuzzy matching that prefers plain, raw, whole foods and the food's own name.
Grams come from FDC's measured portion weights for that food when it has one for the unit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from rapidfuzz import fuzz, process

ALIASES = {
    "salt": "Salt, table", "sea salt": "Salt, table", "kosher salt": "Salt, table",
    "garlic": "Garlic, raw", "garlic clove": "Garlic, raw", "garlic cloves": "Garlic, raw",
    "onion": "Onions, raw", "onions": "Onions, raw", "red onion": "Onions, raw", "red onions": "Onions, raw",
    "butter": "Butter, salted", "unsalted butter": "Butter, without salt", "melted butter": "Butter, salted",
    "olive oil": "Oil, olive, salad or cooking", "extra virgin olive oil": "Oil, olive, salad or cooking",
    "vegetable oil": "Oil, canola", "oil": "Oil, canola", "rapeseed oil": "Oil, canola", "canola oil": "Oil, canola",
    "sunflower oil": "Oil, sunflower, linoleic (less than 60%)", "sesame seed oil": "Oil, sesame, salad or cooking",
    "sesame oil": "Oil, sesame, salad or cooking", "coconut oil": "Oil, coconut",
    "sugar": "Sugars, granulated", "caster sugar": "Sugars, granulated", "granulated sugar": "Sugars, granulated",
    "golden caster sugar": "Sugars, granulated", "brown sugar": "Sugars, brown", "muscovado sugar": "Sugars, brown",
    "icing sugar": "Sugars, powdered", "powdered sugar": "Sugars, powdered", "honey": "Honey", "clear honey": "Honey",
    "water": "Beverages, water, tap, drinking", "warm water": "Beverages, water, tap, drinking",
    "milk": "Milk, whole, 3.25% milkfat, with added vitamin D", "whole milk": "Milk, whole, 3.25% milkfat, with added vitamin D",
    "skim milk": "Milk, nonfat, fluid, with added vitamin A and vitamin D (fat free or skim)",
    "egg": "Egg, whole, raw, fresh", "eggs": "Egg, whole, raw, fresh", "egg yolks": "Egg, yolk, raw, fresh",
    "egg yolk": "Egg, yolk, raw, fresh", "egg white": "Egg, white, raw, fresh", "egg whites": "Egg, white, raw, fresh",
    "pepper": "Spices, pepper, black", "black pepper": "Spices, pepper, black",
    "flour": "Wheat flour, white, all-purpose, enriched, bleached", "plain flour": "Wheat flour, white, all-purpose, enriched, bleached",
    "all purpose flour": "Wheat flour, white, all-purpose, enriched, bleached",
    "self-raising flour": "Wheat flour, white, all-purpose, self-rising, enriched", "corn flour": "Cornstarch",
    "cornstarch": "Cornstarch", "potatoes": "Potatoes, flesh and skin, raw", "potato": "Potatoes, flesh and skin, raw",
    "sweet potato": "Sweet potato, raw, unprepared (Includes foods for USDA's Food Distribution Program)", "sweet potatoes": "Sweet potato, raw, unprepared (Includes foods for USDA's Food Distribution Program)",
    "carrots": "Carrots, raw", "carrot": "Carrots, raw", "celery": "Celery, raw", "cucumber": "Cucumber, with peel, raw",
    "soy sauce": "Soy sauce made from soy and wheat (shoyu)", "fish sauce": "Sauce, fish, ready-to-serve",
    "oyster sauce": "Sauce, oyster, ready-to-serve", "worcestershire sauce": "Sauce, worcestershire",
    "coriander": "Coriander (cilantro) leaves, raw", "cilantro": "Coriander (cilantro) leaves, raw",
    "coriander leaves": "Coriander (cilantro) leaves, raw", "parsley": "Parsley, fresh", "basil": "Basil, fresh",
    "basil leaves": "Basil, fresh", "mint": "Spearmint, fresh", "dill": "Dill weed, fresh", "rosemary": "Rosemary, fresh",
    "thyme": "Thyme, fresh", "chives": "Chives, raw", "spring onions": "Onions, spring or scallions (includes tops and bulb), raw",
    "scallions": "Onions, spring or scallions (includes tops and bulb), raw", "shallots": "Shallots, raw",
    "baking powder": "Leavening agents, baking powder, double-acting, sodium aluminum sulfate",
    "bicarbonate of soda": "Leavening agents, baking soda", "baking soda": "Leavening agents, baking soda",
    "yeast": "Leavening agents, yeast, baker's, active dry", "lime": "Limes, raw", "lime juice": "Lime juice, raw",
    "lemon": "Lemons, raw, without peel", "lemon juice": "Lemon juice, raw", "orange": "Oranges, raw, all commercial varieties",
    "paprika": "Spices, paprika", "smoked paprika": "Spices, paprika", "cayenne pepper": "Spices, pepper, red or cayenne",
    "red pepper": "Peppers, sweet, red, raw", "green pepper": "Peppers, sweet, green, raw", "yellow pepper": "Peppers, sweet, yellow, raw",
    "bell pepper": "Peppers, sweet, red, raw", "red chilli": "Peppers, hot chili, red, raw", "green chilli": "Peppers, hot chili, green, raw",
    "chilli": "Peppers, hot chili, red, raw", "scotch bonnet": "Peppers, hot chili, red, raw", "jalapeno": "Peppers, jalapeno, raw",
    "ginger": "Ginger root, raw", "ground ginger": "Spices, ginger, ground", "cinnamon": "Spices, cinnamon, ground",
    "ground cinnamon": "Spices, cinnamon, ground", "cinnamon stick": "Spices, cinnamon, ground", "cumin": "Spices, cumin seed",
    "ground cumin": "Spices, cumin seed", "cumin seeds": "Spices, cumin seed", "turmeric": "Spices, turmeric, ground",
    "nutmeg": "Spices, nutmeg, ground", "ground nutmeg": "Spices, nutmeg, ground", "allspice": "Spices, allspice, ground",
    "cardamom": "Spices, cardamom", "cloves": "Spices, cloves, ground", "bay leaf": "Spices, bay leaf", "bay leaves": "Spices, bay leaf",
    "oregano": "Spices, oregano, dried", "dried oregano": "Spices, oregano, dried", "curry powder": "Spices, curry powder",
    "garam masala": "Spices, curry powder", "garlic powder": "Spices, garlic powder", "chilli powder": "Spices, chili powder",
    "chilli flakes": "Spices, pepper, red or cayenne", "red pepper flakes": "Spices, pepper, red or cayenne", "saffron": "Spices, saffron",
    "vanilla extract": "Vanilla extract", "vanilla": "Vanilla extract",
    "tomato": "Tomatoes, red, ripe, raw, year round average", "tomatoes": "Tomatoes, red, ripe, raw, year round average",
    "cherry tomatoes": "Tomatoes, red, ripe, raw, year round average", "plum tomatoes": "Tomatoes, red, ripe, raw, year round average",
    "tinned tomatos": "Tomatoes, red, ripe, canned, packed in tomato juice", "chopped tomatoes": "Tomatoes, red, ripe, canned, packed in tomato juice",
    "tomato puree": "Tomato products, canned, puree, without salt added", "tomato sauce": "Tomato products, canned, sauce",
    "tomato ketchup": "Catsup", "ketchup": "Catsup", "coconut milk": "Nuts, coconut milk, canned (liquid expressed from grated meat and water)",
    "chicken stock": "Soup, stock, chicken, home-prepared", "beef stock": "Soup, stock, beef, home-prepared",
    "vegetable stock": "Soup, stock, chicken, home-prepared", "rice": "Rice, white, long-grain, regular, raw, enriched",
    "basmati rice": "Rice, white, long-grain, regular, raw, enriched", "jasmine rice": "Rice, white, long-grain, regular, raw, enriched",
    "white rice": "Rice, white, long-grain, regular, raw, enriched", "brown rice": "Rice, brown, long-grain, raw (Includes foods for USDA's Food Distribution Program)",
    "beef": "Beef, ground, 85% lean meat / 15% fat, raw (Includes foods for USDA's Food Distribution Program)", "ground beef": "Beef, ground, 85% lean meat / 15% fat, raw (Includes foods for USDA's Food Distribution Program)",
    "minced beef": "Beef, ground, 85% lean meat / 15% fat, raw (Includes foods for USDA's Food Distribution Program)", "lamb mince": "Lamb, ground, raw", "lamb": "Lamb, ground, raw",
    "ground pork": "Pork, fresh, ground, raw", "pork": "Pork, fresh, loin, whole, separable lean and fat, raw",
    "bacon": "Pork, cured, bacon, unprepared", "ham": "Ham, sliced, regular (approximately 11% fat)", "chorizo": "Sausage, pork, chorizo, link or ground, raw",
    "chicken": "Chicken, broilers or fryers, meat only, raw", "chicken breast": "Chicken, broiler or fryers, breast, skinless, boneless, meat only, raw",
    "chicken breasts": "Chicken, broiler or fryers, breast, skinless, boneless, meat only, raw",
    "chicken thighs": "Chicken, broilers or fryers, dark meat, thigh, meat only, raw", "prawns": "Crustaceans, shrimp, raw",
    "raw king prawns": "Crustaceans, shrimp, raw", "shrimp": "Crustaceans, shrimp, raw", "salmon": "Fish, salmon, Atlantic, farmed, raw",
    "double cream": "Cream, fluid, heavy whipping", "heavy cream": "Cream, fluid, heavy whipping", "sour cream": "Cream, sour, cultured",
    "creme fraiche": "Cream, sour, cultured", "greek yogurt": "Yogurt, Greek, plain, nonfat", "yogurt": "Yogurt, plain, whole milk",
    "parmesan": "Cheese, parmesan, grated", "parmesan cheese": "Cheese, parmesan, grated", "mozzarella": "Cheese, mozzarella, whole milk",
    "cheddar cheese": "Cheese, cheddar", "cheese": "Cheese, cheddar", "feta": "Cheese, feta", "cream cheese": "Cheese, cream",
    "cottage cheese": "Cheese, cottage, lowfat, 2% milkfat", "condensed milk": "Milk, canned, condensed, sweetened",
    "almonds": "Nuts, almonds", "walnuts": "Nuts, walnuts, english", "peanuts": "Peanuts, all types, raw",
    "pine nuts": "Nuts, pine nuts, dried", "sesame seed": "Seeds, sesame seeds, whole, dried", "raisins": "Raisins, dark, seedless (Includes foods for USDA's Food Distribution Program)",
    "mushrooms": "Mushrooms, white, raw", "cabbage": "Cabbage, raw", "white cabbage": "Cabbage, raw", "leek": "Leeks, (bulb and lower leaf-portion), raw",
    "spinach": "Spinach, raw", "kale": "Kale, raw", "broccoli": "Broccoli, raw", "lettuce": "Lettuce, cos or romaine, raw",
    "aubergine": "Eggplant, raw", "eggplant": "Eggplant, raw", "egg plants": "Eggplant, raw", "courgettes": "Squash, summer, zucchini, includes skin, raw",
    "zucchini": "Squash, summer, zucchini, includes skin, raw", "green beans": "Beans, snap, green, raw", "peas": "Peas, green, raw",
    "sweetcorn": "Corn, sweet, yellow, raw", "avocado": "Avocados, raw, all commercial varieties", "beetroot": "Beets, raw",
    "bean sprouts": "Mung beans, mature seeds, sprouted, raw", "banana": "Bananas, raw", "apple": "Apples, raw, gala, with skin (Includes foods for USDA's Food Distribution Program)",
    "strawberries": "Strawberries, raw", "raspberries": "Raspberries, raw", "blueberries": "Blueberries, raw",
    "grapefruit": "Grapefruit, raw, pink and red, all areas", "mayonnaise": "Salad dressing, mayonnaise, regular",
    "mustard": "Mustard, prepared, yellow", "dijon mustard": "Mustard, prepared, yellow", "breadcrumbs": "Bread, crumbs, dry, grated, plain",
    "bread": "Bread, white, commercially prepared (includes soft bread crumbs)", "baguette": "Bread, french or vienna (includes sourdough)",
    "pita bread": "Bread, pita, white, enriched", "whole wheat bread": "Bread, whole-wheat, commercially prepared",
    "white wine": "Alcoholic beverage, wine, table, white", "dry white wine": "Alcoholic beverage, wine, table, white",
    "red wine": "Alcoholic beverage, wine, table, red", "vinegar": "Vinegar, distilled", "white wine vinegar": "Vinegar, distilled",
    "red wine vinegar": "Vinegar, red wine", "apple cider vinegar": "Vinegar, cider", "rice vinegar": "Vinegar, distilled",
    "tofu": "Tofu, raw, firm, prepared with calcium sulfate", "oats": "Cereals, oats, regular and quick, not fortified, dry",
    "rolled oats": "Cereals, oats, regular and quick, not fortified, dry", "quinoa": "Quinoa, uncooked", "lentils": "Lentils, raw",
    "chickpeas": "Chickpeas (garbanzo beans, bengal gram), mature seeds, raw", "pasta": "Pasta, dry, enriched", "spaghetti": "Pasta, dry, enriched",
    "tuna": "Fish, tuna, light, canned in water, drained solids", "peanut butter": "Peanut butter, smooth style, without salt",
    "dark chocolate": "Chocolate, dark, 70-85% cacao solids", "turkey": "Turkey, ground, 93% lean, 7% fat, raw", "ground turkey": "Turkey, ground, 93% lean, 7% fat, raw",
    "kidney beans": "Beans, kidney, red, mature seeds, raw", "black beans": "Beans, black, mature seeds, raw",
    "cannellini beans": "Beans, white, mature seeds, raw",
}
BAD = ("babyfood", "infant", "fast foods", "restaurant", "school", "formula", "usda commodity", "industrial", "toddler",
       "snacks,")
# Words that describe preparation or size, not the food: "2 large floury potatoes, finely sliced" -> "potato"
DESCRIPTORS = {"fresh", "large", "small", "medium", "chopped", "sliced", "diced", "minced", "finely", "roughly", "thinly",
               "floury", "waxy", "ripe", "boneless", "skinless", "free-range", "free", "range", "organic", "frozen", "raw",
               "peeled", "grated", "crushed", "whole", "dried", "hot", "cold", "warm", "melted", "softened", "cooked",
               "baby", "plain", "fine", "coarse", "extra", "good", "quality", "handful", "bunch", "of", "for", "the", "and"}
COUNT_WORDS = {"clove": 3, "cloves": 3, "stick": 3, "sticks": 3, "leaf": 0.2, "leaves": 0.2, "sprig": 1, "sprigs": 1,
               "rasher": 25, "rashers": 25, "slice": 30, "slices": 30, "fillet": 140, "fillets": 140}
NEGLIGIBLE = ("pinch", "dash", "taste", "garnish", "serve", "sprinkle", "sprinkling", "drizzle", "spray")
MASS = {"g": 1, "gram": 1, "grams": 1, "gr": 1, "kg": 1000, "mg": 0.001, "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
        "lb": 453.6, "lbs": 453.6, "pound": 453.6, "pounds": 453.6}
VOLUME_ML = {"ml": 1, "l": 1000, "litre": 1000, "liter": 1000, "tsp": 4.93, "teaspoon": 4.93, "teaspoons": 4.93,
             "tbs": 14.79, "tbsp": 14.79, "tblsp": 14.79, "tablespoon": 14.79, "tablespoons": 14.79, "cup": 236.6, "cups": 236.6,
             "pint": 473, "pints": 473}
FDC_UNIT = {"tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp", "tbs": "tbsp", "tbsp": "tbsp", "tblsp": "tbsp",
            "tablespoon": "tbsp", "tablespoons": "tbsp", "cup": "cup", "cups": "cup"}
COUNT_DEFAULT_G = {"clove": 3, "cloves": 3, "handful": 30, "bunch": 50, "can": 400, "tin": 400, "slice": 30, "slices": 30,
                   "stick": 113, "sprig": 1, "sprigs": 1, "leaf": 0.5, "leaves": 0.5, "piece": 50, "pieces": 50}
_UNICODE = {"½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3", "⅛": "1/8"}
_NUM = re.compile(r"(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)")


def normalize(name: str) -> str:
    name = re.sub(r"\(includes foods for usda.s food distribution program\)", "", name.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 \-]", " ", name)).strip()


def singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("oes", "ches", "shes")):
        return word[:-2]
    return word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word


def core_tokens(name: str) -> set[str]:
    words = {singular(w) for w in normalize(name).split() if w not in DESCRIPTORS}
    return words - set(COUNT_WORDS) or words  # "garlic clove" is garlic; "cloves" alone stays the spice


@dataclass
class Food:
    fdc_id: int
    name: str
    per100: dict[str, float]  # kcal, protein_g, ... per 100 g
    category: str = ""
    tags: set[str] = field(default_factory=set)
    portions: dict[str, float] = field(default_factory=dict)  # "cup" / "tbsp" / "large" ... -> grams


@dataclass
class Parsed:
    quantity: float | None
    unit: str | None


def parse_measure(measure: str) -> Parsed:
    m = measure.lower().strip()
    for k, v in _UNICODE.items():
        m = m.replace(k, f" {v}")
    if not m or any(w in m for w in NEGLIGIBLE):
        return Parsed(None, "negligible")
    match = _NUM.search(m)
    qty = float(sum(Fraction(p) for p in match.group(1).split())) if match else None
    rest = (m[match.end():] if match else m).strip()
    glued = re.match(r"^([a-z]+)", rest)
    unit = glued.group(1) if glued else None
    return Parsed(qty, unit)


class FoodIndex:
    def __init__(self, foods: list[Food]):
        self.foods = foods
        self.by_name = {f.name: f for f in foods}
        self.names = [f.name for f in foods]
        self.norm = [normalize(n) for n in self.names]

    def match(self, ingredient: str) -> tuple[Food | None, float]:
        key = normalize(ingredient)
        core = " ".join(sorted(core_tokens(ingredient))) or key
        for k in (key, singular(key), " ".join(w for w in key.split() if w not in DESCRIPTORS)):
            alias = ALIASES.get(k)
            if alias and alias in self.by_name:
                return self.by_name[alias], 100.0
        wanted = core_tokens(ingredient)
        best, best_score = None, 0.0
        for name, score, i in process.extract(core, self.norm, scorer=fuzz.token_set_ratio, limit=30):
            food = self.foods[i]
            head = core_tokens(name.split(" ")[0]) or {name.split(" ")[0]}
            s = score + (12 if head <= wanted else -8) + (4 if " raw" in f" {name}" else 0)
            s -= 25 * any(b in name for b in BAD) + 0.15 * len(name)
            if s > best_score:
                best, best_score = food, s
        return (best, best_score) if best_score >= 72 else (None, best_score)

    def grams(self, food: Food, measure: str, ingredient: str = "") -> float | None:
        p = parse_measure(measure)
        if p.unit == "negligible":
            return 0.5
        if p.quantity is None and not p.unit:
            return None
        qty = p.quantity if p.quantity is not None else 1.0
        unit = p.unit
        if unit and unit not in MASS and unit not in VOLUME_ML and unit not in COUNT_DEFAULT_G and singular(unit) in (
                *MASS, *VOLUME_ML, *COUNT_DEFAULT_G):
            unit = singular(unit)  # "cans" -> "can"
        if unit in MASS:
            return qty * MASS[unit]
        if unit in FDC_UNIT and FDC_UNIT[unit] in food.portions:
            return qty * food.portions[FDC_UNIT[unit]]
        if unit in VOLUME_ML:
            density = 0.92 if "oil" in food.name.lower() else 1.0 if food.per100.get("kcal", 0) < 120 else 0.6
            return qty * VOLUME_ML[unit] * density
        if unit in COUNT_DEFAULT_G:
            return qty * food.portions.get(singular(unit), COUNT_DEFAULT_G[unit])
        # A bare count ("2", "1 finely sliced", "2 large"): the unit may be in the ingredient ("garlic clove")
        for word in normalize(f"{ingredient} {measure}").split():
            if word in COUNT_WORDS:
                return qty * food.portions.get(singular(word), COUNT_WORDS[word])
        for size in ([unit] if unit else []) + ["medium", "large", "whole", "small", "slice", "clove", "stalk"]:
            if size and size in food.portions:
                return qty * food.portions[size]
        if food.category.startswith("Spices"):
            return qty * 1.0
        return None

    def nutrition(self, food: Food, grams: float) -> dict[str, float]:
        return {k: v * grams / 100 for k, v in food.per100.items()}


# ---- safety and allergen tags --------------------------------------------------------------
PLANT_ALT = r"(peanut|almond|cashew|hazelnut|nut|seed|sunflower|soy|soya|oat|rice|coconut|cocoa|apple|shea|hemp)[ -]?"
TAG_RULES = {  # tag: (pattern, exceptions), matched on the lowercased USDA name with word boundaries
    "dairy": (r"\b(milk|cheese|yogurt|yoghurt|butter|cream|whey|kefir|ghee|casein|buttermilk)\b",
              PLANT_ALT + r"(butter|milk|cream|cheese|yogurt)|butternut|cream of tartar|cream puff shell|ice cream cone"),
    "egg": (r"\beggs?\b|\begg,", r"eggplant|egg noodles? .*egg.?free"),
    "peanut": (r"\bpeanuts?\b", None),
    "tree_nut": (r"\b(almonds?|walnuts?|cashews?|pecans?|pistachios?|hazelnuts?|macadamias?|pine nuts?|brazil ?nuts?|chestnuts?)\b", None),
    "soy": (r"\b(soy|soya|soybeans?|tofu|edamame|tempeh|miso)\b", None),
    "gluten": (r"\b(wheat|barley|rye|spelt|semolina|couscous|bulgur|seitan|bread|breads|pasta|noodles?|crumbs|pita|croissants?|bagels?|crackers?|flour|cereals?)\b",
               r"(rice|corn|potato|almond|coconut|chickpea|tapioca|buckwheat|soy|cassava|sorghum)[ -](flour|noodles?|pasta|bread)|"
               r"buckwheat|cereals?, (oats|corn|rice)|breadfruit|rice cakes?"),
    "fish": (r"\b(fish|salmon|tuna|cod|tilapia|sardines?|anchov(y|ies)|trout|halibut|mackerel|herring|pollock|haddock|catfish|bass|snapper)\b",
             r"shellfish"),
    "shellfish": (r"\b(shrimp|crab|lobster|clams?|mussels?|oysters?|scallops?|crustaceans|squid|prawns?|crayfish|octopus)\b", r"crabapple"),
    "sesame": (r"\bsesame\b", None),
    "grapefruit": (r"\bgrapefruit\b", None),
    "tyramine": (r"\bcheese, (cheddar|parmesan|blue|gouda|swiss|provolone|gruyere|romano|brie|camembert)\b|\b(salami|pepperoni|chorizo|"
                 r"soy sauce|miso|sauerkraut|kimchi|fish sauce|tempeh)\b|\bbeer\b", None),
    "potassium_supplement": (r"salt substitute", None),
}
MEAT_CATEGORIES = ("Beef Products", "Pork Products", "Poultry Products", "Lamb, Veal, and Game Products", "Sausages and Luncheon Meats")


def food_tags(name: str, category: str | None, per100: dict[str, float]) -> set[str]:
    low = name.lower()
    tags = {t for t, (rx, ex) in TAG_RULES.items() if re.search(rx, low) and not (ex and re.search(ex, low))}
    category = category or ""
    if category in MEAT_CATEGORIES or re.search(r"\b(beef|pork|lamb|veal|chicken|turkey|ham|bacon|sausage|duck|goat|venison|bison)\b", low) \
            and not re.search(r"(broth|stock|bouillon) .*(vegetable)|meatless|imitation|vegetarian", low):
        tags.add("meat")
    if category == "Finfish and Shellfish Products" and "shellfish" not in tags:
        tags.add("fish")
    if per100.get("alcohol_g", 0) > 0.5:
        tags.add("alcohol")
    if per100.get("vitamin_k_ug", 0) >= 100:
        tags.add("high_vitamin_k")
    ca = per100.get("calcium_mg", 0)
    if ca >= 100 and ca >= 0.5 * per100.get("kcal", 0):  # dense enough that a normal serving matters: dairy, tofu, sardines
        tags.add("calcium_rich")
    if per100.get("fiber_g", 0) >= 6:
        tags.add("high_fiber")
    if per100.get("potassium_mg", 0) >= 500:
        tags.add("high_potassium")
    return tags
