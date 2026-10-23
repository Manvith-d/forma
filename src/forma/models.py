"""The shared state the agents read and write."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Goal = Literal["lose_fat", "build_muscle", "endurance", "general_fitness"]
Equipment = Literal["full_gym", "home_dumbbells", "bodyweight"]
Level = Literal["beginner", "intermediate", "advanced"]
REQUIRED = ("age", "sex", "height_cm", "weight_kg", "goal", "days_per_week", "minutes_per_session")


class Profile(BaseModel):
    name: str | None = None
    age: int | None = Field(default=None, ge=13, le=100)
    sex: Literal["male", "female"] | None = None
    height_cm: float | None = Field(default=None, ge=120, le=230)
    weight_kg: float | None = Field(default=None, ge=35, le=300)
    goal: Goal | None = None
    target_change_kg: float | None = None  # negative to lose
    weeks: int | None = Field(default=None, ge=2, le=104)
    days_per_week: int | None = Field(default=None, ge=1, le=7)
    minutes_per_session: int | None = Field(default=None, ge=10, le=180)
    modalities: list[str] = Field(default_factory=list)  # strength, running, cycling, swimming, walking, rowing, yoga, hiit
    equipment: Equipment | None = None
    experience: Level | None = None
    activity_level: Literal["sedentary", "light", "moderate", "very"] | None = None  # outside of training
    diet: list[str] = Field(default_factory=list)  # vegetarian, vegan, pescatarian
    allergies: list[str] = Field(default_factory=list)  # dairy, egg, peanut, tree_nut, soy, gluten, fish, shellfish, sesame
    dislikes: list[str] = Field(default_factory=list)

    def missing(self) -> list[str]:
        return [f for f in REQUIRED if getattr(self, f) is None]


class Medical(BaseModel):
    screened: bool = False
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)  # generic names
    unrecognized_medications: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    injuries: list[str] = Field(default_factory=list)
    pregnant: bool | None = None


class Citation(BaseModel):
    source_id: str
    title: str
    location: str | None = None
    url: str | None = None
    snippet: str


class Constraints(BaseModel):
    avoid_food_tags: set[str] = Field(default_factory=set)
    avoid_exercise_tags: set[str] = Field(default_factory=set)
    steady_nutrients: set[str] = Field(default_factory=set)
    spacing: list[dict] = Field(default_factory=list)
    min_carb_g: float | None = None
    max_deficit_pct: float | None = None
    no_deficit: bool = False
    protein_cap_g_per_kg: float | None = None
    hydration: bool = False
    intensity: Literal["rpe", "any"] = "any"
    max_intensity: Literal["light", "moderate", "vigorous"] = "vigorous"
    pre_exercise: list[str] = Field(default_factory=list)
    clearance: Literal["none", "vigorous", "required"] = "none"
    add_balance: bool = False
    notes: list[dict] = Field(default_factory=list)  # {rule_id, message, citations}
