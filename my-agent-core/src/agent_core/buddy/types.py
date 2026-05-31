from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Rarity = Literal["common", "uncommon", "rare", "epic", "legendary"]
Species = Literal[
    "duck", "goose", "blob", "cat", "dragon", "octopus", "owl", "penguin", "turtle",
    "snail", "ghost", "axolotl", "capybara", "cactus", "robot", "rabbit", "mushroom", "chonk",
]
Eye = Literal["·", "✦", "×", "◉", "@", "°"]
Hat = Literal["none", "crown", "tophat", "propeller", "halo", "wizard", "beanie", "tinyduck"]
StatName = Literal["DEBUGGING", "PATIENCE", "CHAOS", "WISDOM", "SNARK"]

RARITIES: tuple[Rarity, ...] = ("common", "uncommon", "rare", "epic", "legendary")
SPECIES: tuple[Species, ...] = (
    "duck", "goose", "blob", "cat", "dragon", "octopus", "owl", "penguin", "turtle",
    "snail", "ghost", "axolotl", "capybara", "cactus", "robot", "rabbit", "mushroom", "chonk",
)
EYES: tuple[Eye, ...] = ("·", "✦", "×", "◉", "@", "°")
HATS: tuple[Hat, ...] = ("none", "crown", "tophat", "propeller", "halo", "wizard", "beanie", "tinyduck")
STAT_NAMES: tuple[StatName, ...] = ("DEBUGGING", "PATIENCE", "CHAOS", "WISDOM", "SNARK")

RARITY_WEIGHTS: dict[Rarity, int] = {"common": 60, "uncommon": 25, "rare": 10, "epic": 4, "legendary": 1}
RARITY_STARS: dict[Rarity, str] = {"common": "★", "uncommon": "★★", "rare": "★★★", "epic": "★★★★", "legendary": "★★★★★"}
RARITY_ACCENT: dict[Rarity, str] = {
    "common": "#8b8378",
    "uncommon": "#7ad66d",
    "rare": "#7fb4ff",
    "epic": "#c084fc",
    "legendary": "#f0bf6a",
}


@dataclass(frozen=True, slots=True)
class CompanionBones:
    rarity: Rarity
    species: Species
    eye: Eye
    hat: Hat
    shiny: bool
    stats: dict[StatName, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StoredCompanion:
    name: str
    personality: str
    hatched_at: int
    muted: bool = False


@dataclass(frozen=True, slots=True)
class Companion:
    name: str
    personality: str
    hatched_at: int
    muted: bool
    rarity: Rarity
    species: Species
    eye: Eye
    hat: Hat
    shiny: bool
    stats: dict[StatName, int]
