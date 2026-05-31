from __future__ import annotations

import hashlib
import time
from typing import Callable, TypeVar

from agent_core.buddy.sprites import render_face, render_sprite, sprite_frame_count
from agent_core.buddy.types import (
    EYES,
    HATS,
    RARITIES,
    RARITY_ACCENT,
    RARITY_STARS,
    RARITY_WEIGHTS,
    SPECIES,
    STAT_NAMES,
    Companion,
    CompanionBones,
    Rarity,
    StatName,
    StoredCompanion,
)

SALT = "friend-2026-401"
_T = TypeVar("_T")

_NAME_PREFIXES = ["Nibi", "Momo", "Kiki", "Taro", "Pip", "Lumo", "Bobo", "Nova", "Zuzu", "Miso", "Orbi", "Ruru"]
_NAME_SUFFIXES = ["bit", "paw", "spark", "loop", "bean", "byte", "moss", "wink", "drift", "node", "bloom", "quack"]
_PERSONALITIES = [
    "quietly sarcastic, loyal, and obsessed with clean diffs",
    "gentle but chaotic; celebrates tiny wins with dramatic timing",
    "a tiny debugger spirit who notices risk before anyone else",
    "warm, sleepy, and weirdly good at spotting missing edge cases",
    "curious, snack-motivated, and suspicious of flaky tests",
]


def _hash_string(value: str) -> int:
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big", signed=False)


def _mulberry32(seed: int) -> Callable[[], float]:
    a = seed & 0xFFFFFFFF

    def rng() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) ^ t)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rng


def _pick(rng: Callable[[], float], values: tuple[_T, ...] | list[_T]) -> _T:
    return values[min(len(values) - 1, int(rng() * len(values)))]


def _roll_rarity(rng: Callable[[], float]) -> Rarity:
    total = sum(RARITY_WEIGHTS.values())
    roll_value = rng() * total
    for rarity in RARITIES:
        roll_value -= RARITY_WEIGHTS[rarity]
        if roll_value < 0:
            return rarity
    return "common"


_RARITY_FLOOR: dict[Rarity, int] = {"common": 5, "uncommon": 15, "rare": 25, "epic": 35, "legendary": 50}


def _roll_stats(rng: Callable[[], float], rarity: Rarity) -> dict[StatName, int]:
    floor = _RARITY_FLOOR[rarity]
    peak = _pick(rng, STAT_NAMES)
    dump = _pick(rng, STAT_NAMES)
    while dump == peak:
        dump = _pick(rng, STAT_NAMES)
    stats: dict[StatName, int] = {}
    for name in STAT_NAMES:
        if name == peak:
            stats[name] = min(100, floor + 50 + int(rng() * 30))
        elif name == dump:
            stats[name] = max(1, floor - 10 + int(rng() * 15))
        else:
            stats[name] = floor + int(rng() * 40)
    return stats


def roll(account_uuid: str) -> CompanionBones:
    rng = _mulberry32(_hash_string(f"{account_uuid or 'anon'}{SALT}"))
    rarity = _roll_rarity(rng)
    return CompanionBones(
        rarity=rarity,
        species=_pick(rng, SPECIES),
        eye=_pick(rng, EYES),
        hat="none" if rarity == "common" else _pick(rng, HATS),
        shiny=rng() < 0.01,
        stats=_roll_stats(rng, rarity),
    )


def default_soul(account_uuid: str, bones: CompanionBones) -> StoredCompanion:
    rng = _mulberry32(_hash_string(f"soul:{account_uuid or 'anon'}:{SALT}"))
    name = f"{_pick(rng, _NAME_PREFIXES)}{_pick(rng, _NAME_SUFFIXES)}"
    return StoredCompanion(
        name=name,
        personality=_pick(rng, _PERSONALITIES),
        hatched_at=int(time.time()),
        muted=False,
    )


def merge_companion(stored: StoredCompanion, bones: CompanionBones) -> Companion:
    return Companion(
        name=stored.name,
        personality=stored.personality,
        hatched_at=stored.hatched_at,
        muted=stored.muted,
        rarity=bones.rarity,
        species=bones.species,
        eye=bones.eye,
        hat=bones.hat,
        shiny=bones.shiny,
        stats=bones.stats,
    )


def get_companion(user_context: dict, *, create: bool = False):
    from agent_core.buddy.store import BuddyStore

    user = user_context["user"]
    org = user_context.get("organization") or {}
    account_uuid = str(user.get("account_uuid") or user.get("accountUuid") or user.get("id") or "anon")
    bones = roll(account_uuid)
    store = BuddyStore()
    stored = store.get(user_id=user["id"])
    if stored is None and create:
        stored = default_soul(account_uuid, bones)
        store.upsert(user_id=user["id"], org_id=org.get("id") or "", companion=stored)
    return merge_companion(stored, bones) if stored else None


def companion_payload(companion: Companion | None) -> dict | None:
    if companion is None:
        return None
    return {
        "name": companion.name,
        "personality": companion.personality,
        "hatched_at": companion.hatched_at,
        "muted": companion.muted,
        "rarity": companion.rarity,
        "rarity_stars": RARITY_STARS[companion.rarity],
        "accent": RARITY_ACCENT[companion.rarity],
        "species": companion.species,
        "eye": companion.eye,
        "hat": companion.hat,
        "shiny": companion.shiny,
        "stats": companion.stats,
        "face": render_face(companion),
        "sprite_frames": [render_sprite(companion, i) for i in range(sprite_frame_count(companion.species))],
    }


def build_companion_prompt(user_context: dict) -> str:
    companion = get_companion(user_context, create=False)
    if companion is None or companion.muted:
        return ""
    return (
        "# Companion\n\n"
        f"A small {companion.species} named {companion.name} sits beside the user's input box and occasionally comments in a speech bubble. "
        f"You're not {companion.name} — it's a separate watcher.\n\n"
        f"When the user addresses {companion.name} directly (by name), its bubble will answer. Your job in that moment is to stay out of the way: "
        f"respond in ONE line or less, or just answer any part of the message meant for you. Don't explain that you're not {companion.name} — they know. "
        f"Don't narrate what {companion.name} might say — the bubble handles that."
    )
