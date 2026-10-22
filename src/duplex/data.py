"""Deterministic simulator and split-before-rendering scenario families."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

from .io import atomic_json, digest, read_jsonl, write_jsonl
from .schema import ContextEvent, Example, Fact, ModelInput, Patch, Request, Truth, Utterance

GENERATOR_VERSION = "appointments-v1"
FAMILIES = {
    "train": ("actually", "instead", "switch", "cancel", "ambiguous", "tool_only"),
    "dev": ("make_that", "hold_on", "cancel_dev", "ambiguous_dev", "tool_only_dev"),
    "test": (
        "on_second_thought",
        "scratch_that",
        "cancel_test",
        "ambiguous_test",
        "tool_only_test",
    ),
}
PHRASES = {
    "actually": "Actually {day}, same time.",
    "instead": "Use {day} at {hour} instead.",
    "switch": "Switch the day to {day}, keep everything else.",
    "make_that": "Make that {day} at {hour}.",
    "hold_on": "Hold on, {day}, same time please.",
    "on_second_thought": "On second thought, {day} at {hour}.",
    "scratch_that": "Scratch that day. {day}, same time.",
}


def solve(initial: Request, patch: Patch, events: list[ContextEvent]) -> Truth:
    values = initial.model_dump()
    for key in ("person", "day", "hour"):
        value = getattr(patch, key)
        if value is not None:
            values[key] = value
    final = Request.model_validate(values)
    latest = max(events, key=lambda e: e.version) if events else None
    relevant = (
        [
            f
            for f in latest.facts
            if (f.person, f.day, f.hour) == (final.person, final.day, final.hour)
        ]
        if latest
        else []
    )
    status = relevant[-1].status if relevant else "unknown"
    alternative = (
        next(
            (
                f.hour
                for f in latest.facts
                if f.person == final.person
                and f.day == final.day
                and f.status == "available"
                and f.hour != final.hour
            ),
            None,
        )
        if latest
        else None
    )
    action: Literal["offer", "conflict", "clarify", "cancel"] = (
        "cancel"
        if patch.cancel
        else "clarify"
        if patch.ambiguous or status == "unknown"
        else "offer"
        if status == "available"
        else "conflict"
    )
    return Truth(
        initial=initial,
        patch=patch,
        final=final,
        action=action,
        alternative=alternative,
        latest_version=latest.version if latest else 0,
        unchanged=[key for key in ("person", "day", "hour") if getattr(patch, key) is None],
    )


def reference(truth: Truth) -> str:
    r = truth.final
    if truth.action == "cancel":
        return f"I will stop this request for {r.person} on {r.day} at {r.hour}. No booking or cancellation has been completed."
    if truth.action == "clarify":
        return f"I need clarification for {r.person} on {r.day} at {r.hour}. Could you confirm the request and availability?"
    if truth.action == "offer":
        return f"{r.day} at {r.hour} with {r.person} is available. Would you like to proceed?"
    if truth.alternative:
        return f"{r.day} at {r.hour} with {r.person} is unavailable. Would {truth.alternative} on {r.day} with {r.person} work?"
    return f"{r.day} at {r.hour} with {r.person} is unavailable. Would you like another time?"


def normalize(text: str) -> str:
    value = text.lower()
    for word, number in (("three", "3"), ("four", "4"), ("five", "5"), ("six", "6")):
        value = re.sub(rf"\b{word}\b", number, value)
    value = re.sub(r"\b([3-6])(?::00)?\s*(?:p\.?m\.?)\b", r"\1", value)
    return re.sub(r"[^a-z0-9?]+", " ", value).strip()


def check_target(text: str, truth: Truth) -> list[str]:
    """Conservative vocabulary checker. Rejection is preferable to unverified supervision.

    This is not semantic proof; shortlisted imported/API targets still merit listening review.
    """
    n = normalize(text)
    errors = []
    r = truth.final
    for value in (r.person, r.day, r.hour):
        if not re.search(rf"\b{value.lower()}\b", n):
            errors.append(f"missing constraint: {value}")
    for value in (
        "Ananya",
        "Maya",
        "Sam",
        "Priya",
        "Noah",
        "Isha",
        "Friday",
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "3",
        "4",
        "5",
        "6",
    ):
        allowed = {r.person, r.day, r.hour}
        if truth.action == "conflict" and truth.alternative:
            allowed.add(truth.alternative)
        if value not in allowed and re.search(rf"\b{value.lower()}\b", n):
            errors.append(f"unsupported value: {value}")
    if re.search(r"\b(booked|confirmed|reserved|scheduled|cancelled|canceled|done)\b", n):
        errors.append("unsupported commitment")
    if truth.action != "cancel" and "?" not in text:
        errors.append("must ask before proceeding")
    if truth.action == "offer" and ("available" not in n or "unavailable" in n):
        errors.append("missing available fact")
    if truth.action == "conflict" and "unavailable" not in n:
        errors.append("missing conflict")
    if truth.action == "clarify" and (
        not re.search(r"clarif|confirm|check|uncertain|unknown", n)
        or re.search(r"\bis available\b", n)
    ):
        errors.append("must clarify uncertainty")
    if (
        truth.action == "conflict"
        and truth.alternative
        and not re.search(rf"\b{truth.alternative}\b", n)
    ):
        errors.append("missing alternative")
    if truth.action == "cancel" and "stop" not in n:
        errors.append("must stop request without claiming side effect")
    return errors


def generate(groups: int = 100, seed: int = 17) -> list[Example]:
    if not 3 <= groups <= 200:
        raise ValueError("Use 3..200 groups (12..800 candidates)")
    rng = random.Random(seed)
    rows = []
    splits: list[Literal["train", "dev", "test"]] = ["train", "dev", "test"]
    for group in range(groups):
        split = splits[group] if group < 3 else rng.choices(splits, [8, 1, 1])[0]
        family = FAMILIES[split][(group // 3) % len(FAMILIES[split])]
        people = {"train": ["Ananya", "Maya"], "dev": ["Sam", "Priya"], "test": ["Noah", "Isha"]}[
            split
        ]
        days = {
            "train": ["Friday", "Saturday"],
            "dev": ["Sunday", "Monday"],
            "test": ["Tuesday", "Wednesday"],
        }[split]
        initial = Request.model_validate(
            dict(person=rng.choice(people), day=days[0], hour=rng.choice(["3", "4"]))
        )
        two = family in ("instead", "make_that", "on_second_thought")
        patch = Patch.model_validate(dict(day=days[1], hour="5" if two else None))
        if family.startswith("cancel"):
            patch = Patch(cancel=True)
        elif family.startswith("ambiguous"):
            patch = Patch(ambiguous=True)
        elif family.startswith("tool_only"):
            patch = Patch()
        request_id = f"req-{seed}-{group:04d}"
        timing: Literal["clean", "overlap"] = "overlap" if group % 2 else "clean"
        final = solve(initial, patch, []).final
        correction = (
            {
                "train": "Stop this request please.",
                "dev": "Please drop this appointment request.",
                "test": "Never mind, stop trying to arrange it.",
            }[split]
            if patch.cancel
            else {
                "train": "Actually, some other time.",
                "dev": "Could we do it later instead?",
                "test": "Something else might work, maybe.",
            }[split]
            if patch.ambiguous
            else PHRASES.get(family, "").format(day=days[1], hour=patch.hour or initial.hour)
        )
        script = [
            Utterance(
                role="user",
                kind="request",
                text=f"Please arrange an appointment with {initial.person} on {initial.day} at {initial.hour}.",
                start_s=0,
            ),
            Utterance(
                role="assistant",
                kind="prefix",
                text=f"Let me check {initial.day} at {initial.hour} with {initial.person}.",
                start_s=5,
            ),
        ]
        if correction:
            script.append(
                Utterance(
                    role="user",
                    kind="correction",
                    text=correction,
                    start_s=6 if timing == "overlap" else 10,
                )
            )
        for variant in ("available", "unavailable", "unknown", "stale"):
            facts = [
                Fact(
                    **final.model_dump(),
                    status=cast(
                        Literal["available", "unavailable", "unknown"],
                        "unavailable" if variant == "stale" else variant,
                    ),
                )
            ]
            if variant in ("unavailable", "stale"):
                facts.append(Fact(person=final.person, day=final.day, hour="6", status="available"))
            # A distractor entity is never an alternative for this request.
            facts.append(
                Fact(
                    person=cast(Any, people[1] if final.person == people[0] else people[0]),
                    day=final.day,
                    hour=final.hour,
                    status="available",
                )
            )
            events = [
                ContextEvent(
                    event_id=f"e-{group}-2",
                    request_id=request_id,
                    arrival_s=12,
                    version=2,
                    facts=facts,
                )
            ]
            if variant == "stale":
                events.append(
                    ContextEvent(
                        event_id=f"e-{group}-1",
                        request_id=request_id,
                        arrival_s=13,
                        version=1,
                        facts=[Fact(**final.model_dump(), status="available")],
                    )
                )
            # Explicit no-update control groups, not merely stale events.
            if group % 11 == 10:
                events = []
            truth = solve(initial, patch, events)
            rows.append(
                Example(
                    id=f"s-{seed}-{group:04d}-{variant}",
                    group_id=request_id,
                    family=family,
                    split=split,
                    seed=seed,
                    timing=timing,
                    axes=[
                        "family_holdout",
                        "entity_day_holdout",
                        "two_fields" if two else "single_or_control",
                        variant,
                    ],
                    provenance={
                        "source": "simulator",
                        "license": "CC0-1.0",
                        "version": GENERATOR_VERSION,
                    },
                    visible=ModelInput(request_id=request_id, events=events, duration_s=45),
                    script=script,
                    truth=truth,
                )
            )
    validate_splits(rows)
    return rows


def validate_splits(rows: list[Example]) -> dict[str, Any]:
    groups: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    ids: set[str] = set()
    for row in rows:
        if row.id in ids:
            raise ValueError("Duplicate example ID")
        ids.add(row.id)
        groups[row.group_id].add(row.split)
        families[row.family].add(row.split)
    if any(len(v) != 1 for v in (*groups.values(), *families.values())):
        raise ValueError("Split leakage across counterfactual group or template family")
    return {
        "examples": len(rows),
        "groups": len(groups),
        "splits": {s: sum(r.split == s for r in rows) for s in ("train", "dev", "test")},
        "fingerprint": digest([r.model_dump() for r in rows]),
    }


def save_examples(path: Path, rows: list[Example]) -> None:
    report = validate_splits(rows)
    write_jsonl(path, [r.model_dump() for r in rows])
    atomic_json(path.with_suffix(".manifest.json"), report)


def load_examples(path: Path) -> list[Example]:
    rows = [Example.model_validate(row) for row in read_jsonl(path)]
    validate_splits(rows)
    return rows
