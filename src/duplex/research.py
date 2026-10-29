"""Research candidates, locked subsets, and readiness checks. No teacher/network/GPU."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import generate, save_examples, solve, validate_splits
from .io import atomic_json, digest, read_json
from .schema import ContextEvent, Example, Fact, Request, Utterance

REQUESTS = {
    "train": [
        "Please arrange an appointment with {p} on {d} at {h}.",
        "Can I see {p} on {d} at {h}?",
        "I would like an appointment on {d} at {h} with {p}.",
        "Please check {p} for {d} at {h}.",
        "Help me find a meeting with {p} on {d} at {h}.",
        "Could we arrange {d} at {h} with {p}?",
    ],
    "dev": [
        "I am hoping to meet {p} at {h} on {d}.",
        "Would {d} at {h} work for an appointment with {p}?",
        "Check whether I can meet {p} on {d} at {h}.",
        "An appointment with {p}, {d} at {h}, please.",
    ],
    "test": [
        "My preferred appointment is {d} at {h}, with {p}.",
        "Find out about seeing {p} at {h} on {d}.",
        "I need some time with {p}; try {d} at {h}.",
        "Could you look into {d}, {h}, for a meeting with {p}?",
    ],
}
DAY_CORRECTIONS = {
    "train": [
        "Actually {d}, same time.",
        "Switch just the day to {d} please.",
        "Use {d} instead and keep the time.",
    ],
    "dev": [
        "Hold on, make the day {d}.",
        "Keep the hour but make that {d}.",
        "I meant {d}; everything else is right.",
    ],
    "test": [
        "Scratch that day. {d}, with the rest unchanged.",
        "On reflection, the day should be {d}.",
        "That day is wrong; replace it with {d}.",
    ],
}
TWO_CORRECTIONS = {
    "train": [
        "Actually {d} at {h} instead.",
        "Change both the day and time to {d} at {h}.",
        "Use {d}, {h}, with the same person.",
    ],
    "dev": [
        "Make that {d} at {h}, please.",
        "Hold on, I meant {h} on {d}.",
        "The day and hour should be {d}, {h}.",
    ],
    "test": [
        "On second thought, {d} at {h}.",
        "Replace those details with {h} on {d}.",
        "Scratch the day and hour: {d}, {h}.",
    ],
}

CANCELS = {
    "train": "Stop, do not continue with this appointment request.",
    "dev": "Hold it, I no longer want to pursue that appointment.",
    "test": "Please abandon this request without booking anything.",
}
AMBIGUOUS = {
    "train": "Actually, that detail might be wrong; ask me before changing anything.",
    "dev": "Wait, I am uncertain about the details. Could we clarify first?",
    "test": "Something in my request needs checking; do not assume what I mean.",
}


def semantic_key(row: Example) -> str:
    return digest(
        {
            "script": [u.model_dump() for u in row.script],
            "events": [
                {
                    "arrival_s": e.arrival_s,
                    "version": e.version,
                    "source": e.source,
                    "facts": [f.model_dump() for f in e.facts],
                }
                for e in row.visible.events
            ],
        }
    )


def candidate_pool(count: int = 400, seed: int = 101) -> list[Example]:
    if not 300 <= count <= 800:
        raise ValueError("Research candidate request must be 300..800")
    grouped: dict[str, list[Example]] = defaultdict(list)
    for row in generate(200, seed):
        grouped[row.group_id].append(row)
    result: list[Example] = []
    seen: dict[str, str] = {}
    for group_index, original in enumerate(grouped.values()):
        first = original[0]
        split = first.split
        initial = Request.model_validate(
            {
                **first.truth.initial.model_dump(),
                "hour": ["3", "4", "5", "6"][(group_index // 6) % 4],
            }
        )
        patch = first.truth.patch
        if patch.hour:
            patch = patch.model_copy(
                update={
                    "hour": [h for h in ("3", "4", "5", "6") if h != initial.hour][group_index % 3]
                }
            )
        final = solve(initial, patch, []).final
        request = REQUESTS[split][group_index % len(REQUESTS[split])].format(
            p=initial.person, d=initial.day, h=initial.hour
        )
        correction = next((u.text for u in first.script if u.kind == "correction"), "")
        if patch.cancel:
            correction = CANCELS[split]
        elif patch.ambiguous:
            correction = AMBIGUOUS[split]
        if not patch.cancel and not patch.ambiguous and (patch.day or patch.hour):
            bank = TWO_CORRECTIONS[split] if patch.hour else DAY_CORRECTIONS[split]
            correction = bank[(group_index // 3) % len(bank)].format(d=final.day, h=final.hour)
        script = [
            Utterance(role="user", kind="request", text=request, start_s=0),
            Utterance(
                role="assistant",
                kind="prefix",
                text=f"I will check {initial.day} at {initial.hour} with {initial.person}.",
                start_s=5,
            ),
        ]
        if correction:
            script.append(
                Utterance(
                    role="user",
                    kind="correction",
                    text=correction,
                    start_s=6 if first.timing == "overlap" else 12,
                )
            )
        siblings = []
        for row in original:
            variant = row.axes[-1]
            events = []
            if row.visible.events:
                status = "unavailable" if variant == "stale" else variant
                facts = [Fact.model_validate({**final.model_dump(), "status": status})]
                if status == "unavailable":
                    alternative = [h for h in ("3", "4", "5", "6") if h != final.hour][
                        (group_index // 4) % 3
                    ]
                    facts.append(
                        Fact.model_validate(
                            dict(
                                person=final.person,
                                day=final.day,
                                hour=alternative,
                                status="available",
                            )
                        )
                    )
                # Keep the distractor separate from the requested person.
                other = next(
                    f.person
                    for e in row.visible.events
                    for f in e.facts
                    if f.person != row.truth.final.person
                )
                # Include the original slot too: facts must not identify the
                # user's corrected choice as an oracle state update.
                prior = Fact(**initial.model_dump(), status="available")
                if all(
                    (f.person, f.day, f.hour) != (prior.person, prior.day, prior.hour)
                    for f in facts
                ):
                    facts.append(prior)
                else:
                    facts.append(
                        Fact(person=other, day=final.day, hour=final.hour, status="available")
                    )
                events = [
                    ContextEvent(
                        event_id=f"r3-{group_index}-2",
                        request_id=row.group_id,
                        version=2,
                        arrival_s=5.25 if group_index % 3 == 0 else 14,
                        facts=facts,
                    )
                ]
                if variant == "stale":
                    events.append(
                        ContextEvent(
                            event_id=f"r3-{group_index}-1",
                            request_id=row.group_id,
                            version=1,
                            arrival_s=15,
                            facts=[Fact(**final.model_dump(), status="available")],
                        )
                    )
            changed = row.model_copy(
                update={
                    "id": "r3-" + row.id,
                    "script": script,
                    "truth": solve(initial, patch, events),
                    "visible": row.visible.model_copy(update={"events": events, "duration_s": 60}),
                    "provenance": {
                        "source": "deterministic-simulator",
                        "license": "CC0-1.0",
                        "version": "appointments-research-v3",
                        "context_timing": "during_prefix"
                        if group_index % 3 == 0
                        else "after_correction",
                    },
                }
            )
            key = semantic_key(changed)
            if key in seen:
                if seen[key] != digest(changed.truth.model_dump()):
                    raise ValueError("Identical observable scenario has contradictory labels")
                continue
            seen[key] = digest(changed.truth.model_dump())
            siblings.append(changed)
        result.extend(siblings)
        if len(result) >= count:
            break
    if len(result) < 300:
        raise ValueError("Generator did not produce enough distinct candidates")
    validate_splits(result)
    return result


def pool_report(rows: list[Example]) -> dict[str, Any]:
    return {
        **validate_splits(rows),
        "semantic_unique": len({semantic_key(r) for r in rows}),
        "status": "candidate_pool_not_validated_audio",
        "rendered_audio_s": None,
        "families": dict(Counter(f"{r.split}:{r.family}" for r in rows)),
        "timing": dict(Counter(r.timing for r in rows)),
        "no_update_controls": sum(not r.visible.events for r in rows),
        "no_correction_controls": sum(
            not any(u.kind == "correction" for u in r.script) for r in rows
        ),
    }


def select_groups(rows: list[Example], split: str, groups: int) -> list[Example]:
    candidates: dict[str, list[Example]] = defaultdict(list)
    for row in rows:
        if row.split == split:
            candidates[row.group_id].append(row)
    selected: list[Example] = []
    covered: set[str] = set()
    for _ in range(min(groups, len(candidates))):

        def axes(items: list[Example]) -> set[str]:
            return (
                {f"family:{r.family}" for r in items}
                | {r.timing for r in items}
                | {a for r in items for a in r.axes}
                | ({"no-update"} if not items[0].visible.events else set())
            )

        group_id = max(sorted(candidates), key=lambda g: len(axes(candidates[g]) - covered))
        items = candidates.pop(group_id)
        covered |= axes(items)
        selected.extend(items)
    return selected


def write_pool(directory: Path, count: int = 400, seed: int = 101) -> dict[str, Any]:
    rows = candidate_pool(count, seed)
    report = pool_report(rows)
    if (directory / "candidate-report.json").exists() and read_json(
        directory / "candidate-report.json"
    )["fingerprint"] != report["fingerprint"]:
        raise ValueError(
            "Candidate/locked subsets changed; preserve them and use a new dataset version"
        )
    directory.mkdir(parents=True, exist_ok=True)
    save_examples(directory / "candidates.jsonl", rows)
    save_examples(directory / "train-dev-candidates.jsonl", [r for r in rows if r.split != "test"])
    atomic_json(directory / "candidate-report.json", report)
    for split in ("train", "dev", "test"):
        subset = [r for r in rows if r.split == split]
        save_examples(directory / f"{split}-candidates.jsonl", subset)
        if split != "train":
            locked = select_groups(rows, split, 3)
            save_examples(directory / f"{split}-locked.jsonl", locked)
            atomic_json(
                directory / f"{split}-lock.json",
                {
                    "kind": "baseline_development" if split == "dev" else "final_held_out",
                    "ids": [r.id for r in locked],
                    "fingerprint": digest([r.model_dump() for r in locked]),
                    "seed": seed,
                    "policy": "Selected by deterministic coverage before rendering/teacher/evaluation; no test-based tuning.",
                },
            )
    return report
