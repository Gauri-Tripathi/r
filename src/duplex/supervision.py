"""Provider-free teacher import contract and deterministic supervision validation.

This module performs no network calls and has no teacher SDK dependency.
The teacher is an offline preparation role, never part of student training.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from .data import GENERATOR_VERSION, check_target
from .io import digest, read_jsonl, write_jsonl
from .schema import Example, Strict, TeacherAnswer

PROMPT_VERSION: Literal["scenario-verbalization-v2"] = "scenario-verbalization-v2"
CONTRACT: Literal["duplex.teacher-response.v1"] = "duplex.teacher-response.v1"


class TeacherIdentity(Strict):
    model: str = Field(min_length=1, max_length=160)
    revision: str = Field(min_length=1, max_length=160)
    access: Literal["external", "local", "api", "fixture"]

    @field_validator("model", "revision")
    @classmethod
    def identifier(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,159}", value) or "://" in value:
            raise ValueError(
                "Use a model/revision identifier, never a URL, credential, or free-form config"
            )
        if re.search(r"\b(?:sk-|hf_)[A-Za-z0-9_-]{16,}", value):
            raise ValueError("Credentials do not belong in teacher metadata")
        return value


class TeacherRecord(Strict):
    schema_version: Literal["duplex.teacher-response.v1"] = CONTRACT
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: Literal["scenario-verbalization-v2"] = PROMPT_VERSION
    teacher: TeacherIdentity
    text: str = Field(min_length=1, max_length=400)


def scenario_sha256(row: Example) -> str:
    return digest(
        {
            key: row.model_dump()[key]
            for key in (
                "id",
                "group_id",
                "family",
                "split",
                "seed",
                "timing",
                "visible",
                "script",
                "truth",
            )
        }
    )


def teacher_request(row: Example) -> dict[str, Any]:
    return {
        "schema_version": "duplex.teacher-request.v1",
        "id": row.id,
        "scenario_sha256": scenario_sha256(row),
        "prompt_version": PROMPT_VERSION,
        "instruction": "Write the desired short assistant response after the spoken correction and supplied calendar updates. Preserve unchanged constraints. Never claim a completed booking/cancellation. Use available/unavailable explicitly for those actions, ask for clarification when unknown/ambiguous, and say stop for cancellations. Ask before proceeding. Return text only in the response schema; do not render speech.",
        "scenario": {
            "conversation": [u.model_dump() for u in row.script],
            "timestamped_facts": [e.model_dump() for e in row.visible.events],
            "simulator_expected": row.truth.model_dump(),
        },
        "response_schema": TeacherRecord.model_json_schema(),
    }


def export_requests(rows: list[Example], path: Path) -> None:
    if any(row.split == "test" for row in rows):
        raise ValueError("Teacher preparation excludes final test examples")
    write_jsonl(path, [teacher_request(row) for row in rows])


def checked(
    row: Example, text: str, teacher: TeacherIdentity | None = None, contract: str = "fixture"
) -> Example:
    TeacherAnswer(text=text)
    if re.search(r"[<>\x00-\x08\x0b-\x1f]", text):
        raise ValueError("Teacher output contains control tokens/characters")
    errors = check_target(text, row.truth)
    if errors:
        raise ValueError(f"Teacher target rejected for {row.id}: {', '.join(errors)}")
    identity = teacher or TeacherIdentity(
        model="deterministic-reference", revision=GENERATOR_VERSION, access="fixture"
    )
    provenance = {
        **row.provenance,
        "supervision_kind": "fixture_reference"
        if identity.access == "fixture"
        else "checked_teacher",
        "teacher_model": identity.model,
        "teacher_revision": identity.revision,
        "teacher_access": identity.access,
        "teacher_contract": contract,
        "teacher_prompt_version": PROMPT_VERSION,
        "teacher_scenario_sha256": scenario_sha256(row),
    }
    return row.model_copy(update={"supervision": text, "provenance": provenance})


def imported(rows: list[Example], path: Path) -> list[Example]:
    by_id = {r.id: r for r in rows}
    answers: dict[str, Example] = {}
    for raw in read_jsonl(path):
        if raw.get("id") in answers or raw.get("id") not in by_id:
            raise ValueError("Unknown or duplicated teacher target ID")
        row = by_id[raw["id"]]
        if set(raw) == {"id", "text"}:
            identity = TeacherIdentity(
                model="unspecified", revision="unspecified", access="external"
            )
            answer = checked(row, raw["text"], identity, contract="legacy-unversioned")
        else:
            record = TeacherRecord.model_validate(raw)
            if record.scenario_sha256 != scenario_sha256(row):
                raise ValueError(f"Teacher scenario fingerprint mismatch: {row.id}")
            answer = checked(row, record.text, record.teacher, contract=CONTRACT)
        answers[row.id] = answer
    if set(answers) != set(by_id):
        raise ValueError("Imported target IDs must match dataset exactly")
    return [answers[row.id] for row in rows]


def accepted_records(rows: list[Example], path: Path) -> None:
    result = []
    for row in rows:
        if not row.supervision:
            raise ValueError("No accepted response")
        p = row.provenance
        record = TeacherRecord(
            id=row.id,
            scenario_sha256=p["teacher_scenario_sha256"],
            teacher=TeacherIdentity.model_validate(
                {
                    "model": p["teacher_model"],
                    "revision": p["teacher_revision"],
                    "access": p["teacher_access"],
                }
            ),
            text=row.supervision,
        )
        result.append(record.model_dump())
    write_jsonl(path, result)


def require_teacher(rows: list[Example]) -> None:
    if any(
        r.provenance.get("supervision_kind") != "checked_teacher"
        or r.provenance.get("teacher_contract") != CONTRACT
        or r.provenance.get("teacher_model") == "unspecified"
        or r.provenance.get("teacher_revision") == "unspecified"
        for r in rows
    ):
        raise ValueError(
            "Research preparation needs versioned, checked teacher outputs with model/revision provenance. Reference fixtures and unversioned imports are not the research dataset."
        )
