"""Labels and supervision cannot be serialized through the runtime input schema."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Day = Literal["Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"]
Person = Literal["Ananya", "Maya", "Sam", "Priya", "Noah", "Isha"]
Hour = Literal["3", "4", "5", "6"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Request(Strict):
    person: Person
    day: Day
    hour: Hour


class Patch(Strict):
    person: Person | None = None
    day: Day | None = None
    hour: Hour | None = None
    cancel: bool = False
    ambiguous: bool = False


class Fact(Strict):
    person: Person
    day: Day
    hour: Hour
    status: Literal["available", "unavailable", "unknown"]


class ContextEvent(Strict):
    event_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    source: Literal["calendar_mock"] = "calendar_mock"
    request_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    arrival_s: float = Field(ge=0, le=300, allow_inf_nan=False)
    version: int = Field(ge=1, le=10000)
    facts: list[Fact] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_facts(self) -> ContextEvent:
        keys = [(f.person, f.day, f.hour) for f in self.facts]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate or contradictory facts in one revision")
        return self


class Utterance(Strict):
    role: Literal["user", "assistant"]
    kind: Literal["request", "prefix", "correction", "response"]
    text: str = Field(min_length=1, max_length=500)
    start_s: float = Field(ge=0, le=300, allow_inf_nan=False)

    @field_validator("text")
    @classmethod
    def safe_text(cls, value: str) -> str:
        if re.search(r"[<>\x00-\x08\x0b-\x1f]", value):
            raise ValueError("Control tokens/characters are not permitted")
        return value


class ModelInput(Strict):
    request_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    user_audio: str | None = None
    events: list[ContextEvent] = Field(max_length=8)
    duration_s: float = Field(gt=0, le=300, allow_inf_nan=False)

    @model_validator(mode="after")
    def scope(self) -> ModelInput:
        if any(e.request_id != self.request_id for e in self.events):
            raise ValueError("Event scope mismatch")
        if len({e.event_id for e in self.events}) != len(self.events):
            raise ValueError("Duplicate event IDs")
        return self


class Truth(Strict):
    initial: Request
    patch: Patch
    final: Request
    action: Literal["offer", "conflict", "clarify", "cancel"]
    alternative: Hour | None = None
    latest_version: int = 0
    unchanged: list[Literal["person", "day", "hour"]]


class Example(Strict):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    group_id: str
    family: str
    split: Literal["train", "dev", "test"]
    seed: int
    timing: Literal["clean", "overlap"]
    axes: list[str]
    provenance: dict[str, str]
    visible: ModelInput
    script: list[Utterance]  # data construction only, never passed to generation
    supervision: str | None = None
    truth: Truth  # evaluation only; never passed to runtime


class Word(Strict):
    text: str
    start_s: float = Field(ge=0, allow_inf_nan=False)
    end_s: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def duration(self) -> Word:
        if self.end_s <= self.start_s:
            raise ValueError("Nonpositive word duration")
        return self


class TeacherAnswer(Strict):
    text: str = Field(min_length=1, max_length=400)
