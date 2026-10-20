from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from .budget import BudgetConfig
from .io import read_yaml
from .schema import Strict


class TrainConfig(Strict):
    rank: Literal[8, 16] = 8
    alpha: float = Field(gt=0, le=64, default=8)
    dropout: float = Field(ge=0, le=0.2, default=0.0)
    layers: list[int] = Field(default_factory=lambda: [28, 29, 30, 31], min_length=1, max_length=32)
    lr: float = Field(gt=0, le=0.0001, default=0.00001)
    accumulation: int = Field(ge=1, le=32, default=4)
    max_frames: int = Field(ge=8, le=1500, default=750)
    max_steps: int | None = Field(ge=1, le=500, default=None)
    checkpoint_every: int = Field(ge=1, le=100, default=25)
    post_event_weight: float = Field(ge=1, le=3, default=1.0)
    text_pad_weight: float = Field(gt=0, le=1, default=0.1)
    replay_frame_fraction: float = Field(ge=0.1, le=0.2, default=0.15)
    wandb_mode: Literal["offline", "online", "disabled"] = "offline"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    seed: int = 17
    second_pass: Literal[False] = False


class Config(Strict):
    experiment_kind: Literal[
        "smoke_test", "research", "conversation_pilot", "conversation_full"
    ] = "smoke_test"
    min_train_examples: int = Field(ge=4, le=800, default=64)
    max_effective_epochs: float = Field(gt=0, le=3, default=2)
    groups: int = Field(ge=3, le=200, default=100)
    seed: int = 17
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


def load_config(path: Path) -> Config:
    return Config.model_validate(read_yaml(path))
