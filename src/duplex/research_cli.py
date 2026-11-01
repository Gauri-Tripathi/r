"""Correction and grounding experiment commands."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from .audio import SpeechRenderer
from .baseline import score_baseline
from .config import load_config
from .data import load_examples
from .experiment import aggregate_plan
from .io import atomic_json, read_json, redact
from .model import FILES, native, tokenizer, verify_assets
from .preparation import prepare_evaluation, prepare_training
from .research import write_pool
from .supervision import export_requests, require_teacher

P = click.Path(path_type=Path)
E = click.Path(path_type=Path, exists=True)


def emit(value: object) -> None:
    click.echo(json.dumps(redact(value), indent=2))


@click.group("research")
def group() -> None:
    """Prepare and run the correction and grounding experiment."""


@group.command("candidates")
@click.option("--out", type=P, required=True)
@click.option("--count", type=click.IntRange(300, 800), default=400)
@click.option("--seed", type=int, default=101)
def candidates(out: Path, count: int, seed: int) -> None:
    report = write_pool(out, count, seed)
    rows = load_examples(out / "train-dev-candidates.jsonl")
    export_requests(rows, out / "teacher-requests.jsonl")
    emit(
        {
            **report,
            "teacher_requests": str(out / "teacher-requests.jsonl"),
            "teacher_outputs_available": False,
        }
    )


@group.command("teacher-requests")
@click.option("--data", type=E, required=True)
@click.option("--out", type=P, required=True)
def requests(data: Path, out: Path) -> None:
    rows = load_examples(data)
    export_requests(rows, out)
    emit({"requests": len(rows), "provider_selected": False, "network_calls": 0})


@group.command("prepare-eval")
@click.option("--data", type=E, required=True)
@click.option("--root", type=P, required=True)
@click.option("--assets", type=E, required=True)
def prepare_eval(data: Path, root: Path, assets: Path) -> None:
    verify_assets(assets, include_base=False, include_codec=False)
    emit(prepare_evaluation(load_examples(data), root, SpeechRenderer("espeak"), tokenizer(assets)))


@group.command("prepare")
@click.option(
    "--data",
    type=E,
    required=True,
    help="Versioned accepted teacher targets from duplex targets --import-jsonl.",
)
@click.option("--root", type=P, required=True)
@click.option("--assets", type=E, required=True)
@click.option("--cache", type=P, required=True)
@click.option("--max-audio-s", type=click.FloatRange(1, 10800), required=True)
@click.option("--measure-groups", type=click.IntRange(1, 3))
@click.option("--measurement", type=E)
def prepare(
    data: Path,
    root: Path,
    assets: Path,
    cache: Path,
    max_audio_s: float,
    measure_groups: int | None,
    measurement: Path | None,
) -> None:
    rows = load_examples(data)
    require_teacher(rows)
    if measure_groups is None:
        if measurement is None:
            raise ValueError(
                "Measure a small preparation subset first with --measure-groups; then select actual audio duration"
            )
        measured = read_json(measurement)
        if (
            measured["kind"] != "preparation_measurement_only"
            or measured["codec"]["cache_misses"] < 1
        ):
            raise ValueError(
                "Need actual cold codec preparation measurement, not warm-cache throughput"
            )
    verify_assets(assets, include_base=False)
    loaders, _, _ = native()
    started = time.perf_counter()
    mimi = loaders.get_mimi(assets / FILES["codec"], device="cpu").requires_grad_(False)
    load_s = time.perf_counter() - started
    result = prepare_training(
        rows,
        root,
        SpeechRenderer("espeak"),
        tokenizer(assets),
        mimi,
        cache,
        max_audio_s,
        measure_groups,
    )
    result["codec_loading_s"] = load_s
    atomic_json(root / "prepared.json", result)
    emit(result)


@group.command("baseline")
@click.option("--data", type=E, required=True)
@click.option("--root", type=E, required=True)
@click.option("--generations", type=E, required=True)
@click.option(
    "--asr",
    type=E,
    required=True,
    help="Externally cached speech transcription, hash-matched to each generated WAV.",
)
@click.option("--out", type=P, required=True)
def baseline(data: Path, root: Path, generations: Path, asr: Path, out: Path) -> None:
    emit(score_baseline(load_examples(data), root, generations, asr, out))


@group.command("plan")
@click.option("--data-manifest", type=E, required=True)
@click.option("--profile", type=E, required=True)
@click.option("--ledger", type=E, required=True)
@click.option("--config", type=E, default="configs/research.yaml")
@click.option(
    "--remaining-credit",
    type=click.FloatRange(min=0.01),
    required=True,
    help="Current dashboard amount; do not reuse the original balance.",
)
@click.option("--extra-preparation-usd", type=click.FloatRange(min=0), default=0)
@click.option(
    "--previous-metered-usd",
    type=click.FloatRange(min=0),
    required=True,
    help="Cumulative project usage; workspace total is a conservative upper bound if attribution is unavailable.",
)
@click.option("--out", type=P, required=True)
@click.option("--dev-evaluation-lock", type=E, required=True)
@click.option("--test-evaluation-lock", type=E, required=True)
def plan(
    data_manifest: Path,
    profile: Path,
    ledger: Path,
    config: Path,
    remaining_credit: float,
    previous_metered_usd: float,
    extra_preparation_usd: float,
    dev_evaluation_lock: Path,
    test_evaluation_lock: Path,
    out: Path,
) -> None:
    result = aggregate_plan(
        data_manifest,
        profile,
        ledger,
        load_config(config),
        remaining_credit,
        previous_metered_usd,
        dev_evaluation_lock,
        test_evaluation_lock,
        extra_preparation_usd,
    )
    atomic_json(out, result)
    emit(result)
