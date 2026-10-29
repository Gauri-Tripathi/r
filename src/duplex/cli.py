"""One resumable local/Modal CLI. Paid stages are explicit, never fallbacks."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import click
import torch

from . import checkpoint
from .audio import SpeechRenderer, import_audio, render_example
from .budget import Deadline, Ledger, derive_plan
from .config import TrainConfig, load_config
from .data import load_examples, reference, save_examples, validate_splits
from .infer_cli import infer_command
from .io import atomic_json, digest, file_hash, read_json, redact
from .model import (
    ROOT,
    attach,
    load_adapter,
    load_real,
    native,
    prepare_assets,
    tokenizer,
    verify_assets,
)
from .openyap_cli import group as openyap_group
from .research_cli import group as research_group
from .supervision import accepted_records, checked, imported

PATH = click.Path(path_type=Path)
EXISTING = click.Path(exists=True, path_type=Path)


def emit(value: Any) -> None:
    click.echo(json.dumps(redact(value), indent=2))


@click.group()
def cli() -> None:
    """Budgeted speech adaptation; fixtures and real checkpoint stages are distinct."""


@cli.command()
def doctor() -> None:
    """Inspect local capabilities without downloads, secrets, or paid calls."""
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "modal", "wandb", "pydantic", "safetensors")
    }
    import shutil

    emit(
        {
            "versions": versions,
            "python": sys.version.split()[0],
            "cuda": torch.cuda.is_available(),
            "espeak_ng": bool(shutil.which("espeak-ng")),
            "shell_supported": sys.platform != "win32",
            "real_checkpoint_verified": False,
            "prices_checked": "2026-10-31",
            "paid_requests": 0,
        }
    )


@cli.command("prepare-assets")
@click.option("--assets", type=PATH, required=True)
def prepare(assets: Path) -> None:
    """CPU download of exactly pinned gated assets; requires accepted terms/HF read token."""
    emit(prepare_assets(assets))


@cli.command("scenarios")
@click.option("--config", type=EXISTING, default="configs/pilot.yaml")
@click.option("--out", type=PATH, required=True)
def scenarios(config: Path, out: Path) -> None:
    from .data import generate

    cfg = load_config(config)
    rows = generate(cfg.groups, cfg.seed)
    save_examples(out, rows)
    emit(validate_splits(rows))


@cli.command("select")
@click.option("--data", type=EXISTING, required=True)
@click.option("--out", type=PATH, required=True)
@click.option("--split", type=click.Choice(["train", "dev", "test", "train-dev"]), required=True)
@click.option("--groups", type=click.IntRange(1, 200), default=3)
def select(data: Path, out: Path, split: str, groups: int) -> None:
    rows = [
        r
        for r in load_examples(data)
        if r.split in (["train", "dev"] if split == "train-dev" else [split])
    ]
    chosen = sorted({r.group_id for r in rows})[:groups]
    save_examples(out, [r for r in rows if r.group_id in chosen])


@cli.command("targets")
@click.option("--data", type=EXISTING, required=True)
@click.option("--out", type=PATH, required=True)
@click.option("--import-jsonl", type=EXISTING)
@click.option("--teacher-config", type=EXISTING)
@click.option("--cache", type=PATH, default="cache/teacher")
@click.option("--reference-fixture", is_flag=True)
def targets(
    data: Path,
    out: Path,
    import_jsonl: Path | None,
    teacher_config: Path | None,
    cache: Path,
    reference_fixture: bool,
) -> None:
    rows = load_examples(data)
    if sum([bool(import_jsonl), bool(teacher_config), reference_fixture]) != 1:
        raise ValueError(
            "Choose imported JSONL, explicitly budgeted teacher config, or deterministic reference fixture"
        )
    if import_jsonl:
        result = imported(rows, import_jsonl)
    elif teacher_config:
        from .io import read_yaml
        from .teacher import TeacherConfig, api_targets

        result = api_targets(rows, TeacherConfig.model_validate(read_yaml(teacher_config)), cache)
    else:
        result = [checked(r, reference(r.truth)) for r in rows]
    save_examples(out, result)
    accepted_records(result, out.with_suffix(".accepted.jsonl"))
    emit(
        {
            "checked_targets": len(result),
            "source": "deterministic reference (no teacher transfer claim)"
            if reference_fixture
            else "imported"
            if import_jsonl
            else "API",
        }
    )


@cli.command("render")
@click.option("--data", type=EXISTING, required=True)
@click.option("--root", type=PATH, required=True)
@click.option("--assets", type=EXISTING)
@click.option("--backend", type=click.Choice(["espeak", "tone"]), default="espeak")
@click.option("--voice", type=click.Choice(["en-us", "en-gb"]), default="en-us")
def render(data: Path, root: Path, assets: Path | None, backend: str, voice: str) -> None:
    if assets:
        verify_assets(assets, include_base=False)
        tok = tokenizer(assets)
    elif backend == "tone":
        from .fixtures import FixtureTokenizer

        tok = FixtureTokenizer()
    else:
        raise ValueError(
            "Real rendering needs --assets with pinned tokenizer to schedule context delivery"
        )
    renderer = SpeechRenderer(backend, voice)
    rows = [render_example(r, root, renderer, tok) for r in load_examples(data)]
    save_examples(root / "rendered.jsonl", rows)
    emit(
        {
            "rendered": len(rows),
            "backend": backend,
            "version": renderer.version,
            "duration_s": sum(r.visible.duration_s for r in rows),
        }
    )


@cli.command("audio-import")
@click.option("--data", type=EXISTING, required=True)
@click.option("--manifest", type=EXISTING, required=True)
@click.option("--source-root", type=EXISTING, required=True)
@click.option("--root", type=PATH, required=True)
def audio_import(data: Path, manifest: Path, source_root: Path, root: Path) -> None:
    specs = read_json(manifest)
    rows = load_examples(data)
    if set(specs) != {r.id for r in rows}:
        raise ValueError("Audio import IDs must match dataset")
    save_examples(
        root / "rendered.jsonl", [import_audio(r, source_root, specs[r.id], root) for r in rows]
    )


@cli.command("validate")
@click.option("--data", type=EXISTING, required=True)
def validate(data: Path) -> None:
    rows = load_examples(data)
    for row in rows:
        if row.supervision:
            checked(row, row.supervision)
    emit(validate_splits(rows))


@cli.command("encode")
@click.option("--data", type=EXISTING, required=True)
@click.option("--root", type=EXISTING, required=True)
@click.option("--assets", type=EXISTING, required=True)
@click.option("--out", type=PATH, required=True)
@click.option("--cache", type=PATH, required=True)
@click.option("--voice", type=EXISTING)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
def encode(
    data: Path, root: Path, assets: Path, out: Path, cache: Path, voice: Path | None, device: str
) -> None:
    from .codec import build_sample, encode_channel, save_sample
    from .model import FILES

    verify_assets(assets, include_base=False)
    loaders, _, _ = native()
    mimi = loaders.get_mimi(assets / FILES["codec"], device=device).requires_grad_(False)
    tok = tokenizer(assets)
    start = time.perf_counter()
    voice_codes = encode_channel(mimi, voice, "licensed-voice", cache, device) if voice else None
    rows = load_examples(data)
    for row in rows:
        sample = build_sample(row, root, mimi, tok, cache, device, voice_codes)
        save_sample(out / f"{row.id}.safetensors", sample, row)
    elapsed = time.perf_counter() - start
    atomic_json(
        out / "encoding.json",
        {
            "duration_s": sum(r.visible.duration_s for r in rows),
            "wall_s": elapsed,
            "codec_audio_s_per_s": sum(r.visible.duration_s * 2 for r in rows) / elapsed,
            "voice_sha256": None
            if voice is None
            else __import__("duplex.io", fromlist=["file_hash"]).file_hash(voice),
        },
    )


@cli.command("replay-encode")
@click.option("--manifest", type=EXISTING, required=True)
@click.option("--source-root", type=EXISTING, required=True)
@click.option("--assets", type=EXISTING, required=True)
@click.option("--out", type=PATH, required=True)
@click.option("--cache", type=PATH, required=True)
@click.option("--max-audio-s", type=click.FloatRange(1, 300), default=180)
def replay_encode(
    manifest: Path, source_root: Path, assets: Path, out: Path, cache: Path, max_audio_s: float
) -> None:
    """Encode a small licensed ordinary-conversation subset on CPU; no corpus download."""
    from .model import FILES
    from .replay import encode_replay

    verify_assets(assets, include_base=False)
    loaders, _, _ = native()
    mimi = loaders.get_mimi(assets / FILES["codec"], device="cpu").requires_grad_(False)
    emit(
        encode_replay(
            manifest, source_root, out, mimi, tokenizer(assets), cache, "cpu", max_audio_s
        )
    )


def ledger_for(config: Path, ledger: Path) -> tuple[Any, Ledger]:
    cfg = load_config(config)
    return cfg, Ledger(ledger, cfg.budget)


@cli.command("profile")
@click.option("--config", type=EXISTING, default="configs/train.yaml")
@click.option("--assets", type=EXISTING, required=True)
@click.option("--encoded", type=EXISTING, required=True)
@click.option("--data", type=EXISTING, required=True)
@click.option("--root", type=EXISTING, required=True)
@click.option("--run", type=PATH, required=True)
@click.option("--ledger", type=PATH, default="/runs/budget.json")
@click.option("--voice", type=EXISTING)
@click.option(
    "--generation-only",
    is_flag=True,
    help="Reuse a successful smoke test and time one complete train conversation.",
)
@click.option(
    "--eval-examples",
    type=click.IntRange(4, 48),
    default=12,
    help="Predeclared evaluation examples per model, in groups of four.",
)
def profile(
    config: Path,
    assets: Path,
    encoded: Path,
    data: Path,
    root: Path,
    run: Path,
    ledger: Path,
    voice: Path | None,
    generation_only: bool,
    eval_examples: int,
) -> None:
    from .train import calibrate_generation
    from .train import profile as real_profile

    cfg, book = ledger_for(config, ledger)
    if eval_examples % 4:
        raise ValueError("Evaluation examples must be in complete groups of four")
    run.mkdir(parents=True, exist_ok=True)
    # Train example only; test never used to choose profile settings.
    row = next(r for r in load_examples(data) if r.split == "train")
    if generation_only:
        emit(
            calibrate_generation(
                assets,
                encoded,
                run,
                cfg,
                Deadline.from_env(),
                book,
                row.visible,
                root,
                voice,
                eval_examples,
            )
        )
    else:
        emit(
            real_profile(
                assets,
                encoded,
                run,
                cfg,
                Deadline.from_env(),
                book,
                row.visible,
                root,
                voice,
                eval_examples,
            )
        )


@cli.command("plan")
@click.option("--config", type=EXISTING, default="configs/train.yaml")
@click.option("--profile", "profile_path", type=EXISTING, required=True)
@click.option("--remaining-s", type=click.FloatRange(min=1), required=True)
def plan(config: Path, profile_path: Path, remaining_s: float) -> None:
    emit(derive_plan(read_json(profile_path), remaining_s, load_config(config).budget))


@cli.command("train")
@click.option("--config", type=EXISTING, default="configs/train.yaml")
@click.option("--assets", type=EXISTING, required=True)
@click.option("--encoded", type=EXISTING, required=True)
@click.option("--run", type=PATH, required=True)
@click.option("--profile", "profile_path", type=EXISTING, required=True)
@click.option("--ledger", type=PATH, default="/runs/budget.json")
@click.option("--resume", type=EXISTING)
@click.option(
    "--baseline",
    type=EXISTING,
    help="Required locked audible base-baseline report for research runs.",
)
@click.option(
    "--workload-plan", type=EXISTING, help="Complete feasible plan required for research training."
)
def train(
    config: Path,
    assets: Path,
    encoded: Path,
    run: Path,
    profile_path: Path,
    ledger: Path,
    resume: Path | None,
    baseline: Path | None,
    workload_plan: Path | None,
) -> None:
    from .train import run_training

    cfg, book = ledger_for(config, ledger)
    run.mkdir(parents=True, exist_ok=True)
    try:
        emit(
            {
                "checkpoint": str(
                    run_training(
                        assets,
                        encoded,
                        run,
                        cfg,
                        Deadline.from_env(),
                        book,
                        profile_path,
                        resume,
                        baseline,
                        workload_plan,
                    )
                )
            }
        )
    except torch.OutOfMemoryError:
        book.stage(
            "oom",
            {
                "recovery": "No automatic retries; profile shorter COMPLETE sessions; at most one recorded recovery via budget-oom-recovery"
            },
        )
        raise ValueError(
            "A100 OOM. Last completed checkpoint remains valid. Reduce complete-session length, record one recovery, and reprofile; no automatic GPU change."
        ) from None


@cli.command("replay")
@click.option("--assets", type=EXISTING, required=True)
@click.option("--data", type=EXISTING, required=True)
@click.option("--root", type=EXISTING, required=True)
@click.option("--out", type=PATH, required=True)
@click.option("--adapter", type=EXISTING)
@click.option("--voice", type=EXISTING)
@click.option("--limit", type=click.IntRange(1, 48), default=1)
def replay(
    assets: Path,
    data: Path,
    root: Path,
    out: Path,
    adapter: Path | None,
    voice: Path | None,
    limit: int,
) -> None:
    from .inference import generate

    deadline = Deadline.from_env(reserve_s=60)
    lm, mimi, tok = load_real(assets)
    adapter_hash = None
    if adapter:
        meta = checkpoint.adapter_metadata(adapter)
        attach(lm, TrainConfig.model_validate(meta["train_config"]))
        load_adapter(lm, adapter / "adapter.safetensors")
        adapter_hash = file_hash(adapter / "adapter.safetensors")
    for row in load_examples(data)[:limit]:
        if not deadline.can_step(row.visible.duration_s * 2):
            raise ValueError("Not enough reserved evaluation time; stop and persist")
        metadata = read_json(root / "audio" / row.id / "manifest.json")
        if metadata["backend"] == "tone":
            raise ValueError("Tone fixtures are not valid speech evaluation inputs")
        dest = out / row.id
        if (dest / "generation.json").exists():
            prior = read_json(dest / "generation.json")
            if prior["input_fingerprint"] != digest(row.visible.model_dump()):
                raise ValueError("Replay manifest changed")
            if prior.get("user_audio_sha256") != file_hash(root / str(row.visible.user_audio)):
                raise ValueError("Replay user waveform changed or lacks a recorded checksum")
            if prior.get("adapter_sha256") != adapter_hash or prior.get("voice_sha256") != (
                file_hash(voice) if voice else None
            ):
                raise ValueError("Replay model/voice changed; choose a new run directory")
            if prior["generated_audio_sha256"] != file_hash(dest / "generated.wav"):
                raise ValueError("Generated audio checksum differs")
            continue
        generate(
            lm, mimi, tok, row.visible, root, dest, voice, row.seed, adapter_sha256=adapter_hash
        )


@cli.command("evaluate")
@click.option("--data", type=EXISTING, required=True)
@click.option("--root", type=EXISTING, required=True)
@click.option("--base", type=EXISTING, required=True)
@click.option("--adapted", type=EXISTING, required=True)
@click.option("--out", type=PATH, required=True)
@click.option("--base-asr", type=EXISTING)
@click.option("--adapted-asr", type=EXISTING)
@click.option("--training-run", type=EXISTING)
@click.option("--budget-ledger", type=EXISTING)
def evaluate(
    data: Path,
    root: Path,
    base: Path,
    adapted: Path,
    out: Path,
    base_asr: Path | None,
    adapted_asr: Path | None,
    training_run: Path | None,
    budget_ledger: Path | None,
) -> None:
    from .evaluate import audible_text, import_asr, report, score

    transcripts = [import_asr(p) if p else {} for p in (base_asr, adapted_asr)]
    if bool(base_asr) != bool(adapted_asr):
        raise ValueError("Both runs need ASR, or both must be labeled internal proxies")
    rows = load_examples(data)
    if any(r.split != "test" for r in rows):
        raise ValueError("Final held-out report requires test-only examples")
    scored: list[list[dict[str, Any]]] = [[], []]
    sensitivity = []
    for row in rows:
        runs = [read_json(p / row.id / "generation.json") for p in (base, adapted)]
        if runs[0].get("adapter_sha256") is not None or runs[1].get("adapter_sha256") is None:
            raise ValueError("Require a genuine adapter-disabled baseline and an adapted run")
        for key in (
            "input_fingerprint",
            "seed",
            "decoding",
            "voice_sha256",
            "base_revision",
            "runtime_fingerprint",
            "user_audio_sha256",
        ):
            if runs[0][key] != runs[1][key]:
                raise ValueError(f"Unmatched baseline/adapted {key}")
        if runs[0]["user_audio_sha256"] != file_hash(root / str(row.visible.user_audio)):
            raise ValueError("Evaluation user waveform differs from recorded generation")
        if runs[0]["input_fingerprint"] != digest(row.visible.model_dump()):
            raise ValueError("Evaluation inputs do not match the locked manifest")
        sensitivity.append(
            {
                "id": row.id,
                "generated_audio_changed": runs[0]["generated_audio_sha256"]
                != runs[1]["generated_audio_sha256"],
                "internal_text_changed": runs[0]["internal_text_proxy"]
                != runs[1]["internal_text_proxy"],
                "meaning": "Output sensitivity only; not capability improvement",
            }
        )
        timeline = read_json(root / "audio" / row.id / "manifest.json")
        latest = max(row.visible.events, key=lambda e: e.version) if row.visible.events else None
        available = max(timeline["correction_end_s"], latest.arrival_s if latest else 0)
        for index, directory in enumerate((base, adapted)):
            transcript = transcripts[index].get(row.id)
            if base_asr and transcript is None:
                raise ValueError("Missing ASR row")
            text, words, evidence = audible_text(
                runs[index], directory / row.id / "generated.wav", transcript
            )
            if transcript is None:
                text = " ".join(
                    t["piece"] for t in runs[index]["text_tokens"] if t["at_s"] >= available
                )
            item = score(row, text, words, available)
            item.update(
                rtf=runs[index]["rtf"],
                silence_fraction=runs[index]["silence_fraction"],
                truncation=runs[index]["truncated"],
                asr_uncertainty=transcript.uncertainty if transcript else None,
                listening_audio=str(directory / row.id / "generated.wav"),
            )
            scored[index].append(item)
    experiment: dict[str, Any] = {}
    if training_run:
        for name in ("profile.json", "plan.json", "best.json", "replay-selection.json"):
            if (training_run / name).exists():
                experiment[name] = read_json(training_run / name)
    if budget_ledger:
        budget_data = read_json(budget_ledger)
        bc = budget_data["config"]
        elapsed = sum(s["charged_s"] for s in budget_data["sessions"])
        experiment["budget"] = {
            "ledger_sha256": file_hash(budget_ledger),
            "reserved_or_elapsed_s": elapsed,
            "estimated_usd": elapsed * bc["rate_usd_s"] * (1 + bc["safety_margin"])
            + bc["non_gpu_reserve_usd"],
            "exact_billing": False,
            "stages": budget_data["stages"],
            "accounting": budget_data.get("accounting", []),
        }
    report(
        scored[0],
        scored[1],
        out,
        {
            "evidence": "audible_asr" if base_asr else "internal_text_proxy",
            "data": digest([r.model_dump() for r in rows]),
            "provenance": sorted({str(r.provenance) for r in rows}),
            "experiment": experiment,
            "base_revision": runs[0]["base_revision"],
            "runtime_fingerprint": runs[0]["runtime_fingerprint"],
            "adapter_sha256": runs[1]["adapter_sha256"],
            "output_sensitivity": sensitivity,
        },
    )
    emit(
        {
            "result": str(out / "results.json"),
            "evidence": "audible_asr" if base_asr else "internal_text_proxy",
        }
    )


@cli.command("fixture")
@click.option("--out", type=PATH, default="runs/cpu-fixture")
def cpu_fixture(out: Path) -> None:
    from .fixtures import fixture

    emit(fixture(out))


@cli.command("benchmark-replay")
@click.option("--assets", type=EXISTING, required=True)
@click.option("--inputs", type=EXISTING, required=True)
@click.option("--out", type=PATH, required=True)
@click.option("--limit", type=click.IntRange(1, 8), default=2)
@click.option("--voice", type=EXISTING)
def benchmark_replay(assets: Path, inputs: Path, out: Path, limit: int, voice: Path | None) -> None:
    from .benchmark import limited_replay

    lm, mimi, tok = load_real(assets)
    emit(limited_replay(lm, mimi, tok, inputs, out, limit, Deadline.from_env(60), voice))


@cli.command("shell")
@click.option("--config", type=EXISTING, default="configs/train.yaml")
@click.option("--ledger", type=PATH, default="runs/budget.json")
@click.option("--seconds", type=click.IntRange(1981, 3300), default=3300)
@click.option(
    "--stage", type=click.Choice(["interactive", "doctor", "openyap"]), default="interactive"
)
def shell(config: Path, ledger: Path, seconds: int, stage: str) -> None:
    from .launcher import launch

    launch(config, ledger, seconds, stage)


@cli.command("budget-status")
@click.option("--ledger", type=EXISTING, required=True)
def budget_status(ledger: Path) -> None:
    emit(read_json(ledger))


@cli.command("budget-account")
@click.option("--config", type=EXISTING, default="configs/train.yaml")
@click.option("--ledger", type=PATH, default="runs/budget.json")
@click.option("--remaining-credit", type=click.FloatRange(min=0), required=True)
@click.option("--metered-usd", type=click.FloatRange(min=0), required=True)
def budget_account(config: Path, ledger: Path, remaining_credit: float, metered_usd: float) -> None:
    """Local append-only accounting; no paid execution or remote mutation."""
    _, book = ledger_for(config, ledger)
    book.account(remaining_credit, metered_usd)
    emit(read_json(ledger)["accounting"][-1])


@cli.command("budget-close")
@click.option("--config", type=EXISTING, default="configs/train.yaml")
@click.option("--ledger", type=EXISTING, required=True)
@click.option("--session", required=True)
@click.option("--confirmed-stopped", is_flag=True)
@click.option("--release-cloud-lease", is_flag=True)
def budget_close(
    config: Path, ledger: Path, session: str, confirmed_stopped: bool, release_cloud_lease: bool
) -> None:
    _, book = ledger_for(config, ledger)
    book.close(session, confirmed_stopped)
    if release_cloud_lease:
        import modal

        leases = modal.Dict.from_name("duplex-gpu-reservation")
        value = leases.get("active")
        if not value or value["session"] != session:
            raise ValueError("Cloud lease differs; do not clear another reservation")
        leases.pop("active")


@cli.command("budget-oom-recovery")
@click.option("--config", type=EXISTING, default="configs/train.yaml")
@click.option("--ledger", type=EXISTING, required=True)
def budget_oom(config: Path, ledger: Path) -> None:
    _, book = ledger_for(config, ledger)
    book.oom_recovery()
    emit({"recorded": "one allowed recovery; reduce complete session length and reprofile"})


@cli.command("budget-reconcile")
@click.option("--config", type=EXISTING, default="configs/train.yaml")
@click.option("--ledger", type=EXISTING, required=True)
@click.option("--session", required=True)
@click.option("--confirmed-stopped", is_flag=True)
def budget_reconcile(config: Path, ledger: Path, session: str, confirmed_stopped: bool) -> None:
    """After dashboard termination check, close BOTH ledgers and clear the project lease."""
    if not confirmed_stopped:
        raise ValueError(
            "Verify the actual GPU shell/app is stopped in Modal before reconciliation"
        )
    import modal

    cfg, local = ledger_for(config, ledger)
    cloud = modal.Dict.from_name("duplex-gpu-reservation")
    active = cloud.get("active")
    if not active or active["session"] != session:
        raise ValueError("Cloud lease differs from requested session")
    volume = modal.Volume.from_name("duplex-runs")
    payload = b"".join(volume.read_file("budget.json"))
    if len(payload) > 1024 * 1024:
        raise ValueError("Oversized remote budget ledger")
    remote_file = ledger.parent / "remote-budget.json"
    atomic_json(remote_file, json.loads(payload))
    remote_book = Ledger(remote_file, cfg.budget)
    stopped = time.time()
    remote_book.close(session, True, now=stopped)
    with volume.batch_upload(force=True) as upload:
        upload.put_file(remote_file, "budget.json")
    local.close(session, True, now=stopped)
    cloud.pop("active")
    emit(
        {
            "session": session,
            "both_ledgers_closed": True,
            "termination": "operator confirmed; no automatic billing guarantee",
        }
    )


@cli.command("export")
@click.option("--checkpoint", "source", type=EXISTING, required=True)
@click.option("--out", type=PATH, required=True)
def export(source: Path, out: Path) -> None:
    if not checkpoint.complete(source).get("real_checkpoint"):
        raise ValueError("Fixture adapters are not publishable speech adapters")
    checkpoint.export(source, out, ROOT / "THIRD_PARTY_NOTICES.md")


@cli.command("upload")
@click.option("--release", type=EXISTING, required=True)
@click.option("--repo-id", required=True)
@click.option("--terms-reviewed", is_flag=True)
def upload(release: Path, repo_id: str, terms_reviewed: bool) -> None:
    if not terms_reviewed:
        raise ValueError("Review model/data/voice redistribution terms before explicit upload")
    from huggingface_hub import HfApi

    from .io import file_hash

    expected = {"adapter.safetensors", "adapter.json", "THIRD_PARTY_NOTICES.md", "README.md"}
    manifest = read_json(release / "release.json")
    if set(manifest["files"]) != expected:
        raise ValueError("Release allowlist differs")
    for name, checksum in manifest["files"].items():
        if file_hash(release / name) != checksum:
            raise ValueError("Release checksum mismatch")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_folder(
        repo_id=repo_id, folder_path=release, allow_patterns=sorted(expected | {"release.json"})
    )


def main() -> None:
    try:
        cli()
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        # Do not stringify structured validation inputs, HTTP bodies, or arbitrary dataset payloads.
        message = str(exc) if type(exc) in (ValueError, FileNotFoundError) else type(exc).__name__
        click.echo("Error: " + str(redact(message))[:1000], err=True)
        raise SystemExit(1) from None


cli.add_command(research_group)
cli.add_command(infer_command)
cli.add_command(openyap_group)


if __name__ == "__main__":
    main()
