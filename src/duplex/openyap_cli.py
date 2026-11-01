"""OpenYap preparation, training and reporting commands."""

from __future__ import annotations

import time
from pathlib import Path

import click
import torch

from .io import atomic_json, file_hash, read_jsonl, write_jsonl
from .model import FILES, native, tokenizer, verify_assets
from .openyap import CONVERSATION_SYSTEM, prepare
from .replay import encode_replay

P = click.Path(path_type=Path)
E = click.Path(path_type=Path, exists=True)


@click.group("openyap")
def group() -> None:
    """Real speech conversation pilot; no fabricated task labels or teacher."""


@group.command("prepare")
@click.option("--source", type=E, default="data/openyap/source")
@click.option("--root", type=P, default="data/openyap/pilot-v1")
@click.option("--assets", type=E, default="cache/openyap-assets")
def prepare_command(source: Path, root: Path, assets: Path) -> None:
    verify_assets(assets, include_base=False, include_codec=False)
    click.echo(prepare(source, root, tokenizer(assets)))


@group.command("full-prepare")
@click.option("--source", type=E, default="data/openyap/source")
@click.option("--root", type=P, default="data/openyap/full-v1")
@click.option("--assets", type=E, default="cache/openyap-assets")
@click.option(
    "--split-mode", type=click.Choice(["conversation", "speaker"]), default="conversation"
)
@click.option("--plan-only", is_flag=True)
def full_prepare_command(
    source: Path, root: Path, assets: Path, split_mode: str, plan_only: bool
) -> None:
    from .openyap_full import full_prepare

    verify_assets(assets, include_base=False, include_codec=False)
    click.echo(full_prepare(source, root, tokenizer(assets), split_mode, plan_only))


@group.command("full-encode")
@click.option("--root", type=E, required=True)
@click.option("--assets", type=E, required=True)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
@click.option("--chunk-frames", type=click.IntRange(1, 100), default=25)
@click.option("--max-wall-s", type=click.IntRange(60, 86400), default=2400)
@click.option("--limit", type=click.IntRange(1, 10000))
def full_encode_command(
    root: Path, assets: Path, device: str, chunk_frames: int, max_wall_s: int, limit: int | None
) -> None:
    from .openyap_full_encode import full_encode

    click.echo(full_encode(root, assets, device, chunk_frames, max_wall_s, limit))


@group.command("encode")
@click.option("--root", type=E, default="data/openyap/pilot-v1")
@click.option("--assets", type=E, default="cache/openyap-assets")
@click.option("--limit", type=click.IntRange(1, 100))
def encode_command(root: Path, assets: Path, limit: int | None) -> None:
    verify_assets(assets, include_base=False)
    torch.set_num_threads(2)
    rows = read_jsonl(root / "replay.jsonl")
    selected = rows[:limit] if limit else rows
    out = root / ("codec-measurement" if limit else "encoded")
    write_jsonl(out / "selection.jsonl", selected)
    started = time.perf_counter()
    loaders, _, _ = native()
    mimi = loaders.get_mimi(assets / FILES["codec"], device="cpu").requires_grad_(False)
    load_s = time.perf_counter() - started
    result = encode_replay(
        out / "selection.jsonl",
        root,
        out,
        mimi,
        tokenizer(assets),
        root / "mimi-cache",
        "cpu",
        sum(r["duration_s"] for r in selected),
        system_prompt=CONVERSATION_SYSTEM,
    )
    result.update(
        kind="openyap_codec_measurement" if limit else "openyap_encoded_pilot",
        loading_s=load_s,
        wall_s=time.perf_counter() - started,
        source_manifest_sha256=file_hash(root / "replay.jsonl"),
        encoded={p.name: file_hash(p) for p in out.glob("*.safetensors")},
    )
    atomic_json(out / "encoding.json", result)
    click.echo(result)


@group.command("pilot")
@click.option("--root", type=E, required=True)
@click.option("--assets", type=E, required=True)
@click.option("--run", type=P, required=True)
@click.option("--ledger", type=E, default="/runs/budget.json")
def pilot_command(root: Path, assets: Path, run: Path, ledger: Path) -> None:
    from .openyap_train import pilot

    click.echo(pilot(root, assets, run, ledger))


@group.command("full-train")
@click.option("--root", type=E, required=True)
@click.option("--assets", type=E, required=True)
@click.option("--run", type=P, required=True)
@click.option("--config", type=E, default="configs/openyap-full.yaml")
@click.option("--ledger", type=E, default="/runs/budget.json")
@click.option("--resume", is_flag=True)
def full_train_command(
    root: Path, assets: Path, run: Path, config: Path, ledger: Path, resume: bool
) -> None:
    from .config import load_config
    from .openyap_train import pilot

    if load_config(config).experiment_kind != "conversation_full":
        raise ValueError("full-train requires the full-data configuration")
    result = pilot(root, assets, run, ledger, config, resume)
    click.echo(
        {
            key: result[key]
            for key in ("kind", "optimizer_steps", "test_loss_delta")
            if key in result
        }
        if result.get("kind") == "openyap_full_training"
        else result
    )


@group.command("report")
@click.option("--run", type=E, default="runs/openyap/pilot-v1")
def report_command(run: Path) -> None:
    """Verify saved files and create a local audio comparison page; no GPU/network."""
    from .openyap_report import report

    click.echo(report(run))


@group.command("full-session")
@click.option("--root", type=E, required=True)
@click.option("--assets", type=E, required=True)
@click.option("--run", type=P, required=True)
@click.option("--config", type=E, default="configs/openyap-full.yaml")
@click.option("--ledger", type=E, default="/runs/budget.json")
def full_session_command(root: Path, assets: Path, run: Path, config: Path, ledger: Path) -> None:
    """Encode/resume, then train/resume within this existing budgeted GPU shell."""
    from .budget import Deadline
    from .io import read_json
    from .openyap_full_encode import full_encode
    from .openyap_train import pilot

    if (run / "results.json").exists():
        click.echo("Already complete. Use openyap report; no training repeated.")
        return
    encoded = full_encode(root, assets, "cuda", 25, 2400)
    if not encoded["complete"]:
        click.echo(
            "Encoding checkpointed. Exit, reconcile, and repeat full-session in the next budgeted shell."
        )
        return
    if Deadline.from_env().remaining() < 900:
        click.echo(
            "Encoding complete. Exit/reconcile and repeat full-session in a fresh budgeted shell for training."
        )
        return
    resume = any((run / "checkpoints").glob("step-*/complete.json"))
    result = pilot(root, assets, run, ledger, config, resume)
    click.echo(
        read_json(run / "run-status.json") if (run / "run-status.json").exists() else result["kind"]
    )
