"""CPU asset preparation and GPU shell specs; no deployments or public endpoints."""

import os

import modal

app = modal.App("duplex-budgeted")
cache = modal.Volume.from_name("duplex-cache", create_if_missing=True)
data = modal.Volume.from_name("duplex-data", create_if_missing=True)
runs = modal.Volume.from_name("duplex-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "libsndfile1")
    .pip_install("uv==0.8.10")
    .add_local_file("requirements-modal.txt", "/opt/duplex/requirements-modal.txt", copy=True)
    .run_commands(
        "uv venv --python /usr/local/bin/python3 /opt/duplex/.venv",
        "uv pip install --python /opt/duplex/.venv/bin/python --require-hashes -r /opt/duplex/requirements-modal.txt",
    )
    .add_local_file("pyproject.toml", "/opt/duplex/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/opt/duplex/uv.lock", copy=True)
    .add_local_dir("src", "/opt/duplex/src", copy=True)
    .add_local_dir("vendor", "/opt/duplex/vendor", copy=True)
    .add_local_dir("configs", "/opt/duplex/configs", copy=True)
    .add_local_file("upstream.lock.json", "/opt/duplex/upstream.lock.json", copy=True)
    .add_local_file("THIRD_PARTY_NOTICES.md", "/opt/duplex/THIRD_PARTY_NOTICES.md", copy=True)
    .run_commands(
        "cd /opt/duplex && uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "NO_TORCH_COMPILE": "1",
            "NO_CUDA_GRAPH": "1",
            "WANDB_MODE": "offline",
            "DUPLEX_MODAL": "1",
        }
    )
)
mounts = {"/cache": cache, "/data": data, "/runs": runs}


@app.function(
    image=image,
    cpu=2,
    memory=32768,
    timeout=3600,
    volumes=mounts,
    secrets=[modal.Secret.from_name("duplex-hf-read", required_keys=["HF_TOKEN"])],
)
def cpu_prepare() -> None:
    """Spec for modal shell modal_app.py::cpu_prepare; HF read secret only here."""
    return None


@app.function(
    image=image,
    cpu=2,
    memory=32768,
    gpu="A100-80GB",
    timeout=3300,
    volumes=mounts,
    max_containers=1,
    scaledown_window=2,
)
def gpu_shell() -> None:
    """Use duplex shell. No HF, teacher, upload, or W&B secrets mounted on GPU."""
    return None


# Import guard prevents accidentally allocating via the raw GPU shell reference.
if (
    os.environ.get("DUPLEX_REQUIRE_LAUNCHER") == "1"
    and os.environ.get("DUPLEX_LAUNCH_ALLOWED") != "1"
):
    raise RuntimeError("Use duplex shell for persistent reservation accounting")
