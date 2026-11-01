# PersonaPlex LoRA experiments

Fine-tuning PersonaPlex on OpenYap conversations, with a separate pipeline for spoken corrections and timestamped context updates.

The full OpenYap run completed 136 optimizer steps over 544 training windows: about 7.15 source hours, one pass. The remaining audio was held out for development and testing.

| Test loss | Base | Adapter |
|---|---:|---:|
| Audio | 2.66712 | 2.64902 |
| Text | 1.29411 | 1.27428 |

These are modeling losses, not speech-quality scores. All 56 test windows come from one conversation; speakers overlap across splits. The earlier pilot used different held-out data, so its scores cannot be compared directly with this run.

## Setup

Python 3.11 and Linux/WSL are required for the native audio runtime. GPU commands use one A100 80GB.

```bash
bash scripts/setup_cpu_linux.sh
source .venv-linux/bin/activate
duplex --help
```

Model access requires accepting the PersonaPlex terms and supplying a read-only `HF_TOKEN` during download. Revisions are pinned in `upstream.lock.json`. Datasets, weights and local runs are excluded from Git.

## Usage

- [Training and resume](docs/OPENYAP_FULL_TRAINING.md)
- [Inference from a WAV or microphone recording](docs/INFERENCE.md)
- [Pilot results](docs/OPENYAP_PILOT.md)
- [Correction and grounding experiment](docs/RESEARCH_RUNBOOK.md)
- [Teacher response format](docs/TEACHER_CONTRACT.md)
- [Data formats](docs/DATA.md) and [runtime details](docs/COMPATIBILITY.md)

The correction/grounding dataset still needs accepted external teacher responses. OpenYap supplies ordinary conversations, not correction or tool-use labels. The original four-clip run is documented in the [smoke-test audit](docs/SMOKE_AUDIT.md).

## Checks

```bash
python -m pytest -q
python -m ruff check src tests scripts modal_app.py
python -m mypy src/duplex
python scripts/secret_scan.py
```

CPU native tests use small random models. See [security](docs/SECURITY.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
