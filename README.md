# Budgeted duplex speech post-training

**Run the trained adapter on your voice/WAV:** [INFERENCE.md](docs/INFERENCE.md), using `duplex infer`. Optional local microphone recording is included; no teacher runs at inference.

**Full-data training commands:** [OPENYAP_FULL_TRAINING.md](docs/OPENYAP_FULL_TRAINING.md). This removes the pilot quotas, supports resumable causal encoding/training, and uses all eligible source windows after whole-conversation holdouts.

**Completed: [OpenYap real-conversation pilot](docs/OPENYAP_PILOT.md).** The public 8.9-hour sample was downloaded and a speaker-disjoint 47-window subset was tested. The 17-step real A100 run completed, with essentially flat held-out losses: no meaningful improvement demonstrated. Checkpoints, export and a local speech comparison page are saved; GPU shutdown and ledger closure are verified. This separately tests basic conversational fine-tuning, not correction/live grounding.

**The existing four-clip, 100-step run is a smoke test. It is not a completed research experiment or evidence of improved capability.** Actual adapter weights trained and saved; matched speech improvement has not been measured.

Start with the [executed-artifact audit](docs/SMOKE_AUDIT.md), [current status](STATUS.md), and [research commands](docs/RESEARCH_RUNBOOK.md). Older instructions are preserved as [historical smoke-test instructions](docs/LEGACY_SMOKE_RUNBOOK.md); their four-clip workload must not be used as the research experiment.

The implemented workflow is deterministic Python simulation → offline text-teacher responses → validated versioned JSONL → separate speech rendering/alignment and Mimi encoding → PersonaPlex selective-LoRA supervised training → matched speech evaluation. Inference uses incoming user audio and arrived facts. No text teacher generates runtime answers.

No teacher provider is selected. [Externally generated responses can be imported](docs/TEACHER_CONTRACT.md). Train/resume/export require no teacher SDK, API key, logits or network access. Model-download authentication is separate. Local eSpeak supplies audio only; it is not the reasoning teacher.

`data/research-v3/` currently contains **403 distinct candidates across 109 groups**, split 318 train / 32 development / 53 test. These are not all validated audio or a finished teacher dataset. Twelve development and twelve test user-only inputs were rendered locally. Teacher responses, base speech outputs, research training and matched results are still missing.

Research mode rejects fixtures, unversioned supervision, preparation-only measurements, undersized datasets and missing audible baselines. It derives steps from measured data/time and limits reuse to two effective passes by default. The existing rank-8 LoRA recipe targets temporal output projections in layers 28–31. The validated runtime expects one **A100 80GB**; no architecture changes or RL are added.

The user reports $23.87 remaining Modal credit. The original $18 project ceiling remains. The ledger preserves both closed sessions and appended billing evidence; workspace usage is conservatively counted at $6.1834525, with another $1.50 reserved for preparation/non-GPU work. No paid compute, teacher/TTS call, upload or deployment was launched in this continuation.

## Local checks

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests scripts modal_app.py
.venv\Scripts\mypy.exe src/duplex
.venv\Scripts\python.exe scripts/security_check.py
.venv\Scripts\python.exe scripts/secret_scan.py
```

For actual upstream CPU tests, use WSL/Linux and `.venv-linux` (`bash scripts/setup_cpu_linux.sh` creates it if absent):

```bash
source .venv-linux/bin/activate
python -m pytest -q -p no:cacheprovider
duplex research --help
```

Linux is required for `sphn`/Mimi/native generation; Windows runs owned-code checks. CPU tests use tiny random weights and labeled fixtures, not the real 7B model. Source/assets are pinned in `upstream.lock.json`. See [compatibility](docs/COMPATIBILITY.md), [data](docs/DATA.md), [security](docs/SECURITY.md) and [notices](THIRD_PARTY_NOTICES.md) for implementation and license boundaries.
