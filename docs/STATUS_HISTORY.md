# Implementation handoff — 2026-10-31

The implementation is in this repository. Work stopped after the user's timing feedback. No paid GPU session, teacher/TTS API request, model upload or public endpoint was launched.

## Verified locally

* Windows owned-code suite: **32 passed, 2 Linux-native tests skipped**.
* Linux CPU suite: **34 passed in 35.49s** (`linux-tests-final.log`, ignored local artifact). Includes actual upstream tiny LMModel/LMGen forward/backward, audio-loss gradients through frozen depth parameters, stream delays, context forcing, session resets and adapter round trips. This uses random tiny parameters, not real PersonaPlex weights.
* `duplex fixture`: deterministic scenarios, tone timeline, two tiny optimizer steps, frozen-weight checks, atomic checkpoint and adapter reload. Outputs under `runs/cpu-fixture-verified/`; explicitly not speech-model training.
* Real eSpeak NG **1.52.0** rendered clean and overlapping English examples on CPU. Timeline/audio checks passed; artifacts under `runs/espeak-check/`. This check used a fixture tokenizer, not the gated native SentencePiece asset.
* Real W&B **0.31.0 offline files** produced and scanned for environment credential sentinels; none found. Latest verification is referenced by `runs/wandb-check/latest.json`.
* Latest Ruff checks/format and mypy passed for owned code. Bandit retained/reviewed 14 scoped low-severity diagnostics; repository secret scan passed. Final Windows dependency audit returned **no known vulnerabilities** after replacing the initially vulnerable Torch 2.8 pin with **2.13.0**.
* Locked CPU Linux setup completed using official hashed CPU Torch wheels and non-CUDA dependencies from `uv.lock`.
* Generated/validated the 400-candidate pilot (300 train / 56 dev / 44 test), with group/family split checks.
* Audited/pinned official runtime, community trainer, model/tokenizer/codec, OpenYap metadata and Full-Duplex-Bench revisions. `modal_app.py` imports without allocation. Installed Modal **1.6.3** help/source confirms command syntax, Windows restriction and hardcoded 3600-second managed-shell timeout.

The final small report-metadata additions passed Ruff/mypy; they were not exercised with real GPU generation. CPU checks and scanners do not establish production readiness or absence of vulnerabilities.

## Implemented but unverified with real assets/services

Gated PersonaPlex/Mimi/tokenizer download and loading; actual A100 80GB forward/backward/memory/throughput; native SentencePiece audio alignment; streaming generated speech quality; matched audible base/adapted evaluation; trained speech-adapter export; managed Modal image build, disconnect cleanup, atomic cloud lease/ledger reconciliation and fresh-shell Volume persistence; online W&B; paid teacher endpoint; optional benchmark bridge and licensed replay media.

These are explicit execution gates, not claimed successes. No trained PersonaPlex adapter or improvement result exists yet. Accept model terms and supply licensed assets before the CPU preparation stage; separately authorize any paid execution. The repository refuses model/GPU substitution and requires an active budget reservation for real GPU stages.

## Budget plan

User-reported balance $24.51; configured initial ceiling $18; target four aggregate GPU hours, maximum five. At checked rates, one A100 80GB plus conservatively priced 2 CPU cores/32GiB, 15% margin and $1.50 non-GPU reserve estimates **$17.82 for four hours**. The recommended shell uses <=3300-second reservations with persistent local/Volume ledgers, a project cloud lease, watchdog and explicit termination reconciliation. These are estimates/best-effort controls, not exact billing or a perfect cap. Teacher/API budget defaults to zero.

## Start here

Read [README.md](README.md) for exact setup, CPU preparation, budgeted Modal shell, real-checkpoint smoke, training/resume, identical-runtime evaluation and export commands. See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md), [docs/DATA.md](docs/DATA.md), [docs/SECURITY.md](docs/SECURITY.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for evidence and limitations.

No Git commit was created; all source files remain available for review in this initially empty repository.

## Follow-up: Modal image lock failure

The user reported `uv sync --locked` rejecting the lock in the managed image. The rejection did not reproduce locally with exact Python 3.11.12/uv 0.8.10, so its cloud-specific cause remains undetermined. The image now installs a committed hash-locked export of the same dependencies instead of re-resolving the project during the build. Local verification confirmed export equality with `uv.lock`, a hash-enforced installation dry run on Python 3.11.12, and the image specification import. No Modal build/GPU session was launched to test this patch; retry the CPU shell to verify it remotely.

## Follow-up: shell Volume flush

The user completed CPU rendering (four train and four dev clips), then the manual flush failed because `Volume.commit()` rejects calls from the managed shell process. Replaced that RPC with bounded filesystem `sync` on /data, /runs and /cache. Modal documents explicit sync commits for v2 and background/final-shutdown commits for v1; existing volumes are not changed. Three focused local tests, Ruff, mypy, security review and image-spec import passed. Actual persisted files and the patched remote flush still require an external/fresh-shell check; do not infer loss or durability from the prior exception alone. Dependency installation was moved ahead of source copies so future source-only fixes can reuse the expensive dependency layer.

## Real A100 smoke subsequently reported by the user

The user's Modal console output confirms actual checkpoint load, backward/optimizer, frozen parameter checks, adapter save/reload and short generation on the configured A100 80GB. Reported optimizer-step time: 2.559s; peak GPU allocation: 19,694,738,944 bytes; trainable parameters: 262,144; smoke stage: 134.46s. Eight encoded examples and pinned model/codec/tokenizer files were also visible through the external Volume CLI after the CPU shell exited. This is actual compatibility/persistence evidence from the user's run, not an improvement or audio-quality result. Main training refused before starting; only the one-step smoke adapter exists at this point.

The old evaluation estimate extrapolated total prompt-plus-generation time from a truncated 1.6-second sample, reporting 5153.88s for 12 paired examples. Replaced that measurement with complete-conversation calibration and model-loading allowance. Existing smoke runs can use `profile --generation-only` without repeating backward/frozen checks. New local tests cover rejecting truncated estimates, retaining evaluation reserves, refusing genuinely slow full-session plans, and reusing the existing checkpoint without training. No new paid session was launched by the agent; the patched full-conversation calibration awaits the user's next run.
