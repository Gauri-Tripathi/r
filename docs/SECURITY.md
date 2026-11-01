# Credentials and local artifacts

Use a read-only Hugging Face token for model downloads. Teacher credentials are needed only for the optional teacher preparation command. Training and inference read prepared local files and cached model assets.

Keep keys in the environment, outside datasets, YAML, logs and checkpoints. `.env`, data, weights, generated audio and local runs are ignored. W&B defaults to offline mode and receives only allowlisted numeric or hashed metrics.

Checkpoint adapters use safetensors. Optimizer resume state uses `torch.load(weights_only=True)` and checksum checks; accept it only from your own runs. Dataset paths must remain beneath the declared root. Owned loaders do not execute remote dataset code or use `trust_remote_code`.

The vendored runtime includes server and import utilities that the project does not use. Do not use those utilities to load untrusted archives or voice embeddings.

## Checks

```bash
python scripts/secret_scan.py
python scripts/security_check.py
python -m pip_audit --local
```

Bandit findings are stored in `runs/security/bandit.json`. The allowlist in `scripts/security_check.py` covers reviewed subprocess calls and deterministic sampling. CI also runs tests, Ruff and mypy with read-only repository permissions.

Budget reservations include startup and shell idle time. A disconnected client does not prove that the GPU stopped: confirm termination in Modal before reconciling the reservation. The ledger and watchdog are estimates and shutdown controls, not substitutes for checking billing.
