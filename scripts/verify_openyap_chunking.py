"""Verify faster causal Mimi streaming on an actual source clip, locally on CPU."""

from pathlib import Path

import torch

from duplex.io import atomic_json, read_jsonl
from duplex.model import FILES, native, verify_assets
from duplex.openyap_full_encode import verify_chunks

assets = Path("cache/openyap-assets")
root = Path("data/openyap/pilot-v1")
verify_assets(assets, include_base=False)
torch.set_num_threads(2)
loaders, _, _ = native()
mimi = loaders.get_mimi(assets / FILES["codec"], device="cpu").requires_grad_(False)
row = read_jsonl(root / "replay.jsonl")[0]
result = verify_chunks(mimi, root / row["user"], 25, "cpu")
atomic_json(Path("runs/openyap/full-chunk-verification.json"), result)
print(result)
