"""Bounded public metadata probe; never downloads model weights or dataset shards."""

from pathlib import Path

import httpx

from duplex.io import atomic_json

records = {}
for kind, url in {
    "base": "https://huggingface.co/api/models/nvidia/personaplex-7b-v1",
    "openyap": "https://huggingface.co/api/datasets/TheAgenticDataCompany/open-yap-1k",
}.items():
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        response = client.get(url)
    response.raise_for_status()
    if len(response.content) > 1024 * 1024:
        raise ValueError("Unexpected metadata size")
    value = response.json()
    records[kind] = {
        "repo": value["id"],
        "sha": value["sha"],
        "gated": value.get("gated"),
        "files": [x["rfilename"] for x in value["siblings"]],
        "license": value.get("cardData", {}).get("license"),
    }
atomic_json(Path("docs/upstream-metadata.json"), records)
print(
    {name: {"revision": item["sha"], "files": len(item["files"])} for name, item in records.items()}
)
