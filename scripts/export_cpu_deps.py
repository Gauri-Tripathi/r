"""Export Torch's non-CUDA dependencies from the SAME uv lock for the CPU wheel."""

import tomllib
from pathlib import Path

from packaging.markers import Marker

root = Path(__file__).resolve().parents[1]
packages = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))["package"]
environment = {
    "python_version": "3.11",
    "python_full_version": "3.11.15",
    "sys_platform": "linux",
    "platform_machine": "x86_64",
}
by_name = {
    p["name"]: p
    for p in packages
    if not p.get("resolution-markers")
    or any(Marker(m).evaluate(environment) for m in p["resolution-markers"])
}
pending = list(by_name["torch"]["dependencies"])
seen = set()
lines = []
while pending:
    dependency = pending.pop()
    name = dependency["name"]
    if (
        name in seen
        or name.startswith(("nvidia-", "cuda-"))
        or name == "triton"
        or (dependency.get("marker") and not Marker(dependency["marker"]).evaluate(environment))
    ):
        continue
    seen.add(name)
    package = by_name[name]
    hashes = sorted(
        {item["hash"] for item in package.get("wheels", [])}
        | ({package["sdist"]["hash"]} if "sdist" in package else set())
    )
    lines.append(f"{name}=={package['version']} " + " ".join("--hash=" + h for h in hashes))
    pending.extend(package.get("dependencies", []))
print("\n".join(sorted(lines)))
