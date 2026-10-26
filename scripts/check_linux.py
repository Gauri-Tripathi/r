import importlib.util
import sys

print(sys.version)
for name in ("torch", "numpy", "sphn", "scipy", "soundfile", "pydantic"):
    spec = importlib.util.find_spec(name)
    print(name, bool(spec))
    if name == "torch" and spec:
        import torch

        print(torch.__version__)
