#!/usr/bin/env python3
"""Check both runtime environments without downloading weights or running a design."""
import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def probe(role, device):
    # Preserve stdout for one machine-readable result, including noisy imports.
    result = dict(role=role, python=sys.executable, device=device, status="FAIL")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            import numpy
            import scipy
            import Bio
            import yaml
            import lmdb
            import easydict
            import abnumber
            result.update(python_version=sys.version.split()[0], torch=torch.__version__,
                          cuda_runtime=torch.version.cuda, numpy=numpy.__version__,
                          scipy=scipy.__version__, biopython=Bio.__version__)
            if sys.version_info[:2] != (3, 8) or torch.__version__.split("+")[0] != "1.12.1":
                raise RuntimeError("The bundled runtime requires Python 3.8 and PyTorch 1.12.1; use the supplied environment YAMLs.")
            if torch.version.cuda != "11.3":
                raise RuntimeError("The bundled runtime requires the CUDA 11.3 PyTorch build.")
            if role == "gearbind":
                import rdkit
                from rdkit.Chem import rdchem
                import torch_scatter
                import torch_cluster
                import torchdrug
                from torchdrug import core
                from diffab.tools.GearBind.energy import G_EXTENSION_SHA256
                result.update(rdkit=rdkit.__version__,
                              torchdrug=getattr(torchdrug, "__version__", "unknown"))
                if torch._C._GLIBCXX_USE_CXX11_ABI:
                    raise RuntimeError("The bundled GearBind extension requires PyTorch C++ ABI 0.")
                extension = ROOT / "diffab/tools/GearBind/native/torch_ext.so"
                if hashlib.sha256(extension.read_bytes()).hexdigest() != G_EXTENSION_SHA256:
                    raise RuntimeError("Bundled GearBind extension hash mismatch.")
                spec = importlib.util.spec_from_file_location("torch_ext", extension)
                native = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(native)
                importlib.import_module("torchdrug.utils.torch").torch_ext = native
                from diffab.tools.GearBind.gearbind import dataset, model, task, util
                result["native_extension"] = "PASS"
            selected = torch.device(device)
            if selected.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable to this process. Check GPU visibility and the NVIDIA driver.")
                capability = torch.cuda.get_device_capability(selected)
                if capability not in ((8, 0), (8, 6), (8, 9)):
                    raise RuntimeError("GPU compute capability %s is outside the bundled kernels (8.0/8.6/8.9)." % (capability,))
                value = torch.ones(4, device=selected).sum().item()
                if value != 4:
                    raise RuntimeError("CUDA tensor check failed.")
                result.update(gpu=torch.cuda.get_device_name(selected), capability=list(capability),
                              cuda_tensor_check="PASS")
            elif selected.type != "cpu":
                raise ValueError("Use --device cpu or cuda:N.")
            maps = Path("/proc/self/maps")
            if maps.is_file():
                result["libstdcxx"] = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                                               if "libstdc++.so" in line})
            result["status"] = "PASS"
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu checks imports; cuda:0 also checks GPU access")
    parser.add_argument("--guidance", choices=("dg", "d-only"), default="dg")
    parser.add_argument("--gearbind-python", default=os.environ.get("AFFIGUIDE_GEARBIND_PYTHON"))
    parser.add_argument("--probe", choices=("main", "gearbind"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        return probe(args.probe, args.device)
    from diffab.tools.GearBind.client import gearbind_environment, resolve_python
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    checks = [("main", sys.executable, env)]
    missing_gearbind = args.guidance == "dg" and not args.gearbind_python
    if args.guidance == "dg" and not missing_gearbind:
        try:
            python = resolve_python(args.gearbind_python)
            checks.append(("gearbind", python, gearbind_environment(python, ROOT)))
        except (OSError, ValueError) as exc:
            print(json.dumps(dict(role="gearbind", status="FAIL", error=str(exc))))
            return 1
    failed = False
    for role, python, child in checks:
        try:
            completed = subprocess.run(
                [python, "-B", str(Path(__file__).resolve()), "--probe", role, "--device", args.device],
                cwd=str(ROOT), env=child, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=120)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            print(completed.stdout, end="")
            failed |= completed.returncode != 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(json.dumps(dict(role=role, status="FAIL", error=str(exc))))
            failed = True
    if missing_gearbind:
        print(json.dumps(dict(role="gearbind", status="FAIL", error=
            "Set AFFIGUIDE_GEARBIND_PYTHON to the affiguide-gearbind environment's bin/python.")))
        failed = True
    if not failed:
        print("Environment checks passed. Full design validation is still a separate step.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
