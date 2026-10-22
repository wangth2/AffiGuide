"""Lightweight persistent GearBind client: no TorchDrug/Torch imports."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .wire import primitive, read_frame, write_frame

COST_FIELDS = ("forward_count", "hard_forward_count", "soft_forward_count",
               "backward_count", "model_load_count")


class GearBindError(RuntimeError):
    pass


def resolve_python(python):
    """Resolve an executable without following a virtual environment's symlinks."""
    executable = shutil.which(os.path.expanduser(str(python)))
    if not executable:
        raise FileNotFoundError(
            "GearBind Python not found: %s. Create envs/affiguide-gearbind.yml "
            "and set AFFIGUIDE_GEARBIND_PYTHON to its bin/python." % python)
    return str(Path(executable).absolute())


def gearbind_environment(python, root, parent=None):
    """Use the selected Conda runtime in this child only; retain driver paths."""
    child = dict(os.environ if parent is None else parent)
    executable = Path(resolve_python(python))
    child.pop("PYTHONHOME", None)
    child["PYTHONPATH"] = str(root)
    child["PYTHONNOUSERSITE"] = "1"
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    child["PATH"] = os.pathsep.join([str(executable.parent)] + [
        path for path in child.get("PATH", "").split(os.pathsep)
        if path and path != str(executable.parent)])
    prefix = executable.parent.parent
    if sys.platform.startswith("linux") and (prefix / "conda-meta").is_dir():
        # An inherited system/CUDA path can otherwise select an older libstdc++
        # before RDKit imports Boost. Set this before exec, not after importing
        # Torch. Keep the remaining paths for external NVIDIA driver libraries.
        library = str(prefix / "lib")
        child["LD_LIBRARY_PATH"] = os.pathsep.join([library] + [
            path for path in child.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if path and path != library])
    return child


class GearBindClient:
    def __init__(self, python, checkpoint, device="cpu", backend="production_atomic",
                 log_path=None, extension_path=None, config_path=None, torch_threads=None):
        torch_threads = int(torch_threads if torch_threads is not None else os.environ.get("AFFIGUIDE_GEARBIND_THREADS", "2"))
        if torch_threads < 1:
            raise ValueError("GearBind Torch thread count must be positive")
        here = Path(__file__).resolve().parent
        root = here.parents[2]
        python = resolve_python(python)
        child_env = gearbind_environment(python, root)
        self.log_handle = open(log_path, "ab", buffering=0) if log_path else None
        command = [str(python), "-B", "-u", "-m", "diffab.tools.GearBind.server",
                   "--device", str(device), "--checkpoint", str(checkpoint),
                   "--backend", backend, "--torch-threads", str(torch_threads)]
        if extension_path:
            command += ["--extension-path", str(extension_path)]
        if config_path:
            command += ["--config-path", str(config_path)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log_handle or sys.stderr.buffer,
            env=child_env, cwd=str(root), bufsize=0)
        self.request_id = 0
        self.last_cost = {key: 0 for key in COST_FIELDS}
        self.hello = self._request({"op": "hello"})

    def _request(self, request):
        self.request_id += 1
        request = dict(request, request_id=self.request_id)
        try:
            write_frame(self.process.stdin, request)
            response = read_frame(self.process.stdout)
        except (BrokenPipeError, EOFError) as exc:
            raise GearBindError("GearBind sidecar terminated; inspect its log") from exc
        if response is None or response.get("request_id") != self.request_id:
            raise GearBindError("GearBind sidecar response identity mismatch")
        # These are process-cumulative counters reported by the worker. A
        # caller takes copies before/after a job; native rebinding does not
        # reset them. Batched responses carry counters on the last candidate.
        snapshots = [response]
        if isinstance(response.get("results"), list) and response["results"]:
            snapshots.append(response["results"][-1])
        for snapshot in snapshots:
            for key in COST_FIELDS:
                if key in snapshot:
                    self.last_cost[key] = int(snapshot[key])
        if response.get("status") != "OK":
            raise GearBindError(response.get("error", "GearBind sidecar failed"))
        return response

    def initialize(self, native_batch, s_native=None, p_native_A=None, R_native=None):
        from .identity import batch_contract
        native_batch = dict(native_batch)
        native_batch["design_cdr_contract"] = batch_contract(native_batch)
        request = {"op": "init", "native_batch": native_batch}
        for key, value in [("s_native", s_native), ("p_native_A", p_native_A),
                           ("R_native", R_native)]:
            if value is not None:
                request[key] = value
        return self._request(primitive(request))

    def score(self, s_geo, p_A, R, q=None, gradient=False,
              gradient_blocks=None, gradient_objective="historical_blocks"):
        candidate = {"s_geo": s_geo, "p_A": p_A, "R": R, "gradient": bool(gradient)}
        if gradient_blocks is not None:
            candidate["gradient_blocks"] = list(gradient_blocks)
        candidate["gradient_objective"] = gradient_objective
        if q is not None:
            candidate["q"] = q
        return self._request({"op": "score", "candidate": primitive(candidate)})

    def score_many(self, candidates):
        return self._request({"op": "score_many", "candidates": primitive(candidates)})["results"]

    def p_many(self, candidates):
        """Independent actual candidates, each requesting a hard-position gradient."""
        requests = [dict(candidate, gradient_blocks=["p"],
                         gradient_objective="historical_blocks") for candidate in candidates]
        return self.score_many(requests)

    def close(self):
        if self.process.poll() is None:
            try:
                self._request({"op": "close"})
            except GearBindError:
                pass
            self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        if self.log_handle:
            self.log_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
