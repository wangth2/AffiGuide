from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torchdrug import core

from .gearbind import util as gearbind_util
from .gearbind_bridge_fixed import GearBindBridge


_REGISTRY_READY = False
_BRIDGES: Dict[str, GearBindBridge] = {}


def _ensure_gearb_registry() -> None:
    global _REGISTRY_READY
    if _REGISTRY_READY:
        return
    try:
        from .gearbind import dataset as _dataset  # noqa: F401
        from .gearbind import model as _model  # noqa: F401
        from .gearbind import task as _task  # noqa: F401
    except Exception as exc:
        raise ImportError("Failed to import GearBind modules.") from exc
    _REGISTRY_READY = True


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    get = getattr(cfg, "get", None)
    if callable(get):
        return get(key, default)
    return getattr(cfg, key, default)


def _gearb_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(path_like: Any) -> Path:
    path = Path(os.path.expanduser(str(path_like)))
    return path if path.is_absolute() else _gearb_root() / path


def _checkpoint_paths(cfg: Any) -> List[Path]:
    env_ckpt = os.environ.get("GEARBIND_CKPT", "").strip()
    if env_ckpt:
        return [_resolve_path(p) for p in env_ckpt.split(os.pathsep) if p]

    model_checkpoint = _cfg_get(cfg, "model_checkpoint")
    if model_checkpoint is not None:
        return [_resolve_path(model_checkpoint)]

    checkpoints = _cfg_get(cfg, "checkpoints")
    if checkpoints is not None:
        return [_resolve_path(p) for p in checkpoints]

    raise KeyError("GearBind config must define 'model_checkpoint' or 'checkpoints'.")


def _checkpoint_fallback(path: Path) -> Path:
    if path.exists():
        return path
    for old, new in (("gearbindP", "gearbind"), ("bindddgP", "bindddg")):
        if path.name.startswith(old) and path.name.endswith(".pth"):
            alt = path.parent / path.name.replace(old, new, 1)
            if alt.exists():
                return alt
    return path


def _load_task(cfg: Any, checkpoint: Path, device: torch.device):
    _ensure_gearb_registry()

    task_cfg = _cfg_get(cfg, "task")
    if task_cfg is None:
        raise KeyError("GearBind config is missing 'task'.")

    task = core.Configurable.load_config_dict(task_cfg)
    state = torch.load(str(checkpoint), map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected checkpoint format: {checkpoint}")

    task_like = any(str(k).startswith("model.") for k in state_dict) or any(
        k in state_dict for k in ("mean", "std", "weight")
    )

    loaded = False
    if task_like:
        try:
            task.load_state_dict(state_dict, strict=False)
            loaded = True
        except Exception:
            loaded = False

    if not loaded:
        if any(str(k).startswith("model.") for k in state_dict):
            state_dict = {
                str(k)[len("model."):]: v
                for k, v in state_dict.items()
                if str(k).startswith("model.")
            }
        task.model.load_state_dict(state_dict, strict=False)

    return task.to(device).eval()


def _init_bridge(device: torch.device) -> GearBindBridge:
    default_cfg = _gearb_root() / "config" / "predict" / "CR3022_GearBindP.yaml"
    cfg_path = _resolve_path(os.environ.get("GEARBIND_CFG", default_cfg))
    if not cfg_path.exists():
        raise FileNotFoundError(f"GearBind config not found: {cfg_path}")

    cfg = gearbind_util.load_config(str(cfg_path), context={})
    checkpoints = [_checkpoint_fallback(p) for p in _checkpoint_paths(cfg)]
    checkpoints = [p for p in checkpoints if p.exists()]
    if not checkpoints:
        raise FileNotFoundError(f"No GearBind checkpoint found for config: {cfg_path}")

    tasks = [_load_task(cfg, checkpoint, device) for checkpoint in checkpoints]
    chain_a = os.environ.get("GEARBIND_CHAIN_A", "BA")
    chain_b = os.environ.get("GEARBIND_CHAIN_B", "C")
    return GearBindBridge(tasks, device=device, chain_a=chain_a, chain_b=chain_b)


def _bridge(device: torch.device) -> GearBindBridge:
    device = torch.device(device)
    key = f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}" if device.type == "cuda" else str(device)
    if key not in _BRIDGES:
        _BRIDGES[key] = _init_bridge(device)
    return _BRIDGES[key]


def _require_grad(name: str, grad: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    if grad is None:
        raise RuntimeError(f"GearBind did not return {name}.")
    if grad.shape != like.shape:
        raise RuntimeError(f"{name} shape mismatch: got {tuple(grad.shape)}, expected {tuple(like.shape)}.")
    return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).detach()


@torch.enable_grad()
def predict_ddg_batch(
    batch_ref: Dict[str, Any],
    aa_gen: torch.Tensor,
    ca_gen: torch.Tensor,
    v_next: torch.Tensor,
    aa_ref: Optional[torch.Tensor] = None,
    pos_ref: Optional[torch.Tensor] = None,
    v_ref: Optional[torch.Tensor] = None,
    seq_prob_gen: Optional[torch.Tensor] = None,
    clamp_pred_min: float = -80.0,
    clamp_pred_max: float = 80.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    return _bridge(ca_gen.device).score_and_ca_grad_from_diffab(
        batch_ref=batch_ref,
        aa_gen=aa_gen,
        ca_gen=ca_gen,
        v_next=v_next,
        aa_ref=aa_ref,
        ca_ref=pos_ref,
        v_ref=v_ref,
        seq_prob_gen=seq_prob_gen,
        clamp_pred_min=clamp_pred_min,
        clamp_pred_max=clamp_pred_max,
    )


@torch.enable_grad()
def predict_ddg_position(
    batch_ref: Dict[str, Any],
    aa_gen: torch.Tensor,
    pos_gen: torch.Tensor,
    v_next: torch.Tensor,
    aa_ref: Optional[torch.Tensor] = None,
    pos_ref: Optional[torch.Tensor] = None,
    v_ref: Optional[torch.Tensor] = None,
    clamp_pred_min: float = -8.0,
    clamp_pred_max: float = 8.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ddg, pos_grad, _ = predict_ddg_batch(
        batch_ref=batch_ref,
        aa_gen=aa_gen,
        ca_gen=pos_gen,
        v_next=v_next,
        aa_ref=aa_ref,
        pos_ref=pos_ref,
        v_ref=v_ref,
        clamp_pred_min=clamp_pred_min,
        clamp_pred_max=clamp_pred_max,
    )
    return ddg.detach(), _require_grad("coordinate gradient", pos_grad, pos_gen)


@torch.enable_grad()
def predict_ddg_sequence(
    batch_ref: Dict[str, Any],
    aa_prob: torch.Tensor,
    pos_gen: torch.Tensor,
    v_next: torch.Tensor,
    aa_ref: Optional[torch.Tensor] = None,
    pos_ref: Optional[torch.Tensor] = None,
    v_ref: Optional[torch.Tensor] = None,
    clamp_pred_min: float = -8.0,
    clamp_pred_max: float = 8.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if aa_prob.dim() != 3 or aa_prob.size(-1) != 20:
        raise ValueError(f"aa_prob must have shape (N, L, 20), got {tuple(aa_prob.shape)}.")

    aa_hard = aa_prob.detach().argmax(dim=-1)
    ddg, _, seq_grad = predict_ddg_batch(
        batch_ref=batch_ref,
        aa_gen=aa_hard,
        ca_gen=pos_gen,
        v_next=v_next,
        aa_ref=aa_ref,
        pos_ref=pos_ref,
        v_ref=v_ref,
        seq_prob_gen=aa_prob,
        clamp_pred_min=clamp_pred_min,
        clamp_pred_max=clamp_pred_max,
    )
    return ddg.detach(), _require_grad("sequence gradient", seq_grad, aa_prob)


__all__ = ["predict_ddg_batch", "predict_ddg_position", "predict_ddg_sequence"]
