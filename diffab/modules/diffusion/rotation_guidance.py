"""Deterministic right-local SO(3) guidance; legacy SO(3) paths stay unchanged.

DiffAb uses the negative of the usual cross-product skew matrix. Both the
increment and the log below use that convention. The log is used only after a
nonzero guidance update, never to rewrite unaffected/zero-step residues.
"""
import math
import json
import contextlib
import os
import warnings

import torch

from diffab.modules.common.so3 import so3vec_to_rotation, so3vec_to_skewsym


@contextlib.contextmanager
def matrix_exp_warning_scope():
    """Quiet only PyTorch 1.12.1's verified matrix-exp buffer-resize warning.

    Its degree-4/8 Taylor branches resize an internal [B,n,n] buffer to
    [1,B,n,n]. n=3 is the forward rotation, n=6 its block-matrix derivative.
    Keep the original matrix_exp/autograd operations and all other diagnostics.
    """
    if (torch.__version__.split("+")[0] != "1.12.1"
            or os.environ.get("AFFIGUIDE_SHOW_MATRIX_EXP_WARNINGS") == "1"):
        yield
        return
    message = (
        r"An output with one or more elements was resized since it had shape "
        r"\[(?P<batch>\d+), (?P<size>3|6), (?P=size)\], which does not match "
        r"the required output shape \[1, (?P=batch), (?P=size), (?P=size)\]\. "
        r"This behavior is deprecated,"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=message, category=UserWarning)
        yield


def local_rotation_exp(omega):
    """Accurate derivative at omega=0, in the project's existing convention."""
    with matrix_exp_warning_scope():
        return torch.matrix_exp(so3vec_to_skewsym(omega))


def rotation_autograd_grad(*args, **kwargs):
    """Unchanged autograd call, scoped to derivatives containing matrix_exp."""
    with matrix_exp_warning_scope():
        return torch.autograd.grad(*args, **kwargs)


@torch.no_grad()
def rotation_to_legacy_rotvec(R):
    """Matrix -> quaternion -> principal rotvec, robust at zero and near pi.

    Select the largest quaternion component to avoid divisions by a vanishing
    trace branch. Quaternion normalization handles only legacy roundoff on this
    new nonzero-update path; no global projection or legacy log replacement.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    q_abs = torch.sqrt(torch.clamp(torch.stack([
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ], dim=-1), min=0))
    candidates = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1),
    ], dim=-2) / (2 * q_abs.clamp_min(0.1)[..., :, None])
    idx = q_abs.argmax(dim=-1)
    q = candidates.gather(-2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    q = torch.where((q[..., :1] < 0), -q, q)
    sin_half = torch.linalg.norm(q[..., 1:], dim=-1)
    angle = 2 * torch.atan2(sin_half, q[..., 0])
    factor = torch.where(sin_half > 1e-7, angle / sin_half.clamp_min(1e-30),
                         2 + sin_half.square() / 3)
    return -q[..., 1:] * factor[..., None]


def require_finite(value, mask, name):
    selected = value[mask.bool()]
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError("Non-finite {} on generated residues".format(name))


@torch.no_grad()
def apply_local_rotation(v_next, g_omega, mask_generate, sigma_rot, gamma_rot,
                         max_step_deg=5.0):
    """Return v_guided and detached diagnostics, without consuming any RNG."""
    if not math.isfinite(float(gamma_rot)) or float(gamma_rot) < 0:
        raise ValueError("gradient_scale_rot must be finite and nonnegative")
    if not math.isfinite(float(max_step_deg)) or float(max_step_deg) <= 0:
        raise ValueError("rot_max_step_deg must be finite and positive")
    mask = mask_generate.bool()
    require_finite(g_omega, mask, "local rotation gradient")
    sigma = torch.as_tensor(sigma_rot, device=v_next.device, dtype=v_next.dtype)
    if sigma.numel() == v_next.size(0):
        sigma = sigma.reshape(v_next.size(0), 1, 1)
    require_finite(sigma, torch.ones_like(sigma, dtype=torch.bool), "rotation sigma")
    raw_delta = -float(gamma_rot) * sigma * g_omega
    raw_delta = torch.where(mask[..., None], raw_delta, torch.zeros_like(raw_delta))
    require_finite(raw_delta, mask, "raw local rotation increment")
    raw_norm = torch.linalg.norm(raw_delta, dim=-1)
    require_finite(raw_norm, mask, "raw local rotation increment norm")
    cap = math.radians(float(max_step_deg))
    scale = torch.minimum(torch.ones_like(raw_norm), cap / (raw_norm + 1e-12))
    delta = raw_delta * scale[..., None]
    require_finite(delta, mask, "capped local rotation increment")
    norm = torch.linalg.norm(delta, dim=-1)
    changed = mask & (norm > 0)
    v_guided = v_next.detach().clone()
    actual_angle = torch.zeros_like(norm)
    base_orth_error = torch.zeros_like(norm)
    base_det_error = torch.zeros_like(norm)
    roundtrip_error = torch.zeros_like(norm)
    if bool(changed.any()):
        base = so3vec_to_rotation(v_next[changed]).detach()
        updated = base @ local_rotation_exp(delta[changed])
        v_updated = rotation_to_legacy_rotvec(updated)
        require_finite(v_updated, torch.ones(v_updated.shape[:-1], dtype=torch.bool,
                                           device=v_updated.device), "guided rotation")
        v_guided[changed] = v_updated
        restored = so3vec_to_rotation(v_updated)
        relative = base.transpose(-1, -2) @ restored
        skew = torch.stack([relative[..., 2, 1] - relative[..., 1, 2],
                            relative[..., 0, 2] - relative[..., 2, 0],
                            relative[..., 1, 0] - relative[..., 0, 1]], dim=-1)
        actual_angle[changed] = torch.atan2(torch.linalg.norm(skew, dim=-1) / 2,
                                          (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2)
        ident = torch.eye(3, device=base.device, dtype=base.dtype)
        base_orth_error[changed] = (base.transpose(-1, -2) @ base - ident).abs().amax(dim=(-1, -2))
        # Explicit 3x3 determinant avoids CUDA 11.3's NVRTC reduction-product
        # compilation for Ada GPUs (sm_89); this is a diagnostic only.
        det = (base[..., 0, 0] * (base[..., 1, 1] * base[..., 2, 2] - base[..., 1, 2] * base[..., 2, 1])
               - base[..., 0, 1] * (base[..., 1, 0] * base[..., 2, 2] - base[..., 1, 2] * base[..., 2, 0])
               + base[..., 0, 2] * (base[..., 1, 0] * base[..., 2, 1] - base[..., 1, 1] * base[..., 2, 0]))
        base_det_error[changed] = (det - 1).abs()
        roundtrip_error[changed] = (restored - updated).abs().amax(dim=(-1, -2))
    return v_guided, {
        "raw_angle_rad": raw_norm,
        "angle_rad": norm,
        "actual_angle_rad": actual_angle,
        "cap_hit": raw_norm > cap,
        "base_orthogonality_error": base_orth_error,
        "base_determinant_error": base_det_error,
        "roundtrip_error": roundtrip_error,
    }


def write_rotation_diagnostics(path, t, sigma_rot, gamma_rot, mask_generate,
                               g_omega, geometry, update, detailed=False,
                               legacy_pred=None, legacy_pos_grad=None,
                               geom_pred=None, geom_pos_grad=None):
    """Append detached statistics; detailed residue values are pilot-only."""
    if path is None:
        return
    mask = mask_generate.detach().bool()
    norms = torch.linalg.norm(g_omega.detach(), dim=-1)
    sigma = torch.as_tensor(sigma_rot, device=norms.device, dtype=norms.dtype)
    if sigma.numel() == norms.size(0):
        sigma = sigma.reshape(norms.size(0), 1)
    u = sigma * norms

    def summary(value):
        flat = value.detach()[mask].float().cpu()
        if flat.numel() == 0:
            return {"count": 0}
        return {"count": flat.numel(), "median": float(torch.quantile(flat, .5)),
                "p95": float(torch.quantile(flat, .95)), "max": float(flat.max()),
                "mean": float(flat.mean())}

    selected = norms[mask]
    rec = {
        "t": int(t), "sigma_rot": float(torch.as_tensor(sigma_rot).flatten()[0]),
        "gamma_rot": float(gamma_rot), "probe_only": bool(detailed and gamma_rot == 0),
        "batch_size": int(mask.size(0)), "generated_residues": int(mask.sum()),
        "gradient_norm_stats": summary(norms), "u_stats": summary(u),
        "zero_gradient_ratio": float((selected == 0).float().mean()) if selected.numel() else 0.,
        "nonfinite_gradient_ratio": float((~torch.isfinite(selected)).float().mean()) if selected.numel() else 0.,
        "raw_pred": geometry["raw_pred"].detach().cpu().flatten().tolist(),
        "clamp_saturated_ratio": float(geometry["clamp_saturated"].float().mean()),
        "cap_hit_ratio": float(update["cap_hit"][mask].float().mean()) if selected.numel() else 0.,
    }
    for key in ("raw_angle_rad", "angle_rad", "actual_angle_rad", "base_orthogonality_error",
                "base_determinant_error", "roundtrip_error"):
        rec[key + "_stats"] = summary(update[key])
    if detailed:
        rec["positions_internal"] = [torch.nonzero(item, as_tuple=False).flatten().cpu().tolist() for item in mask]
        rec["gradient_norms"] = norms[mask].cpu().tolist()
        rec["u"] = u[mask].cpu().tolist()
    if legacy_pred is not None and geom_pred is not None:
        rec["legacy_energy_abs_error_max"] = float((legacy_pred - geom_pred).abs().max())
    if legacy_pos_grad is not None and geom_pos_grad is not None:
        rec["legacy_position_gradient_abs_error_max"] = float((legacy_pos_grad - geom_pos_grad).abs().max())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, allow_nan=False) + "\n")
