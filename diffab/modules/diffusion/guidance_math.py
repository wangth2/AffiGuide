"""Frozen guidance arithmetic for shared-proposal AffiGuide.

Methods: calibrated minimum-norm consensus, TV-matched mirror descent,
right-local rotation increments, and calibrated minimax-regret selection.
Float64 diagnostic/consensus reductions and float32 state casts intentionally
retain the production operation order. These are not interchangeable with
normalized-posterior mirror variants used in older experiments.
"""
import hashlib
import json
import math
import numpy as np
import torch
from .rotation_guidance import apply_local_rotation, local_rotation_exp, rotation_to_legacy_rotvec
EPSILON = 1e-08

def effective_sampling_probs(post):
    """Observational: reproduce native float32 `_sample` weights, normalized.

    Native sampling actually passes post+EPS to multinomial. This helper never
    feeds the sampler, which retains that literal old call and its RNG draws.
    """
    if hasattr(post, 'detach'):
        q = post.detach() + EPSILON
        return q / q.sum(-1, keepdim=True)
    q = np.asarray(post) + np.asarray(EPSILON, dtype=np.asarray(post).dtype)
    return q / q.sum(-1, keepdims=True)

def additive_update(q, gradient, eta, mask=None):
    """Bitwise old R0 `post-eta*g; where; clamp(1e-8); normalize`.

    `eta` may be the actual original tensor 1-alpha_bar[t]. It is deliberately
    NOT converted to Python float before multiplication in the torch path.
    """
    if hasattr(q, 'detach'):
        import torch
        p = q - gradient * eta
        if mask is not None:
            p = torch.where(mask[..., None].expand_as(p), p, q.detach())
        p = torch.clamp(p, min=EPSILON)
        p = p / p.sum(-1, keepdim=True)
        info = {'rule': 'ADD', 'eta_seq': float(eta.detach().flatten()[0]) if torch.is_tensor(eta) else float(eta), 'clipped_channels': int((q - gradient * eta < EPSILON)[mask].sum()) if mask is not None else int((q - gradient * eta < EPSILON).sum()), 'fallback': None, 'probability_evaluations': 1}
        return (p.detach(), info)
    q = np.asarray(q)
    g = np.asarray(gradient, dtype=q.dtype)
    p = q - np.asarray(eta, dtype=q.dtype) * g
    if mask is not None:
        p = np.where(np.asarray(mask)[..., None], p, q)
    count = int((p < EPSILON).sum())
    p = np.maximum(p, np.asarray(EPSILON, dtype=p.dtype))
    p = p / p.sum(-1, keepdims=True)
    return (p, {'rule': 'ADD', 'eta_seq': float(eta), 'clipped_channels': count, 'fallback': None, 'probability_evaluations': 1})

def stable_bytes(values):
    return json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode('utf-8')

def derive_seed(base_seed, case_id, sample_id, t, stream_name):
    """Original named independent candidate stream; method labels and scores never enter it."""
    values = [int(base_seed), str(case_id), f'{int(sample_id):04d}', int(t), str(stream_name)]
    return int.from_bytes(hashlib.sha256(stable_bytes(values)).digest()[:8], 'big') % 2 ** 63

def _array(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)

def combine(gD, gG, mask, block, calibration):
    """Calibrated minimum-norm weight, followed by the original raw-gradient update.

Only q weights use row-tangent gradients; the applied gradient retains its
raw offset. Float64 reductions and the lambda=1 exact-copy branch are part
of the frozen numerical protocol."""
    proof = calibration.get('blocks', {}).get(block, {})
    if proof.get('status') != 'FROZEN':
        raise RuntimeError('COMMON_CALIBRATION_BLOCKED ' + block + ': ' + str(proof.get('reason', 'missing frozen block')))
    policy = calibration.get('numerical', {})
    required = ('difference_atol', 'difference_rtol', 'common_zero_atol', 'common_zero_rtol', 'scale_min')
    if any((k not in policy for k in required)):
        raise RuntimeError('COMMON_NUMERICAL_POLICY_MISSING')
    if any((not math.isfinite(float(policy[k])) or float(policy[k]) < 0 for k in required)):
        raise ValueError('COMMON_INVALID_NUMERICAL_POLICY')
    sD = float(proof['s_D'])
    sG = float(proof['s_G'])
    if any((not math.isfinite(x) or x <= float(policy['scale_min']) for x in (sD, sG))):
        raise RuntimeError('COMMON_NEAR_ZERO_OR_NONFINITE_FROZEN_SCALE ' + block)
    d = _array(gD)
    g = _array(gG)
    m = _array(mask).astype(bool)
    if d.shape != g.shape or d.shape[:-1] != m.shape or (not m.any()):
        raise ValueError('COMMON_BLOCK_SHAPE')
    if not np.isfinite(d[m]).all() or not np.isfinite(g[m]).all():
        raise FloatingPointError('COMMON_NONFINITE_GRADIENT ' + block)
    dr = d[m]
    gr = g[m]
    dt = dr - dr.mean(-1, keepdims=True) if block == 'q' else dr
    gt = gr - gr.mean(-1, keepdims=True) if block == 'q' else gr
    u = dt / sD
    v = gt / sG
    w = u - v
    uu = float(np.square(u).sum())
    vv = float(np.square(v).sum())
    ww = float(np.square(w).sum())
    difference_tolerance = float(policy['difference_atol']) ** 2 + float(policy['difference_rtol']) ** 2 * max(uu, vv)
    lam = 1.0 if ww <= difference_tolerance else float(np.clip(-float((v * w).sum()) / ww, 0.0, 1.0))
    tangent = lam * u + (1 - lam) * v
    cn = float(np.linalg.norm(tangent))
    zero_tolerance = float(policy['common_zero_atol']) + float(policy['common_zero_rtol']) * max(math.sqrt(uu), math.sqrt(vv))
    near_zero = cn <= zero_tolerance
    if near_zero:
        implemented = np.zeros_like(d)
    elif lam == 1.0:
        implemented = d.copy()
    else:
        implemented = lam * d + (1 - lam) * (sD / sG) * g
    implemented = np.where(m[..., None], implemented, 0.0)
    if hasattr(gD, 'detach'):
        import torch
        implemented = torch.as_tensor(implemented, device=gD.device, dtype=gD.dtype)
    diag = dict(block=block, lambda_=lam, cosine=float((dt * gt).sum() / (np.linalg.norm(dt) * np.linalg.norm(gt))) if np.linalg.norm(dt) > 0 and np.linalg.norm(gt) > 0 else None, raw_D_norm=float(np.linalg.norm(dr)), raw_G_norm=float(np.linalg.norm(gr)), tangent_D_norm=float(np.linalg.norm(dt)), tangent_G_norm=float(np.linalg.norm(gt)), common_tangent_norm=cn, raw_common_offset=(lam * dr + (1 - lam) * (sD / sG) * gr).mean(-1).tolist() if block == 'q' else None, near_zero=near_zero, scale_D=sD, scale_G=sG, difference_degenerate=ww <= difference_tolerance, common_zero_tolerance=zero_tolerance, weights_scope='one weight for flattened complete design block', implementation='raw convex combination with frozen scale ratio')
    diag['lambda'] = diag.pop('lambda_')
    return (implemented, diag)

def tv(a, b):
    return float(0.5 * np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).sum(-1).mean())

def solve(q0, gradient, target, policy):
    """Match a literal additive-shadow TV target along a support-preserving mirror path.

Do not normalize the input q0: its pre-epsilon row mass is meaningful.
Constant-gradient and zero-target cases preserve literal q0. The finite
solver retains the lower feasible bisection branch."""
    q = np.asarray(q0, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    if q.ndim != 2 or q.shape != g.shape or q.shape[-1] != 20 or (not len(q)):
        raise ValueError('MTV_SHAPE')
    if not np.isfinite(q).all() or not np.isfinite(g).all() or (q < 0).any() or (q.sum(-1) <= 0).any():
        raise FloatingPointError('MTV_NONFINITE_OR_INVALID_INPUT')
    if not math.isfinite(float(target)) or target < 0:
        raise ValueError('MTV_TARGET_INVALID')
    required = ('tv_atol', 'tv_rtol', 'bracket_steps', 'bisection_steps', 'eta_scaled_max', 'initial_eta_scaled', 'minimum_gradient_scale')
    if any((k not in policy for k in required)):
        raise RuntimeError('MTV_FROZEN_NUMERICAL_POLICY_MISSING')
    if min((float(policy[k]) for k in required)) < 0 or int(policy['bracket_steps']) < 1 or int(policy['bisection_steps']) < 1:
        raise ValueError('MTV_INVALID_NUMERICAL_POLICY')
    support = q > 0
    zero = int((~support).sum())
    tol = float(policy['tv_atol']) + float(policy['tv_rtol']) * float(target)
    centered = g - g.mean(-1, keepdims=True)
    scale = float(np.abs(centered).max())
    positive_min = np.where(support, g, np.inf).min(-1, keepdims=True)
    amin = support & (g == positive_min)
    limiting = np.where(amin, q, 0.0)
    limiting /= limiting.sum(-1, keepdims=True)
    reachable = tv(limiting, q)
    diag = dict(target_tv=float(target), achieved_tv=0.0, error=-float(target), eta=0.0, bracket_steps=0, iterations=0, reachable_tv=reachable, zero_support_count=zero, underflow_count=0, floor_count=0, q0_row_mass_max_error=float(np.abs(q.sum(-1) - 1).max()), per_position_tv=np.zeros(len(q)).tolist(), status='MTV_MATCHED', reason=None, source_pattern='old bridge finite bracket40/bisect52; strict zero support variant')
    if target == 0:
        return (q.copy(), diag)
    if scale <= float(policy['minimum_gradient_scale']):
        normalized_baseline = q / q.sum(-1, keepdims=True)
        diag.update(status='MTV_LIMITED', reason='constant_or_numerically_zero_gradient_no_action', reachable_tv=0.0, normalized_baseline_TV=tv(normalized_baseline, q), normalized_baseline_committed=False, gradient_scale=scale, row_mass_policy='literal q0 no action; no constant-gradient normalization')
        return (q.copy(), diag)
    logq = np.full_like(q, -np.inf)
    np.log(q, out=logq, where=support)
    normalized_gradient = centered / scale

    def path(z):
        logits = logq - z * normalized_gradient
        logits -= np.max(logits, axis=-1, keepdims=True)
        out = np.exp(logits)
        out /= out.sum(-1, keepdims=True)
        if not np.isfinite(out).all() or (out[~support] != 0).any():
            raise FloatingPointError('MTV_MIRROR_PATH_INVALID')
        return (out, tv(out, q))
    low = 0.0
    high = float(policy['initial_eta_scaled'])
    upper = float(policy['eta_scaled_max'])
    if high <= 0 or upper < high or (not math.isfinite(upper)):
        raise ValueError('MTV_ETA_BOUND_INVALID')
    best = q.copy()
    value = 0.0
    bracketed = False
    for i in range(int(policy['bracket_steps'])):
        (out, amount) = path(high)
        diag['bracket_steps'] = i + 1
        if amount >= target:
            bracketed = True
            break
        low = high
        best = out
        value = amount
        if high == upper:
            break
        high = min(high * 2, upper)
    if bracketed:
        for i in range(int(policy['bisection_steps'])):
            mid = (low + high) / 2
            (out, amount) = path(mid)
            diag['iterations'] = i + 1
            if amount <= target:
                low = mid
                best = out
                value = amount
            else:
                high = mid
    limited = target > reachable + tol or not bracketed or abs(value - target) > tol
    diag.update(eta=low / scale, achieved_tv=value, error=value - target, status='MTV_LIMITED' if limited else 'MTV_MATCHED', reason='target_exceeds_support_limit' if target > reachable + tol else 'finite_eta_or_numeric_limit' if limited else None, underflow_count=int((support & (best == 0)).sum()), per_position_tv=(0.5 * np.abs(best - q).sum(-1)).tolist(), eta_scaled=low, eta_scaled_upper=upper, gradient_scale=scale)
    return (best, diag)

def apply_mtv(q0, gradient, eta, mask, numerical_policy):
    """Compute the original float32 ADD shadow, then commit only the TV-matched mirror result."""
    import torch
    if numerical_policy.get('status') != 'FROZEN':
        raise RuntimeError('MTV_NUMERICAL_POLICY_NOT_FROZEN')
    (shadow, shadow_info) = additive_update(q0, gradient, eta, mask)
    result = q0.detach().clone()
    details = []
    for n in range(len(q0)):
        m = mask[n].bool()
        before = q0[n, m].detach().cpu().double().numpy()
        g = gradient[n, m].detach().cpu().double().numpy()
        target = tv(shadow[n, m].detach().cpu().double().numpy(), before)
        (proposed, info) = solve(before, g, target, numerical_policy['MTV'])
        result[n, m] = torch.as_tensor(proposed, device=q0.device, dtype=q0.dtype)
        actual = result[n, m].detach().cpu().double().numpy()
        achieved = tv(actual, before)
        tol = float(numerical_policy['MTV']['tv_atol']) + float(numerical_policy['MTV']['tv_rtol']) * target
        if target == 0 and (not torch.equal(result[n, m], q0[n, m])):
            raise RuntimeError('MTV_ZERO_ACTION_NOT_LITERAL_Q0')
        info.update(achieved_tv=achieved, error=achieved - target, per_position_tv=(0.5 * np.abs(actual - before).sum(-1)).tolist(), underflow_count=int(((before > 0) & (actual == 0)).sum()), batch_item=n, shadow_ADD_eta=shadow_info['eta_seq'], shadow_ADD_committed=False)
        if abs(achieved - target) > tol:
            info['status'] = 'MTV_LIMITED'
            info['reason'] = info['reason'] or 'output_dtype_roundoff_limit'
        details.append(info)
    return (result.detach(), dict(rule='MTV', per_sample=details, scalar_scope='one scalar per trajectory', shadow_ADD=shadow_info))

def select(d, g, calibration):
    """Select by calibrated Chebyshev regret with the frozen four-component tie key.

Keep location subtraction before scale division even though the location
cancels algebraically in the regret."""
    d = np.asarray(d, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    valid = np.isfinite(d) & np.isfinite(g)
    if not valid.any():
        raise FloatingPointError('All candidates unscoreable')
    zd = (d - calibration['D']['location']) / calibration['D']['scale']
    zg = (g - calibration['G']['location']) / calibration['G']['scale']
    rd = zd - zd[valid].min()
    rg = zg - zg[valid].min()
    winner = min(np.flatnonzero(valid).tolist(), key=lambda k: (max(rd[k], rg[k]), rd[k] + rg[k], zd[k], k))
    return (winner, dict(rule='frozen_continuous_regret_Chebyshev', selection_key=[float(max(rd[winner], rg[winner])), float(rd[winner] + rg[winner]), float(zd[winner]), winner]))

def saved_delta(v, g, mask, sigma, gamma):
    """Compute one capped right-local rotation delta for reuse by every candidate."""
    from diffab.modules.diffusion.rotation_guidance import apply_local_rotation
    (result, info) = apply_local_rotation(v, g, mask, sigma, gamma, 5.0)
    sig = torch.as_tensor(sigma, device=v.device, dtype=v.dtype)
    if sig.numel() == v.size(0):
        sig = sig.reshape(v.size(0), 1, 1)
    raw = torch.where(mask[..., None], -float(gamma) * sig * g, torch.zeros_like(g))
    norm = torch.linalg.norm(raw, dim=-1)
    delta = raw * torch.minimum(torch.ones_like(norm), math.radians(5) / (norm + 1e-12))[..., None]
    info.update(gamma=float(gamma), delta_used=delta, RMS_angle_rad=float(torch.linalg.norm(delta[mask], dim=-1).square().mean().sqrt()), cap_fraction=float(info['cap_hit'][mask].float().mean()))
    return (result, delta, info)

@torch.no_grad()
def apply_saved_rotation(v, delta, mask):
    """Apply the already computed local delta once to a candidate rotation."""
    from diffab.modules.common.so3 import so3vec_to_rotation
    from diffab.modules.diffusion.rotation_guidance import local_rotation_exp, rotation_to_legacy_rotvec
    out = v.detach().clone()
    changed = mask & (torch.linalg.norm(delta, dim=-1) > 0)
    if bool(changed.any()):
        out[changed] = rotation_to_legacy_rotvec(so3vec_to_rotation(v[changed]) @ local_rotation_exp(delta[changed]))
    return out
