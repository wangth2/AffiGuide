"""Small explicit guidance settings; checkpoints and model definitions are separate."""
import copy
import json
import math
from pathlib import Path

_DEFAULT = dict(protocol='shared_proposal', gradient_source='DG', gradient_start_t=50,
    channels=['p', 'omega', 'q'], rotation_start_t=50,
    candidate_count=20, candidate_start_t=50, selector='auto',
    sequence_operator='MTV', coordinate_scale=4., rotation_scale=1., sequence_scale=1.,
    rotation_cap_degrees=5., hydro_prior_b=.8, backend='paper_v4',
    calibration='configs/calibration/H3.json',
    hard_calibration='configs/calibration/hard.json',
    numerical_policy='configs/calibration/numerical_policy.json',
    d_window=None, dg_window=None, channel_windows={}, candidate_window=None)


def _window(value, name):
    """Reverse-time inclusive interval; [] explicitly disables an interval."""
    if not isinstance(value, (list, tuple)) or len(value) not in (0, 2):
        raise ValueError(name + ' must be [] or [start_t, end_t]')
    if not value:
        return []
    if any(type(x) is not int for x in value) or not 1 <= value[1] <= value[0] <= 100:
        raise ValueError(name + ' must satisfy 100 >= start_t >= end_t >= 1')
    return list(value)


def _contains(window, t):
    return bool(window) and window[1] <= t <= window[0]


def _overlap(left, right):
    return bool(left and right) and max(left[1], right[1]) <= min(left[0], right[0])


def normalize_guidance(value=None):
    """Preserve caller paths and reject ambiguous old/new parameter mixtures."""
    value = copy.deepcopy(dict(value or {}))
    aliases = {'gradient_scale': 'coordinate_scale', 'gradient_scale_pos': 'coordinate_scale',
        'gradient_scale_cord': 'coordinate_scale', 'gradient_scale_categ': 'sequence_scale',
        'gradient_scale_seq': 'sequence_scale', 'gradient_scale_rot': 'rotation_scale',
        'rot_max_step_deg': 'rotation_cap_degrees'}
    for old, new in aliases.items():
        if old in value:
            if new in value and value[new] != value[old]:
                raise ValueError('Conflicting guidance options: %s / %s' % (old, new))
            value[new] = value.pop(old)
    result = copy.deepcopy(_DEFAULT)
    result.update(value)
    if result['protocol'] not in ('shared_proposal', 'legacy_k1'):
        raise ValueError('Unknown guidance protocol')
    if result['gradient_source'] not in ('D', 'DG', 'none'):
        raise ValueError('gradient_source must be D, DG, or none')
    channels = list(result['channels'])
    if len(channels) != len(set(channels)) or not set(channels) <= {'p', 'omega', 'q'}:
        raise ValueError('channels must be a unique subset of p, omega, q')
    result['channels'] = channels
    for key in ('gradient_start_t', 'rotation_start_t', 'candidate_start_t'):
        if type(result[key]) is not int or not 0 <= result[key] <= 100:
            raise ValueError(key + ' must be an integer in [0,100]')
    if type(result['candidate_count']) is not int or result['candidate_count'] < 1:
        raise ValueError('candidate_count must be a positive integer')
    if result['selector'] == 'auto':
        result['selector'] = 'd_minimum' if result['gradient_source'] == 'D' else 'dg_minimax'
    if result['selector'] not in ('d_minimum', 'dg_minimax', 'single'):
        raise ValueError('Unknown candidate selector')
    if result['gradient_source'] == 'none' and result['selector'] == 'single':
        raise ValueError('For unguided K1, use an original YAML without guidance and --no_sample_ddg; the registered selection-only protocol retains D/G scoring')
    if result['selector'] == 'single' and result['candidate_count'] != 1:
        raise ValueError('selector=single requires candidate_count=1')
    if result['sequence_operator'] not in ('ADD', 'MTV', 'none'):
        raise ValueError('Unknown sequence operator')
    if result['backend'] not in ('production_atomic', 'deterministic_gpu', 'paper_v4'):
        raise ValueError('Unknown numerical backend; no automatic fallback is allowed')
    for key in ('coordinate_scale', 'rotation_scale', 'sequence_scale'):
        if not math.isfinite(float(result[key])) or float(result[key]) < 0:
            raise ValueError(key + ' must be finite and nonnegative')
    if not math.isfinite(float(result['rotation_cap_degrees'])) or result['rotation_cap_degrees'] <= 0:
        raise ValueError('rotation_cap_degrees must be finite and positive')
    if result['hydro_prior_b'] is not None and not math.isfinite(float(result['hydro_prior_b'])):
        raise ValueError('hydro_prior_b must be finite, or null for the ordinary posterior')
    if result['protocol'] == 'legacy_k1' and (result['candidate_count'] != 1 or
            result['gradient_source'] != 'D' or result['sequence_operator'] != 'ADD'):
        raise ValueError('The historical K1 path requires D-only, K1, and original ADD')
    if result.get('tyr_control') is not None:
        raise ValueError('Tyr ablation operators are not included in this runtime distribution')
    source = result['gradient_source']
    dg = result['dg_window']
    if dg is None:
        start = result['gradient_start_t']
        dg = [start, 1] if source == 'DG' and start else []
    dg = _window(dg, 'dg_window')
    if source != 'DG' and dg:
        raise ValueError('A nonempty dg_window requires gradient_source=DG')
    d = result['d_window']
    if d is None:
        if source == 'none':
            d = []
        elif source == 'DG' and dg:
            d = [100, dg[0] + 1] if dg[0] < 100 else []
        else:
            d = [100, 1]
    d = _window(d, 'd_window')
    if source == 'none' and d:
        raise ValueError('A nonempty d_window requires D or DG guidance')
    if _overlap(d, dg):
        raise ValueError('D-only and DG windows must not overlap; both endpoints are inclusive')
    windows = result['channel_windows']
    if not isinstance(windows, dict) or set(windows) - {'p', 'omega', 'q'}:
        raise ValueError('channel_windows must map p, omega, q to intervals')
    rotation = [result['rotation_start_t'], 1] if result['rotation_start_t'] else []
    channel_windows = {name: _window(windows.get(name, default), 'channel_windows.' + name)
                       for name, default in (('p', [100, 1]), ('omega', rotation), ('q', [100, 1]))}
    candidate = result['candidate_window']
    if candidate is None:
        candidate = [result['candidate_start_t'], 1] if result['candidate_start_t'] else []
    result['resolved_windows'] = dict(D=d, DG=dg, channels=channel_windows,
                                      candidates=_window(candidate, 'candidate_window'))
    return result


def step_schedule(guidance, t):
    """Inclusive reverse-time gates; no RNG or model calls occur here."""
    cfg = normalize_guidance(guidance)
    if type(t) is not int or not 1 <= t <= 100:
        raise ValueError('Reverse step t must be in [1,100]')
    windows = cfg['resolved_windows']
    common = _contains(windows['DG'], t)
    active = common or _contains(windows['D'], t)
    directions = {}
    scales = {'p': 'coordinate_scale', 'omega': 'rotation_scale', 'q': 'sequence_scale'}
    for block in ('p', 'omega', 'q'):
        if (not active or block not in cfg['channels'] or float(cfg[scales[block]]) == 0
                or not _contains(windows['channels'][block], t)):
            directions[block] = 'OFF'
        else:
            directions[block] = 'COMMON' if common else 'D'
    d = {b: int(directions[b] != 'OFF') for b in directions}
    g = {b: int(directions[b] == 'COMMON') for b in directions}
    return dict(t=t, K=cfg['candidate_count'] if _contains(windows['candidates'], t) else 1,
        DG_active=any(g.values()), directions=directions, active_channels={b: bool(d[b]) for b in d},
        D_logical_requests=d, G_logical_requests=g, D_backward_calls=sum(d.values()),
        G_gradient_requests=sum(g.values()), selector=cfg['selector'])


schedule = step_schedule


def uses_dg_gradients(guidance):
    cfg = normalize_guidance(guidance)
    windows = cfg['resolved_windows']
    scales = {'p': 'coordinate_scale', 'omega': 'rotation_scale', 'q': 'sequence_scale'}
    return any(float(cfg[scales[b]]) > 0 and _overlap(windows['DG'], windows['channels'][b])
               for b in cfg['channels'])


def needs_gearbind(guidance):
    cfg = normalize_guidance(guidance)
    return cfg['selector'] == 'dg_minimax' or uses_dg_gradients(cfg)


def schedule_segments(guidance):
    """Compact, executable-plan preview without importing model libraries."""
    result = []
    for t in range(100, 0, -1):
        step = step_schedule(guidance, t)
        state = {name: step[name] for name in ('directions', 'K', 'selector')}
        if result and all(result[-1][name] == value for name, value in state.items()):
            result[-1]['to_t'] = t
        else:
            result.append(dict(from_t=t, to_t=t, **state))
    return result


def load_calibration(guidance, cdr=None, root=None):
    """Load frozen assets; a missing per-CDR scale never falls back to H3."""
    cfg = normalize_guidance(guidance)
    need_common = uses_dg_gradients(cfg)
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    common_path = cfg['calibration']
    if 'calibration_by_cdr' in cfg:
        common_path = cfg['calibration_by_cdr'].get(cdr)
        if need_common and not common_path:
            raise FileNotFoundError('MISSING_FROZEN_CDR_CALIBRATION ' + str(cdr))
    def read(path):
        if isinstance(path, dict):
            return copy.deepcopy(path)
        p = Path(path)
        p = p if p.is_absolute() else root / p
        with p.open() as stream:
            return json.load(stream)
    common = read(common_path) if need_common else None
    hard = read(cfg['hard_calibration']) if cfg['selector'] == 'dg_minimax' else None
    if hard is not None and 'hard' in hard:
        hard = hard['hard']
    policy = read(cfg['numerical_policy']) if cfg['sequence_operator'] == 'MTV' else None
    assets = dict(common=common, hard=hard, numerical=policy)
    return assets
