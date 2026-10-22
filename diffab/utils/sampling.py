"""Trajectory persistence and RNG bookkeeping, independent of scientific updates."""
import contextlib
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def cpu(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return type(value)(cpu(v) for v in value)
    return value


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.random.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(value, device):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.random.set_rng_state(value['cpu'])
    if device.type == 'cuda': torch.cuda.set_rng_state(value['cuda'], device)


@contextlib.contextmanager
def candidate_rng(device, seed):
    # Candidate streams fork Torch only; the versioned portable SO(3) stream
    # continues across slots, as in the frozen paper implementation.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.random.default_generator.manual_seed(int(seed))
        if device.type == 'cuda': torch.cuda.manual_seed(int(seed))
        yield


def tensor_hash(value):
    x=value.detach().cpu().contiguous()
    h=hashlib.sha256(str(x.dtype).encode()+str(tuple(x.shape)).encode())
    h.update(x.numpy().tobytes()); return h.hexdigest()


def clean(value):
    if torch.is_tensor(value): return clean(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save_checkpoint(path, value):
    if path is None: return
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    torch.save(cpu(value),tmp); os.replace(tmp,path)
