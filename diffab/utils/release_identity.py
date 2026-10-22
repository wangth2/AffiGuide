"""Release provenance outside the scientific sampling loop."""
import functools
import hashlib
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def source_manifest():
    root=Path(__file__).resolve().parents[2]
    paths=[]
    for folder in ('diffab','configs'):
        for path in (root/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json','.yml','.yaml','.cu','.fatbin','.so'):
                paths.append(path)
    files={str(p.relative_to(root)):sha256(p) for p in sorted(paths)}
    import json
    digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(sha256=digest,files=files)


def numerical_environment():
    import torch
    return dict(torch=torch.__version__,cuda=torch.version.cuda,
        dtype=str(torch.get_default_dtype()),
        matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic))
