"""Versioned portable SO3 categorical RNG; no historical output injection.

Only the categorical draw is replaced. The original ApproxAngularDistribution
sample body, histogram construction, bin interpolation and Gaussian branch stay
unchanged. Float32 left-to-right CDF and counter-based uniform are shared by the
CPU reference and a no-atomic CUDA implementation. This intentionally establishes
new RNG consumption semantics; it cannot reproduce the old torch RNG trajectory.
"""
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

VERSION = 'AFFIGUIDE_PORTABLE_SO3_SPLITMIX64_F32CDF_V1'
MASK64 = (1 << 64) - 1
STREAM = 'so3_angle'
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'kernels/portable_rng/portable_categorical.cu'
DEFAULT_LIBRARY = ROOT / 'build/portable_rng/portable_rng_v1.fatbin'
_LIBRARIES = {}


class PortableRNGError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def splitmix64(value):
    x = (int(value) + 0x9e3779b97f4a7c15) & MASK64
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK64
    return (x ^ (x >> 31)) & MASK64


def _u64(value, label):
    if type(value) is not int or not 0 <= value <= MASK64:
        raise PortableRNGError(label + '_MUST_BE_UINT64')
    return value


def stream_hash(stream):
    if not isinstance(stream, str) or not stream:
        raise PortableRNGError('NONEMPTY_VERSIONED_STREAM_REQUIRED')
    x = 0xcbf29ce484222325
    for byte in stream.encode('utf-8'):
        x = ((x ^ byte) * 0x100000001b3) & MASK64
    return x


def counter_key(task_seed, call_index, stream=STREAM):
    return splitmix64(_u64(task_seed, 'TASK_SEED') ^ stream_hash(stream)
                      ^ splitmix64(_u64(call_index, 'CALL_INDEX')))


def uniform_bits(task_seed, call_index, rows, stream=STREAM):
    """Top 24 bits; integer results identical on CPU and GPU, no global state."""
    import numpy as np
    if type(rows) is not int or rows < 1:
        raise PortableRNGError('POSITIVE_INTEGER_ROW_COUNT_REQUIRED')
    key = counter_key(task_seed, call_index, stream)
    x = np.arange(rows, dtype=np.uint64)
    def mixed(a):
        with np.errstate(over='ignore'):
            a = a + np.uint64(0x9e3779b97f4a7c15)
            a = (a ^ (a >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
            a = (a ^ (a >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
            return a ^ (a >> np.uint64(31))
    return (mixed(np.uint64(key) ^ mixed(x)) >> np.uint64(40)).astype(np.int32)


def _probability(prob):
    import torch
    if not torch.is_tensor(prob) or prob.dtype != torch.float32 or prob.ndim != 2:
        raise PortableRNGError('ORIGINAL_2D_FLOAT32_PROBABILITY_REQUIRED')
    if prob.shape[0] < 1 or prob.shape[1] < 1 or prob.stride(0) < 0 or prob.stride(1) <= 0:
        raise PortableRNGError('NONEMPTY_POSITIVE_COLUMN_STRIDE_REQUIRED')
    if prob.device.type not in ('cpu', 'cuda'):
        raise PortableRNGError('UNSUPPORTED_DEVICE')
    return prob.detach()


def select_from_bits_cpu(prob, bits):
    """Diagnostic CPU oracle, accepts explicit integer uniforms, not old answers."""
    import numpy as np
    p = np.asarray(prob)
    bits = np.asarray(bits)
    if p.dtype != np.float32 or p.ndim != 2 or min(p.shape) < 1:
        raise PortableRNGError('CPU_2D_FLOAT32_REQUIRED')
    if bits.shape != (p.shape[0],) or bits.dtype.kind not in 'iu' or np.any(bits < 0) or np.any(bits >= 1 << 24):
        raise PortableRNGError('EXPLICIT_UNIFORM_BITS_INVALID')
    if not np.isfinite(p).all() or np.any(p < 0):
        raise PortableRNGError('INVALID_PROBABILITY_NEGATIVE_OR_NONFINITE')
    with np.errstate(over='ignore', invalid='ignore'):
        cumulative = np.cumsum(p, axis=1, dtype=np.float32)
        mass = cumulative[:, -1].copy()
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise PortableRNGError('INVALID_PROBABILITY_ZERO_OR_OVERFLOW_MASS')
    uniform = bits.astype(np.float32) * np.float32(2.0 ** -24)
    target = np.multiply(uniform, mass, dtype=np.float32)
    greater = cumulative > target[:, None]
    chosen = greater.argmax(axis=1).astype(np.int64)
    # FP32 multiplication may round target to total. Keep trailing zero bins
    # impossible by choosing the last positive bin only in this rounding case.
    fallback = (~greater.any(axis=1))
    if np.any(fallback):
        last_positive = p.shape[1] - 1 - (p[:, ::-1] > 0).argmax(axis=1)
        chosen[fallback] = last_positive[fallback]
    return chosen[:, None], mass, target


def build_cuda_library(output=None, nvcc='/usr/local/cuda/bin/nvcc',
                       architecture=None):
    """Explicit device-only offline build; no new host ELF/glibc/cudart ABI."""
    output = Path(output or DEFAULT_LIBRARY).resolve()
    if output.suffix != '.fatbin':
        raise PortableRNGError('DEVICE_ONLY_FATBIN_OUTPUT_REQUIRED')
    output.parent.mkdir(parents=True, exist_ok=True)
    architectures = [80, 86, 89] if architecture is None else [int(str(architecture).replace('sm_', '').replace('compute_', ''))]
    if any(x not in (70, 75, 80, 86, 89) for x in architectures):
        raise PortableRNGError('EXPLICIT_SUPPORTED_ARCHITECTURE_REQUIRED')
    command = [str(nvcc), '--fatbin', '--cudart=none', '-O2', '--fmad=false',
               '--ftz=false', '--prec-div=true', '--prec-sqrt=true']
    for cap in architectures:
        command += ['-gencode', 'arch=compute_%d,code=sm_%d' % (cap, cap)]
    command += [str(SOURCE), '-o', str(output)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    if result.returncode:
        raise PortableRNGError('OFFLINE_CUDA_BUILD_FAILED ' + result.stderr[-4000:])
    manifest = dict(version=VERSION, schema='AFFIGUIDE_PORTABLE_RNG_DEVICE_ONLY_FATBIN_V2_DRIVER_API',
                    source=str(SOURCE), source_sha256=sha(SOURCE),
                    library=str(output), library_sha256=sha(output), command=command,
                    compiled_compute_capabilities=architectures, PTX_JIT_fallback=False,
                    new_host_ELF_or_glibc_or_cudart_dependency=False,
                    returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                    cross_device_validation='NOT_PERFORMED_BY_BUILD')
    output.with_suffix('.build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


class _DriverBackend:
    """Launch device-only kernel using the existing machine's CUDA Driver API."""
    def __init__(self, path, metadata):
        self.path, self.metadata = path, metadata
        self.lib = ctypes.CDLL('libcuda.so.1')
        P, U, I = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int
        signatures = [
            ('cuInit', [U]), ('cuCtxGetCurrent', [ctypes.POINTER(P)]),
            ('cuCtxPushCurrent_v2', [P]), ('cuCtxPopCurrent_v2', [ctypes.POINTER(P)]),
            ('cuPointerGetAttribute', [P, I, ctypes.c_uint64]),
            ('cuModuleLoadData', [ctypes.POINTER(P), P]),
            ('cuModuleGetFunction', [ctypes.POINTER(P), P, ctypes.c_char_p]),
            ('cuLaunchKernel', [P,U,U,U,U,U,U,U,P,ctypes.POINTER(P),ctypes.POINTER(P)]),
            ('cuGetErrorString', [I, ctypes.POINTER(ctypes.c_char_p)])]
        for name, arguments in signatures:
            fn = getattr(self.lib, name); fn.argtypes = arguments; fn.restype = I
        self.modules, self.lock = {}, threading.Lock()

    def check(self, code):
        if code:
            msg = ctypes.c_char_p()
            self.lib.cuGetErrorString(code, ctypes.byref(msg))
            raise PortableRNGError('PORTABLE_CUDA_DRIVER_ERROR %d ' % code +
                                   (msg.value.decode(errors='replace') if msg.value else 'UNKNOWN'))

    def launch(self, probability, nr, nc, s0, s1, key, out, bits, mass, target, status, stream):
        self.check(self.lib.cuInit(0))
        context = ctypes.c_void_p()
        # Query the allocation's actual context; a new Python/autograd thread
        # need not already have a current CUDA context despite torch.device().
        self.check(self.lib.cuPointerGetAttribute(ctypes.byref(context), 1,
                                                 ctypes.c_uint64(probability)))
        if not context.value:
            raise PortableRNGError('ACTUAL_PROBABILITY_CUDA_CONTEXT_UNRESOLVED')
        self.check(self.lib.cuCtxPushCurrent_v2(context))
        try:
            with self.lock:
                if context.value not in self.modules:
                    image = ctypes.create_string_buffer(self.path.read_bytes())
                    module = ctypes.c_void_p(); function = ctypes.c_void_p()
                    self.check(self.lib.cuModuleLoadData(ctypes.byref(module), ctypes.cast(image, ctypes.c_void_p)))
                    self.check(self.lib.cuModuleGetFunction(ctypes.byref(function), module, b'categorical_kernel'))
                    self.modules[context.value] = (image, module, function)
                function = self.modules[context.value][2]
            values = [ctypes.c_void_p(probability)] + [ctypes.c_int64(v) for v in (nr,nc,s0,s1)] + [ctypes.c_uint64(key)] + [ctypes.c_void_p(v) for v in (out,bits,mass,target,status)]
            arguments = (ctypes.c_void_p * len(values))(*[
                ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in values])
            self.check(self.lib.cuLaunchKernel(function, (nr+127)//128, 1, 1,
                      128, 1, 1, 0, ctypes.c_void_p(int(stream)), arguments, None))
            return 0
        finally:
            popped = ctypes.c_void_p(); self.check(self.lib.cuCtxPopCurrent_v2(ctypes.byref(popped)))


def _library():
    import torch
    path = Path(os.environ.get('AFFIGUIDE_PORTABLE_RNG_LIBRARY', str(DEFAULT_LIBRARY))).resolve()
    if not path.is_file():
        raise PortableRNGError('PORTABLE_CUDA_LIBRARY_NOT_BUILT_NO_ATOMIC_FALLBACK ' + str(path))
    if path not in _LIBRARIES:
        metadata = json.loads(path.with_suffix('.build.json').read_text())
        if metadata['version'] != VERSION or metadata['source_sha256'] != sha(SOURCE) or metadata['library_sha256'] != sha(path):
            raise PortableRNGError('PORTABLE_CUDA_LIBRARY_IDENTITY_MISMATCH')
        if metadata.get('schema') != 'AFFIGUIDE_PORTABLE_RNG_DEVICE_ONLY_FATBIN_V2_DRIVER_API':
            raise PortableRNGError('DEVICE_ONLY_ARTIFACT_REQUIRED_NO_HOST_ELF_FALLBACK')
        backend = _DriverBackend(path, metadata)
        _LIBRARIES[path] = (backend, backend.launch, metadata)
    metadata = _LIBRARIES[path][2]
    capability = torch.cuda.get_device_capability()
    arch = 10 * capability[0] + capability[1]
    if arch not in metadata['compiled_compute_capabilities']:
        raise PortableRNGError('PORTABLE_CUDA_ARCHITECTURE_NOT_COMPILED '+str(arch))
    return _LIBRARIES[path]


def portable_multinomial(prob, *, task_seed, call_index, stream=STREAM,
                         backend=None, return_audit=False):
    """Exactly one sample per row, without replacement semantics for n=1.

    No torch/Python/NumPy RNG is consumed. Caller supplies the task-scoped,
    versioned call counter; same seed alone is not a sufficient protocol lock.
    """
    import numpy as np
    import torch
    p = _probability(prob)
    backend = backend or p.device.type
    if backend != p.device.type:
        raise PortableRNGError('EXPLICIT_BACKEND_MUST_MATCH_INPUT_DEVICE')
    key = counter_key(task_seed, call_index, stream)
    nr, nc = p.shape
    if backend == 'cpu':
        bits = uniform_bits(task_seed, call_index, int(nr), stream)
        chosen, mass, target = select_from_bits_cpu(p.numpy(), bits)
        result = torch.from_numpy(chosen)
        audit = dict(uniform_bits=torch.from_numpy(bits), mass=torch.from_numpy(mass),
                     target=torch.from_numpy(target), status=torch.zeros(nr, dtype=torch.int32),
                     actual_probability_host_transfer_bytes=0)
    elif backend == 'cuda':
        with torch.cuda.device(p.device):
            _, fn, metadata = _library()
            result = torch.empty((nr, 1), device=p.device, dtype=torch.int64)
            bits = torch.empty(nr, device=p.device, dtype=torch.int32)
            mass = torch.empty(nr, device=p.device, dtype=torch.float32)
            target = torch.empty(nr, device=p.device, dtype=torch.float32)
            status = torch.empty(nr, device=p.device, dtype=torch.int32)
            code = fn(p.data_ptr(), nr, nc, p.stride(0), p.stride(1), key,
                      result.data_ptr(), bits.data_ptr(), mass.data_ptr(), target.data_ptr(),
                      status.data_ptr(), torch.cuda.current_stream(p.device).cuda_stream)
            if code:
                raise PortableRNGError('PORTABLE_CUDA_KERNEL_LAUNCH_FAILED ' + str(code))
            # Small explicit validation transfer only. Full probabilities never
            # leave GPU in this path; error detection is not silently omitted.
            status_cpu = status.cpu().numpy()
            if np.any(status_cpu):
                raise PortableRNGError('INVALID_GPU_PROBABILITY_ROWS ' + repr(np.flatnonzero(status_cpu).tolist()))
        audit = dict(uniform_bits=bits, mass=mass, target=target, status=status,
                     actual_probability_host_transfer_bytes=0,
                     validation_host_transfer_bytes=int(nr) * 4,
                     library_sha256=metadata['library_sha256'])
    else:
        raise PortableRNGError('NO_SUPPORTED_BACKEND_NO_ATOMIC_FALLBACK')
    audit.update(version=VERSION, task_seed=task_seed, call_index=call_index,
                 stream=stream, counter_key=key, rows=int(nr), columns=int(nc),
                 Torch_RNG_consumed=False, old_results_injected=False,
                 float32_CDF_order='original columns left-to-right, rounded after each add',
                 boundary='first cumulative > float32(U*total); rounding-only last-positive fallback')
    return (result, audit) if return_audit else result


class ProtocolState:
    def __init__(self, task_seed=0, stream=STREAM):
        self.stream = stream
        stream_hash(stream)
        self.reset(task_seed)

    def reset(self, task_seed):
        self.task_seed = _u64(task_seed, 'TASK_SEED')
        self.call_index = 0
        return self

    def state_dict(self):
        return dict(version=VERSION, task_seed=self.task_seed, call_index=self.call_index,
                    stream=self.stream, Torch_RNG_consumed=False,
                    downstream_Torch_RNG='original remaining operations consume their original stream; removed multinomial consumption is not emulated')

    def load_state_dict(self, packet):
        if packet.get('version') != VERSION or packet.get('stream') != self.stream:
            raise PortableRNGError('PORTABLE_RNG_PROTOCOL_VERSION_OR_STREAM_MISMATCH')
        self.task_seed = _u64(packet['task_seed'], 'TASK_SEED')
        self.call_index = _u64(packet['call_index'], 'CALL_INDEX')
        return self

    def hash(self):
        return hashlib.sha256(json.dumps(self.state_dict(), sort_keys=True,
                           separators=(',', ':')).encode()).hexdigest()
