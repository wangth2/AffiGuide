"""Primitive-only, framed local IPC. This module uses the standard library only."""
import io
import pickle
import struct

MAX_FRAME = 512 * 1024 * 1024
ARRAY_MARKER = "__primitive_array_v1__"


def primitive(value):
    """Convert tensors/arrays without importing their implementation packages."""
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return {ARRAY_MARKER: True, "dtype": str(tensor.dtype).split(".")[-1],
                "shape": list(tensor.shape), "values": tensor.tolist()}
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tolist"):
        return {ARRAY_MARKER: True, "dtype": str(value.dtype), "shape": list(value.shape),
                "values": value.tolist()}
    if value is None or type(value) in (str, bool, int, float, bytes):
        return value
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise TypeError("IPC dictionary keys must be primitive strings or integers")
        return {key: primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [primitive(item) for item in value]
    if hasattr(value, "item"):
        return primitive(value.item())
    raise TypeError("IPC payload cannot contain " + type(value).__name__)


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Only primitive IPC payloads are accepted")


def _read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Sidecar pipe closed during a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream):
    header = stream.read(8)
    if not header:
        return None
    if len(header) != 8:
        header += _read_exact(stream, 8 - len(header))
    size, = struct.unpack("!Q", header)
    if size > MAX_FRAME:
        raise ValueError("Sidecar frame exceeds its fixed bound")
    return PrimitiveUnpickler(io.BytesIO(_read_exact(stream, size))).load()


def write_frame(stream, value):
    encoded = pickle.dumps(primitive(value), protocol=4)
    if len(encoded) > MAX_FRAME:
        raise ValueError("Sidecar frame exceeds its fixed bound")
    for block in (struct.pack("!Q", len(encoded)), encoded):
        remaining = memoryview(block)
        while remaining:
            written = stream.write(remaining)
            if not written:
                raise BrokenPipeError("Sidecar pipe closed while writing a frame")
            remaining = remaining[written:]
    stream.flush()
