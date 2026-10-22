"""Deterministic float32 graph sums using GPU CSR segmentation.

No CUDA compiler or additional package is needed: torch_scatter.segment_csr is
already present in the frozen GearBind environment.  Sum ordering is stable
within a segment, and the installed CSR kernel has a fixed reduction schedule.
The first derivative is a direct gather, not an atomic scatter.  This module
never edits a model source file.  TargetedGPUHandle replaces only explicit,
source-hashed callsites in an experiment process and can restore them.

The CUDA implementation and this wrapper still require empirical reproducibility
verification on each supported software / device environment.  Determinism does
not imply bitwise identity to the original atomic sum or CPU accumulation.
"""
import hashlib
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable
import torch_scatter


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def _check_arguments(src, index, dim_size, check_index):
    if src.dtype != torch.float32 or index.dtype != torch.int64:
        raise TypeError('DETERMINISTIC_SUM_REQUIRES_FLOAT32_SRC_INT64_INDEX')
    if src.ndim != 2 or index.ndim != 1 or src.shape[0] != index.numel():
        raise ValueError('DETERMINISTIC_SUM_REQUIRES_MATCHING_2D_SRC_1D_INDEX')
    if src.device != index.device:
        raise ValueError('DETERMINISTIC_SUM_INDEX_DEVICE_MISMATCH')
    if isinstance(dim_size, bool) or not isinstance(dim_size, int) or dim_size < 0:
        raise ValueError('DETERMINISTIC_SUM_NONNEGATIVE_INTEGER_DIM_SIZE_REQUIRED')
    if index.numel() and dim_size == 0:
        raise ValueError('DETERMINISTIC_SUM_NONEMPTY_INDEX_WITH_ZERO_OUTPUT')
    if check_index and index.numel():
        # No .item(), CPU transfer or host synchronization on valid CUDA input.
        # Like other CUDA indexing operations, invalid indices fail the process;
        # the CUDA context must not be reused after a device assertion.
        torch._assert_async(((index >= 0) & (index < dim_size)).all())


class _StableCSRSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, index, dim_size, check_index):
        _check_arguments(src, index, dim_size, check_index)
        ctx.save_for_backward(index)
        ctx.input_shape = tuple(src.shape)
        if not index.numel() or not src.shape[1]:
            return src.new_zeros((dim_size, src.shape[1]))
        # Stable GPU sorting preserves the original slot order among equal
        # destinations, including entirely dynamic graph edge lists.
        sorted_index, permutation = torch.sort(index, stable=True)
        ordered_src = src.index_select(0, permutation)
        destinations = torch.arange(dim_size + 1, device=index.device,
                                    dtype=index.dtype)
        indptr = torch.searchsorted(sorted_index, destinations, right=False)
        return torch_scatter.segment_csr(ordered_src, indptr, reduce='sum')

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (index,) = ctx.saved_tensors
        # d sum(src[index == i]) / d src[j] = 1 for i == index[j].
        # Reading the same grad_output row repeatedly is safe; each src row has
        # one dedicated output position.  No write contention or atomic add.
        if not index.numel() or not ctx.input_shape[1]:
            grad_src = grad_output.new_zeros(ctx.input_shape)
        else:
            grad_src = grad_output.index_select(0, index)
        return grad_src, None, None, None


def deterministic_sum(src, index, dim=0, out=None, dim_size=None,
                      check_index=True):
    """Drop-in sum for the two observed 2D float32, dim=0 graph callsites.

    `check_index=False` is for separately audited trusted graph inputs and
    performance decomposition only.  Default checking stays on the device.
    First derivatives through model inputs are supported; second derivatives
    are explicitly unsupported, because they are outside production usage.
    """
    if dim != 0 or out is not None or dim_size is None:
        raise ValueError('DETERMINISTIC_SUM_ONLY_DIM0_OUT_NONE_EXPLICIT_SIZE')
    return _StableCSRSum.apply(src, index, dim_size, bool(check_index))
