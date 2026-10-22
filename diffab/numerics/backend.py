"""Explicit numerical policies; scientific sampler settings are independent.

`deterministic_gpu` changes only the two GearBind reductions. `paper_v4` adds
its separately versioned SO(3) stream and the accepted named Linear operations.
No policy is installed by importing this module, and no fallback is implicit.
"""
import contextlib
import contextvars
import json
from pathlib import Path
import torch
from torch import nn

BACKENDS = ('production_atomic', 'deterministic_gpu', 'paper_v4')
_CURRENT = contextvars.ContextVar('affiguide_numerical_context', default=None)
GENERATOR_LINEAR_TARGETS = ('residue_embed.mlp.0',) + tuple(
    'diffusion.eps_net.encoder.blocks.%d.out_transform' % i for i in range(6))


def validate_backend(name):
    if name not in BACKENDS:
        raise ValueError('Unknown numerical backend %r; choose %s' % (name, ', '.join(BACKENDS)))
    return name


def execution_flags():
    return dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def apply_execution_flags(role):
    """Explicit frozen process-role policy; importing this module changes nothing."""
    profiles = json.loads((Path(__file__).parent / 'configs/PAPER_V4_EXECUTION_FLAGS.json').read_text())
    if role not in ('generator', 'gearbind'):
        raise ValueError('Execution flag role must be generator or gearbind')
    flags = profiles[role]
    torch.backends.cudnn.benchmark = flags['cudnn_benchmark']
    torch.backends.cudnn.deterministic = flags['cudnn_deterministic']
    torch.backends.cudnn.allow_tf32 = flags['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32 = flags['matmul_allow_tf32']
    torch.use_deterministic_algorithms(flags['deterministic_algorithms'])
    return execution_flags()


def graph_sum(src, index, count, backend='production_atomic'):
    validate_backend(backend)
    if backend == 'production_atomic':
        from torch_scatter import scatter_add
        return scatter_add(src, index, dim=0, dim_size=count)
    from .reductions import deterministic_sum
    return deterministic_sum(src, index, dim_size=int(count))


class GraphSumReadout(nn.Module):
    """The original TorchDrug SumReadout with an explicit sum policy."""
    def __init__(self, backend='production_atomic'):
        super().__init__()
        self.numerical_backend = validate_backend(backend)

    def forward(self, graph, input):
        # PackedGraph.num_nodes shape is host metadata; no GPU scalar readback.
        return graph_sum(input, graph.node2graph, graph.num_nodes.shape[0], self.numerical_backend)


class PortableLinear(nn.Linear):
    """Same parameters/state_dict names; accepted FP32 one-output-thread kernel."""
    @classmethod
    def from_linear(cls, original):
        # Avoid nn.Linear.__init__: no random initialization or RNG consumption.
        obj = cls.__new__(cls)
        super(nn.Linear, obj).__init__()
        obj.in_features, obj.out_features = original.in_features, original.out_features
        obj.weight, obj.bias = original.weight, original.bias
        obj.training = original.training
        obj.calls = 0
        return obj

    def forward(self, input):
        from .portable_linear import linear
        self.calls += 1
        return linear(input, self.weight, self.bias)


def _linear_targets(model, names):
    modules = dict(model.named_modules())
    for name in names:
        if name not in modules or not isinstance(modules[name], nn.Linear):
            raise ValueError('Required frozen Linear module is missing: ' + name)
        if isinstance(modules[name], PortableLinear):
            continue
        parent_name, _, child = name.rpartition('.')
        parent = modules[parent_name] if parent_name else model
        parent._modules[child] = PortableLinear.from_linear(modules[name])
    return tuple(names)


def _frozen_targets(kind):
    name = {'ddg': 'DDGPREDICTOR_LINEAR_TARGETS.json', 'gearbind': 'GEARBIND_LINEAR_TARGETS.json'}[kind]
    return json.loads((Path(__file__).parent / 'configs' / name).read_text())['target_names']


class NumericalContext:
    def __init__(self, backend, task_seed=0):
        self.backend = validate_backend(backend)
        from .portable_rng import ProtocolState
        self.portable = ProtocolState(int(task_seed)) if backend == 'paper_v4' else None

    def reset(self, task_seed):
        if self.portable is not None:
            self.portable.reset(int(task_seed))
        return self

    def state_dict(self):
        return {'backend': self.backend, 'portable_SO3': self.portable.state_dict() if self.portable else None}

    def load_state_dict(self, value):
        if value['backend'] != self.backend:
            raise ValueError('Checkpoint numerical backend differs from selected policy')
        if self.portable is not None:
            self.portable.load_state_dict(value['portable_SO3'])

    @contextlib.contextmanager
    def activate(self):
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    @contextlib.contextmanager
    def temporary_seed(self, device, seed):
        # Frozen V4 forks Torch only. Its portable SO3 counter continues across
        # candidate slots; resetting it here would change the paper protocol.
        device = torch.device(device)
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
            torch.random.default_generator.manual_seed(int(seed))
            if device.type == 'cuda':
                torch.cuda.manual_seed(int(seed))
            yield


def sample_so3_categories(prob):
    """Call only at ApproxAngularDistribution's original categorical draw."""
    context = _CURRENT.get()
    if context is None or context.backend != 'paper_v4':
        return torch.multinomial(prob, num_samples=1)
    from .portable_rng import portable_multinomial
    state = context.portable
    out = portable_multinomial(prob, task_seed=state.task_seed,
                              call_index=state.call_index, stream=state.stream)
    state.call_index += 1
    return out


def configure_generator(model, backend='production_atomic', task_seed=0):
    context = NumericalContext(backend, task_seed)
    if backend == 'paper_v4':
        apply_execution_flags('generator')
        _linear_targets(model, GENERATOR_LINEAR_TARGETS)
    return context


def configure_ddg(model, backend='production_atomic'):
    validate_backend(backend)
    if backend == 'paper_v4':
        _linear_targets(model, _frozen_targets('ddg'))
    return model


def configure_gearbind(model, backend='production_atomic'):
    validate_backend(backend)
    for module in model.modules():
        if isinstance(module, GraphSumReadout) or type(module).__name__ == 'GearBind':
            module.numerical_backend = backend
    if backend == 'paper_v4':
        _linear_targets(model, _frozen_targets('gearbind'))
    return model


install_generator = configure_generator
install_ddg = configure_ddg
