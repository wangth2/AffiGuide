"""Shared CLI planning and independent-design workers; no model math lives here."""
import argparse
import ctypes
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]


def split_designs(start, count, workers):
    if any(type(v) is not int for v in (start, count, workers)) or start < 0 or min(count, workers) < 1:
        raise ValueError('start_id must be nonnegative; num_samples and workers must be positive integers')
    workers = min(workers, count)
    size, extra = divmod(count, workers)
    result = []
    for index in range(workers):
        n = size + int(index < extra)
        result.append(dict(worker=index, start_id=start, num_samples=n))
        start += n
    return result


def effective_plan(args):
    import yaml
    from easydict import EasyDict
    from .design_for_pdb import _apply_sampling_overrides, _guided_cli_options, _read_sampling_options
    from diffab.modules.diffusion.guidance_config import schedule_segments
    from diffab.utils.model_assets import required_models
    path = Path(args.config).expanduser()
    if not path.is_file() and not path.is_absolute():
        path = REPO / path
    config = EasyDict(yaml.safe_load(path.read_text()))
    _apply_sampling_overrides(config, args)
    count = config.sampling.num_samples
    if type(count) is not int or count < 1:
        raise ValueError('sampling.num_samples must be a positive integer')
    execution = dict(config.get('execution', {}))
    workers = args.workers if getattr(args, 'workers', None) is not None else execution.get('workers', 1)
    mps = args.mps if getattr(args, 'mps', None) is not None else execution.get('mps', False)
    if type(workers) is not int or workers < 1 or type(mps) is not bool:
        raise ValueError('execution.workers must be a positive integer and execution.mps a boolean')
    if mps and not re.fullmatch(r'cuda(?::[0-9]+)?', str(args.device)):
        raise ValueError('--mps requires --device cuda or cuda:N')
    ranges = split_designs(getattr(args, 'start_id', 0), count, workers)
    result = dict(config_path=str(path.resolve()), config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  configuration=config, profile='updated' if 'guidance' in config else 'legacy',
                  execution=dict(requested_workers=workers, workers=len(ranges), mps=mps,
                                 device=str(args.device)), worker_ranges=ranges)
    if 'guidance' in config:
        guidance = _guided_cli_options(args, config.guidance)
        result.update(guidance=guidance, schedule=schedule_segments(guidance),
                      required_models=list(required_models(dict(guidance=guidance))))
    else:
        if workers != 1 or mps:
            raise ValueError('Multiple workers and managed MPS require an updated-method YAML; legacy entries retain their original execution')
        result['sampling_options'] = _read_sampling_options(config, args)
    return result


def visible_gpu_uuids():
    """Map CUDA-visible ordinals through PCI identity, never assume physical indices."""
    import torch
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError('No CUDA-visible GPU in the current allocation')
    candidates = glob.glob(str(Path(sys.prefix) / 'lib/libcudart.so*'))
    candidates += glob.glob(str(Path(torch.__file__).parent / 'lib/libcudart.so*'))
    buses = None
    for library in list(dict.fromkeys(candidates + ['libcudart.so'])):
        try:
            cuda = ctypes.CDLL(library)
            fn = cuda.cudaDeviceGetPCIBusId
            fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            fn.restype = ctypes.c_int
            current = []
            for ordinal in range(count):
                buffer = ctypes.create_string_buffer(64)
                if fn(buffer, 64, ordinal):
                    raise RuntimeError('CUDA runtime PCI lookup failed')
                current.append(buffer.value.decode())
            buses = current
            break
        except (OSError, RuntimeError):
            continue
    if buses is None:
        raise RuntimeError('Cannot prove actual CUDA device identities; no index fallback')
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,pci.bus_id',
                                      '--format=csv,noheader,nounits'], universal_newlines=True)
    key = lambda value: tuple(int(x, 16) for x in re.split('[:.]', value.strip()))
    lookup = {key(line.split(',')[1]): line.split(',')[0].strip()
              for line in output.splitlines() if line.strip()}
    return [lookup[key(bus)] for bus in buses]


def _prepare_parallel_input(args, kind, plan, directory):
    if kind == 'pdb' and not args.input_manifest and not args.no_renumber:
        # Renumber once, avoiding multiple writers to the same input-adjacent PDB.
        from diffab.tools.renumber import renumber
        source = Path(args.pdb_path).resolve()
        numbered = directory / (source.stem + '_chothia.pdb')
        heavy, light = renumber(str(source), str(numbered))
        args.pdb_path = str(numbered)
        args.no_renumber = True
        args.heavy = args.heavy or (heavy[0] if heavy else None)
        args.light = args.light or (light[0] if light else None)
    elif kind == 'testset':
        # Populate/validate the original dataset cache once before worker readers.
        from . import design_for_testset as entry
        from easydict import EasyDict
        entry._load_dependencies()
        seed = args.seed if args.seed is not None else plan['configuration']['sampling']['seed']
        entry.seed_all(seed)
        dataset = entry.get_dataset(EasyDict(plan['configuration']['dataset']['test']))
        if args.index < 0 or args.index >= len(dataset):
            raise IndexError('Test-set index out of range')
        dataset[args.index]


def _stop_owned_workers(children):
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for child in children:
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            # Only a process group created by this controller is targeted.
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass


def launch(args, kind, plan):
    from .private_mps import PrivateMPS, atomic
    parent = Path(args.output_dir or args.out_root).resolve() / '_execution'
    directory = parent / ('%s-%s' % (time.time_ns(), os.getpid()))
    directory.mkdir(parents=True, exist_ok=False)
    atomic(directory / 'PLAN.json', plan)
    print('Output: ' + str(Path(args.output_dir or args.out_root).resolve()), flush=True)
    _prepare_parallel_input(args, kind, plan, directory)
    service = None
    children, handles = [], []
    failure = None
    previous_term = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Execution interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        environment['PYTHONPATH'] = str(REPO) + os.pathsep + environment.get('PYTHONPATH', '')
        if plan['execution']['mps']:
            ordinal = int(str(args.device).split(':')[1]) if ':' in str(args.device) else 0
            visible = visible_gpu_uuids()
            if ordinal >= len(visible):
                raise ValueError('Selected CUDA ordinal is outside the visible allocation')
            service = PrivateMPS(directory / 'mps', visible[ordinal])
            environment.update(service.start())
            args.device = 'cuda:0'  # The selected physical UUID is now the sole visible GPU.
        for allocation in plan['worker_ranges']:
            values = dict(vars(args), **{k:v for k,v in allocation.items() if k != 'worker'})
            values.update(_execution_child=True, _effective_execution=plan['execution'],
                          _expected_config_sha256=plan['config_sha256'])
            options = directory / ('worker_%d.json' % allocation['worker'])
            atomic(options, values)
            handle = (directory / ('worker_%d.log' % allocation['worker'])).open('w')
            handles.append(handle)
            command = [sys.executable, '-B', '-m', 'diffab.tools.runner.execution',
                       '--worker-kind', kind, '--options', str(options)]
            children.append(subprocess.Popen(command, env=environment, stdout=handle,
                                             stderr=subprocess.STDOUT, start_new_session=True))
        while True:
            codes = [child.poll() for child in children]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError('Design worker failed; see ' + str(directory))
            if all(code == 0 for code in codes):
                break
            time.sleep(0.2)
    except BaseException as error:
        failure = error
        _stop_owned_workers(children)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        for handle in handles:
            handle.close()
        cleanup = service.close() if service is not None else dict(status='MPS_NOT_STARTED')
        if service is not None and cleanup['status'] not in ('OWN_PRIVATE_QUIT_AND_PIPE_REMOVAL_COMPLETE', 'NO_STARTED_PRIVATE_MPS'):
            failure = failure or RuntimeError('Private MPS cleanup incomplete; see ' + str(directory))
        atomic(directory / 'RESULT.json', dict(status='FAILED' if failure else 'SUCCESS',
               worker_exitcodes=[child.poll() for child in children], mps_cleanup=cleanup,
               error=repr(failure) if failure else None))
    if failure:
        raise failure
    return str(directory)


def dispatch(args, kind):
    if getattr(args, '_execution_child', False):
        return False, None
    plan = effective_plan(args)
    args.config = plan['config_path']
    args._effective_execution = plan['execution']
    if getattr(args, 'print_config', False):
        print(json.dumps(plan, indent=2))
        return True, plan
    if plan['execution']['workers'] > 1 or plan['execution']['mps']:
        return True, launch(args, kind, plan)
    return False, None


def main():
    parser = argparse.ArgumentParser(description='Internal worker for the two public design entries.')
    parser.add_argument('--worker-kind', choices=['pdb', 'testset'], required=True)
    parser.add_argument('--options', type=Path, required=True)
    options = parser.parse_args()
    values = json.loads(options.options.read_text())
    observed = hashlib.sha256(Path(values['config']).read_bytes()).hexdigest()
    if observed != values['_expected_config_sha256']:
        raise RuntimeError('Configuration changed after the execution plan was created')
    if options.worker_kind == 'pdb':
        from .design_for_pdb import args_factory, design_for_pdb
        design_for_pdb(args_factory(**values))
    else:
        from .design_for_testset import args_factory, design_for_testset
        design_for_testset(args_factory(**values))


if __name__ == '__main__':
    main()
