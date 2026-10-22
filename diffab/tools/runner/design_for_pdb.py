import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def _load_dependencies():
    # Argument help stays usable before optional GPU/scorer packages are installed.
    global np, torch, DataLoader, tqdm, preprocess_antibody_structure, get_model
    global reconstruct_backbone_partially, so3vec_to_rotation, renumber_antibody
    global RemoveNative, save_pdb, recursive_to, EasyDict, load_config, seed_all
    global get_new_log_dir, get_logger, PaddingCollate, apply_patch_to_tensor
    global Compose, MaskSingleCDR, MaskMultipleCDRs, MaskAntibody, MergeChains, PatchAroundAnchor
    global find_cdrs, get_residue_first_last
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from easydict import EasyDict
    from diffab.datasets.custom import preprocess_antibody_structure
    from diffab.models import get_model
    from diffab.modules.common.geometry import reconstruct_backbone_partially
    from diffab.modules.common.so3 import so3vec_to_rotation
    from diffab.tools.renumber import renumber as renumber_antibody
    from diffab.utils.inference import RemoveNative, find_cdrs, get_residue_first_last
    from diffab.utils.protein.writers import save_pdb
    from diffab.utils.train import recursive_to
    from diffab.utils.misc import load_config, seed_all, get_new_log_dir, get_logger
    from diffab.utils.data import PaddingCollate, apply_patch_to_tensor
    from diffab.utils.transforms import (Compose, MaskSingleCDR, MaskMultipleCDRs,
                                         MaskAntibody, MergeChains, PatchAroundAnchor)


def serialize_data(obj):
    if isinstance(obj, dict):
        return {k: serialize_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_data(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(serialize_data(v) for v in obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def create_data_variants(config, structure_factory):
    structure = structure_factory()
    structure_id = structure['id']

    data_variants = []
    if config.mode == 'single_cdr':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            data_variants.append({
                'data': data_var,
                'name': f'{structure_id}-{cdr_name}',
                'tag': f'{cdr_name}',
                'cdr': cdr_name,
                'residue_first': residue_first,
                'residue_last': residue_last,
            })

    elif config.mode == 'multiple_cdrs':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        residue_first_list = []
        residue_last_list = []
        for cdr_name in cdrs:
            transform_cdr = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_cdr = transform_cdr(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_cdr)
            residue_first_list.append(residue_first)
            residue_last_list.append(residue_last)

        transform = Compose([
            MaskMultipleCDRs(selection=cdrs, augmentation=False),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-MultipleCDRs',
            'tag': 'MultipleCDRs',
            'cdrs': cdrs,
            'residue_first_list': residue_first_list,
            'residue_last_list': residue_last_list,
        })

    elif config.mode == 'full':
        transform = Compose([
            MaskAntibody(),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-Full',
            'tag': 'Full',
            'residue_first': None,
            'residue_last': None,
        })

    elif config.mode == 'abopt':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            for opt_step in config.sampling.optimize_steps:
                data_variants.append({
                    'data': data_var,
                    'name': f'{structure_id}-{cdr_name}-O{opt_step}',
                    'tag': f'{cdr_name}-O{opt_step}',
                    'cdr': cdr_name,
                    'opt_step': opt_step,
                    'residue_first': residue_first,
                    'residue_last': residue_last,
                })
    else:
        raise ValueError(f'Unknown mode: {config.mode}.')

    return data_variants



def _read_sampling_options(config, args):
    sampling = config.sampling
    sample_ddg = sampling.get('sample_ddg', True)
    sample_hydro = sampling.get('sample_hydro', True)
    hydro_prior_b = sampling.get('hydro_prior_b', 0.8)
    ddg_guidance_model = sampling.get('ddg_guidance_model', 'ddgpred')
    ddg_ref = sampling.get('ddg_ref', 'scaled')
    gradient_scale_cord = sampling.get('gradient_scale_cord', 0.0)
    gradient_scale_categ = sampling.get('gradient_scale_categ', 0.0)
    task = sampling.get('task', 'redesign')

    if args.sample_ddg is not None:
        sample_ddg = args.sample_ddg
    if args.sample_hydro is not None:
        sample_hydro = args.sample_hydro
    if args.hydro_prior_b is not None:
        hydro_prior_b = args.hydro_prior_b
    if args.ddg_guidance_model is not None:
        ddg_guidance_model = args.ddg_guidance_model
    if args.ddg_ref is not None:
        ddg_ref = args.ddg_ref
    if args.gradient_scale_cord is not None:
        gradient_scale_cord = args.gradient_scale_cord
    if args.gradient_scale_categ is not None:
        gradient_scale_categ = args.gradient_scale_categ
    if args.task is not None:
        task = args.task

    if task not in ('redesign', 'external'):
        raise ValueError(f"Unknown task: {task!r}. Use 'redesign' or 'external'.")

    if ddg_guidance_model not in ('ddgpred', 'gearbind', 'gearb'):
        raise ValueError(
            f"Unknown ddg_guidance_model: {ddg_guidance_model!r}. "
            "Use 'ddgpred', 'gearbind', or 'gearb'."
        )

    if ddg_guidance_model == 'ddgpred':
        if ddg_ref not in ('scaled', 'prev'):
            raise ValueError("ddg_ref must be 'scaled' or 'prev' when using ddgpred.")
    else:
        ddg_ref = 'native'

    if hydro_prior_b is not None and not (0.0 <= float(hydro_prior_b) <= 1.0):
        raise ValueError(f'hydro_prior_b must be in [0, 1], got {hydro_prior_b}.')

    return {
        'sample_ddg': bool(sample_ddg),
        'sample_hydro': bool(sample_hydro),
        'hydro_prior_b': None if hydro_prior_b is None else float(hydro_prior_b),
        'ddg_guidance_model': ddg_guidance_model,
        'ddg_ref': ddg_ref,
        'gradient_scale_cord': float(gradient_scale_cord),
        'gradient_scale_categ': float(gradient_scale_categ),
        'task': task,
    }


def _prepare_legacy_model_assets(config, args, sampling_opt):
    """Resolve and verify the exact files used later by the legacy loaders."""
    from diffab.utils.model_assets import ensure_model
    options = dict(offline=getattr(args, 'offline', False),
                   source_dir=getattr(args, 'model_source_dir', None))
    generator = ensure_model('DiffAb', _resolve_path(config.model.checkpoint, args.config), **options)
    if sampling_opt['sample_ddg']:
        if sampling_opt['ddg_guidance_model'] == 'ddgpred':
            explicit = (getattr(args, 'ddg_checkpoint', None)
                        or config.sampling.get('ddg_checkpoint')
                        or os.environ.get('AFFIGUIDE_D_CHECKPOINT'))
            checkpoint = ensure_model('D', _resolve_path(explicit, args.config) if explicit else None, **options)
            from diffab.tools.ddg.predictor import configure
            configure(checkpoint=str(checkpoint), device=args.device)
        else:
            # The optional legacy adapter independently verifies each actual
            # checkpoint at its load point; it may not substitute P/non-P files.
            ensure_model('G', **options)
    return generator


def design_for_pdb(args):
    from diffab.tools.runner.execution import dispatch
    handled, result = dispatch(args, 'pdb')
    if handled:
        return result
    _load_dependencies()
    # Load configs
    config, config_name = load_config(args.config)
    _apply_sampling_overrides(config, args)
    if 'guidance' in config:
        return _design_guided(args, config)
    seed_all(args.seed if args.seed is not None else config.sampling.seed)
    sampling_opt = _read_sampling_options(config, args)
    generator_checkpoint_path = _prepare_legacy_model_assets(config, args, sampling_opt)

    # Structure loading
    data_id = os.path.basename(args.pdb_path)
    if args.no_renumber:
        pdb_path = args.pdb_path
    else:
        in_pdb_path = args.pdb_path
        out_pdb_path = os.path.splitext(in_pdb_path)[0] + '_chothia.pdb'
        heavy_chains, light_chains = renumber_antibody(in_pdb_path, out_pdb_path, verbose=False)
        pdb_path = out_pdb_path

        if args.heavy is None and len(heavy_chains) > 0:
            args.heavy = heavy_chains[0]
        if args.light is None and len(light_chains) > 0:
            args.light = light_chains[0]

    if args.heavy is None and args.light is None:
        raise ValueError('Neither heavy chain id (--heavy) nor light chain id (--light) is specified.')

    get_structure = lambda: preprocess_antibody_structure({
        'id': data_id,
        'pdb_path': pdb_path,
        'heavy_id': args.heavy,
        # If the input is a nanobody, the light chain will be ignores
        'light_id': args.light,
    })

    # Logging
    structure_ = get_structure()
    structure_id = structure_['id']
    tag_postfix = '_%s' % args.tag if args.tag else ''
    log_dir = get_new_log_dir(
        os.path.join(args.out_root, config_name + tag_postfix),
        prefix=data_id,
    )
    logger = get_logger('sample', log_dir, stream_level='WARNING')
    logger.info(f'Data ID: {structure_["id"]}')
    logger.info(f'Results will be saved to {log_dir}')
    logger.info(
        'Guidance mode: '
        f"task={sampling_opt['task']}, "
        f"sample_ddg={sampling_opt['sample_ddg']}, "
        f"sample_hydro={sampling_opt['sample_hydro']}, "
        f"hydro_prior_b={sampling_opt['hydro_prior_b']}, "
        f"ddg_guidance_model={sampling_opt['ddg_guidance_model']}, "
        f"ddg_ref={sampling_opt['ddg_ref']}, "
        f"gradient_scale_cord={sampling_opt['gradient_scale_cord']}, "
        f"gradient_scale_categ={sampling_opt['gradient_scale_categ']}"
    )

    data_native = MergeChains()(structure_)
    save_pdb(data_native, os.path.join(log_dir, 'reference.pdb'))

    # Load checkpoint and model
    logger.info('Loading model config and checkpoints: %s' % generator_checkpoint_path)
    ckpt = torch.load(str(generator_checkpoint_path), map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    logger.info(str(lsd))

    # Make data variants
    data_variants = create_data_variants(
        config=config,
        structure_factory=get_structure,
    )

    # Save metadata
    metadata = {
        'identifier': structure_id,
        'index': data_id,
        'config': args.config,
        'items': [{kk: vv for kk, vv in var.items() if kk != 'data'} for var in data_variants],
        'sampling_options': sampling_opt,
    }
    with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Start sampling
    collate_fn = PaddingCollate(eight=False)
    inference_tfm = [PatchAroundAnchor()]
    if 'abopt' not in config.mode:
        inference_tfm.append(RemoveNative(
            remove_structure=config.sampling.sample_structure,
            remove_sequence=config.sampling.sample_sequence,
        ))
    inference_tfm = Compose(inference_tfm)

    for variant in data_variants:
        variant_dir = os.path.join(log_dir, variant['tag'])
        os.makedirs(variant_dir, exist_ok=True)
        logger.info(f"Start sampling for: {variant['tag']}")
        save_pdb(data_native, os.path.join(variant_dir, 'REF1.pdb'))

        data_cropped = inference_tfm(copy.deepcopy(variant['data']))
        data_list_repeat = [data_cropped] * config.sampling.num_samples
        loader = DataLoader(
            data_list_repeat,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        data_cropped_save_path = os.path.join(variant_dir, 'data_cropped.json')
        with open(data_cropped_save_path, 'w') as f:
            json.dump(serialize_data(data_cropped), f, indent=4)

        count = 0
        for batch in tqdm(loader, desc=variant['name'], dynamic_ncols=True):
            torch.set_grad_enabled(False)
            model.eval()
            batch = recursive_to(batch, args.device)

            if 'abopt' in config.mode:
                # Antibody optimization starting from native
                traj_batch = model.optimize(batch, opt_step=variant['opt_step'], optimize_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                })
            else:
                # De novo design
                # If compare with external dataset, than save post probability in outfile
                outfile = os.path.join(log_dir, 'post_probs.jsonl') if sampling_opt['task'] == 'external' else None
                sample_opt = {
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                    'sample_ddg': sampling_opt['sample_ddg'],
                    'sample_hydro': sampling_opt['sample_hydro'],
                    'hydro_prior_b': sampling_opt['hydro_prior_b'],
                    'ddg_guidance_model': sampling_opt['ddg_guidance_model'],
                    'ddg_ref': sampling_opt['ddg_ref'],
                    'gradient_scale_cord': sampling_opt['gradient_scale_cord'],
                    'gradient_scale_categ': sampling_opt['gradient_scale_categ'],
                }
                traj_batch = model.sample(batch, outfile=outfile, sample_opt=sample_opt)

            aa_new = traj_batch[0][2]
            pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx=batch['pos_heavyatom'],
                R_new=so3vec_to_rotation(traj_batch[0][0]),
                t_new=traj_batch[0][1],
                aa=aa_new,
                chain_nb=batch['chain_nb'],
                res_nb=batch['res_nb'],
                mask_atoms=batch['mask_heavyatom'],
                mask_recons=batch['generate_flag'],
            )
            aa_new = aa_new.cpu()
            pos_atom_new = pos_atom_new.cpu()
            mask_atom_new = mask_atom_new.cpu()

            for i in range(aa_new.size(0)):
                data_tmpl = variant['data']
                aa = apply_patch_to_tensor(data_tmpl['aa'], aa_new[i], data_cropped['patch_idx'])
                mask_ha = apply_patch_to_tensor(
                    data_tmpl['mask_heavyatom'],
                    mask_atom_new[i],
                    data_cropped['patch_idx'],
                )
                pos_ha = apply_patch_to_tensor(
                    data_tmpl['pos_heavyatom'],
                    pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(),
                    data_cropped['patch_idx'],
                )

                save_path = os.path.join(variant_dir, '%04d.pdb' % count)
                save_pdb({
                    'chain_nb': data_tmpl['chain_nb'],
                    'chain_id': data_tmpl['chain_id'],
                    'resseq': data_tmpl['resseq'],
                    'icode': data_tmpl['icode'],
                    # Generated
                    'aa': aa,
                    'mask_heavyatom': mask_ha,
                    'pos_heavyatom': pos_ha,
                }, path=save_path)
                count += 1

        logger.info('Finished.\n')

    print('Output: ' + str(Path(log_dir).resolve()), flush=True)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def _resolve_path(value, config_path):
    """Retain repository-relative YAML paths; allow config-local external assets."""
    expanded = os.path.expandvars(str(value))
    if '$' in expanded:
        raise ValueError('Unconfigured environment variable in path: ' + str(value))
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return path
    repository = Path(__file__).resolve().parents[3]
    candidates = [Path.cwd() / path, repository / path, Path(config_path).resolve().parent / path]
    return next((p.resolve() for p in candidates if p.exists()), (repository / path).resolve())



def _design_lock(job):
    """Local/shared-filesystem advisory lock for one immutable logical task."""
    import contextlib
    import fcntl
    @contextlib.contextmanager
    def held():
        job.mkdir(parents=True, exist_ok=True)
        with (job / 'DESIGN.lock').open('a') as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Logical design is already claimed: ' + str(job)) from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return held()


def _record_design_failure(job):
    import contextlib
    import time
    import traceback
    @contextlib.contextmanager
    def checked():
        try:
            yield
        except BaseException as exc:
            _atomic_json(job / 'FAILED.json', dict(status='FAILED', error=repr(exc),
                traceback=traceback.format_exc(), unix_time=time.time(),
                checkpoint_preserved=(job / 'trajectory_checkpoint.pt').exists()))
            raise
    return checked()


def _trajectory_seeds(case_id, cdr, design_id, base_seed):
    # Frozen Table 1 derivation deliberately excludes method, GPU and worker rank.
    def seed_for(*keys):
        return int.from_bytes(hashlib.sha256(':'.join(map(str, keys)).encode()).digest()[:4], 'big') % (2**31 - 1)
    parent = seed_for('benchmark19_' + cdr, base_seed, case_id)
    return parent, seed_for(parent, 'trajectory', design_id)


def _make_scorers(config, guidance, args):
    from diffab.tools.ddg.scorer import DdgScorer, Scorers
    from diffab.numerics.backend import configure_ddg
    from diffab.modules.diffusion.guidance_config import needs_gearbind
    settings = config.get('scorers', {})
    dcfg = settings.get('D', {})
    checkpoint = args.ddg_checkpoint or dcfg.get('checkpoint') or config.sampling.get('ddg_checkpoint')
    if checkpoint is None:
        checkpoint = 'diffab/tools/ddg/data/model.pt'
    need_d = True  # The updated sampler always records a raw D hard score.
    D = DdgScorer(_resolve_path(checkpoint, args.config), args.device,
                  dcfg.get('sha256')) if need_d else None
    if D is not None:
        configure_ddg(D.model, guidance['backend'])
    G = None
    if needs_gearbind(guidance):
        from diffab.tools.GearBind.client import GearBindClient
        gcfg = settings.get('G', {})
        if not gcfg.get('checkpoint') or not gcfg.get('python'):
            raise ValueError('D/G configuration requires scorers.G.python and scorers.G.checkpoint')
        kwargs = {k: str(_resolve_path(gcfg[k], args.config)) for k in
                  ('extension_path', 'config_path', 'log_path') if gcfg.get(k)}
        g_sha = _sha(_resolve_path(gcfg['checkpoint'], args.config))
        if gcfg.get('sha256') and g_sha != gcfg['sha256']:
            raise ValueError('GearBind checkpoint SHA256 mismatch')
        G = GearBindClient(str(_resolve_path(gcfg['python'], args.config)),
                           str(_resolve_path(gcfg['checkpoint'], args.config)),
                           device=args.device, backend=guidance['backend'], **kwargs)
        G.checkpoint_sha256 = g_sha
    return Scorers(D, G)


def _apply_sampling_overrides(config, args):
    """Options shared by the original PDB and test-set entry points."""
    if 'guidance' not in config:
        for name in ('candidate_count', 'gradient_source', 'selector', 'guidance_mode',
                     'coord_guidance', 'rot_guidance', 'seq_guidance', 'rotation_scale',
                     'd_window', 'dg_window', 'coord_window', 'rotation_window',
                     'seq_window', 'candidate_window'):
            if getattr(args, name, None) is not None:
                raise ValueError('--' + name.replace('_', '-') +
                    ' requires a guidance configuration; use '
                    '--config configs/test/codesign_single_dg.yml')
    if getattr(args, 'num_samples', None) is not None:
        if args.num_samples < 1:
            raise ValueError('num_samples must be positive')
        config.sampling.num_samples = args.num_samples
    if getattr(args, 'cdr', None) is not None:
        names = {x + str(i): x + '_CDR' + str(i) for x in ('H', 'L') for i in (1, 2, 3)}
        config.sampling.cdrs = [names.get(x, x) for x in args.cdr]


def _guided_cli_options(args, value):
    """Apply explicit CLI overrides without mixing legacy reference protocols."""
    for name in ('ddg_ref', 'ddg_guidance_model'):
        if getattr(args, name, None) is not None:
            raise ValueError('--' + name + ' is only supported by legacy YAML configurations; '
                             'select guidance.gradient_source and the frozen method reference in the modern configuration')
    from diffab.modules.diffusion.guidance_config import normalize_guidance
    guidance = copy.deepcopy(dict(value))
    mode = getattr(args, 'guidance_mode', None)
    source = getattr(args, 'gradient_source', None)
    if mode is not None:
        selected = {'dg': 'DG', 'd-only': 'D', 'selection-only': 'none'}[mode]
        if source is not None and source != selected:
            raise ValueError('--guidance and --gradient-source disagree')
        source = selected
    if source is not None:
        guidance['gradient_source'] = source
    for name in ('candidate_count', 'gradient_source', 'selector'):
        override = getattr(args, name, None)
        if override is not None:
            guidance[name] = override
    for name in ('d_window', 'dg_window', 'candidate_window'):
        override = getattr(args, name, None)
        if override is not None:
            guidance[name] = override
    channels = list(guidance.get('channels', ['p', 'omega', 'q']))
    for name, block in (('coord_guidance', 'p'), ('rot_guidance', 'omega'), ('seq_guidance', 'q')):
        toggle = getattr(args, name, None)
        if toggle == 'off' and block in channels:
            channels.remove(block)
        elif toggle == 'on' and block not in channels:
            channels.append(block)
    guidance['channels'] = channels
    windows = dict(guidance.get('channel_windows', {}))
    for name, block in (('coord_window', 'p'), ('rotation_window', 'omega'), ('seq_window', 'q')):
        override = getattr(args, name, None)
        if override is not None:
            windows[block] = override
    guidance['channel_windows'] = windows
    if getattr(args, 'rotation_scale', None) is not None:
        guidance['rotation_scale'] = args.rotation_scale
    if args.backend is not None:
        guidance['backend'] = args.backend
    if args.gradient_scale_cord is not None:
        guidance['coordinate_scale'] = args.gradient_scale_cord
    if args.gradient_scale_categ is not None:
        guidance['sequence_scale'] = args.gradient_scale_categ
    if args.sample_hydro is False:
        guidance['hydro_prior_b'] = None
    elif args.hydro_prior_b is not None:
        guidance['hydro_prior_b'] = args.hydro_prior_b
    elif args.sample_hydro is True and guidance.get('hydro_prior_b', 0.8) is None:
        guidance['hydro_prior_b'] = 0.8
    if args.sample_ddg is True and guidance.get('gradient_source', 'DG') == 'none':
        raise ValueError('--sample_ddg cannot choose a gradient source for selection-only; '
                         'select a configuration with guidance.gradient_source D or DG')
    if args.sample_ddg is False:
        if source is not None and source != 'none':
            raise ValueError('--no_sample_ddg conflicts with the requested guidance source')
        guidance['gradient_source'] = 'none'
        guidance['channels'] = []
    return normalize_guidance(guidance)


def _prepared_structure_identity(structure):
    """Hash dataset values without serialization-specific tensor storage IDs."""
    encoded = json.dumps(serialize_data(structure), sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(b'AFFIGUIDE_DATASET_INPUT_V1\n' + encoded).hexdigest()


def _design_guided(args, config, prepared_structure=None):
    """Original PDB/CDR entry with one independently seeded logical design at a time."""
    from diffab.modules.diffusion.guidance_config import load_calibration
    guidance = _guided_cli_options(args, config.guidance)
    from diffab.modules.diffusion.guidance_config import schedule_segments
    # Store the portable effective configuration before absolute path resolution.
    effective = dict(config=serialize_data(config), guidance=copy.deepcopy(guidance),
                     schedule=schedule_segments(guidance),
                     execution=getattr(args, '_effective_execution', dict(workers=1, mps=False)))
    for field in ('calibration', 'hard_calibration', 'numerical_policy'):
        if guidance.get(field):
            guidance[field] = str(_resolve_path(guidance[field], args.config))
    if guidance.get('calibration_by_cdr'):
        guidance['calibration_by_cdr'] = {k: str(_resolve_path(v, args.config)) if v else None
                                           for k, v in guidance['calibration_by_cdr'].items()}
    base_seed = int(args.seed if args.seed is not None else config.sampling.seed)
    names = {x + str(i): x + '_CDR' + str(i) for x in ('H', 'L') for i in (1, 2, 3)}
    if args.cdr is not None:
        config.sampling.cdrs = [names.get(x, x) for x in args.cdr]
    count = int(args.num_samples if args.num_samples is not None else config.sampling.num_samples)
    if count < 1 or args.start_id < 0:
        raise ValueError('num_samples must be positive and start_id nonnegative')
    if config.mode != 'single_cdr':
        raise ValueError('Registered shared-proposal protocols use single_cdr; legacy YAML retains original multicdr/abopt entry')
    input_record = None
    if prepared_structure is not None:
        structure = copy.deepcopy(prepared_structure)
        case_id = args.case_id or str(structure['id'])
        input_sha = _prepared_structure_identity(structure)
    elif args.input_manifest:
        import pickle
        manifest = Path(args.input_manifest).resolve()
        input_record = json.loads(manifest.read_text())
        source = Path(input_record['source_pickle'])
        source = source if source.is_absolute() else manifest.parent / source
        if _sha(source) != input_record['source_sha256']:
            raise ValueError('Frozen source pickle SHA256 mismatch')
        structure = pickle.loads(source.read_bytes())
        case_id = args.case_id or input_record['case_id']
        input_sha = _sha(source)
    else:
        pdb = Path(args.pdb_path).resolve()
        if not args.no_renumber:
            numbered = pdb.with_name(pdb.stem + '_chothia.pdb')
            heavy, light = renumber_antibody(str(pdb), str(numbered), verbose=False)
            pdb = numbered
            args.heavy = args.heavy or (heavy[0] if heavy else None)
            args.light = args.light or (light[0] if light else None)
        if args.heavy is None and args.light is None:
            raise ValueError('Specify --heavy and/or --light chain identity')
        case_id = args.case_id or pdb.stem
        structure = preprocess_antibody_structure(dict(id=case_id, pdb_path=str(pdb),
                                                      heavy_id=args.heavy, light_id=args.light))
        input_sha = _sha(pdb)
    if structure is None:
        raise ValueError('Antibody structure preprocessing failed')
    if input_record and input_record.get('cdr_name') and args.cdr is None:
        config.sampling.cdrs = [input_record['cdr_name']]
    variants = create_data_variants(config, lambda: copy.deepcopy(structure))
    if not variants:
        raise ValueError('No requested CDR is present in the input')
    for variant in variants:
        load_calibration(guidance, cdr=variant['cdr'].replace('_CDR', ''))
    # Asset acquisition belongs to the entry point. Every file is verified
    # before deserialization; unavailable frozen resources never change method.
    from diffab.utils.model_assets import ensure_config_models
    assets = copy.deepcopy(config)
    assets.guidance = guidance
    assets.model.checkpoint = str(_resolve_path(config.model.checkpoint, args.config))
    for name in ('D', 'G'):
        settings = assets.get('scorers', {}).get(name, {})
        checkpoint = args.ddg_checkpoint if name == 'D' and args.ddg_checkpoint else settings.get('checkpoint')
        if checkpoint:
            settings['checkpoint'] = str(_resolve_path(checkpoint, args.config))
    ensure_config_models(assets, offline=getattr(args, 'offline', False),
                         source_dir=getattr(args, 'model_source_dir', None))
    destination = Path(args.output_dir or (Path(args.out_root) / Path(args.config).stem / case_id))
    destination.mkdir(parents=True, exist_ok=True)
    logger = get_logger('sample', str(destination), stream_level='WARNING')
    effective['sampling'] = dict(cdrs=list(config.sampling.cdrs), seed=base_seed,
                                 start_id=args.start_id, num_samples=count)
    native = MergeChains()(copy.deepcopy(structure))
    checkpoint_path = _resolve_path(config.model.checkpoint, args.config)
    checkpoint_sha = _sha(checkpoint_path)
    if config.model.get('sha256') and checkpoint_sha != config.model.sha256:
        raise ValueError('DiffAb checkpoint SHA256 mismatch')
    saved = torch.load(str(checkpoint_path), map_location='cpu')
    model = get_model(saved['config'].model).to(args.device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False)
    scorers = _make_scorers(config, guidance, args)
    from diffab.utils.release_identity import source_manifest, numerical_environment
    if guidance['backend'] == 'paper_v4':
        from diffab.numerics.backend import apply_execution_flags
        apply_execution_flags('generator')
    sources = source_manifest()
    _atomic_json(destination / 'SOURCE_MANIFEST.json', sources)
    runtime_identity = dict(source_signature=sources['sha256'],
        generator_checkpoint_sha256=checkpoint_sha,
        D_checkpoint_sha256=scorers.D.checkpoint_sha256 if scorers.D else None,
        G_checkpoint_sha256=getattr(scorers.G, 'checkpoint_sha256', None),
        numerical_environment=numerical_environment())
    try:
        for variant in variants:
            cdr = variant['cdr'].replace('_CDR', '')
            for design_id in range(args.start_id, args.start_id + count):
                if getattr(args, 'before_design', None) is not None:
                    if args.before_design(case_id, cdr, design_id) is False:
                        return str(destination)
                parent_seed, trajectory_seed = _trajectory_seeds(case_id, cdr, design_id, base_seed)
                guidance = dict(guidance, cdr=cdr)
                # Final per-task reset follows all model construction. Metadata
                # checks below consume no scientific random draws.
                seed_all(trajectory_seed)
                runtime_identity = dict(runtime_identity, numerical_environment=numerical_environment())
                identity = dict(case_id=case_id, cdr=cdr, design_id=design_id, seed=base_seed,
                                parent_seed=parent_seed, trajectory_seed=trajectory_seed,
                                input_sha256=input_sha, model_sha256=checkpoint_sha, guidance=guidance,
                                runtime_identity=runtime_identity)
                fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                job = destination / cdr / ('%04d' % design_id)
                with _design_lock(job):
                    success = job / 'SUCCESS.json'
                    if success.is_file():
                        old = json.loads(success.read_text())
                        if old.get('input_fingerprint') != fingerprint or _sha(job / 'raw_MT.pdb') != old['structure_sha256']:
                            raise ValueError('Existing SUCCESS conflicts with requested design identity: ' + str(job))
                        if args.resume:
                            continue
                        raise FileExistsError('Design exists; use --resume: ' + str(job))
                    job.mkdir(parents=True, exist_ok=True)
                    with _record_design_failure(job):
                        _atomic_json(job / 'INPUT.json', identity)
                        _atomic_json(job / 'EFFECTIVE_CONFIG.json', effective)
                        # RemoveNative consumes this task's stream; its randomized
                        # geometry also defines the frozen historical D reference.
                        cropped = PatchAroundAnchor()(copy.deepcopy(variant['data']))
                        native_crop = copy.deepcopy(cropped)
                        contract = None
                        if input_record is not None:
                            from diffab.tools.GearBind.identity import crop_contract
                            contract = crop_contract(input_record, cropped)
                        removed = RemoveNative(remove_structure=config.sampling.sample_structure,
                                               remove_sequence=config.sampling.sample_sequence)(cropped)
                        batch = recursive_to(PaddingCollate(eight=False)([removed]), args.device)
                        if contract is not None:
                            batch['design_cdr_contract'] = contract
                        native_batch = copy.deepcopy(batch)
                        native_batch['aa'] = recursive_to(native_crop['aa'].unsqueeze(0), args.device)
                        native_batch['pos_heavyatom'] = recursive_to(native_crop['pos_heavyatom'].unsqueeze(0), args.device)
                        sample_opt = dict(guidance=guidance, scorers=scorers, native_batch=native_batch,
                                          case_id=case_id, design_id=design_id, seed=base_seed,
                                          trajectory_seed=trajectory_seed, generation_seed=base_seed, runtime_identity=runtime_identity,
                                          checkpoint=str(job / 'trajectory_checkpoint.pt'),
                                          sample_structure=config.sampling.sample_structure,
                                          sample_sequence=config.sampling.sample_sequence, pbar=True)
                        outfile = str(job / 'post_probs.jsonl') if (args.task or config.sampling.get('task')) == 'external' else None
                        traj = model.sample(batch, outfile=outfile, sample_opt=sample_opt)
                        v, p, s = traj[0]
                        xyz, masks = reconstruct_backbone_partially(
                            pos_ctx=batch['pos_heavyatom'], R_new=so3vec_to_rotation(v), t_new=p, aa=s,
                            chain_nb=batch['chain_nb'], res_nb=batch['res_nb'], mask_atoms=batch['mask_heavyatom'],
                            mask_recons=batch['generate_flag'])
                        local = torch.where(batch['generate_flag'][0])[0].cpu()
                        target = removed['patch_idx'][local]
                        result = copy.deepcopy(native)
                        result['aa'][target] = s[0].detach().cpu()[local]
                        result['mask_heavyatom'][target] = masks[0].detach().cpu()[local]
                        result['pos_heavyatom'][target] = xyz[0].detach().cpu()[local] + batch['origin'][0].detach().cpu().view(1, 1, 3)
                        path = job / 'raw_MT.pdb'
                        save_pdb(result, str(path))
                        state = model.diffusion.last_sampling_result
                        if state.get('q_used') is not None:
                            _atomic_json(job / 'posterior.json', dict(probs=state['q_used'][0, local].detach().cpu().tolist(),
                                positions_internal=local.tolist(), aa_order='ACDEFGHIKLMNPQRSTVWY',
                                semantics='literal selected final q_used before categorical epsilon'))
                        _atomic_json(job / 'SUCCESS.json', dict(status='SUCCESS', **identity,
                            input_fingerprint=fingerprint, structure_sha256=_sha(path),
                            actual_sequence=''.join('ACDEFGHIKLMNPQRSTVWY'[int(x)] for x in s[0, local]),
                            trace=serialize_data(state.get('trace', []))))
                    logger.info('SUCCESS %s/%s/%04d', case_id, cdr, design_id)
                    if getattr(args, 'after_design', None) is not None:
                        args.after_design(case_id, cdr, design_id, str(success))
    finally:
        scorers.close()
    print('Output: ' + str(destination.resolve()), flush=True)
    return str(destination)


def argument_parser(input_kind='pdb'):
    parser = argparse.ArgumentParser()
    if input_kind == 'testset':
        parser.add_argument('index', type=int, help='Index in the configured test dataset.')
        parser.set_defaults(pdb_path=None, heavy=None, light=None,
                            no_renumber=True, input_manifest=None)
    else:
        parser.add_argument('pdb_path', type=str, nargs='?', help='Input antibody-antigen PDB (or use --input-manifest).')
        parser.add_argument('--heavy', type=str, default=None, help='Chain id of the heavy chain.')
        parser.add_argument('--light', type=str, default=None, help='Chain id of the light chain.')
        parser.add_argument('--no_renumber', action='store_true', default=False)
        parser.add_argument('--input-manifest', default=None, help='Frozen local input JSON with source_pickle/source_sha256.')
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_single_dg.yml')
    parser.add_argument('-o', '--out_root', type=str, default='./results')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('--task', type=str, default=None, choices=['redesign', 'external'])
    parser.add_argument('--sample_ddg', dest='sample_ddg', action='store_true', default=None)
    parser.add_argument('--no_sample_ddg', dest='sample_ddg', action='store_false', default=None)
    parser.add_argument('--sample_hydro', dest='sample_hydro', action='store_true', default=None)
    parser.add_argument('--no_sample_hydro', dest='sample_hydro', action='store_false', default=None)
    parser.add_argument('--hydro_prior_b', type=float, default=None)
    parser.add_argument('--ddg_guidance_model', type=str, default=None, choices=['ddgpred', 'gearbind', 'gearb'])
    parser.add_argument('--ddg_ref', type=str, default=None, choices=['scaled', 'prev'])
    parser.add_argument('--gradient_scale_cord', '--coord-scale', type=float, default=None)
    parser.add_argument('--gradient_scale_categ', '--seq-scale', type=float, default=None)
    parser.add_argument('--rotation-scale', type=float, default=None)
    parser.add_argument('--candidate-count', '--K', dest='candidate_count', type=int,
                        default=None, help='Override guidance.candidate_count in an updated-method YAML.')
    parser.add_argument('--gradient-source', choices=['D', 'DG', 'none'], default=None,
                        help='Override guidance.gradient_source; the selector is independent.')
    parser.add_argument('--guidance', dest='guidance_mode', choices=['dg', 'd-only', 'selection-only'],
                        default=None, help='Online gradient mode; selector=auto follows this mode.')
    parser.add_argument('--selector', choices=['auto', 'single', 'd_minimum', 'dg_minimax'], default=None,
                        help='Override guidance.selector in an updated-method YAML.')
    for flag in ('coord-guidance', 'rot-guidance', 'seq-guidance'):
        parser.add_argument('--' + flag, choices=['on', 'off'], default=None,
                            help='Enable/disable this energy-gradient channel, not diffusion itself.')
    for flag in ('d-window', 'dg-window', 'coord-window', 'rotation-window', 'seq-window', 'candidate-window'):
        parser.add_argument('--' + flag, nargs=2, type=int, default=None, metavar=('START_T', 'END_T'),
                            help='Inclusive reverse-time interval, 100 >= START_T >= END_T >= 1.')
    parser.add_argument('--cdr', nargs='+', default=None, help='CDR names H1/H2/H3/L1/L2/L3 or legacy H_CDR1 names.')
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--start-id', type=int, default=0)
    parser.add_argument('--case-id', default=None)
    parser.add_argument('--output-dir', default=None, help='Stable output directory, suitable for --resume.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ddg-checkpoint', default=None)
    parser.add_argument('--offline', action='store_true', help='Verify local model files and forbid automatic downloads.')
    parser.add_argument('--model-source-dir', default=None, help='Explicit directory of matching frozen weights for offline installation.')
    parser.add_argument('--backend', choices=['production_atomic', 'deterministic_gpu', 'paper_v4'], default=None)
    parser.add_argument('--workers', type=int, default=None, help='Independent design workers on the selected device; default 1.')
    mps = parser.add_mutually_exclusive_group()
    mps.add_argument('--mps', dest='mps', action='store_true', default=None, help='Use a private MPS service on the selected CUDA GPU.')
    mps.add_argument('--no-mps', dest='mps', action='store_false')
    parser.add_argument('--print-config', action='store_true', help='Print effective settings and time intervals without loading models or starting MPS.')
    return parser


def args_from_cmdline():
    parser = argument_parser()
    args = parser.parse_args()
    if args.pdb_path is None and args.input_manifest is None and not args.print_config:
        parser.error('provide pdb_path or --input-manifest')
    return args


def args_factory(**kwargs):
    from easydict import EasyDict
    default_args = EasyDict(
        heavy='D',
        light='E',
        no_renumber=False,
        config='./configs/test/codesign_single_dg.yml',
        out_root='./results',
        tag='',
        seed=None,
        device='cuda',
        batch_size=16,
        task=None,
        sample_ddg=None,
        sample_hydro=None,
        hydro_prior_b=None,
        ddg_guidance_model=None,
        ddg_ref=None,
        gradient_scale_cord=None,
        gradient_scale_categ=None,
        cdr=None, num_samples=None, start_id=0, case_id=None, input_manifest=None,
        output_dir=None, resume=False, ddg_checkpoint=None, backend=None,
        offline=False, model_source_dir=None,
        candidate_count=None, gradient_source=None, selector=None,
        guidance_mode=None, coord_guidance=None, rot_guidance=None, seq_guidance=None,
        rotation_scale=None, d_window=None, dg_window=None, coord_window=None,
        rotation_window=None, seq_window=None, candidate_window=None,
        workers=None, mps=None, print_config=False,
    )
    default_args.update(kwargs)
    return default_args


if __name__ == '__main__':
    design_for_pdb(args_from_cmdline())
