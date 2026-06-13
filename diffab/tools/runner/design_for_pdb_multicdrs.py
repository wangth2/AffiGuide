import argparse
import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffab.datasets.custom import preprocess_antibody_structure
from diffab.models import get_model
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.tools.renumber import renumber as renumber_antibody
from diffab.utils.inference import RemoveNative
from diffab.utils.protein.writers import save_pdb
from diffab.utils.train import recursive_to

from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.transforms import *
from diffab.utils.inference import *



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
            'flexible_regions': list(zip(residue_first_list, residue_last_list)),
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


def design_for_pdb(args):
    config, config_name = load_config(args.config)
    seed_all(args.seed if args.seed is not None else config.sampling.seed)
    sampling_opt = _read_sampling_options(config, args)

    data_id = os.path.basename(args.pdb_path)
    if args.no_renumber:
        pdb_path = args.pdb_path
    else:
        in_pdb_path = args.pdb_path
        out_pdb_path = os.path.splitext(in_pdb_path)[0] + '_chothia.pdb'
        heavy_chains, light_chains = renumber_antibody(in_pdb_path, out_pdb_path)
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
        'light_id': args.light,
    })

    structure_ = get_structure()
    structure_id = structure_['id']
    tag_postfix = '_%s' % args.tag if args.tag else ''
    log_dir = get_new_log_dir(
        os.path.join(args.out_root, config_name + tag_postfix),
        prefix=data_id,
    )
    logger = get_logger('sample', log_dir)
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

    logger.info('Loading model config and checkpoints: %s' % config.model.checkpoint)
    ckpt = torch.load(config.model.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    logger.info(str(lsd))

    data_variants = create_data_variants(
        config=config,
        structure_factory=get_structure,
    )

    metadata = {
        'identifier': structure_id,
        'index': data_id,
        'config': args.config,
        'items': [{kk: vv for kk, vv in var.items() if kk != 'data'} for var in data_variants],
        'sampling_options': sampling_opt,
    }
    with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

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
                traj_batch = model.optimize(batch, opt_step=variant['opt_step'], optimize_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                })
            else:
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
                    'aa': aa,
                    'mask_heavyatom': mask_ha,
                    'pos_heavyatom': pos_ha,
                }, path=save_path)
                count += 1

        logger.info('Finished.\n')


def args_from_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_path', type=str)
    parser.add_argument('--heavy', type=str, default=None, help='Chain id of the heavy chain.')
    parser.add_argument('--light', type=str, default=None, help='Chain id of the light chain.')
    parser.add_argument('--no_renumber', action='store_true', default=False)
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_multicdrs_guided.yml')
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
    parser.add_argument('--gradient_scale_cord', type=float, default=None)
    parser.add_argument('--gradient_scale_categ', type=float, default=None)
    return parser.parse_args()


def args_factory(**kwargs):
    default_args = EasyDict(
        heavy='H',
        light='L',
        no_renumber=False,
        config='./configs/test/codesign_multicdrs_guided.yml',
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
    )
    default_args.update(kwargs)
    return default_args


if __name__ == '__main__':
    design_for_pdb(args_from_cmdline())
