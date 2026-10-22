import torch
import torch.nn as nn

from diffab.modules.common.geometry import construct_3d_basis
from diffab.modules.common.so3 import rotation_to_so3vec
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.modules.encoders.pair import PairEmbedding
from diffab.modules.diffusion.dpm_full import FullDPM
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from ._base import register_model


resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms,
}



@register_model('diffab')
class DiffusionAntibodyDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)

        self.diffusion = FullDPM(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            **cfg.diffusion,
        )

    def encode(self, batch, remove_structure, remove_sequence):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
            R:          (N, L, 3, 3)
            p:          (N, L, 3)
        """
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA],
            ~batch['generate_flag'],
        )

        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        res_feat = self.residue_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            fragment_type=batch['fragment_type'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        pair_feat = self.pair_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p

    def _build_single_reference_state(self, aa, pos_heavyatom):
        R_ref = construct_3d_basis(
            pos_heavyatom[:, :, BBHeavyAtom.CA],
            pos_heavyatom[:, :, BBHeavyAtom.C],
            pos_heavyatom[:, :, BBHeavyAtom.N],
        )
        return {
            's_ref': aa,
            'p_ref': pos_heavyatom[:, :, BBHeavyAtom.CA],
            'v_ref': rotation_to_so3vec(R_ref),
        }

    def build_reference_state(self, batch):
        current_ref = self._build_single_reference_state(
            batch['aa'],
            batch['pos_heavyatom'],
        )

        if 'aa_native' in batch and 'pos_heavyatom_native' in batch:
            native_ref = self._build_single_reference_state(
                batch['aa_native'],
                batch['pos_heavyatom_native'],
            )
        else:
            native_ref = current_ref

        return {
            'current': current_ref,
            'native': native_ref,
        }

    def forward(self, batch):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure=self.cfg.get('train_structure', True),
            remove_sequence=self.cfg.get('train_sequence', True),
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        loss_dict = self.diffusion(
            v_0,
            p_0,
            s_0,
            res_feat,
            pair_feat,
            mask_generate,
            mask_res,
            denoise_structure=self.cfg.get('train_structure', True),
            denoise_sequence=self.cfg.get('train_sequence', True),
        )
        return loss_dict

    @torch.no_grad()
    def sample(self, batch, outfile=None, sample_opt=None):
        if sample_opt is None:
            sample_opt = {}
        else:
            sample_opt = dict(sample_opt)


        if 'guidance' in sample_opt:
            from diffab.numerics.backend import configure_generator
            from diffab.modules.diffusion.guidance_config import normalize_guidance
            guidance=normalize_guidance(sample_opt['guidance'])
            scorers=sample_opt.pop('scorers')
            native=sample_opt.pop('native_batch', None)
            if native is None:
                raise ValueError('Guided sampling requires the native crop before RemoveNative')
            context=configure_generator(self,guidance['backend'],sample_opt.get('trajectory_seed',0))
            # Backend construction is RNG-neutral. Preprocessing already consumed
            # the frozen trajectory stream; do not reseed before initialization.
            with context.activate():
                encoded=self.encode(batch,remove_structure=True,remove_sequence=True)
                res,pair,R,p=encoded
                scorers.reference=scorers.D.bind_old({'batch':batch,'native_batch':native},self,encoded=encoded)
                if scorers.G is not None:
                    scorers.G.initialize(native,native['aa'],native['pos_heavyatom'][:,:,BBHeavyAtom.CA])
                return self.diffusion.sample(batch,rotation_to_so3vec(R),p,batch['aa'],res,pair,
                    batch['generate_flag'],batch['mask'],outfile,scorers=scorers,
                    numerical_context=context,**sample_opt)

        sample_opt.setdefault('sample_structure', True)
        sample_opt.setdefault('sample_sequence', True)
        sample_opt.setdefault('sample_ddg', True)
        sample_opt.setdefault('sample_hydro', True)
        sample_opt.setdefault('hydro_prior_b', 0.8)
        sample_opt.setdefault('ddg_guidance_model', 'ddgpred')
        sample_opt.setdefault('ddg_ref', 'scaled')
        sample_opt.setdefault('gradient_scale_cord', 0.0)
        sample_opt.setdefault('gradient_scale_categ', 0.0)

        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure=sample_opt.get('sample_structure', True),
            remove_sequence=sample_opt.get('sample_sequence', True),
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        reference_state = self.build_reference_state(batch)

        traj = self.diffusion.sample(
            batch,
            v_0,
            p_0,
            s_0,
            res_feat,
            pair_feat,
            mask_generate,
            mask_res,
            outfile,
            reference_state=reference_state,
            **sample_opt,
        )
        return traj

    @torch.no_grad()
    def optimize(self, batch, opt_step, optimize_opt=None):
        if optimize_opt is None:
            optimize_opt = {
                'sample_structure': True,
                'sample_sequence': True,
            }
        else:
            optimize_opt = dict(optimize_opt)

        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure=optimize_opt.get('sample_structure', True),
            remove_sequence=optimize_opt.get('sample_sequence', True),
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.optimize(
            v_0,
            p_0,
            s_0,
            opt_step,
            res_feat,
            pair_feat,
            mask_generate,
            mask_res,
            **optimize_opt,
        )
        return traj
