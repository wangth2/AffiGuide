import functools
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffab.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from diffab.modules.encoders.ga import GAEncoder
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition


def _get_ddg_predictor(ddg_guidance_model):
    """Return ddG guidance functions for the requested predictor."""
    if ddg_guidance_model == 'ddgpred':
        from diffab.tools.ddg.predictor import (
            predict_ddg_batch,
            predict_ddg_position,
            predict_ddg_sequence,
        )
        return predict_ddg_batch, predict_ddg_position, predict_ddg_sequence

    if ddg_guidance_model in ('gearbind', 'gearb'):
        from diffab.tools.GearBind.predictor_gearb import (
            predict_ddg_position,
            predict_ddg_sequence,
        )
        return None, predict_ddg_position, predict_ddg_sequence

    raise ValueError(f"Unknown ddG guidance model: {ddg_guidance_model!r}")


def _select_reference_state(reference_state, name):
    """Select a named reference state from a nested reference_state dict."""
    if reference_state is None:
        return None
    if isinstance(reference_state, dict) and name in reference_state:
        return reference_state[name]
    return reference_state


def _to_device_ref(ref_state, device):
    """Move a reference state dict to the target device and detach it."""
    required = ('s_ref', 'p_ref', 'v_ref')
    missing = [key for key in required if key not in ref_state]
    if missing:
        raise KeyError(f"Reference state is missing required keys: {missing}")
    return (
        ref_state['s_ref'].to(device).detach(),
        ref_state['p_ref'].to(device).detach(),
        ref_state['v_ref'].to(device).detach(),
    )


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3)

    ones = torch.ones([ncol], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')
    loss = loss.reshape(size + [3]).sum(dim=-1)
    return loss


class EpsilonNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}):
        super().__init__()
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3),
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3),
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20), nn.Softmax(dim=-1),
        )

    def forward(self, v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res):
        """
        Args:
            v_t: Orientations, (N, L, 3).
            p_t: Positions in normalized diffusion scale, (N, L, 3).
            s_t: Amino-acid indices, (N, L).
            res_feat: Residue features, (N, L, res_dim).
            pair_feat: Pair features, (N, L, L, pair_dim).
            beta: Variance schedule values, (N,).
            mask_generate: Generated residue mask, (N, L).
            mask_res: Valid residue mask, (N, L).
        Returns:
            v_next: Predicted next orientations, (N, L, 3).
            R_next: Predicted next rotation matrices, (N, L, 3, 3).
            eps_pos: Predicted position noise, (N, L, 3).
            c_denoised: Predicted denoised sequence probabilities, (N, L, 20).
        """
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t)

        res_feat = self.res_feat_mixer(
            torch.cat([res_feat, self.current_sequence_embedding(s_t)], dim=-1)
        )
        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res)

        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        t_embed = t_embed[:, None, :].expand(N, L, 3)
        in_feat = torch.cat([res_feat, t_embed], dim=-1)

        eps_crd = self.eps_crd_net(in_feat)
        eps_pos = apply_rotation_to_vector(R, eps_crd)
        eps_pos = torch.where(
            mask_generate[:, :, None].expand_as(eps_pos),
            eps_pos,
            torch.zeros_like(eps_pos),
        )

        eps_rot = self.eps_rot_net(in_feat)
        U = quaternion_1ijk_to_rotation_matrix(eps_rot)
        R_next = R @ U
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        c_denoised = self.eps_seq_net(in_feat)
        return v_next, R_next, eps_pos, c_denoised


class FullDPM(nn.Module):

    def __init__(
        self,
        res_feat_dim,
        pair_feat_dim,
        num_steps,
        eps_net_opt={},
        trans_rot_opt={},
        trans_pos_opt={},
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
    ):
        super().__init__()
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)

        trans_seq_opt = dict(trans_seq_opt)
        prior_b = trans_seq_opt.pop('prior_b', 0.8)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt, prior_b=prior_b)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0]))

    def _normalize_position(self, p):
        return (p - self.position_mean) / self.position_scale

    def _unnormalize_position(self, p_norm):
        return p_norm * self.position_scale + self.position_mean

    def forward(
        self,
        v_0,
        p_0,
        s_0,
        res_feat,
        pair_feat,
        mask_generate,
        mask_res,
        denoise_structure,
        denoise_sequence,
        t=None,
    ):
        N = res_feat.shape[0]
        if t is None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        p_0 = self._normalize_position(p_0)

        if denoise_structure:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        beta = self.trans_pos.var_sched.betas[t]
        _, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res
        )

        loss_dict = {}

        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred,
            target=post_true,
            reduction='none',
            log_target=False,
        ).sum(dim=-1)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        return loss_dict

    def sample(
        self,
        batch_ref,
        v,
        p,
        s,
        res_feat,
        pair_feat,
        mask_generate,
        mask_res,
        _outfile=None,
        reference_state=None,
        sample_structure=True,
        sample_sequence=True,
        sample_ddg=True,
        sample_hydro=True,
        hydro_prior_b=0.8,
        ddg_guidance_model='ddgpred',
        ddg_ref='scaled',
        gradient_scale_cord=0.0,
        gradient_scale_categ=0.0,
        pbar=False,
    ):
        """
        Args:
            v: Orientations of contextual residues, (N, L, 3).
            p: Positions of contextual residues in input coordinate scale, (N, L, 3).
            s: Sequence of contextual residues, (N, L).

        ddG reference modes:
            - ddgpred + ddg_ref='scaled':
                Use the fixed scaled input state as the reference.  This is the
                default ddG-predictor guidance mode.
            - ddgpred + ddg_ref='prev':
                Use traj[t] from the previous recorded sampling state as the
                reference at each step.
            - gearbind:
                Always use the native reference state from reference_state.
        """
        N, L = v.shape[:2]
        device = self._dummy.device

        if hydro_prior_b is None:
            hydro_prior_b = self.trans_seq.prior_b

        p = self._normalize_position(p)

        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}

        if pbar:
            pbar_fn = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar_fn = lambda x: x

        use_ddg = sample_ddg and (
            (sample_structure and gradient_scale_cord != 0.0)
            or (sample_sequence and gradient_scale_categ != 0.0)
        )

        predict_ddg_batch_fn = None
        predict_ddg_position_fn = None
        predict_ddg_sequence_fn = None
        aa_ref_fixed = pos_ref_fixed = v_ref_fixed = None

        if use_ddg:
            (
                predict_ddg_batch_fn,
                predict_ddg_position_fn,
                predict_ddg_sequence_fn,
            ) = _get_ddg_predictor(ddg_guidance_model)

            if ddg_guidance_model == 'ddgpred':
                if ddg_ref == 'scaled':
                    aa_ref_fixed = s.to(device).detach()
                    pos_ref_fixed = p.to(device).detach()
                    v_ref_fixed = v.to(device).detach()
                elif ddg_ref == 'prev':
                    pass
                else:
                    raise ValueError(
                        f"Unknown ddg_ref={ddg_ref!r} for ddgpred. Use 'scaled' or 'prev'."
                    )
            elif ddg_guidance_model in ('gearbind', 'gearb'):
                ref_state = _select_reference_state(reference_state, 'native')
                if ref_state is None:
                    raise ValueError(
                        "reference_state with a native entry is required when "
                        "ddg_guidance_model='gearbind'."
                    )
                aa_ref_fixed, pos_ref_fixed, v_ref_fixed = _to_device_ref(ref_state, device)
            else:
                raise ValueError(f"Unknown ddG guidance model: {ddg_guidance_model!r}")

        for t in pbar_fn(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            v_t = v_t.to(device)
            p_t = p_t.to(device)
            s_t = s_t.to(device)

            if use_ddg and ddg_guidance_model == 'ddgpred' and ddg_ref == 'prev':
                aa_ref_step = s_t.detach()
                pos_ref_step = p_t.detach()
                v_ref_step = v_t.detach()
            else:
                aa_ref_step = aa_ref_fixed
                pos_ref_step = pos_ref_fixed
                v_ref_step = v_ref_fixed

            p_t_norm = self._normalize_position(p_t)

            beta = self.trans_pos.var_sched.betas[t].expand([N])
            t_tensor = torch.full([N], fill_value=t, dtype=torch.long, device=device)

            with torch.no_grad():
                v_next, _, eps_p, c_denoised = self.eps_net(
                    v_t, p_t_norm, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
                )
                v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)

                if sample_hydro:
                    post, s_next = self.trans_seq.denoise(
                        s_t, c_denoised, mask_generate, t_tensor, prior_b=hydro_prior_b
                    )
                else:
                    post, s_next = self.trans_seq.denoise(
                        s_t, c_denoised, mask_generate, t_tensor
                    )

                mu_norm, p_next_norm, z = self.trans_pos.denoise(
                    p_t_norm, eps_p, mask_generate, t_tensor
                )
                mu = self._unnormalize_position(mu_norm)
                p_next_unnorm = self._unnormalize_position(p_next_norm)

            post_update = post
            mu_guided = mu.detach()

            if use_ddg and t > 0:
                with torch.enable_grad():
                    if sample_structure and gradient_scale_cord != 0.0:
                        mu_for_pos = mu.detach().clone().requires_grad_(True)

                        _, pos_grad = predict_ddg_position_fn(
                            batch_ref=batch_ref,
                            aa_gen=s_next,
                            pos_gen=mu_for_pos,
                            v_next=v_next,
                            aa_ref=aa_ref_step,
                            pos_ref=pos_ref_step,
                            v_ref=v_ref_step,
                            clamp_pred_min=-8,
                            clamp_pred_max=8,
                        )

                        sigma_t = self.trans_pos.var_sched.sigmas[t].expand(N, 1, 1)
                        gradient_scale_cord_t = gradient_scale_cord * sigma_t

                        mask_broadcast = mask_generate[..., None].expand_as(mu_for_pos)
                        pos_update = -pos_grad * gradient_scale_cord_t
                        mu_guided = torch.where(
                            mask_broadcast,
                            mu_for_pos + pos_update,
                            mu_for_pos,
                        ).detach()

                        with torch.no_grad():
                            p_next_unnorm = torch.where(
                                mask_generate[..., None].expand_as(mu_guided),
                                mu_guided + self.trans_pos.var_sched.sigmas[t].view(-1, 1, 1) * z,
                                p_t,
                            )

                    if sample_sequence and gradient_scale_categ != 0.0:
                        post_soft = post.detach().clone().requires_grad_(True)

                        _, aa_grad = predict_ddg_sequence_fn(
                            batch_ref=batch_ref,
                            aa_prob=post_soft,
                            pos_gen=mu_guided.detach(),
                            v_next=v_next,
                            aa_ref=aa_ref_step,
                            pos_ref=pos_ref_step,
                            v_ref=v_ref_step,
                            clamp_pred_min=-8,
                            clamp_pred_max=8,
                        )

                        alpha_bar = self.trans_seq.var_sched.alpha_bars[t].expand(N, 1, 1)
                        gradient_scale_categ_t = gradient_scale_categ * (1 - alpha_bar)

                        post_update = post_soft - gradient_scale_categ_t * aa_grad
                        post_update = torch.where(
                            mask_generate[..., None].expand_as(post_update),
                            post_update,
                            post_soft.detach(),
                        )
                        post_update = torch.clamp(post_update, min=1e-8)
                        post_update = post_update / post_update.sum(dim=-1, keepdim=True)

                        with torch.no_grad():
                            s_next_sampled = self.trans_seq._sample(post_update)
                            s_next = torch.where(mask_generate, s_next_sampled, s_t)


            else:
                mu = mu.detach()
                p_next_unnorm = p_next_unnorm.detach()

            if (t == 1) and (_outfile is not None):
                with torch.no_grad():
                    post_cpu = post_update.detach().float().cpu()
                    mask_cpu = mask_generate.detach().bool().cpu()
                    with open(_outfile, 'a', encoding='utf-8') as f:
                        for n in range(N):
                            pos_idx = torch.nonzero(mask_cpu[n], as_tuple=False).view(-1).tolist()
                            rec = {
                                'batch_item_index': int(n),
                                't_final': 0,
                                'positions_internal': pos_idx,
                                'probs': post_cpu[n, pos_idx, :].tolist(),
                                'mask_len': len(pos_idx),
                                'mask_idx': pos_idx,
                                'mask_bool': mask_cpu[n].view(-1).tolist(),
                            }
                            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

            if use_ddg and predict_ddg_batch_fn is not None and (t - 1) % 5 == 0:
                with torch.no_grad():
                    pred_ddg_eval, _ = predict_ddg_batch_fn(
                        batch_ref=batch_ref,
                        aa_gen=s_next,
                        pos_gen=mu_guided.detach(),
                        v_next=v_next,
                        aa_ref=None,
                        pos_ref=None,
                        v_ref=None,
                    )
                    print(
                        f"Step {t}, ΔΔG between generated structure and reference: {pred_ddg_eval}\n"
                        f"Step {t}, ΔΔG mean: {pred_ddg_eval.mean()}"
                    )

            if not sample_structure:
                v_next, p_next_unnorm = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t - 1] = (
                v_next.detach(),
                p_next_unnorm.detach(),
                s_next.detach(),
            )
            traj[t] = tuple(x.detach().cpu() for x in traj[t])

            if t % 5 == 0:
                torch.cuda.empty_cache()

        return traj

    @torch.no_grad()
    def optimize(
        self,
        v,
        p,
        s,
        opt_step: int,
        res_feat,
        pair_feat,
        mask_generate,
        mask_res,
        sample_structure=True,
        sample_sequence=True,
        pbar=False,
    ):
        """
        First add noise to the given structure, then denoise it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        if sample_structure:
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x

        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)

            beta = self.trans_pos.var_sched.betas[t].expand([N])
            t_tensor = torch.full([N], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, _, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            _, p_next, _ = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t - 1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])

        return traj
