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


def _get_ddg_predictor(ddg_guidance_model, device=None):
    """Return ddG guidance functions for the requested predictor."""
    if ddg_guidance_model == 'ddgpred':
        from diffab.tools.ddg.predictor import (
            predict_ddg_batch,
            predict_ddg_position,
            predict_ddg_sequence,
        )
        # Retain the legacy predictor-load point. Modern guidance supplies an
        # already-loaded scorer and never reaches this compatibility branch.
        from diffab.tools.ddg.predictor import get_model
        get_model(device)
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

    @torch.no_grad()
    def sample(self, batch_ref, v, p, s, res_feat, pair_feat, mask_generate, mask_res,
               _outfile=None, guidance=None, scorers=None, case_id='', design_id=0,
               seed=2022, trajectory_seed=None, generation_seed=2022, checkpoint=None, capture=None,
               numerical_context=None, cdr=None, stop_after=None, **legacy_options):
        """Reverse diffusion with visible shared-proposal guidance and selection.

        See Methods: calibrated consensus, TV-matched mirror descent, and
        multi-candidate selection. Old YAMLs without `guidance` retain their
        original single-candidate entry and checkpoint parameter names.
        """
        if guidance is None:
            return self._sample_legacy(batch_ref, v, p, s, res_feat, pair_feat,
                                       mask_generate, mask_res, _outfile, **legacy_options)
        from .guidance_config import normalize_guidance, load_calibration, step_schedule, needs_gearbind
        from .guidance_math import derive_seed, effective_sampling_probs, select
        from .guidance_math import apply_saved_rotation
        from diffab.utils.sampling import (candidate_rng, rng_state, restore_rng,
            save_checkpoint, tensor_hash, clean)
        from pathlib import Path
        import math
        cfg=normalize_guidance(guidance)
        if v.shape[0] != 1 or self.num_steps != 100:
            raise ValueError('Paper guidance requires one independent T100 trajectory per call')
        if not legacy_options.get('sample_structure', True) or not legacy_options.get('sample_sequence', True):
            raise ValueError('Paper guidance requires structure and sequence sampling; use the legacy entry for abopt')
        if scorers is None or scorers.D is None:
            raise ValueError('The configured D scorer is required')
        if needs_gearbind(cfg) and scorers.G is None:
            raise ValueError('The selected configuration requires GearBind')
        calibration=load_calibration(cfg,cdr=cdr or cfg.get('cdr'))
        mask=mask_generate.bool(); device=p.device
        signature=dict(case_id=case_id, design_id=int(design_id), seed=int(seed),
            generation_seed=int(generation_seed), trajectory_seed=trajectory_seed,
            runtime_identity=legacy_options.get('runtime_identity'),
            guidance=cfg, calibration=calibration, reference=scorers.reference.reference_hash,
            input={k:tensor_hash(x) for k,x in dict(v=v,p=p,s=s,mask=mask,
                    res=res_feat,pair=pair_feat).items()})
        job=dict(case_id=case_id,design_id=int(design_id),generation_seed=int(generation_seed))
        trace=[]; next_t=100; traj={}
        if checkpoint is not None and Path(checkpoint).is_file():
            saved=torch.load(checkpoint,map_location='cpu')
            if saved['signature'] != signature:
                raise ValueError('Checkpoint input, task or scientific configuration changed')
            current={k:x.to(device) if torch.is_tensor(x) else x for k,x in saved['state'].items()}
            next_t=saved['next_t']; trace=saved['trace']
            restore_rng(saved['rng'],device)
            if numerical_context is not None: numerical_context.load_state_dict(saved['numerical_context'])
        else:
            # Preserve the published initialization and random draw order,
            # including its historical exclusive upper category bound.
            pn=self._normalize_position(p)
            vrand=random_uniform_so3(v.shape[:2],device=device)
            prand=torch.randn_like(pn)
            vi=torch.where(mask[...,None],vrand,v)
            pi=torch.where(mask[...,None],prand,pn)
            srand=torch.randint_like(s,low=0,high=19)
            si=torch.where(mask,srand,s)
            current=dict(v=vi,R=so3vec_to_rotation(vi),p=self._unnormalize_position(pi),s=si)
        traj[next_t]=(current['v'],current['p'],current['s'])
        steps=range(next_t,0,-1)
        if legacy_options.get('pbar',False): steps=tqdm(steps,desc='Sampling')
        for step_index,t in enumerate(steps):
            if stop_after is not None and step_index >= stop_after: break
            before=current
            schedule=step_schedule(cfg,t)
            # The public proposal is differentiated once, in p -> R -> q order.
            proposal=self._guided_proposal(current,t,res_feat,pair_feat,mask,mask_res,
                                            scorers,cfg,calibration,schedule,job=job)
            states=[]
            for slot in range(schedule['K']):
                def realize():
                    if slot==0:
                        vc,z=proposal['v'],proposal['z']
                    else:
                        vc=self.trans_rot.denoise(current['v'],proposal['vhead'],mask,proposal['tt'])
                        z=torch.randn_like(current['p']) if t>1 else torch.zeros_like(current['p'])
                        if proposal['rotation_gamma']:
                            vc=apply_saved_rotation(vc,proposal['rotation_delta_used'],mask)
                    sampled=self.trans_seq._sample(proposal['q_used'])
                    sc=torch.where(mask,sampled,current['s'])
                    pc=torch.where(mask[...,None],proposal['mu']+proposal['sigma']*z,current['p'])
                    return dict(v=vc.detach(),R=so3vec_to_rotation(vc).detach(),p=pc.detach(),s=sc.detach(),
                        post=proposal['q_used'].detach(),q_used=proposal['q_used'].detach(),
                        q_before=proposal['q_before'].detach(),q_D_plus=proposal['q_D_plus'].detach(),
                        sampling_probs_used=effective_sampling_probs(proposal['q_used']).detach())
                if slot==0: state=realize()
                else:
                    slot_seed=derive_seed(seed,case_id,design_id,t,'MC20_R0_MC_%d'%slot)
                    with candidate_rng(device,slot_seed): state=realize()
                for name in ('p','v','s'):
                    if not torch.equal(state[name][~mask],current[name][~mask]):
                        raise RuntimeError('Non-design context changed: '+name)
                states.append(state)
            # Literal B1 hard scoring: no batching, gradient recomputation,
            # candidate replacement, or normalization of the raw D objective.
            invalid=[i for i,x in enumerate(states) if not all(bool(torch.isfinite(x[k]).all())
                     for k in ('p','v','post','sampling_probs_used'))]
            ds=[float('nan') if i in invalid else float(scorers.D.hard(scorers.reference,
                    x['s'],x['p'],x['R']).reshape(-1)[0]) for i,x in enumerate(states)]
            gs=[None]*len(states)
            if cfg['selector']=='dg_minimax':
                for i,x in enumerate(states):
                    if i in invalid: continue
                    try: gs[i]=float(scorers.G.score(x['s'],x['p'],x['R'])['hard_score'])
                    except RuntimeError as error:
                        if 'Nonfinite hard GearBind score' not in str(error): raise
                winner,selection=select(ds,[float('nan') if x is None else x for x in gs],calibration['hard'])
            else:
                eligible=[i for i,x in enumerate(ds) if math.isfinite(x)]
                if not eligible: raise FloatingPointError('All candidates unscoreable')
                winner=min(eligible,key=lambda i:(ds[i],i)); selection=dict(rule='raw_D_minimum')
            current=states[winner]; current['D_hard']=ds[winner]
            if gs[winner] is not None: current['G_hard']=gs[winner]
            row=clean(dict(t=t,K=len(states),winner_id=winner,D_scores=ds,G_scores=gs,
                selection=selection,schedule=schedule,direct=proposal['diagnostics'],
                state_hashes={k:tensor_hash(current[k]) for k in ('p','v','R','s','q_used')},
                invalid_candidates=invalid))
            trace.append(row); next_t=t-1
            if capture is not None: capture(t,before,proposal,states,winner,row,current)
            save_checkpoint(checkpoint,dict(signature=signature,state=current,next_t=next_t,
                trace=trace,rng=rng_state(device),numerical_context=(numerical_context.state_dict()
                    if numerical_context is not None else None)))
            traj[t-1]=(current['v'],current['p'],current['s'])
            traj[t]=tuple(x.cpu() for x in traj[t])
        if next_t==0 and _outfile is not None:
            positions=torch.nonzero(mask[0],as_tuple=False).view(-1).cpu().tolist()
            with open(_outfile,'a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(batch_item_index=0,t_final=0,
                    positions_internal=positions,mask_idx=positions,mask_len=len(positions),
                    mask_bool=mask[0].cpu().tolist(),probs=current['q_used'][0,mask[0]].cpu().tolist()))+'\n')
        self.last_sampling_result=dict(status='GENERATED' if next_t==0 else 'PARTIAL',
            state=current,q_used=current.get('q_used'),trace=trace,next_t=next_t,signature=signature)
        return traj

    @torch.no_grad()
    def _guided_proposal(self,current,t,res,pair,mask,mask_res,scorers,cfg,calibration,schedule,job=None):
        """One public proposal. The three derivative evaluation points are explicit."""
        from .guidance_math import combine, saved_delta, additive_update, apply_mtv
        from .rotation_guidance import local_rotation_exp, rotation_autograd_grad
        from diffab.utils.sampling import clean
        device=current['p'].device
        tt=torch.full((1,),t,dtype=torch.long,device=device)
        beta=self.trans_pos.var_sched.betas[t].expand(1)
        vhead,_,eps,qhead=self.eps_net(current['v'],self._normalize_position(current['p']),
                                     current['s'],res,pair,beta,mask,mask_res)
        v=self.trans_rot.denoise(current['v'],vhead,mask,tt)
        q,sgeo=self.trans_seq.denoise(current['s'],qhead,mask,tt,prior_b=cfg['hydro_prior_b'])
        mu_norm,_,z=self.trans_pos.denoise(self._normalize_position(current['p']),eps,mask,tt)
        mu=self._unnormalize_position(mu_norm); vbar=v.detach().clone()
        sigma=self.trans_pos.var_sched.sigmas[t].expand(1,1,1)
        gradients={}; common={}; d_info={}; directions=schedule['directions']
        def g_gradient(block,position,rotation):
            response=scorers.G.score(sgeo,position,so3vec_to_rotation(rotation),q=q,
                gradient_blocks=[block],gradient_objective='historical_blocks')
            key={'p':'position_gradient','omega':'omega_gradient','q':'sequence_gradient'}[block]
            value=torch.as_tensor(response[key],device=device,dtype=q.dtype)
            if value.shape != (q.shape if block=='q' else mu.shape) or not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Invalid GearBind '+block+' gradient')
            if bool((value[~mask]!=0).any()): raise ValueError('GearBind gradient escaped design mask')
            return value.detach()
        def consensus(block,gd,position,rotation):
            gradients['D_'+block]=gd
            if directions[block]=='COMMON':
                gg=g_gradient(block,position,rotation); gradients['G_'+block]=gg
                gd,common[block]=combine(gd,gg,mask,block,calibration['common'])
            gradients['used_'+block]=gd
            return gd
        # Position: differentiate the common denoised mean before adding noise.
        gp=torch.zeros_like(mu); joint_omega=None
        if cfg['protocol']=='legacy_k1' and directions['omega']!='OFF':
            # Historical K1 computes p and omega jointly at the pre-update
            # mean. It is a distinct published protocol, not DG50 with K=1.
            with torch.enable_grad():
                leaf=mu.detach().clone().requires_grad_(True)
                omega=torch.zeros_like(v,requires_grad=True)
                value,d_info['joint_geometry']=scorers.D.hard_details(scorers.reference,sgeo.detach(),leaf,
                    so3vec_to_rotation(v.detach())@local_rotation_exp(omega))
                gp,joint_omega=rotation_autograd_grad(value.sum(),(leaf,omega),allow_unused=False)
            gp=consensus('p',gp.detach(),mu,v)
        elif directions['p']!='OFF':
            with torch.enable_grad():
                leaf=mu.detach().clone().requires_grad_(True)
                value,d_info['p']=scorers.D.hard_details(scorers.reference,sgeo.detach(),leaf,
                                                        so3vec_to_rotation(v.detach()))
                gp,=torch.autograd.grad(value.sum(),leaf,allow_unused=False)
            gp=consensus('p',gp.detach(),mu,v)
        guided=(torch.where(mask[...,None],mu-float(cfg['coordinate_scale'])*sigma*gp,mu)
                if directions['p']!='OFF' else mu.detach().clone())
        if cfg['protocol']=='legacy_k1' and directions['p']!='OFF':
            pos_update=-gp*(float(cfg['coordinate_scale'])*sigma)
            guided=torch.where(mask[...,None].expand_as(mu),mu+pos_update,mu).detach()
        # Rotation: DG50 differentiates at p+; legacy K1 retains its joint leaf.
        rotation_gradient=rotation_delta=None; rotation_info=None; rotgamma=0.
        if directions['omega']!='OFF':
            if joint_omega is not None:
                gd=joint_omega
            else:
                with torch.enable_grad():
                    omega=torch.zeros_like(v,requires_grad=True)
                    value,d_info['omega']=scorers.D.hard_details(scorers.reference,sgeo.detach(),guided.detach(),
                        so3vec_to_rotation(v.detach())@local_rotation_exp(omega))
                    gd,=rotation_autograd_grad(value.sum(),omega,allow_unused=False)
            rotation_gradient=consensus('omega',gd.detach(),guided,v)
            rotgamma=float(cfg['rotation_scale'])
            if float(cfg['rotation_cap_degrees']) != 5.:
                raise ValueError('This frozen rotation implementation has a 5 degree cap')
            v,rotation_delta,rotation_info=saved_delta(v,rotation_gradient,mask,
                self.trans_rot.var_sched.sigmas[t],rotgamma)
        # Sequence: D uses argmax(q) geometry; G retains sgeo. These identities
        # intentionally differ. Both derivatives see updated p+ and R+.
        gq=torch.zeros_like(q)
        if directions['q']!='OFF':
            with torch.enable_grad():
                leaf=q.detach().clone().requires_grad_(True)
                value,d_info['q']=scorers.D.soft_details(scorers.reference,leaf,leaf.argmax(-1),
                                                       guided.detach(),so3vec_to_rotation(v.detach()))
                gq,=torch.autograd.grad(value.sum(),leaf,allow_unused=False)
            gq=consensus('q',gq.detach(),guided,v)
        eta=float(cfg['sequence_scale'])*(1-self.trans_seq.var_sched.alpha_bars[t])
        if directions['q']=='OFF' or cfg['sequence_operator']=='none':
            updated=q.detach().clone(); uinfo=dict(rule='DISABLED_CHANNEL_LITERAL_Q0')
        elif cfg['sequence_operator']=='MTV':
            updated,uinfo=apply_mtv(q,gq,eta,mask,calibration['numerical'])
        elif cfg['protocol']=='legacy_k1':
            eta=float(cfg['sequence_scale'])*(1-self.trans_seq.var_sched.alpha_bars[t].expand(1,1,1))
            updated=q-eta*gq
            updated=torch.where(mask[...,None].expand_as(updated),updated,q.detach())
            updated=torch.clamp(updated,min=1e-8)
            updated=(updated/updated.sum(dim=-1,keepdim=True)).detach()
            uinfo=dict(rule='HISTORICAL_K1_ADD',eta_seq=float(eta.flatten()[0]))
        else: updated,uinfo=additive_update(q,gq,eta,mask)
        q_D_plus=updated
        for name,g in gradients.items():
            if not bool(torch.isfinite(g[mask]).all()): raise FloatingPointError('Nonfinite '+name)
        # The per-step JSON trace contains scalar diagnostics only.
        sequence_summary={k:value for k,value in uinfo.items() if not torch.is_tensor(value)}
        return dict(mu=guided,v=v,vhead=vhead,q_before=q,q_D_plus=q_D_plus,q_used=updated,
            sgeo=sgeo,z=z,sigma=sigma,tt=tt,v_unrot=vbar,mu_unguided=mu,gradients=gradients,
            rotation_gradient=rotation_gradient,rotation_gamma=rotgamma,rotation_delta_used=rotation_delta,
            sequence_details=uinfo,
            diagnostics=clean(dict(common=common,D=d_info,rotation=rotation_info,sequence_update=sequence_summary)))

    def _sample_legacy(
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
            ) = _get_ddg_predictor(ddg_guidance_model, device)

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
