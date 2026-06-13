from copy import deepcopy
import torch
from .models.predictor import DDGPredictor
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.modules.common.geometry import reconstruct_backbone_partially
import torch.nn.functional as F

MODELNAME = "./diffab/tools/ddg/data/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = torch.load(MODELNAME, map_location=DEVICE)
MODEL = DDGPredictor(CKPT["config"].model)
MODEL.load_state_dict(CKPT["model"])
MODEL.to(DEVICE)
MODEL.eval()


def _num_aa_classes(batch_ref):
    # DiffAb categorical head is typically 20. Predictor residue encoder may use 21.
    # We keep soft aa in 20-dim here and let residue.py handle optional padding to 21.
    return 20


def ensure_aa_prob(aa, num_classes=20):
    """
    Accepts either:
      - hard aa index: (N, L)
      - soft aa prob:   (N, L, K)
    Returns:
      aa_prob:  (N, L, K)
      aa_index: (N, L)
    """
    if aa.dim() == 2:
        aa_index = aa.long()
        aa_prob = F.one_hot(aa_index.clamp(min=0, max=num_classes - 1), num_classes=num_classes).float()
    elif aa.dim() == 3:
        aa_prob = aa.float()
        aa_index = aa_prob.argmax(dim=-1)
    else:
        raise ValueError(f"Unsupported aa shape: {tuple(aa.shape)}")
    return aa_prob, aa_index


def convert_batch_wt(batch_ref):
    batch_wt = dict()
    batch_wt["chain_id"] = batch_ref["chain_id"]
    batch_wt["chain_seq"] = batch_ref["chain_nb"][:, :]
    batch_wt["resseq"] = batch_ref["resseq"][:, :]
    batch_wt["seq"] = batch_ref["res_nb"][:, :]
    batch_wt["aa"] = batch_ref["aa"][:, :]                # keep hard index for geometry/crd_feat path
    batch_wt["aa_index"] = batch_ref["aa"][:, :]
    batch_wt["aa_prob"] = None                              # optional, for aa_emb soft path in residue.py
    batch_wt["pos14"] = batch_ref["pos_heavyatom"][:, :, :-1, :]
    mask_heavyatom = batch_ref["mask_heavyatom"][:, :, :-1]
    batch_wt["pos14_mask"] = mask_heavyatom.unsqueeze(-1).repeat(1, 1, 1, 3)

    mutation_mask = batch_ref["generate_flag"][:, :]
    mask = batch_ref["mask"][:, :]
    return batch_wt, mutation_mask, mask


def create_batch_mut(batch_wt, aa_mut, pos_mut, v_next, mutation_mask, chain_nb, res_nb, mask_atoms, aa_mode="hard"):
    """
    aa_mode:
      - 'hard': predictor sees hard aa index only
      - 'soft_emb': predictor keeps hard aa for geometry/crd_feat path, but also carries aa_prob
                    for residue encoder aa-embedding path.
    """
    batch_mut = {}
    batch_mut["chain_id"] = batch_wt["chain_id"]
    batch_mut["chain_seq"] = batch_wt["chain_seq"]
    batch_mut["resseq"] = batch_wt["resseq"]
    batch_mut["seq"] = batch_wt["seq"]

    aa_wt_index = batch_wt["aa_index"]
    aa_mut_prob, aa_mut_index = ensure_aa_prob(aa_mut, num_classes=_num_aa_classes({}))

    # geometry / crd_feat still uses hard aa index
    batch_mut["aa_index"] = torch.where(mutation_mask.bool(), aa_mut_index, aa_wt_index)
    batch_mut["aa"] = batch_mut["aa_index"]

    # optional soft aa only for aa_emb path in residue.py
    if aa_mode == "soft_emb":
        aa_wt_prob, _ = ensure_aa_prob(aa_wt_index, num_classes=aa_mut_prob.size(-1))
        mutation_mask_prob = mutation_mask.unsqueeze(-1).expand_as(aa_mut_prob)
        batch_mut["aa_prob"] = torch.where(mutation_mask_prob.bool(), aa_mut_prob, aa_wt_prob)
    else:
        batch_mut["aa_prob"] = None

    with torch.no_grad():
        R_new = so3vec_to_rotation(v_next)
    t_new = pos_mut

    pos_new, mask_new = reconstruct_backbone_partially(
        pos_ctx=batch_wt["pos14"],
        R_new=R_new,
        t_new=t_new,
        aa=batch_mut["aa_index"],
        chain_nb=chain_nb,
        res_nb=res_nb,
        mask_atoms=mask_atoms,
        mask_recons=mutation_mask,
    )

    batch_mut["pos14"] = pos_new
    batch_mut["pos14_mask"] = mask_new.unsqueeze(-1).expand_as(pos_new)
    return batch_mut


def _predict_ddg_common(batch_ref, aa_gen, pos_gen, v_next, aa_ref=None, pos_ref=None, v_ref=None,
                        clamp_pred_min=None, clamp_pred_max=None, aa_mode="hard", track_pos_grad=True,
                        track_seq_grad=False):
    batch_wt, mutation_mask, mask = convert_batch_wt(batch_ref)

    if (aa_ref is not None) and (pos_ref is not None):
        pos_ref = pos_ref[:, :, :]
        batch_wt = create_batch_mut(
            batch_wt, aa_ref, pos_ref, v_ref[:, :], mutation_mask,
            batch_ref["chain_nb"][:, :], batch_ref["res_nb"][:, :], batch_ref["mask_heavyatom"][:, :, :-1],
            aa_mode="hard",
        )

    if track_seq_grad:
        if aa_gen.dim() != 3:
            raise ValueError("Sequence-gradient branch expects soft aa prob with shape (N, L, K).")
        aa_gen = aa_gen.clone().detach().requires_grad_(True)
    else:
        aa_gen = aa_gen.clone().detach()

    if track_pos_grad:
        pos_gen = pos_gen.clone().detach().requires_grad_(True)
    else:
        pos_gen = pos_gen.clone().detach()

    v_next = v_next.clone().detach()

    batch_mut = create_batch_mut(
        batch_wt, aa_gen, pos_gen[:, :, :], v_next[:, :, :], mutation_mask,
        batch_ref["chain_nb"][:, :], batch_ref["res_nb"][:, :], batch_ref["mask_heavyatom"][:, :, :-1],
        aa_mode=aa_mode,
    )
    batch = {"wt": batch_wt, "mut": batch_mut, "mutation_mask": mutation_mask, "mask": mask}

    with torch.set_grad_enabled(True):
        pred = MODEL(batch["wt"], batch["mut"])
        if clamp_pred_min is not None and clamp_pred_max is not None:
            pred = torch.clamp(pred, min=clamp_pred_min, max=clamp_pred_max)
        pred.backward(gradient=torch.ones_like(pred))

    aa_grad = aa_gen.grad if track_seq_grad else None
    pos_grad = pos_gen.grad if track_pos_grad else None
    return pred.detach(), aa_grad, pos_grad


def predict_ddg_position(batch_ref, aa_gen, pos_gen, v_next, aa_ref=None, pos_ref=None, v_ref=None,
                         clamp_pred_min=None, clamp_pred_max=None):
    """Position guidance: hard aa only, gradient only wrt positions."""
    pred, _, pos_grad = _predict_ddg_common(
        batch_ref, aa_gen, pos_gen, v_next,
        aa_ref=aa_ref, pos_ref=pos_ref, v_ref=v_ref,
        clamp_pred_min=clamp_pred_min, clamp_pred_max=clamp_pred_max,
        aa_mode="hard", track_pos_grad=True, track_seq_grad=False,
    )
    return pred, pos_grad


def predict_ddg_sequence(batch_ref, aa_prob, pos_gen, v_next, aa_ref=None, pos_ref=None, v_ref=None,
                         clamp_pred_min=None, clamp_pred_max=None):
    """Sequence guidance: geometry path stays hard; aa_emb path can use aa_prob via residue.py."""
    pred, aa_grad, _ = _predict_ddg_common(
        batch_ref, aa_prob, pos_gen, v_next,
        aa_ref=aa_ref, pos_ref=pos_ref, v_ref=v_ref,
        clamp_pred_min=clamp_pred_min, clamp_pred_max=clamp_pred_max,
        aa_mode="soft_emb", track_pos_grad=False, track_seq_grad=True,
    )
    return pred, aa_grad


def predict_ddg_batch(batch_ref, aa_gen, pos_gen, v_next, aa_ref=None, pos_ref=None, v_ref=None,
                      clamp_pred_min=None, clamp_pred_max=None):
    """Backward-compatible wrapper: hard-aa position guidance."""
    return predict_ddg_position(
        batch_ref, aa_gen, pos_gen, v_next,
        aa_ref=aa_ref, pos_ref=pos_ref, v_ref=v_ref,
        clamp_pred_min=clamp_pred_min, clamp_pred_max=clamp_pred_max,
    )
