import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PositionalEncoding, construct_3d_basis, global_to_local


class PerResidueEncoder(nn.Module):

    def __init__(self, feat_dim):
        super().__init__()
        self.aatype_embed = nn.Embedding(21, feat_dim)
        self.torsion_embed = PositionalEncoding()
        self.mlp = nn.Sequential(
            nn.Linear(21 * 14 * 3 + feat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def _normalize_aa_prob(self, aa_prob, device, dtype):
        """
        Convert optional aa_prob to shape (N, L, 21).
        Supports either 20-way or 21-way probabilities.
        """
        if aa_prob is None:
            return None

        aa_prob = aa_prob.to(device=device, dtype=dtype)

        if aa_prob.dim() != 3:
            raise ValueError(f"aa_prob must have shape (N, L, 20/21), got {aa_prob.shape}")

        if aa_prob.size(-1) == 20:
            pad = torch.zeros(*aa_prob.shape[:-1], 1, device=device, dtype=dtype)
            aa_prob = torch.cat([aa_prob, pad], dim=-1)
        elif aa_prob.size(-1) != 21:
            raise ValueError(f"aa_prob last dim must be 20 or 21, got {aa_prob.size(-1)}")

        # Keep it numerically well-formed.
        aa_prob = torch.clamp(aa_prob, min=0.0)
        aa_prob = aa_prob / aa_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return aa_prob

    def forward(self, aa, pos14, atom_mask, aa_prob=None):
        """
        Args:
            aa:           (N, L) hard residue indices. Used by crd_feat path.
            pos14:        (N, L, 14, 3)
            atom_mask:    (N, L, 14)
            aa_prob:      Optional (N, L, 20/21), used ONLY by aa embedding path.
        """
        if aa.dim() != 2:
            raise ValueError(
                f"PerResidueEncoder expects hard aa indices with shape (N, L); got {aa.shape}. "
                "Pass soft probabilities through aa_prob instead."
            )

        N, L = aa.size()

        R = construct_3d_basis(pos14[:, :, 1], pos14[:, :, 2], pos14[:, :, 0])  # (N, L, 3, 3)
        t = pos14[:, :, 1]  # (N, L, 3)
        crd14 = global_to_local(R, t, pos14)    # (N, L, 14, 3)
        crd14_mask = atom_mask[:, :, :, None].expand_as(crd14)
        crd14 = torch.where(crd14_mask, crd14, torch.zeros_like(crd14))

        # Keep original hard-index gating for crd_feat.
        aa_expand = aa[:, :, None, None, None].expand(N, L, 21, 14, 3)
        rng_expand = torch.arange(0, 21, device=aa.device)[None, None, :, None, None].expand(N, L, 21, 14, 3)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd14[:, :, None, :, :].expand(N, L, 21, 14, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, 21 * 14 * 3)

        # Only aa embedding path becomes soft/differentiable.
        aa_prob = self._normalize_aa_prob(aa_prob, device=aa.device, dtype=self.aatype_embed.weight.dtype)
        if aa_prob is None:
            aa_feat = self.aatype_embed(aa)  # original behavior
        else:
            aa_feat = torch.matmul(aa_prob, self.aatype_embed.weight)  # (N, L, feat)

        out_feat = self.mlp(torch.cat([crd_feat, aa_feat], dim=-1))
        return out_feat
