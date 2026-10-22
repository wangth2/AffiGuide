"""Frozen single2 GearBind energies on native WT and generated backbone MT.

Only deployment paths and model setup are configurable. The original candidate
reconstruction, soft-residue features and gradient expressions are retained.
"""
import hashlib
import importlib
import importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F

G_SHA256 = "743b953a0091badf8291033f7fa8eafa2e05f993425473a3990a5f5dad736712"
G_CONFIG_SHA256 = "29c88cbcbc38956191aaf1d99853a661166b39af091b056f04c8dd1f54a286db"
G_EXTENSION_SHA256 = "6c73049ddd712710a30683fc9d20e8f5088f63af35fcdd4c5ec2e3db15a24f3c"
def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _freeze(model, device):
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model

def _move(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value

class DirectEnergyPair:
    def _load_g(self):
        from torchdrug import core
        # Reuse the already validated binary exactly as the offline deployment
        # does. Avoid implicit JIT compilation in an inference worker.
        if _sha(self.extension_path) != G_EXTENSION_SHA256:
            raise RuntimeError("Validated TorchDrug native extension hash changed")
        native_spec = importlib.util.spec_from_file_location("torch_ext", self.extension_path)
        native_extension = importlib.util.module_from_spec(native_spec)
        native_spec.loader.exec_module(native_extension)
        importlib.import_module("torchdrug.utils.torch").torch_ext = native_extension
        from diffab.tools.GearBind.gearbind import dataset, model, task, util
        from diffab.tools.GearBind.gearbind_bridge_fixed import GearBindBridge, _extract_chain_ids

        if _sha(self.checkpoint) != G_SHA256:
            raise RuntimeError("Fixed GearBind single2 checkpoint hash changed")
        if _sha(self.config_path) != G_CONFIG_SHA256:
            raise RuntimeError("Fixed non-P GearBind architecture/graph config hash changed")
        cfg = util.load_config(str(self.config_path), context={})
        if cfg.task["model"]["class"] != "BindModel" or cfg.task["normalization"]:
            raise RuntimeError("Expected fixed non-P GearBind raw-ddG task")
        loaded = core.Configurable.load_config_dict(cfg.task)
        state = torch.load(self.checkpoint, map_location="cpu")["model"]
        for key in ("mean", "std", "weight"):
            if key in state and key not in loaded._buffers:
                loaded.register_buffer(key, state[key].clone())
        loaded.load_state_dict(state, strict=True)
        self.G = _freeze(loaded, self.device)
        from diffab.numerics.backend import configure_gearbind
        configure_gearbind(self.G, self.backend)
        if "fragment_type" not in self.batch:
            raise ValueError("Explicit fragment_type required to identify GearBind partners")
        partners = []
        for i in range(self.s_native.shape[0]):
            chain_ids = _extract_chain_ids(self.batch["chain_id"], i, self.s_native.shape[1])
            fragment = self.batch["fragment_type"][i].detach().cpu().tolist()
            antibody = list(dict.fromkeys(chain for chain, kind in zip(chain_ids, fragment) if kind in (1, 2)))
            antigen = list(dict.fromkeys(chain for chain, kind in zip(chain_ids, fragment) if kind == 3))
            if not antibody or not antigen or set(antibody) & set(antigen):
                raise ValueError("Invalid explicit antibody / antigen partner chain mapping")
            partners.append((antibody, antigen))
        self.partners = partners
        self.bridge = GearBindBridge(self.G, device=self.device)
        self.metadata["G_state_keys_strict_loaded"] = len(state)
        self.metadata["G_partner_chains"] = partners
        self.metadata["G_native_extension"] = str(self.extension_path)
        self.metadata["G_native_extension_sha256"] = G_EXTENSION_SHA256

    def _candidate(self, s_geo, p_A, R, q=None):
        if s_geo.shape != self.s_native.shape or p_A.shape != self.p_native_A.shape or R.shape != self.R_native.shape:
            raise ValueError("Candidate must have full cropped native batch shape")
        if p_A.device != self.device or R.device != self.device or s_geo.device != self.device:
            raise ValueError("Candidate device must equal the model device")
        s = torch.where(self.mask, s_geo.long(), self.s_native)
        if bool(((s[self.mask] < 0) | (s[self.mask] >= 20)).any()):
            raise ValueError("Generated hard AA must be an explicit standard DiffAb index 0..19")
        p = torch.where(self.mask[..., None], p_A, self.p_native_A)
        R = torch.where(self.mask[..., None, None], R, self.R_native)
        pos, atom_mask = self._reconstruct(
            pos_ctx=self.native_pos, R_new=R, t_new=p, aa=s,
            chain_nb=self.batch["chain_nb"], res_nb=self.batch["res_nb"],
            mask_atoms=self.native_mask, mask_recons=self.mask,
        )
        if q is not None:
            if q.shape != (*self.mask.shape, 20) or q.device != self.device:
                raise ValueError("q must be [N,L,20] on the model device")
            # Unknown native residues are represented by the optional 21st
            # embedding class in D; generated q still has exactly 20 classes.
            base = F.one_hot(self.s_native.clamp(max=20), num_classes=21).to(q.dtype)
            extended = F.pad(q, (0, 1))
            q = torch.where(self.mask[..., None], extended, base)
        return s, pos, atom_mask, q

    def _g_forward(self, s, pos, atom_mask, q):
        from torchdrug import data
        from diffab.tools.GearBind.gearbind_bridge_fixed import (
            _diffab_seq_prob_to_torchdrug_feature, attach_features_like_dataset,
        )

        if self.G is None:
            raise RuntimeError("GearBind was not loaded for this adapter")
        mutations = (s != self.s_native) & self.mask
        wild_types, mutants = [], []
        full_indices = []
        for i, (chain_a, chain_b) in enumerate(self.partners):
            wt, mt = self.bridge._make_protein_pair(
                batch_ref=self.batch, sample_index=i,
                aa_wt=self.s_native, aa_mut=s, wt_pos=self.native_pos,
                wt_mask=self.native_mask, mut_pos=pos, mut_mask=atom_mask,
                seq_prob=None, is_mutation=mutations, chain_a=chain_a, chain_b=chain_b,
            )
            if q is not None:
                # Pair retains residue IDs (Protein.subgraph does not crop the
                # residue table). Replace only generated residue features;
                # context including any X retains the native hard features.
                if int(mt.num_residue) != self.s_native.shape[1]:
                    raise RuntimeError("GearBind residue cropping changed explicit full-residue mapping")
                soft = _diffab_seq_prob_to_torchdrug_feature(q[i, :, :20])
                features = torch.where(self.mask[i, :, None], soft, mt.residue_feature)
                mt = attach_features_like_dataset(mt, residue_feature=features)
            wild_types.append(wt)
            mutants.append(mt)
            full_indices.append(int(mt.num_node))
        packed_wt = data.Protein.pack(wild_types)
        packed_mt = data.Protein.pack(mutants)
        graph_counts = []
        def count_graph(module, args, output):
            graph_counts.append(int(output.num_node))
        handle = self.G.graph_construction_model.register_forward_hook(count_graph)
        try:
            predicted = self.G.predict({"wild_type": packed_wt, "mutant": packed_mt})
        finally:
            handle.remove()
        self.last_details = {"G_pre_graph_atom_counts": full_indices,
                             "G_interface_graph_atom_counts_WT_MT": graph_counts,
                             "mutated_design_residue_count": int(mutations.sum()),
                             "cdr": self.design_cdr_contract["cdr"], "unit_id": self.design_cdr_contract["unit_id"],
                             "frozen_geometry_identity": True}
        return predicted.sum()

    @torch.enable_grad()
    def energy_hard(self, which, s_geo, p_A, R):
        s, pos, atom_mask, _ = self._candidate(s_geo, p_A, R)
        if which.upper() == "G":
            return self._g_forward(s, pos, atom_mask, None)
        raise ValueError("GearBind energy accepts only G; use the independent D scorer for D")

    @torch.enable_grad()
    def energy_soft(self, which, q, s_geo, p_fixed_A, R):
        s, pos, atom_mask, full_q = self._candidate(s_geo, p_fixed_A, R, q)
        if which.upper() == "G":
            return self._g_forward(s, pos, atom_mask, full_q)
        raise ValueError("GearBind energy accepts only G; use the independent D scorer for D")

    @torch.enable_grad()
    def position_gradient(self, which, s_geo, p_A, R):
        p = p_A.detach().clone().requires_grad_(True)
        energy = self.energy_hard(which, s_geo.detach(), p, R.detach())
        gradient, = torch.autograd.grad(energy, (p,), allow_unused=False)
        if not bool(torch.isfinite(energy)) or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Nonfinite true position energy or derivative")
        if bool((gradient[~self.mask] != 0).any()):
            raise RuntimeError("Position gradient escaped the design mask")
        return float(energy.detach()), gradient.detach(), dict(self.last_details)

    @torch.enable_grad()
    def sequence_gradient(self, which, q, s_geo, p_fixed_A, R):
        probabilities = q.detach().clone().requires_grad_(True)
        energy = self.energy_soft(which, probabilities, s_geo.detach(), p_fixed_A.detach(), R.detach())
        gradient, = torch.autograd.grad(energy, (probabilities,), allow_unused=False)
        if not bool(torch.isfinite(energy)) or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Nonfinite true sequence energy or derivative")
        if bool((gradient[~self.mask] != 0).any()):
            raise RuntimeError("Sequence gradient escaped the design mask")
        return float(energy.detach()), gradient.detach(), dict(self.last_details)
