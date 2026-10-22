"""Differentiable bridge between DiffAb sampling states and GearBind.

The bridge converts DiffAb tensors into the TorchDrug ``Protein`` objects used by
GearBind, runs one or more loaded GearBind tasks, and returns gradients with
respect to generated C-alpha coordinates and, optionally, soft amino-acid
probabilities.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torchdrug import data

try:
    from .gearbind import residue_constants
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Cannot import GearBind residue constants. Make sure "
        "diffab/tools/GearBind/gearbind is importable."
    ) from exc

try:
    from diffab.modules.common.geometry import reconstruct_backbone_partially
    from diffab.modules.common.so3 import so3vec_to_rotation
    from diffab.utils.protein.constants import (
        ressymb_to_resindex,
        restype_to_heavyatom_names,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "GearBindBridge requires DiffAb geometry utilities and residue constants."
    ) from exc


ChainSpec = Union[str, Sequence[str]]
OpenFoldDict = Dict[str, torch.Tensor]

ATOM_NAMES_37: List[str] = list(residue_constants.atom_types)
_HEAVY_TO_ATOM37_CACHE: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
_DIFFAB_TO_TORCHDRUG_CACHE: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}


def _as_task_list(task_or_tasks: Union[torch.nn.Module, Sequence[torch.nn.Module]]) -> List[torch.nn.Module]:
    if isinstance(task_or_tasks, (list, tuple)):
        return list(task_or_tasks)
    return [task_or_tasks]


def _split_chain_spec(chains: Optional[ChainSpec]) -> List[str]:
    if chains is None:
        return []
    if isinstance(chains, str):
        return list(chains)

    out: List[str] = []
    for item in chains:
        if not isinstance(item, str):
            raise TypeError(f"Chain IDs must be strings, got {type(item)!r}")
        out.extend(list(item))
    return out


def _decode_alphabet_tensor(values: torch.Tensor, default: str) -> List[str]:
    id_to_char = {v: k for k, v in data.Protein.alphabet2id.items()}
    return [id_to_char.get(int(x), default) for x in values.detach().cpu().view(-1).tolist()]


def _to_python_sequence(value):
    if torch.is_tensor(value):
        return value
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _select_residue_row(field, batch_index: int, length: int, *, default=None):
    """Extract a per-residue row from common DiffAb collation layouts."""
    if field is None:
        return [default] * length

    field = _to_python_sequence(field)

    if isinstance(field, str):
        if len(field) == length:
            return list(field)
        return [field] * length

    if torch.is_tensor(field):
        if field.dim() == 1:
            if field.numel() != length:
                raise ValueError(f"Expected tensor length {length}, got {field.numel()}")
            return field
        if field.dim() == 2:
            if field.shape[1] == length:
                return field[batch_index, :]
            if field.shape[0] == length:
                return field[:, batch_index]
        raise ValueError(f"Cannot interpret tensor field with shape {tuple(field.shape)}")

    if isinstance(field, (list, tuple)):
        if len(field) == 0:
            return []
        first = field[0]

        if not isinstance(first, (list, tuple)):
            if len(field) == length:
                return list(field)
            if isinstance(first, str) and len(first) == length:
                return list(field[batch_index])
            raise ValueError(f"Expected field length {length}, got {len(field)}")

        outer = len(field)
        inner = len(first)
        if inner == length:
            return list(field[batch_index])
        if outer == length:
            return [field[r][batch_index] for r in range(length)]

        # Fallback for ambiguous legacy batches: the residue dimension is usually larger.
        if outer > inner:
            return [field[r][batch_index] for r in range(outer)]
        return list(field[batch_index])

    raise TypeError(f"Unsupported residue field type: {type(field)!r}")


def _extract_chain_ids(chain_field, batch_index: int, length: int) -> List[str]:
    row = _select_residue_row(chain_field, batch_index, length, default="A")
    if torch.is_tensor(row):
        return _decode_alphabet_tensor(row, default="A")
    return [str(x) for x in row]


def _normalize_icode(value) -> str:
    if value is None:
        return " "
    text = str(value)
    if text in {"", " ", "None", "nan", "NaN"}:
        return " "
    return text[0]


def _extract_icodes(icode_field, batch_index: int, length: int) -> List[str]:
    row = _select_residue_row(icode_field, batch_index, length, default=" ")
    if torch.is_tensor(row):
        decoded = _decode_alphabet_tensor(row, default=" ")
        return [_normalize_icode(x) for x in decoded]
    return [_normalize_icode(x) for x in row]


def _ca_atom_index() -> int:
    try:
        return int(restype_to_heavyatom_names[0].index("CA"))
    except Exception:
        return 1


def _as_ca_positions(pos: torch.Tensor) -> torch.Tensor:
    """Accept either CA coordinates ``(N,L,3)`` or heavy atoms ``(N,L,H,3)``."""
    if pos.dim() == 3:
        return pos
    if pos.dim() == 4:
        return pos[:, :, _ca_atom_index(), :]
    raise ValueError(f"Expected CA or heavy-atom positions, got shape {tuple(pos.shape)}")


def _diffab_aa_to_one_letter(aa: torch.Tensor) -> List[str]:
    index_to_symbol = {int(v): str(k).upper() for k, v in ressymb_to_resindex.items()}
    return [index_to_symbol.get(int(i), "X") for i in aa.detach().cpu().tolist()]


def _one_letter_to_openfold_restype(symbol: str) -> int:
    symbol = symbol.upper()
    if symbol in residue_constants.restype_order:
        return int(residue_constants.restype_order[symbol])
    return int(residue_constants.unk_restype_index)


def _heavy_to_atom37_mapping(num_heavy_atoms: int, device: torch.device) -> Dict[str, torch.Tensor]:
    key = (int(num_heavy_atoms), str(device))
    if key in _HEAVY_TO_ATOM37_CACHE:
        return _HEAVY_TO_ATOM37_CACHE[key]

    mapping: Dict[str, torch.Tensor] = {}
    for symbol, residue_index in ressymb_to_resindex.items():
        heavy_atom_names = restype_to_heavyatom_names[int(residue_index)][:num_heavy_atoms]
        atom37_indices = []
        for name in heavy_atom_names:
            atom_name = "" if name is None else str(name).strip()
            atom37_indices.append(residue_constants.atom_order.get(atom_name, -1))
        mapping[str(symbol).upper()] = torch.tensor(atom37_indices, dtype=torch.long, device=device)

    mapping.setdefault("X", torch.full((num_heavy_atoms,), -1, dtype=torch.long, device=device))
    _HEAVY_TO_ATOM37_CACHE[key] = mapping
    return mapping


def _dataset_mappings(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    atom_type_mapping = torch.tensor(
        [data.feature.atom_vocab.get(name[0], 0) for name in ATOM_NAMES_37],
        dtype=torch.long,
        device=device,
    )
    atom_name_mapping = torch.tensor(
        [data.Protein.atom_name2id[name] for name in ATOM_NAMES_37],
        dtype=torch.long,
        device=device,
    )
    residue_type_mapping = torch.tensor(
        [data.Protein.residue_symbol2id.get(sym, 0) for sym in residue_constants.restypes_with_x],
        dtype=torch.long,
        device=device,
    )
    return atom_type_mapping, atom_name_mapping, residue_type_mapping


def _diffab_prob_to_torchdrug_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype)
    if key in _DIFFAB_TO_TORCHDRUG_CACHE:
        return _DIFFAB_TO_TORCHDRUG_CACHE[key]

    matrix = torch.zeros((20, len(data.Protein.residue2id) + 1), device=device, dtype=dtype)
    for symbol, diffab_index in ressymb_to_resindex.items():
        torchdrug_index = data.Protein.residue_symbol2id.get(str(symbol).upper())
        if torchdrug_index is not None and int(diffab_index) < 20:
            matrix[int(diffab_index), int(torchdrug_index)] = 1.0

    _DIFFAB_TO_TORCHDRUG_CACHE[key] = matrix
    return matrix


def _diffab_seq_prob_to_torchdrug_feature(seq_prob: torch.Tensor) -> torch.Tensor:
    matrix = _diffab_prob_to_torchdrug_matrix(seq_prob.device, seq_prob.dtype)
    return seq_prob @ matrix


def apply_patch_batched(full: torch.Tensor, patch: torch.Tensor, patch_idx: torch.Tensor) -> torch.Tensor:
    """Replace ``full[i, patch_idx[i]]`` with ``patch[i]`` without breaking gradients."""
    if full.shape[0] != patch.shape[0] or full.shape[0] != patch_idx.shape[0]:
        raise ValueError(
            "Batch size mismatch: "
            f"full={tuple(full.shape)}, patch={tuple(patch.shape)}, patch_idx={tuple(patch_idx.shape)}"
        )
    return torch.stack(
        [full[i].index_copy(0, patch_idx[i].long(), patch[i]) for i in range(full.shape[0])],
        dim=0,
    )


def _encode_insertion_codes(icodes: Optional[Sequence[str]], length: int, device: torch.device) -> torch.Tensor:
    if icodes is None:
        return torch.full((length,), -1, dtype=torch.long, device=device)
    if len(icodes) != length:
        raise ValueError(f"Expected {length} insertion codes, got {len(icodes)}")

    encoded = []
    for code in icodes:
        code = _normalize_icode(code)
        encoded.append(-1 if code == " " else int(data.Protein.alphabet2id.get(code, -1)))
    return torch.tensor(encoded, dtype=torch.long, device=device)


def diffab_to_openfold_like_dict(*, aa_full, pos_heavyatom_full, mask_heavyatom_full,
               chain_id_full, resseq_full, device, icode_full=None):
    aa_full = aa_full.to(device=device, dtype=torch.long)
    pos_heavyatom_full = pos_heavyatom_full.to(device)
    mask_heavyatom_full = mask_heavyatom_full.to(device=device, dtype=torch.bool)
    resseq_full = resseq_full.to(device=device, dtype=torch.long)
    length, num_heavy_atoms, _ = pos_heavyatom_full.shape
    if len(chain_id_full) != length:
        raise ValueError('Expected %d chain IDs, got %d' % (length, len(chain_id_full)))
    # Preserve the original one bulk AA transfer and exact symbol mapping.
    # This finite engineering change targets O(L) scalar-sync / tiny kernels.
    aa_letters = _diffab_aa_to_one_letter(aa_full)
    by_letter = _heavy_to_atom37_mapping(num_heavy_atoms, device)
    mapping = torch.stack([by_letter.get(letter, by_letter['X']) for letter in aa_letters], dim=0)
    valid = (mapping >= 0) & mask_heavyatom_full
    residue_rows = torch.arange(length, device=device)[:, None].expand_as(mapping)[valid]
    atom_columns = mapping[valid]
    atom_positions = torch.zeros((length, residue_constants.atom_type_num, 3),
                                 dtype=pos_heavyatom_full.dtype, device=device)
    atom_mask = torch.zeros((length, residue_constants.atom_type_num),
                           dtype=torch.bool, device=device)
    # Static original AA->atom mapping has distinct atom indices per residue.
    # Each assignment is a copy; no floating reduction or atom reorder occurs.
    atom_positions[residue_rows, atom_columns] = pos_heavyatom_full[valid]
    atom_mask[residue_rows, atom_columns] = True
    return {
        'atom_positions': atom_positions,
        'atom_mask': atom_mask,
        'aatype': torch.tensor([_one_letter_to_openfold_restype(x) for x in aa_letters], dtype=torch.long, device=device),
        'residue_index': resseq_full,
        'chain_index': torch.tensor([data.Protein.alphabet2id[x] for x in chain_id_full], dtype=torch.long, device=device),
        'insertion_code': _encode_insertion_codes(icode_full, length, device),
        'b_factors': torch.zeros((length, residue_constants.atom_type_num), dtype=pos_heavyatom_full.dtype, device=device),
    }


def load_protein_keep_tensors(openfold_dict: OpenFoldDict, device: Optional[torch.device] = None) -> data.Protein:
    """Create a TorchDrug ``Protein`` while preserving coordinate gradients."""
    if device is None:
        device = openfold_dict["atom_positions"].device

    atom_positions = torch.as_tensor(openfold_dict["atom_positions"], device=device)
    atom_mask = torch.as_tensor(openfold_dict["atom_mask"], device=device).bool()
    aatype = torch.as_tensor(openfold_dict["aatype"], device=device).long()
    residue_index = torch.as_tensor(openfold_dict["residue_index"], device=device).long()
    chain_index = torch.as_tensor(openfold_dict["chain_index"], device=device).long()
    insertion_code = torch.as_tensor(openfold_dict["insertion_code"], device=device).long()
    b_factors = torch.as_tensor(
        openfold_dict.get("b_factors", torch.zeros_like(atom_positions[..., 0])),
        device=device,
    )

    atom_type_map, atom_name_map, residue_type_map = _dataset_mappings(device)
    atom2residue = torch.arange(aatype.shape[0], device=device)[:, None].expand_as(atom_mask)[atom_mask]

    protein = data.Protein(
        edge_list=torch.zeros((1, 3), dtype=torch.long, device=device),
        atom_type=atom_type_map[None, :].expand_as(atom_mask)[atom_mask],
        bond_type=torch.zeros((1,), dtype=torch.long, device=device),
        residue_type=residue_type_map[aatype],
        atom_name=atom_name_map[None, :].expand_as(atom_mask)[atom_mask],
        atom2residue=atom2residue,
        bond_feature=None,
        residue_number=residue_index,
        insertion_code=insertion_code,
        b_factor=b_factors[atom_mask],
        chain_id=chain_index,
        node_position=atom_positions[atom_mask],
        num_node=int(atom_mask.sum().item()),
        num_residue=int(aatype.shape[0]),
    )
    return protein


def attach_features_like_dataset(
    protein: data.Protein,
    residue_feature: Optional[torch.Tensor] = None,
) -> data.Protein:
    """Attach GearBind-style atom and residue features."""
    if residue_feature is None:
        residue_feature = F.one_hot(protein.residue_type, len(data.Protein.residue2id) + 1).float()
    else:
        residue_feature = torch.as_tensor(residue_feature, device=protein.device, dtype=torch.float32)
        if residue_feature.shape[0] != protein.num_residue:
            raise ValueError(
                f"residue_feature length mismatch: got {tuple(residue_feature.shape)}, "
                f"expected {protein.num_residue} residues"
            )

    atom_feature = torch.cat(
        [
            F.one_hot(protein.atom_name, residue_constants.atom_type_num).float(),
            residue_feature[protein.atom2residue],
        ],
        dim=-1,
    )
    with protein.node():
        protein.node_feature = atom_feature
    with protein.residue():
        protein.residue_feature = residue_feature
    return protein


def attach_annotations_like_dataset(
    wild_type: data.Protein,
    mutant: data.Protein,
    *,
    chain_a: ChainSpec,
    chain_b: ChainSpec,
    is_mutation_residue_full: torch.Tensor,
) -> Tuple[data.Protein, data.Protein]:
    """Attach GearBind entity and mutation masks, then crop to the two partners."""
    chain_a_ids = _split_chain_spec(chain_a)
    chain_b_ids = _split_chain_spec(chain_b)
    is_mutation = torch.as_tensor(is_mutation_residue_full, device=mutant.device).bool()
    is_mutation = is_mutation[: mutant.num_residue]

    def annotate(protein: data.Protein) -> data.Protein:
        entity_a = torch.zeros(protein.num_residue, dtype=torch.bool, device=protein.device)
        entity_b = torch.zeros_like(entity_a)
        for chain in chain_a_ids:
            entity_a |= protein.chain_id == protein.alphabet2id[chain]
        for chain in chain_b_ids:
            entity_b |= protein.chain_id == protein.alphabet2id[chain]

        with protein.node():
            protein.entity_a = entity_a[protein.atom2residue]
            protein.entity_b = entity_b[protein.atom2residue]
            protein.is_mutation = is_mutation[protein.atom2residue]
        return protein.subgraph(protein.entity_a | protein.entity_b).clone()

    return annotate(wild_type), annotate(mutant)



class GearBindBridge:
    """Run GearBind on DiffAb states and return differentiable guidance signals."""

    required_batch_keys = ("aa", "pos_heavyatom", "mask_heavyatom", "generate_flag", "resseq", "chain_id")

    def __init__(
        self,
        task: Union[torch.nn.Module, Sequence[torch.nn.Module]],
        device: Union[str, torch.device] = "cpu",
        *,
        chain_a: ChainSpec = "H",
        chain_b: ChainSpec = "A",
    ):
        self.device = torch.device(device)
        self.tasks = [task_i.to(self.device).eval() for task_i in _as_task_list(task)]
        if len(self.tasks) == 0:
            raise ValueError("At least one GearBind task is required")

        self.task = self.tasks[0]  # compatibility with older scripts
        self.chain_a = chain_a
        self.chain_b = chain_b

    def _check_batch(self, batch_ref: Dict[str, torch.Tensor]) -> None:
        missing = [key for key in self.required_batch_keys if key not in batch_ref]
        if missing:
            raise KeyError(f"batch_ref is missing required keys: {missing}")

    def _build_full_inputs(
        self,
        batch_ref: Dict[str, torch.Tensor],
        aa_gen: torch.Tensor,
        ca_gen: torch.Tensor,
        v_next: torch.Tensor,
        aa_ref: torch.Tensor,
        ca_ref: torch.Tensor,
        v_ref: torch.Tensor,
        seq_prob_gen: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        device = self.device
        aa_wt = aa_ref.to(device=device, dtype=torch.long)
        ca_wt = _as_ca_positions(ca_ref.to(device))
        v_wt = v_ref.to(device)
        length = aa_wt.shape[1]

        if aa_gen.shape[1] == length:
            aa_mut = aa_gen.to(device=device, dtype=torch.long)
            ca_mut = ca_gen.to(device)
            v_mut = v_next.to(device)
            seq_prob = None if seq_prob_gen is None else seq_prob_gen.to(device)
            return aa_wt, ca_wt, v_wt, aa_mut, ca_mut, v_mut, seq_prob

        if "patch_idx" not in batch_ref:
            raise ValueError(
                f"Generated tensors have length {aa_gen.shape[1]}, but full reference length is "
                f"{length} and batch_ref has no patch_idx."
            )

        patch_idx = batch_ref["patch_idx"].to(device=device, dtype=torch.long)
        aa_mut = apply_patch_batched(aa_wt, aa_gen.to(device=device, dtype=torch.long), patch_idx)
        ca_mut = apply_patch_batched(ca_wt, ca_gen.to(device), patch_idx)
        v_mut = apply_patch_batched(v_wt, v_next.to(device), patch_idx)

        seq_prob = None
        if seq_prob_gen is not None:
            base_prob = F.one_hot(aa_wt, num_classes=20).to(device=device, dtype=seq_prob_gen.dtype)
            seq_prob = apply_patch_batched(base_prob, seq_prob_gen.to(device), patch_idx)

        return aa_wt, ca_wt, v_wt, aa_mut, ca_mut, v_mut, seq_prob

    def _reconstruct_heavy_atoms(
        self,
        batch_ref: Dict[str, torch.Tensor],
        aa_wt: torch.Tensor,
        ca_wt: torch.Tensor,
        v_wt: torch.Tensor,
        aa_mut: torch.Tensor,
        ca_mut: torch.Tensor,
        v_mut: torch.Tensor,
        mutation_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        original_pos = batch_ref["pos_heavyatom"].to(self.device)
        original_mask = batch_ref["mask_heavyatom"].to(device=self.device, dtype=torch.bool)

        wt_pos, wt_mask = reconstruct_backbone_partially(
            pos_ctx=original_pos,
            R_new=so3vec_to_rotation(v_wt),
            t_new=ca_wt,
            aa=aa_wt,
            chain_nb=batch_ref.get("chain_nb"),
            res_nb=batch_ref.get("res_nb"),
            mask_atoms=original_mask,
            mask_recons=mutation_mask,
        )
        mut_pos, mut_mask = reconstruct_backbone_partially(
            pos_ctx=original_pos,
            R_new=so3vec_to_rotation(v_mut),
            t_new=ca_mut,
            aa=aa_mut,
            chain_nb=batch_ref.get("chain_nb"),
            res_nb=batch_ref.get("res_nb"),
            mask_atoms=original_mask,
            mask_recons=mutation_mask,
        )
        return wt_pos, wt_mask, mut_pos, mut_mask

    def _make_protein_pair(
        self,
        batch_ref: Dict[str, torch.Tensor],
        sample_index: int,
        aa_wt: torch.Tensor,
        aa_mut: torch.Tensor,
        wt_pos: torch.Tensor,
        wt_mask: torch.Tensor,
        mut_pos: torch.Tensor,
        mut_mask: torch.Tensor,
        seq_prob: Optional[torch.Tensor],
        is_mutation: torch.Tensor,
        chain_a: ChainSpec,
        chain_b: ChainSpec,
    ) -> Tuple[data.Protein, data.Protein]:
        length = aa_wt.shape[1]
        chain_ids = _extract_chain_ids(batch_ref["chain_id"], sample_index, length)
        icodes = _extract_icodes(batch_ref.get("icode"), sample_index, length)
        resseq = batch_ref["resseq"].to(self.device)

        wt_dict = diffab_to_openfold_like_dict(
            aa_full=aa_wt[sample_index],
            pos_heavyatom_full=wt_pos[sample_index],
            mask_heavyatom_full=wt_mask[sample_index],
            chain_id_full=chain_ids,
            icode_full=icodes,
            resseq_full=resseq[sample_index],
            device=self.device,
        )
        mut_dict = diffab_to_openfold_like_dict(
            aa_full=aa_mut[sample_index],
            pos_heavyatom_full=mut_pos[sample_index],
            mask_heavyatom_full=mut_mask[sample_index],
            chain_id_full=chain_ids,
            icode_full=icodes,
            resseq_full=resseq[sample_index],
            device=self.device,
        )

        wild_type = attach_features_like_dataset(load_protein_keep_tensors(wt_dict, self.device))
        mutant = load_protein_keep_tensors(mut_dict, self.device)
        if seq_prob is None:
            mutant = attach_features_like_dataset(mutant)
        else:
            residue_feature = _diffab_seq_prob_to_torchdrug_feature(seq_prob[sample_index])
            mutant = attach_features_like_dataset(mutant, residue_feature=residue_feature)

        return attach_annotations_like_dataset(
            wild_type,
            mutant,
            chain_a=chain_a,
            chain_b=chain_b,
            is_mutation_residue_full=is_mutation[sample_index],
        )

    def score_and_ca_grad_from_diffab(
        self,
        batch_ref: Dict[str, torch.Tensor],
        aa_gen: torch.Tensor,
        ca_gen: torch.Tensor,
        v_next: torch.Tensor,
        aa_ref: torch.Tensor,
        ca_ref: torch.Tensor,
        v_ref: torch.Tensor,
        seq_prob_gen: Optional[torch.Tensor] = None,
        *,
        clamp_pred_min: Optional[float] = None,
        clamp_pred_max: Optional[float] = None,
        chain_a: Optional[ChainSpec] = None,
        chain_b: Optional[ChainSpec] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return GearBind ddG and gradients for a DiffAb-generated batch."""
        self._check_batch(batch_ref)
        if aa_ref is None or ca_ref is None or v_ref is None:
            raise ValueError("GearBind guidance requires aa_ref, ca_ref, and v_ref.")

        if not ca_gen.requires_grad:
            ca_gen = ca_gen.detach().requires_grad_(True)
        if seq_prob_gen is not None and not seq_prob_gen.requires_grad:
            seq_prob_gen = seq_prob_gen.detach().requires_grad_(True)

        chain_a = self.chain_a if chain_a is None else chain_a
        chain_b = self.chain_b if chain_b is None else chain_b

        aa_wt, ca_wt, v_wt, aa_mut, ca_mut, v_mut, seq_prob = self._build_full_inputs(
            batch_ref=batch_ref,
            aa_gen=aa_gen,
            ca_gen=ca_gen,
            v_next=v_next,
            aa_ref=aa_ref,
            ca_ref=ca_ref,
            v_ref=v_ref,
            seq_prob_gen=seq_prob_gen,
        )

        mutation_mask = batch_ref["generate_flag"].to(device=self.device, dtype=torch.bool)
        is_mutation = (aa_mut != aa_wt) & mutation_mask

        wt_pos, wt_mask, mut_pos, mut_mask = self._reconstruct_heavy_atoms(
            batch_ref=batch_ref,
            aa_wt=aa_wt,
            ca_wt=ca_wt,
            v_wt=v_wt,
            aa_mut=aa_mut,
            ca_mut=ca_mut,
            v_mut=v_mut,
            mutation_mask=mutation_mask,
        )

        wild_types: List[data.Protein] = []
        mutants: List[data.Protein] = []
        for sample_index in range(aa_wt.shape[0]):
            wild_type, mutant = self._make_protein_pair(
                batch_ref=batch_ref,
                sample_index=sample_index,
                aa_wt=aa_wt,
                aa_mut=aa_mut,
                wt_pos=wt_pos,
                wt_mask=wt_mask,
                mut_pos=mut_pos,
                mut_mask=mut_mask,
                seq_prob=seq_prob,
                is_mutation=is_mutation,
                chain_a=chain_a,
                chain_b=chain_b,
            )
            wild_types.append(wild_type)
            mutants.append(mutant)

        packed_wild_type = data.Protein.pack(wild_types)
        packed_mutant = data.Protein.pack(mutants)
        ddg = torch.stack(
            [task.predict({"wild_type": packed_wild_type, "mutant": packed_mutant}).view(-1) for task in self.tasks],
            dim=0,
        ).mean(dim=0)

        if clamp_pred_min is not None or clamp_pred_max is not None:
            ddg = torch.clamp(ddg, min=clamp_pred_min, max=clamp_pred_max)

        grad_targets: List[torch.Tensor] = [ca_gen]
        if seq_prob_gen is not None:
            grad_targets.append(seq_prob_gen)
        grads = torch.autograd.grad(
            ddg.sum(),
            grad_targets,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        ca_grad = grads[0]
        seq_grad = grads[1] if seq_prob_gen is not None else None
        return ddg.detach(), ca_grad, seq_grad
