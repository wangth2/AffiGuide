from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torchdrug import data

from .gearbind import residue_constants
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.utils.protein.constants import ressymb_to_resindex, restype_to_heavyatom_names


ChainSpec = Union[str, Sequence[str]]
AtomDict = Dict[str, torch.Tensor]

_ATOM37 = list(residue_constants.atom_types)
_HEAVY_TO_ATOM37: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
_SEQ_PROB_MAP: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}


def _tasks(tasks):
    if isinstance(tasks, (list, tuple)):
        return list(tasks)
    return [tasks]


def _chain_list(chains: ChainSpec) -> List[str]:
    if isinstance(chains, str):
        return list(chains)
    out = []
    for chain in chains:
        out.extend(list(chain))
    return out


def _aa_letters(aa: torch.Tensor) -> List[str]:
    index_to_symbol = {int(v): str(k).upper() for k, v in ressymb_to_resindex.items()}
    return [index_to_symbol.get(int(x), "X") for x in aa.detach().cpu().tolist()]


def _openfold_restype(symbol: str) -> int:
    return int(residue_constants.restype_order.get(symbol.upper(), residue_constants.unk_restype_index))


def _ca_index() -> int:
    try:
        return int(restype_to_heavyatom_names[0].index("CA"))
    except Exception:
        return 1


def _ca_positions(pos: torch.Tensor) -> torch.Tensor:
    if pos.dim() == 3:
        return pos
    if pos.dim() == 4:
        return pos[:, :, _ca_index(), :]
    raise ValueError(f"Expected CA or heavy-atom coordinates, got {tuple(pos.shape)}.")


def _heavy_to_atom37(num_atoms: int, device: torch.device) -> Dict[str, torch.Tensor]:
    key = (int(num_atoms), str(device))
    if key in _HEAVY_TO_ATOM37:
        return _HEAVY_TO_ATOM37[key]

    mapping = {}
    for symbol, residue_index in ressymb_to_resindex.items():
        atom_ids = []
        for name in restype_to_heavyatom_names[int(residue_index)][:num_atoms]:
            name = "" if name is None else str(name).strip()
            atom_ids.append(residue_constants.atom_order.get(name, -1))
        mapping[str(symbol).upper()] = torch.tensor(atom_ids, dtype=torch.long, device=device)

    mapping.setdefault("X", torch.full((num_atoms,), -1, dtype=torch.long, device=device))
    _HEAVY_TO_ATOM37[key] = mapping
    return mapping


def _protein_maps(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    atom_type = torch.tensor([data.feature.atom_vocab.get(name[0], 0) for name in _ATOM37], dtype=torch.long, device=device)
    atom_name = torch.tensor([data.Protein.atom_name2id[name] for name in _ATOM37], dtype=torch.long, device=device)
    residue_type = torch.tensor(
        [data.Protein.residue_symbol2id.get(symbol, 0) for symbol in residue_constants.restypes_with_x],
        dtype=torch.long,
        device=device,
    )
    return atom_type, atom_name, residue_type


def _seq_prob_map(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype)
    if key in _SEQ_PROB_MAP:
        return _SEQ_PROB_MAP[key]

    mat = torch.zeros((20, len(data.Protein.residue2id) + 1), dtype=dtype, device=device)
    for symbol, diffab_idx in ressymb_to_resindex.items():
        td_idx = data.Protein.residue_symbol2id.get(str(symbol).upper())
        if td_idx is not None and int(diffab_idx) < 20:
            mat[int(diffab_idx), int(td_idx)] = 1.0
    _SEQ_PROB_MAP[key] = mat
    return mat


def _soft_residue_feature(seq_prob: torch.Tensor) -> torch.Tensor:
    return seq_prob @ _seq_prob_map(seq_prob.device, seq_prob.dtype)


def _as_sequence(value):
    if torch.is_tensor(value):
        return value
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _row(field, batch_index: int, length: int, default):
    if field is None:
        return [default] * length

    field = _as_sequence(field)
    if isinstance(field, str):
        return list(field) if len(field) == length else [field] * length

    if torch.is_tensor(field):
        if field.dim() == 1 and field.numel() == length:
            return field
        if field.dim() == 2:
            if field.shape[1] == length:
                return field[batch_index]
            if field.shape[0] == length:
                return field[:, batch_index]
        raise ValueError(f"Cannot read residue field with shape {tuple(field.shape)}.")

    if isinstance(field, (list, tuple)):
        if len(field) == length and (not field or not isinstance(field[0], (list, tuple))):
            return list(field)
        if field and isinstance(field[0], str) and len(field[0]) == length:
            return list(field[batch_index])
        if field and isinstance(field[0], (list, tuple)):
            if len(field[0]) == length:
                return list(field[batch_index])
            if len(field) == length:
                return [field[i][batch_index] for i in range(length)]
        raise ValueError(f"Cannot read residue field of length {len(field)}.")

    raise TypeError(f"Unsupported residue field type: {type(field)!r}")


def _decode_ids(values: torch.Tensor, default: str) -> List[str]:
    table = {v: k for k, v in data.Protein.alphabet2id.items()}
    return [table.get(int(x), default) for x in values.detach().cpu().view(-1).tolist()]


def _chains(batch_ref, sample_index: int, length: int) -> List[str]:
    values = _row(batch_ref["chain_id"], sample_index, length, "A")
    if torch.is_tensor(values):
        return _decode_ids(values, "A")
    return [str(x) for x in values]


def _icode_value(value) -> str:
    if value is None:
        return " "
    value = str(value)
    return " " if value in {"", " ", "None", "nan", "NaN"} else value[0]


def _icodes(batch_ref, sample_index: int, length: int) -> List[str]:
    values = _row(batch_ref.get("icode"), sample_index, length, " ")
    if torch.is_tensor(values):
        values = _decode_ids(values, " ")
    return [_icode_value(x) for x in values]


def _encode_icodes(icodes: Sequence[str], length: int, device: torch.device) -> torch.Tensor:
    if len(icodes) != length:
        raise ValueError(f"Expected {length} insertion codes, got {len(icodes)}.")
    values = []
    for code in icodes:
        code = _icode_value(code)
        values.append(-1 if code == " " else int(data.Protein.alphabet2id.get(code, -1)))
    return torch.tensor(values, dtype=torch.long, device=device)


def _atom_dict(
    aa: torch.Tensor,
    pos_heavyatom: torch.Tensor,
    mask_heavyatom: torch.Tensor,
    chain_id: Sequence[str],
    icode: Sequence[str],
    resseq: torch.Tensor,
    device: torch.device,
) -> AtomDict:
    aa = aa.to(device=device, dtype=torch.long)
    pos_heavyatom = pos_heavyatom.to(device)
    mask_heavyatom = mask_heavyatom.to(device=device, dtype=torch.bool)
    resseq = resseq.to(device=device, dtype=torch.long)

    length, num_atoms, _ = pos_heavyatom.shape
    if len(chain_id) != length:
        raise ValueError(f"Expected {length} chain IDs, got {len(chain_id)}.")

    letters = _aa_letters(aa)
    heavy_to_atom37 = _heavy_to_atom37(num_atoms, device)

    atom_pos, atom_mask = [], []
    for i, letter in enumerate(letters):
        atom37 = heavy_to_atom37.get(letter, heavy_to_atom37["X"])
        valid = (atom37 >= 0) & mask_heavyatom[i]

        pos_row = torch.zeros((residue_constants.atom_type_num, 3), dtype=pos_heavyatom.dtype, device=device)
        mask_row = torch.zeros((residue_constants.atom_type_num,), dtype=torch.bool, device=device)
        if valid.any():
            pos_row = pos_row.index_copy(0, atom37[valid], pos_heavyatom[i][valid])
            mask_row = mask_row.index_fill(0, atom37[valid], True)
        atom_pos.append(pos_row)
        atom_mask.append(mask_row)

    return {
        "atom_positions": torch.stack(atom_pos, dim=0),
        "atom_mask": torch.stack(atom_mask, dim=0),
        "aatype": torch.tensor([_openfold_restype(x) for x in letters], dtype=torch.long, device=device),
        "residue_index": resseq,
        "chain_index": torch.tensor([data.Protein.alphabet2id[x] for x in chain_id], dtype=torch.long, device=device),
        "insertion_code": _encode_icodes(icode, length, device),
        "b_factors": torch.zeros((length, residue_constants.atom_type_num), dtype=pos_heavyatom.dtype, device=device),
    }


def _protein(atom_dict: AtomDict, device: torch.device) -> data.Protein:
    atom_positions = torch.as_tensor(atom_dict["atom_positions"], device=device)
    atom_mask = torch.as_tensor(atom_dict["atom_mask"], device=device).bool()
    aatype = torch.as_tensor(atom_dict["aatype"], device=device).long()

    atom_type_map, atom_name_map, residue_type_map = _protein_maps(device)
    atom2residue = torch.arange(aatype.shape[0], device=device)[:, None].expand_as(atom_mask)[atom_mask]

    return data.Protein(
        edge_list=torch.zeros((1, 3), dtype=torch.long, device=device),
        atom_type=atom_type_map[None, :].expand_as(atom_mask)[atom_mask],
        bond_type=torch.zeros((1,), dtype=torch.long, device=device),
        residue_type=residue_type_map[aatype],
        atom_name=atom_name_map[None, :].expand_as(atom_mask)[atom_mask],
        atom2residue=atom2residue,
        bond_feature=None,
        residue_number=torch.as_tensor(atom_dict["residue_index"], device=device).long(),
        insertion_code=torch.as_tensor(atom_dict["insertion_code"], device=device).long(),
        b_factor=torch.as_tensor(atom_dict["b_factors"], device=device)[atom_mask],
        chain_id=torch.as_tensor(atom_dict["chain_index"], device=device).long(),
        node_position=atom_positions[atom_mask],
        num_node=int(atom_mask.sum().item()),
        num_residue=int(aatype.shape[0]),
    )


def _add_features(protein: data.Protein, residue_feature: Optional[torch.Tensor] = None) -> data.Protein:
    if residue_feature is None:
        residue_feature = F.one_hot(protein.residue_type, len(data.Protein.residue2id) + 1).float()
    else:
        residue_feature = torch.as_tensor(residue_feature, device=protein.device, dtype=torch.float32)

    atom_feature = torch.cat(
        [F.one_hot(protein.atom_name, residue_constants.atom_type_num).float(), residue_feature[protein.atom2residue]],
        dim=-1,
    )
    with protein.node():
        protein.node_feature = atom_feature
    with protein.residue():
        protein.residue_feature = residue_feature
    return protein


def _annotate_pair(
    wild_type: data.Protein,
    mutant: data.Protein,
    chain_a: ChainSpec,
    chain_b: ChainSpec,
    mutation_residue: torch.Tensor,
) -> Tuple[data.Protein, data.Protein]:
    chain_a = _chain_list(chain_a)
    chain_b = _chain_list(chain_b)
    mutation_residue = torch.as_tensor(mutation_residue, device=mutant.device).bool()[: mutant.num_residue]

    def annotate(protein: data.Protein) -> data.Protein:
        entity_a = torch.zeros(protein.num_residue, dtype=torch.bool, device=protein.device)
        entity_b = torch.zeros_like(entity_a)
        for chain in chain_a:
            entity_a |= protein.chain_id == protein.alphabet2id[chain]
        for chain in chain_b:
            entity_b |= protein.chain_id == protein.alphabet2id[chain]
        with protein.node():
            protein.entity_a = entity_a[protein.atom2residue]
            protein.entity_b = entity_b[protein.atom2residue]
            protein.is_mutation = mutation_residue[protein.atom2residue]
        return protein.subgraph(protein.entity_a | protein.entity_b).clone()

    return annotate(wild_type), annotate(mutant)


class GearBindBridge:
    required_batch_keys = ("pos_heavyatom", "mask_heavyatom", "generate_flag", "resseq", "chain_id")

    def __init__(
        self,
        task: Union[torch.nn.Module, Sequence[torch.nn.Module]],
        device: Union[str, torch.device] = "cpu",
        *,
        chain_a: ChainSpec = "H",
        chain_b: ChainSpec = "A",
    ):
        self.device = torch.device(device)
        self.tasks = [task_i.to(self.device).eval() for task_i in _tasks(task)]
        if not self.tasks:
            raise ValueError("At least one GearBind task is required.")
        self.chain_a = chain_a
        self.chain_b = chain_b

    def _check_batch(self, batch_ref: Dict[str, torch.Tensor]) -> None:
        missing = [key for key in self.required_batch_keys if key not in batch_ref]
        if missing:
            raise KeyError(f"batch_ref is missing keys: {missing}")

    def _reconstruct(
        self,
        batch_ref: Dict[str, torch.Tensor],
        aa_wt: torch.Tensor,
        ca_wt: torch.Tensor,
        v_wt: torch.Tensor,
        aa_mut: torch.Tensor,
        ca_mut: torch.Tensor,
        v_mut: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_ctx = batch_ref["pos_heavyatom"].to(self.device)
        mask_atoms = batch_ref["mask_heavyatom"].to(device=self.device, dtype=torch.bool)

        wt_pos, wt_mask = reconstruct_backbone_partially(
            pos_ctx=pos_ctx,
            R_new=so3vec_to_rotation(v_wt),
            t_new=ca_wt,
            aa=aa_wt,
            chain_nb=batch_ref.get("chain_nb"),
            res_nb=batch_ref.get("res_nb"),
            mask_atoms=mask_atoms,
            mask_recons=mask,
        )
        mut_pos, mut_mask = reconstruct_backbone_partially(
            pos_ctx=pos_ctx,
            R_new=so3vec_to_rotation(v_mut),
            t_new=ca_mut,
            aa=aa_mut,
            chain_nb=batch_ref.get("chain_nb"),
            res_nb=batch_ref.get("res_nb"),
            mask_atoms=mask_atoms,
            mask_recons=mask,
        )
        return wt_pos, wt_mask, mut_pos, mut_mask

    def _make_pair(
        self,
        batch_ref: Dict[str, torch.Tensor],
        i: int,
        aa_wt: torch.Tensor,
        aa_mut: torch.Tensor,
        wt_pos: torch.Tensor,
        wt_mask: torch.Tensor,
        mut_pos: torch.Tensor,
        mut_mask: torch.Tensor,
        seq_prob: Optional[torch.Tensor],
        mutation_residue: torch.Tensor,
        chain_a: ChainSpec,
        chain_b: ChainSpec,
    ) -> Tuple[data.Protein, data.Protein]:
        length = aa_wt.shape[1]
        chain_id = _chains(batch_ref, i, length)
        icode = _icodes(batch_ref, i, length)
        resseq = batch_ref["resseq"].to(self.device)

        wt = _protein(
            _atom_dict(aa_wt[i], wt_pos[i], wt_mask[i], chain_id, icode, resseq[i], self.device),
            self.device,
        )
        mt = _protein(
            _atom_dict(aa_mut[i], mut_pos[i], mut_mask[i], chain_id, icode, resseq[i], self.device),
            self.device,
        )

        wt = _add_features(wt)
        mt = _add_features(mt, None if seq_prob is None else _soft_residue_feature(seq_prob[i]))
        return _annotate_pair(wt, mt, chain_a, chain_b, mutation_residue[i])

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
        self._check_batch(batch_ref)
        if aa_ref is None or ca_ref is None or v_ref is None:
            raise ValueError("GearBind requires aa_ref, ca_ref and v_ref.")

        if not ca_gen.requires_grad:
            ca_gen = ca_gen.detach().requires_grad_(True)
        if seq_prob_gen is not None and not seq_prob_gen.requires_grad:
            seq_prob_gen = seq_prob_gen.detach().requires_grad_(True)

        aa_wt = aa_ref.to(device=self.device, dtype=torch.long)
        ca_wt = _ca_positions(ca_ref.to(self.device))
        v_wt = v_ref.to(self.device)
        aa_mut = aa_gen.to(device=self.device, dtype=torch.long)
        ca_mut = ca_gen.to(self.device)
        v_mut = v_next.to(self.device)
        seq_prob = None if seq_prob_gen is None else seq_prob_gen.to(self.device)

        if aa_mut.shape != aa_wt.shape:
            raise ValueError(f"aa_gen and aa_ref shapes differ: {tuple(aa_mut.shape)} vs {tuple(aa_wt.shape)}.")
        if ca_mut.shape != ca_wt.shape or v_mut.shape != v_wt.shape:
            raise ValueError("Generated and reference coordinate/orientation tensors must have the same shape.")

        mask = batch_ref["generate_flag"].to(device=self.device, dtype=torch.bool)
        mutation_residue = (aa_mut != aa_wt) & mask
        wt_pos, wt_mask, mut_pos, mut_mask = self._reconstruct(
            batch_ref, aa_wt, ca_wt, v_wt, aa_mut, ca_mut, v_mut, mask
        )

        wild_types, mutants = [], []
        chain_a = self.chain_a if chain_a is None else chain_a
        chain_b = self.chain_b if chain_b is None else chain_b
        for i in range(aa_wt.shape[0]):
            wt, mt = self._make_pair(
                batch_ref, i, aa_wt, aa_mut, wt_pos, wt_mask, mut_pos, mut_mask,
                seq_prob, mutation_residue, chain_a, chain_b
            )
            wild_types.append(wt)
            mutants.append(mt)

        wt_batch = data.Protein.pack(wild_types)
        mt_batch = data.Protein.pack(mutants)
        ddg = torch.stack(
            [task.predict({"wild_type": wt_batch, "mutant": mt_batch}).view(-1) for task in self.tasks],
            dim=0,
        ).mean(dim=0)

        if clamp_pred_min is not None or clamp_pred_max is not None:
            ddg = torch.clamp(ddg, min=clamp_pred_min, max=clamp_pred_max)

        targets: List[torch.Tensor] = [ca_gen]
        if seq_prob_gen is not None:
            targets.append(seq_prob_gen)
        grads = torch.autograd.grad(ddg.sum(), targets, retain_graph=False, create_graph=False, allow_unused=True)

        return ddg.detach(), grads[0], grads[1] if seq_prob_gen is not None else None
