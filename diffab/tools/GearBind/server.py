"""Persistent GearBind scoring process with the frozen hard/soft gradient protocol."""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from .wire import ARRAY_MARKER, read_frame, write_frame
from . import energy
from .q_validation import validate_literal_q

def decode(value, torch, device):
    if isinstance(value, dict) and value.get(ARRAY_MARKER) is True:
        dtypes = {name: getattr(torch, name) for name in
                  ("float32", "float64", "float16", "int64", "int32", "int16", "int8", "uint8", "bool")}
        if value["dtype"] not in dtypes:
            raise ValueError("Unsupported primitive tensor dtype " + value["dtype"])
        tensor = torch.tensor(value["values"], dtype=dtypes[value["dtype"]], device=device)
        if list(tensor.shape) != value["shape"]:
            raise ValueError("Primitive tensor shape disagrees with payload")
        return tensor
    if isinstance(value, dict):
        return {key: decode(item, torch, device) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item, torch, device) for item in value]
    return value

class GearBindRuntime:
    def __init__(self, checkpoint, device="cpu", backend="production_atomic", extension_path=None,
                 config_path=None, torch_threads=2):
        import torch
        if backend == 'paper_v4':
            from diffab.numerics import apply_execution_flags
            apply_execution_flags('gearbind')
        torch.set_num_threads(int(torch_threads))
        self.torch, self.energy = torch, energy
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        self.config_path = Path(config_path or Path(__file__).parent / "config/predict/frozen_single2.yaml").resolve(strict=True)
        self.extension_path = Path(extension_path or Path(__file__).parent / "native/torch_ext.so").resolve(strict=True)
        self.backend = backend
        self.pair = self.model = None
        self.model_load_count = self.forward_count = self.hard_forward_count = 0
        self.soft_forward_count = self.backward_count = 0
        self.hard_cache = OrderedDict()
        self.diagnostic_frozen_context = None
        self.diagnostic_frozen_evidence = None

    def initialize(self, request):
        """Rebind native tensors; retained frozen GearBind weights load once."""
        torch, energy = self.torch, self.energy
        if self.diagnostic_frozen_context is not None:
            raise ValueError("End frozen topology diagnostic before native rebind")
        from diffab.modules.common.geometry import reconstruct_backbone_partially, construct_3d_basis
        batch = decode(request["native_batch"], torch, self.device)
        native_pos = batch.get("pos_heavyatom_native", batch["pos_heavyatom"])
        s_native = decode(request.get("s_native"), torch, self.device)
        if s_native is None:
            s_native = batch.get("aa_native", batch["aa"])
        p_native = decode(request.get("p_native_A"), torch, self.device)
        if p_native is None:
            p_native = native_pos[:, :, 1]
        R_native = decode(request.get("R_native"), torch, self.device)
        if R_native is None:
            R_native = construct_3d_basis(native_pos[:, :, 1], native_pos[:, :, 2], native_pos[:, :, 0])
        pair = energy.DirectEnergyPair.__new__(energy.DirectEnergyPair)
        pair.device = self.device
        pair.checkpoint, pair.config_path = self.checkpoint, self.config_path
        pair.extension_path, pair.backend = self.extension_path, self.backend
        pair.batch = energy._move(batch, self.device)
        pair.s_native = s_native.detach().long().clone()
        pair.p_native_A = p_native.detach().clone()
        pair.R_native = R_native.detach().clone()
        pair.batch["aa"] = pair.s_native
        pair.batch["pos_heavyatom"] = native_pos.detach().clone()
        pair.native_pos = pair.batch["pos_heavyatom"]
        pair.native_mask = pair.batch["mask_heavyatom"].bool()
        pair.mask = pair.batch["generate_flag"].bool()
        if "mask" in pair.batch:
            pair.mask &= pair.batch["mask"].bool()
        pair.batch["generate_flag"] = pair.mask
        pair._reconstruct = reconstruct_backbone_partially
        if pair.s_native.shape != pair.mask.shape or pair.p_native_A.shape != (*pair.mask.shape, 3):
            raise ValueError("Native AA/CA/CDR mapping shapes disagree")
        if pair.s_native.shape[0] != 1:
            raise ValueError("Each sidecar case must have batch size one; candidate batch is score_many")
        if pair.R_native.shape != (*pair.mask.shape, 3, 3):
            raise ValueError("Native R must be [1,L,3,3]")
        if not bool(pair.mask.any()):
            raise ValueError("Missing generated CDR residues")
        from .identity import verify_batch_contract
        pair.design_cdr_contract = verify_batch_contract(pair.batch)
        if not bool(torch.isfinite(pair.native_pos).all()):
            raise ValueError("Nonfinite native observed atom coordinates")
        if not torch.allclose(pair.native_pos[:, :, 1][pair.mask], pair.p_native_A[pair.mask], atol=1e-4, rtol=1e-5):
            raise ValueError("Native CA must equal actual native heavy atoms in Angstrom")
        pair.D = None
        pair.G = self.model
        pair.bridge = None
        pair.last_details = {}
        pair.metadata = {"protocol": "prior_C_G_DIRECT_nonP_single2_softq_v1",
                         "sign": "raw_MT_minus_WT_lower_is_better",
                         "WT": "native_REF1_observed_heavy_atoms_fixed",
                         "MT": "prior_C_single_CDR_N_CA_C_O_reconstruction_native_context",
                         "cdr": pair.design_cdr_contract["cdr"], "unit_id": pair.design_cdr_contract["unit_id"],
                         "design_cdr_contract_sha256": pair.design_cdr_contract["contract_sha256"],
                         "q_semantics": "soft_design_residue_features_hard_geometry_identity",
                         "position_unit": "Angstrom", "prediction_clamp": None}
        if self.model is None:
            pair._load_g()
            self.model = pair.G
            self.model_load_count += 1
        else:
            # This is literally the prior C partner mapping, without reloading
            # the checkpoint or importing the generator / DDG predictor.
            from diffab.tools.GearBind.gearbind_bridge_fixed import GearBindBridge, _extract_chain_ids
            chain_ids = _extract_chain_ids(pair.batch["chain_id"], 0, pair.s_native.shape[1])
            fragments = pair.batch["fragment_type"][0].detach().cpu().tolist()
            antibody = list(dict.fromkeys(chain for chain, kind in zip(chain_ids, fragments) if kind in (1, 2)))
            antigen = list(dict.fromkeys(chain for chain, kind in zip(chain_ids, fragments) if kind == 3))
            if not antibody or not antigen or set(antibody) & set(antigen):
                raise ValueError("Invalid explicit antibody / antigen partner mapping")
            pair.partners = [(antibody, antigen)]
            pair.bridge = GearBindBridge(pair.G, device=self.device)
        self.pair = pair
        self.hard_cache.clear()
        return {"metadata": pair.metadata, "model_load_count": self.model_load_count}

    def _score(self, candidate):
        if self.pair is None:
            raise ValueError("Initialize a native reference before scoring")
        torch = self.torch
        candidate = decode(candidate, torch, self.device)
        s, p, R = [candidate[key] for key in ("s_geo", "p_A", "R")]
        for name, tensor in [("p_A", p), ("R", R)]:
            if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Invalid candidate " + name)
        requested = candidate.get("gradient_blocks")
        if requested is None:
            requested = ["q"] if candidate.get("gradient", False) else []
        if any(block not in ("p", "q") for block in requested):
            raise ValueError("Only p/q gradients are supported; rotation guidance is fixed off")
        blocks = tuple(block for block in ("p", "q") if block in requested)
        objective = candidate.get("gradient_objective", "historical_blocks")
        if objective not in ("historical_blocks", "soft_joint"):
            raise ValueError("Unknown gradient objective")
        need_q = "q" in blocks or (objective == "soft_joint" and bool(blocks))
        if need_q:
            if "q" not in candidate:
                raise ValueError("Soft sequence-gradient objective requires q")
            q = candidate["q"]
            q_validation = validate_literal_q(q, self.pair.mask, torch)
        historical_p = None
        historical_p_details = None
        # Gradient requests revisit expansion parents that have already been
        # hard-scored. Cache only literal hard geometry/AA under this native
        # binding; no q, condition, rank or label enters this key.
        digest = hashlib.sha256()
        for tensor in (s, p, R):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        cache_key = digest.hexdigest()
        cache_hit = cache_key in self.hard_cache
        if cache_hit:
            hard, hard_details = self.hard_cache[cache_key]
            self.hard_cache.move_to_end(cache_key)
        else:
            if objective == "historical_blocks" and "p" in blocks:
                hard, historical_p, hard_details = self.pair.position_gradient("G", s, p, R)
                historical_p_details = hard_details
                self.backward_count += 1
            else:
                with torch.no_grad():
                    hard = float(self.pair.energy_hard("G", s, p, R).detach())
                hard_details = dict(self.pair.last_details)
            self.forward_count += 1
            self.hard_forward_count += 1
            self.hard_cache[cache_key] = (hard, hard_details)
            if len(self.hard_cache) > 64:
                self.hard_cache.popitem(last=False)
        if not __import__("math").isfinite(hard):
            raise RuntimeError("Nonfinite hard GearBind score")
        result = {"hard_score": hard, "soft_score": None, "gradient": None,
                  "gradient_array": None, "details": hard_details,
                  "hard_cache_hit": cache_hit, "position_gradient": None,
                  "sequence_gradient": None, "position_score": None,
                  "gradient_objective": objective, "gradient_blocks": list(blocks),
                  "rotation_gradient": False}
        if blocks and objective == "soft_joint":
            # Latent VJPs require both derivatives of ONE scalar at ONE actual
            # candidate. No R leaf is created and no R derivative is requested.
            with torch.enable_grad():
                position = p.detach().clone().requires_grad_("p" in blocks)
                probability = q.detach().clone().requires_grad_("q" in blocks)
                energy = self.pair.energy_soft("G", probability, s.detach(), position, R.detach())
                leaves = [leaf for block, leaf in (("p", position), ("q", probability)) if block in blocks]
                derivatives = torch.autograd.grad(energy, leaves, allow_unused=False)
            if not bool(torch.isfinite(energy)):
                raise RuntimeError("Nonfinite joint soft GearBind energy")
            gradients = dict(zip(blocks, derivatives))
            for block, derivative in gradients.items():
                if not bool(torch.isfinite(derivative).all()) or bool((derivative[~self.pair.mask] != 0).any()):
                    raise RuntimeError("Invalid joint soft GearBind " + block + " derivative")
            soft = float(energy.detach())
            result.update(soft_score=soft, position_score=soft if "p" in blocks else None,
                          details=dict(self.pair.last_details), gradient_connected=True)
            if "p" in blocks:
                result["position_gradient"] = gradients["p"].detach().cpu().tolist()
            if "q" in blocks:
                result["sequence_gradient"] = gradients["q"].detach().cpu().tolist()
            self.forward_count += 1
            self.soft_forward_count += 1
            self.backward_count += 1
        elif blocks:
            if "p" in blocks:
                if historical_p is None:
                    p_energy, historical_p, historical_p_details = self.pair.position_gradient("G", s, p, R)
                    if abs(p_energy - hard) > 2e-6:
                        raise RuntimeError("Cached hard score and same-candidate p derivative energy disagree")
                    self.forward_count += 1
                    self.hard_forward_count += 1
                    self.backward_count += 1
                result.update(position_gradient=historical_p.detach().cpu().tolist(),
                              position_score=hard, position_details=historical_p_details,
                              gradient_connected=True)
            if "q" in blocks:
                soft, gradient, details = self.pair.sequence_gradient("G", q, s, p, R)
                self.forward_count += 1
                self.soft_forward_count += 1
                self.backward_count += 1
                result.update(soft_score=soft, sequence_gradient=gradient.detach().cpu().tolist(),
                              details=details, gradient_connected=True)
        if need_q:
            result["q_input_validation"] = q_validation
        if result["sequence_gradient"] is not None:
            result["gradient"] = result["sequence_gradient"]
            result["gradient_array"] = {"dtype": str(q.dtype).split(".")[-1],
                                        "shape": list(q.shape), "values": result["sequence_gradient"]}
            result["gradient_outside_H3_max"] = 0.0
        if result["position_gradient"] is not None:
            result["position_gradient_outside_H3_max"] = 0.0
        result.update(status="OK", model_load_count=self.model_load_count,
                      forward_count=self.forward_count,
                      hard_forward_count=self.hard_forward_count,
                      soft_forward_count=self.soft_forward_count,
                      backward_count=self.backward_count)
        if self.diagnostic_frozen_context is not None:
            result["frozen_topology_for_diagnostic"] = True
            result["diagnostic_frozen_evidence"] = dict(self.diagnostic_frozen_evidence)
        return result

    def score(self, candidate):
        requested = candidate.get('gradient_blocks') or []
        if candidate.get('soft_only'):
            c = decode(candidate, self.torch, self.device)
            with self.torch.no_grad():
                value = self.pair.energy_soft('G', c['q'], c['s_geo'], c['p_A'], c['R'])
            self.forward_count += 1
            self.soft_forward_count += 1
            return dict(status='OK', soft_score=float(value), hard_score=None,
                        gradient=None, model_load_count=self.model_load_count,
                        forward_count=self.forward_count, hard_forward_count=self.hard_forward_count,
                        soft_forward_count=self.soft_forward_count, backward_count=self.backward_count,
                        details=self.pair.last_details)
        if 'omega' not in requested:
            return self._score(candidate)
        if any(block not in ('p', 'q', 'omega') for block in requested):
            raise ValueError('Unknown GearBind gradient block')
        result = self._score(dict(candidate, gradient_blocks=[b for b in requested if b != 'omega']))
        torch = self.torch
        c = decode(candidate, torch, self.device)
        s, p, R = [c[k] for k in ('s_geo', 'p_A', 'R')]
        objective = c.get('gradient_objective', 'historical_blocks')
        from diffab.modules.diffusion.rotation_guidance import local_rotation_exp, rotation_autograd_grad
        with torch.enable_grad():
            omega = torch.zeros_like(p, requires_grad=True)
            local_exp = local_rotation_exp(omega)
            rotated = R.detach() @ local_exp
            if objective == 'soft_joint':
                if 'q' not in c:
                    raise ValueError('Soft omega objective requires q')
                value = self.pair.energy_soft('G', c['q'].detach(), s.detach(), p.detach(), rotated)
            else:
                value = self.pair.energy_hard('G', s.detach(), p.detach(), rotated)
            derivative, = rotation_autograd_grad(value, omega, allow_unused=False)
        if not bool(torch.isfinite(value)) or not bool(torch.isfinite(derivative).all()):
            raise RuntimeError('Nonfinite right-local omega energy or derivative')
        if bool((derivative[~self.pair.mask] != 0).any()):
            raise RuntimeError('Omega gradient escaped the design mask')
        self.forward_count += 1
        self.backward_count += 1
        if objective == 'soft_joint':
            self.soft_forward_count += 1
        else:
            self.hard_forward_count += 1
        result.update(omega_gradient=derivative.detach().cpu().tolist(),
                      rotation_gradient=derivative.detach().cpu().tolist(),
                      rotation_score=float(value.detach()), rotation_gradient_connected=True,
                      rotation_parameterization='Rbar @ torch.matrix_exp(original negative-cross so3vec_to_skewsym(omega)),omega=0',
                      rotation_objective=objective, gradient_blocks=list(requested),
                      forward_count=self.forward_count, hard_forward_count=self.hard_forward_count,
                      soft_forward_count=self.soft_forward_count, backward_count=self.backward_count)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--backend', choices=('production_atomic', 'deterministic_gpu', 'paper_v4'), default='production_atomic')
    parser.add_argument('--extension-path', type=Path)
    parser.add_argument('--config-path', type=Path)
    parser.add_argument('--torch-threads', type=int, default=2)
    args = parser.parse_args()
    protocol_output, protocol_input = sys.stdout.buffer, sys.stdin.buffer
    sys.stdout = sys.stderr
    runtime = None
    while True:
        request = read_frame(protocol_input)
        if request is None:
            break
        response = {'request_id': request.get('request_id'), 'status': 'OK'}
        try:
            op = request.get('op')
            if op == 'hello':
                response.update(protocol='primitive_pickle_uint64_length_v1', pid=os.getpid(),
                                device=args.device, backend=args.backend, model_load_count=0)
            elif op == 'init':
                if runtime is None:
                    runtime = GearBindRuntime(args.checkpoint, args.device, args.backend,
                                              args.extension_path, args.config_path, args.torch_threads)
                response.update(runtime.initialize(request))
                from diffab.numerics import execution_flags
                response['execution_flags'] = execution_flags()
            elif op in ('score', 'score_many'):
                if runtime is None:
                    raise ValueError('Initialize the native reference before scoring')
                if op == 'score':
                    response.update(runtime.score(request['candidate']))
                else:
                    response['results'] = [runtime.score(c) for c in request['candidates']]
            elif op == 'close':
                write_frame(protocol_output, response)
                break
            else:
                raise ValueError('Unsupported sidecar operation ' + str(op))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response.update(status='ERROR', error=type(exc).__name__ + ': ' + str(exc))
        write_frame(protocol_output, response)


if __name__ == '__main__':
    main()
