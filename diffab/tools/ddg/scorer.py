"""Frozen historical D energy and tensor bridge used by AffiGuide.

Gradients use the original clipped objective; hard candidate selection uses
raw scores. The D reference intentionally retains historical scaled geometry,
whereas GearBind owns a separate native reference (Methods: affinity guidance).
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import inspect
import torch
from diffab.tools.ddg.models.predictor import DDGPredictor
from . import predictor as helpers
OLD_PROTOCOL = 'historical_R0_RemoveNative_fixed_scaled_translation_as_Angstrom'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite energy input: ' + name)

def _hard_aa(value, name):
    if value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(name + ' must contain integer AA indices')
    if bool(((value < 0) | (value > 20)).any()):
        raise ValueError(name + ' contains an AA index outside [0,20]')
    return value.long()

@dataclass(frozen=True)
class ReferenceTensors:
    """Detached reference tensors, fixed for every candidate in a search step.

    A single trajectory reference (N=1) expands to a candidate microbatch B.
    Otherwise the requested batch size must equal the reference batch size;
    implicit trajectory/candidate reordering is deliberately forbidden.
    """
    batch_wt: dict
    mutation_mask: torch.Tensor
    chain_nb: torch.Tensor
    res_nb: torch.Tensor
    mask_atoms: torch.Tensor
    mask_res: torch.Tensor
    reference_hash: str
    protocol: str = OLD_PROTOCOL

    @property
    def batch_size(self):
        return int(self.mutation_mask.shape[0])

    @property
    def length(self):
        return int(self.mutation_mask.shape[1])


class LoadedDdgModel:
    def __init__(self, checkpoint, device='cpu', expected_sha256=None):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = torch.device(device)
        self.checkpoint_sha256 = file_sha256(self.checkpoint)
        if expected_sha256 is not None and self.checkpoint_sha256 != expected_sha256:
            raise ValueError('ddG checkpoint SHA256 mismatch')
        from diffab.utils.model_assets import verify_model
        verify_model('D', self.checkpoint)
        # CPU deserialization is explicit even when the eventual requested
        # device is CUDA. This is the sole checkpoint load in this factory.
        ckpt = torch.load(str(self.checkpoint), map_location='cpu')
        if not isinstance(ckpt, dict) or 'config' not in ckpt or 'model' not in ckpt:
            raise ValueError('ddG checkpoint lacks config/model')
        cfg = ckpt['config']
        model_config = cfg['model'] if isinstance(cfg, dict) else cfg.model
        self.model = DDGPredictor(model_config)
        self.model.load_state_dict(ckpt['model'], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        # Resolve an unspecified CUDA index to the device actually holding the
        # parameters, so callers using 'cuda' and tensors on 'cuda:0' agree.
        self.device = next(self.model.parameters()).device
        self._cost = dict(hard_pairs=0, soft_pairs=0, pair_forward_calls=0,
                          network_encoder_calls=0, network_encoder_examples=0,
                          readout_calls=0, readout_examples=0, reference_builds=0)
        # Hooks count executed encoder/readout batch calls, separately from
        # pair examples. A microbatch B performs two encoder calls, not 2*B.
        self._hooks = [self.model.encoder.register_forward_pre_hook(self._encoder_hook),
                       self.model.ddG_readout.register_forward_pre_hook(self._readout_hook)]

    def _encoder_hook(self, module, args):
        self._cost['network_encoder_calls'] += 1
        self._cost['network_encoder_examples'] += int(args[0].shape[0])

    def _readout_hook(self, module, args):
        self._cost['readout_calls'] += 1
        self._cost['readout_examples'] += int(args[0].shape[0])

    def cost_snapshot(self):
        return dict(self._cost)

    def metadata(self):
        return dict(checkpoint=str(self.checkpoint), checkpoint_sha256=self.checkpoint_sha256,
                    model_class_file=inspect.getfile(DDGPredictor), device=str(self.device),
                    eval_mode=not self.model.training,
                    parameters_frozen=all(not p.requires_grad for p in self.model.parameters()),
                    parameter_grads_none=all(p.grad is None for p in self.model.parameters()),
                    hard_clamp=None, soft_clamp=None, protocol=OLD_PROTOCOL,
                    wt_encoder_cached=False,
                    pair_vs_network_cost='pair examples; two encoder batch calls per successful pair forward')


    def _expand(self, value, batch_size):
        if value.shape[0] == batch_size:
            return value
        if value.shape[0] == 1:
            return value.expand((batch_size,)+tuple(value.shape[1:]))
        raise ValueError('candidate batch requires one reference or equal batch size; no implicit remapping')


    def hard(self, reference, aa_index, positions_A, rotations_R):
        return self._forward(reference, aa_index, positions_A, rotations_R, None, 'hard')

    def soft(self, reference, aa_prob, aa_index_for_geometry, positions_A, rotations_R):
        return self._forward(reference, aa_index_for_geometry, positions_A, rotations_R, aa_prob, 'soft')

class BridgePredictor:
 """Old reconstructed/scaled or literal native WT, with the same MT path."""
 def __init__(self,model,helpers):self.model=model;self.helpers=helpers
 def bind_references(self,batch,native_batch,v_input,p_input,s_input,dpm):
  with __import__('torch').no_grad():
   old_args=(s_input.detach(),dpm._normalize_position(p_input).detach(),v_input.detach())
   old_wt,mask,_=self.helpers._prepare_reference(batch,*old_args)
   native_wt,mask_n,_=self.helpers.convert_batch_wt(native_batch)
   if not __import__('torch').equal(mask,mask_n):raise ValueError('Reference mutation mapping mismatch')
  return {'O':dict(kind='O',wt=old_wt,args=old_args,semantics='RemoveNative input fixed scaled p interpreted as Angstrom for H3 reconstruction'),
          'N':dict(kind='N',wt=native_wt,args=None,semantics='literal native observed atoms in centered Angstrom frame')}

class OldReference(ReferenceTensors):
    def metadata(self):
        return dict(protocol=self.protocol,reference_hash=self.reference_hash,
                    reference_batch_size=self.batch_size,cropped_length=self.length,
                    generated_residues=self.mutation_mask.sum(1).cpu().tolist(),
                    reference_object='fixed original RemoveNative input reconstruction',
                    reference_generated_translation='original normalized p interpreted as Angstrom, exactly historical',
                    candidate_translation_unit='Angstrom',soft_geometry_identity='original q.argmax',
                    direct_gradient_clamp=[-8.,8.],hard_selection_score='raw historical-reference D')

class HistoricalDdgEnergy(LoadedDdgModel):
    def bind_old(self,case,model,encoded=None):
        from diffab.modules.common.so3 import rotation_to_so3vec
        batch=case['batch'];dpm=model.diffusion
        with torch.no_grad():
            encoded=model.encode(batch,remove_structure=True,remove_sequence=True) if encoded is None else encoded
            if isinstance(encoded,dict):R_input,p_input=encoded['R_input'],encoded['p_input']
            else:_,_,R_input,p_input=encoded
            v=rotation_to_so3vec(R_input);s=batch['aa'];pn=dpm._normalize_position(p_input)
            self.references=self.predictor.bind_references(batch,case['native_batch'],v,p_input,s,dpm)
            wt=self.references['O']['wt'];mask=batch['generate_flag'].detach().clone().bool()
            digest=hashlib.sha256(OLD_PROTOCOL.encode())
            for name,value in sorted(dict(s_input=s,p_norm_input=pn,v_input=v,mask=mask,wt_aa=wt['aa_index'],wt_pos14=wt['pos14'],wt_pos14_mask=wt['pos14_mask']).items()):
                cpu=value.detach().cpu().contiguous();digest.update(name.encode());digest.update(str(cpu.dtype).encode());digest.update(str(tuple(cpu.shape)).encode());digest.update(cpu.numpy().tobytes())
            self._cost['reference_builds']+=1
            return OldReference({k:x.detach().clone() if torch.is_tensor(x) else x for k,x in wt.items()},mask,batch['chain_nb'].detach().clone(),batch['res_nb'].detach().clone(),batch['mask_heavyatom'][:,:,:-1].detach().clone().bool(),batch['mask'].detach().clone().bool(),digest.hexdigest(),protocol=OLD_PROTOCOL)

    def _forward(self,reference,geometry_aa,positions,rotations,aa_prob,kind):
        if not isinstance(reference,OldReference):raise TypeError('Historical D requires bind_old; native G reference is separate')
        geometry_aa=_hard_aa(geometry_aa,'candidate hard AA');B,L=geometry_aa.shape
        if L!=reference.length or positions.shape!=(B,L,3) or rotations.shape!=(B,L,3,3):raise ValueError('Candidate shape mismatch')
        if positions.device!=self.device or rotations.device!=self.device or geometry_aa.device!=self.device:raise ValueError('Candidate device mismatch')
        _finite(positions,'candidate positions');_finite(rotations,'candidate rotations')
        wt={k:self._expand(v,B) if torch.is_tensor(v) else v for k,v in reference.batch_wt.items()}
        source=geometry_aa if aa_prob is None else aa_prob
        if aa_prob is not None:_finite(aa_prob,'candidate soft q')
        mt=self.predictor.helpers.create_batch_mut(wt,source,positions,None,self._expand(reference.mutation_mask,B),self._expand(reference.chain_nb,B),self._expand(reference.res_nb,B),self._expand(reference.mask_atoms,B),aa_mode='hard' if aa_prob is None else 'soft_emb',R_override=rotations)
        self._cost[kind+'_pairs']+=B;self._cost['pair_forward_calls']+=1
        return self.model(wt,mt)

    def hard_guided(self,reference,aa_index,positions_A,rotations_R):
        return self.hard(reference,aa_index,positions_A,rotations_R).clamp(-8.,8.)
    def soft_guided(self,reference,q,aa_index_for_geometry,positions_A,rotations_R):
        return self.soft(reference,q,aa_index_for_geometry,positions_A,rotations_R).clamp(-8.,8.)
    @staticmethod
    def _details(raw):
        clipped=raw.clamp(-8.,8.)
        return clipped,dict(raw_ddg=raw,clamped_ddg=clipped,clamp_saturated=((raw<-8)|(raw>8)).detach())
    def hard_details(self,reference,aa_index,positions_A,rotations_R):
        return self._details(self.hard(reference,aa_index,positions_A,rotations_R))
    def soft_details(self,reference,q,aa_index_for_geometry,positions_A,rotations_R):
        return self._details(self.soft(reference,q,aa_index_for_geometry,positions_A,rotations_R))
    def metadata(self):
        value=super().metadata();value.update(protocol=OLD_PROTOCOL,hard_clamp=None,soft_clamp=None,guidance_clamp=[-8.,8.],selector_and_calibration='raw old-reference D',soft_geometry_identity='original R0 q.argmax');return value

class DdgScorer(HistoricalDdgEnergy):
    """Load the exact D model once; bind a fresh frozen reference per design."""
    def __init__(self, checkpoint, device='cpu', expected_sha256=None):
        super().__init__(checkpoint, device, expected_sha256)
        self.predictor = BridgePredictor(self.model, helpers)
        self.bridge_predictor = self.predictor
        self.reference = None


@dataclass
class Scorers:
    """Two scorer handles; a D-only configuration leaves G unconstructed."""
    D: object = None
    G: object = None

    def close(self):
        if self.G is not None:
            self.G.close()
