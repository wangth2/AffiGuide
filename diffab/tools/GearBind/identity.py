"""Exact six-single-CDR input identity; no model/graph/sampling changes."""
import copy,hashlib,json
CDRS={'H1':(1,1,'H_CDR1'),'H2':(2,1,'H_CDR2'),'H3':(3,1,'H_CDR3'),
      'L1':(4,2,'L_CDR1'),'L2':(5,2,'L_CDR2'),'L3':(6,2,'L_CDR3')}
SCHEMA='FROZEN_SINGLE_CDR_INPUT_V1'
def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def residue(x):
    if isinstance(x,(list,tuple)):return [str(x[0]),int(x[1]),str(x[2] or '').strip()]
    return [str(x.get('chain',x.get('chain_id'))),int(x.get('resseq',x.get('residue_number'))),str(x.get('icode',x.get('insertion_code','')) or '').strip()]
def region(record):
    cdr=record.get('cdr')
    if cdr not in CDRS:raise ValueError('EXPLICIT_SIX_CDR_IDENTITY_REQUIRED')
    enum,fragment,name=CDRS[cdr]
    if record.get('cdr_enum')!=enum or record.get('fragment_type')!=fragment or record.get('cdr_name')!=name:
        raise ValueError('CDR_ENUM_FRAGMENT_OR_NAME_CONFLICT')
    if record.get('unit_id')!=record.get('case_id','')+'__'+cdr:raise ValueError('CDR_UNIT_IDENTITY_CONFLICT')
    return cdr,enum,fragment,name
def crop_contract(record,cropped):
    cdr,enum,fragment,name=region(record)
    mask=cropped['generate_flag'].bool();idx=mask.nonzero(as_tuple=False).flatten().tolist()
    if not idx:raise ValueError('EMPTY_FROZEN_CDR_MASK')
    positions=[[str(cropped['chain_id'][i]),int(cropped['resseq'][i]),str(cropped['icode'][i] or '').strip()]for i in idx]
    wanted=[residue(x)for x in record.get('design_positions',[])]
    if positions!=wanted:raise ValueError('FROZEN_CDR_RESIDUE_MAPPING_MISMATCH')
    if [residue(x)for x in record.get('design_mask',[])]!=wanted:raise ValueError('FROZEN_CDR_DESIGN_MASK_CONFLICT')
    if idx!=record.get('positions_internal'):raise ValueError('FROZEN_CDR_PATCH_GENERATED_INDICES_MISMATCH')
    patch=cropped['patch_idx'].tolist()
    if patch!=record.get('patch_idx'):raise ValueError('FROZEN_CDR_CROP_PATCH_INDEX_MISMATCH')
    if [patch[i]for i in idx]!=record.get('generate_indices'):raise ValueError('FROZEN_CDR_FULL_GENERATE_INDICES_MISMATCH')
    origin=cropped['origin'].tolist()
    if origin!=record.get('patch_origin'):raise ValueError('FROZEN_CDR_CROP_ORIGIN_MISMATCH')
    if bool((mask & (cropped['cdr_flag']!=enum)).any()) or bool((mask & (cropped['fragment_type']!=fragment)).any()):
        raise ValueError('ACTUAL_CDR_ENUM_OR_CHAIN_ROLE_MISMATCH')
    anchor=cropped['anchor_flag'].nonzero(as_tuple=False).flatten().tolist()
    if [patch[i]for i in anchor]!=record.get('anchor_indices'):raise ValueError('FROZEN_CDR_ANCHOR_INDICES_MISMATCH')
    if [[str(cropped['chain_id'][i]),int(cropped['resseq'][i]),str(cropped['icode'][i] or '').strip()]for i in anchor]!=[residue(x)for x in record.get('anchors',[])]:raise ValueError('FROZEN_CDR_ANCHOR_IDENTITIES_MISMATCH')
    contract=dict(schema=SCHEMA,case_id=record['case_id'],unit_id=record['unit_id'],cdr=cdr,cdr_name=name,
        cdr_enum=enum,fragment_type=fragment,design_positions=wanted,positions_internal=idx,
        generate_indices=[patch[i]for i in idx],patch_idx=patch,patch_origin=origin,
        cropped_generate_mask=mask.tolist(),cropped_anchor_indices=anchor)
    contract['contract_sha256']=canonical(contract)
    return contract
def _chain(values,i,length):
    if len(values)==length:
        value=values[i]
        return str(value[0] if isinstance(value,(tuple,list)) else value)
    if len(values)==1 and isinstance(values[0],(tuple,list)) and len(values[0])==length:return str(values[0][i])
    raise ValueError('COLLATED_CDR_CHAIN_LAYOUT_UNRESOLVED')
def verify_batch_contract(batch):
    contract=copy.deepcopy(batch.get('design_cdr_contract'))
    if not isinstance(contract,dict) or contract.get('schema')!=SCHEMA:raise ValueError('FROZEN_CDR_BATCH_CONTRACT_REQUIRED')
    wanted_hash=contract.pop('contract_sha256',None)
    if canonical(contract)!=wanted_hash:raise ValueError('FROZEN_CDR_BATCH_CONTRACT_HASH_MISMATCH')
    region(contract);mask=batch['generate_flag'].bool()
    if mask.ndim!=2 or mask.shape[0]!=1:raise ValueError('CDR_CONTRACT_REQUIRES_LITERAL_B1')
    if 'mask' in batch and bool((mask & ~batch['mask'].bool()).any()):raise ValueError('DESIGN_CDR_ESCAPED_VALID_CONTEXT')
    if mask[0].tolist()!=contract['cropped_generate_mask']:raise ValueError('FROZEN_CDR_MASK_CHANGED_IN_TRANSPORT')
    idx=mask[0].nonzero(as_tuple=False).flatten().tolist()
    if idx!=contract['positions_internal']:raise ValueError('FROZEN_CDR_MASK_INDEX_CHANGED')
    if not idx:raise ValueError('EMPTY_TRANSPORTED_CDR_MASK')
    if 'cdr_flag'not in batch or 'fragment_type'not in batch:raise ValueError('EXPLICIT_CDR_AND_PARTNER_ROLE_REQUIRED')
    if bool((mask & (batch['cdr_flag']!=contract['cdr_enum'])).any()) or bool((mask & (batch['fragment_type']!=contract['fragment_type'])).any()):
        raise ValueError('TRANSPORTED_CDR_ENUM_OR_CHAIN_ROLE_MISMATCH')
    length=mask.shape[1];positions=[]
    for i in idx:
        positions.append([_chain(batch['chain_id'],i,length),int(batch['resseq'][0,i]),_chain(batch['icode'],i,length).strip()])
    if positions!=contract['design_positions']:raise ValueError('TRANSPORTED_CDR_RESIDUE_IDENTITY_MISMATCH')
    contract['contract_sha256']=wanted_hash
    return contract


def batch_contract(batch, case_id=None):
    """Freeze the actual public input's CDR identity without changing its tensors.

    Historical manifests can provide their already verified contract. A generic
    PDB entry instead records the explicit current mask and residue identities;
    it does not pretend to have a historical registry signature.
    """
    if batch.get('design_cdr_contract') is not None:
        return copy.deepcopy(batch['design_cdr_contract'])

    def values(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu()
        return value.tolist() if hasattr(value, 'tolist') else value

    def row(name):
        value = values(batch[name])
        if len(value) != 1 or not isinstance(value[0], (list, tuple)):
            raise ValueError('Public CDR identity requires a collated batch of one: ' + name)
        return list(value[0])

    mask = [bool(x) for x in row('generate_flag')]
    length = len(mask)
    indices = [i for i, selected in enumerate(mask) if selected]
    if not indices:
        raise ValueError('Public CDR mask is empty')
    if 'mask' in batch and any(mask[i] and not row('mask')[i] for i in indices):
        raise ValueError('Public CDR mask escapes the valid context')
    flags, fragments = row('cdr_flag'), row('fragment_type')
    identities = {(int(flags[i]), int(fragments[i])) for i in indices}
    matches = [(cdr, enum, fragment, name) for cdr, (enum, fragment, name) in CDRS.items()
               if identities == {(enum, fragment)}]
    if len(matches) != 1:
        raise ValueError('Public mask must identify exactly one CDR and its correct chain role')
    cdr, enum, fragment, name = matches[0]
    positions = [[_chain(batch['chain_id'], i, length), int(row('resseq')[i]),
                  _chain(batch['icode'], i, length).strip()] for i in indices]
    if case_id is None:
        case_id = batch.get('case_id', batch.get('id'))
        if isinstance(case_id, (list, tuple)) and len(case_id) == 1:
            case_id = case_id[0]
        if not isinstance(case_id, str) or not case_id:
            case_id = 'input_' + canonical(dict(chain_id=batch['chain_id'],
                resseq=row('resseq'), native_aa=row('aa')))[:16]
    patch = row('patch_idx') if 'patch_idx' in batch else None
    origin = values(batch['origin'])[0] if 'origin' in batch else None
    anchor = [i for i, value in enumerate(row('anchor_flag')) if value] if 'anchor_flag' in batch else None
    contract = dict(schema=SCHEMA, case_id=str(case_id), unit_id=str(case_id)+'__'+cdr,
        cdr=cdr, cdr_name=name, cdr_enum=enum, fragment_type=fragment,
        design_positions=positions, positions_internal=indices,
        generate_indices=[int(patch[i]) for i in indices] if patch is not None else None,
        patch_idx=patch, patch_origin=origin, cropped_generate_mask=mask,
        cropped_anchor_indices=anchor, identity_origin='PUBLIC_CURRENT_INPUT_MASK',
        historical_registry_identity_claimed=False)
    contract['contract_sha256'] = canonical(contract)
    return contract
