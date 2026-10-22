"""Validate literal pre-EPS q under the frozen T100/Hydro .8 posterior.
This module never modifies, returns a replacement for, normalizes or clips q.
The bound is derived before checking experimental values; see accompanying proof.
"""
POSTERIOR_MASS_LOWER = 0.9993834484715303
SUBUNIT_ATOL = 1.0 - POSTERIOR_MASS_LOWER
OVERSHOOT_ATOL = 2e-6
DERIVATION_SCHEMA = 'Q_POSTERIOR_DERIVED_BOUND_T100_HYDRO08_V1'
FROZEN_PROTOCOL = dict(T=100,hydro_b=.8,s=.01,num_classes=20,posterior_epsilon=1e-8,float32_unit_roundoff=2.**-24)

def validate_literal_q(q,mask,torch):
 if not torch.is_tensor(q)or not bool(torch.isfinite(q).all())or bool((q<0).any()):raise ValueError('Invalid sequence probability vector')
 if q.ndim!=3 or tuple(q.shape[:-1])!=tuple(mask.shape)or q.shape[-1]!=20:raise ValueError('Invalid literal pre-EPS probability shape')
 mass=q.sum(-1)[mask]
 if mass.numel()==0:raise ValueError('Missing generated sequence rows')
 if bool(((mass<POSTERIOR_MASS_LOWER)|(mass>1+OVERSHOOT_ATOL)).any()):raise ValueError('Literal pre-EPS q row mass outside analytically derived frozen T100/Hydro08 bound: min='+str(float(mass.min()))+',max='+str(float(mass.max())))
 return dict(q_semantics='literal historical pre-EPS; never renormalized by guard',row_mass_min=float(mass.min()),row_mass_max=float(mass.max()),subunit_atol=SUBUNIT_ATOL,overshoot_atol=OVERSHOOT_ATOL,analytic_mass_lower=POSTERIOR_MASS_LOWER,derivation_schema=DERIVATION_SCHEMA)
