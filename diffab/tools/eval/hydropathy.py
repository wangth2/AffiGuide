import numpy as np

from diffab.tools.eval.base import EvalTask
from diffab.tools.eval.similarity import collect_reslist, entity_to_seq


HYDROPATHY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
    'X': 0.0, '-': 0.0,
}


# The diffusion posterior uses the frozen property prior in the model's AA
# order. Evaluation statistics below are separate from this sampling prior.
from diffab.utils.protein.constants import ressymb_to_resindex
hydropathy_vector = np.zeros((len(ressymb_to_resindex)))
for aa_res, index in ressymb_to_resindex.items():
    hydropathy_vector[index] = HYDROPATHY[aa_res]
hydropathy_prob = -hydropathy_vector[:-1] + hydropathy_vector[:-1].max()
hydropathy_prob /= hydropathy_prob.sum()


def get_hydropathy(sequence):
    return np.asarray([HYDROPATHY.get(aa, 0.0) for aa in sequence], dtype=float)


def _sequence_for_task(model, task):
    residues = collect_reslist(
        model,
        getattr(task, 'flexible_regions', None),
        task.residue_first,
        task.residue_last,
    )
    sequence, _ = entity_to_seq(residues)
    if len(sequence) == 0:
        raise ValueError('No residues found for hydropathy evaluation.')
    return sequence


def eval_hydropathy(task: EvalTask):
    seq_gen = _sequence_for_task(task.get_gen_biopython_model(), task)
    seq_ref = _sequence_for_task(task.get_ref_biopython_model(), task)

    hydro_gen = float(get_hydropathy(seq_gen).mean())
    hydro_ref = float(get_hydropathy(seq_ref).mean())
    task.scores.update({
        'hydro': hydro_gen,
        'hydro_gen': hydro_gen,
        'hydro_ref': hydro_ref,
        'delta_hydro': hydro_gen - hydro_ref,
    })
    return task
