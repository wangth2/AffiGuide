# pyright: reportMissingImports=false
import time
import pyrosetta
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation
from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action
from pyrosetta.rosetta.core.select.residue_selector import (
    ResidueIndexSelector,
    OrResidueSelector,
    NeighborhoodResidueSelector,
    TrueResidueSelector,
)
pyrosetta.init(' '.join([
    '-mute', 'all',
    '-use_input_sc',
    '-ignore_unrecognized_res',
    '-ignore_zero_occupancy', 'false',
    '-load_PDB_components', 'false',
    '-relax:default_repeats', '2',
    '-no_fconfig',
]))

from diffab.tools.relax.base import RelaxTask


def current_milli_time():
    return round(time.time() * 1000)


def _strip_blank_icode(pos):
    if pos is None:
        return None
    pos = tuple(pos)
    if len(pos) >= 3 and pos[-1] == ' ':
        return pos[:-1]
    return pos


def _region_list(flexible_residue_first=None, flexible_residue_last=None, flexible_regions=None):
    if flexible_regions:
        return [(_strip_blank_icode(first), _strip_blank_icode(last)) for first, last in flexible_regions]
    if flexible_residue_first is not None and flexible_residue_last is not None:
        return [(_strip_blank_icode(flexible_residue_first), _strip_blank_icode(flexible_residue_last))]
    return []


def get_scorefxn(scorefxn_name: str):
    import pyrosetta

    corrections = {
        'beta_july15': False,
        'beta_nov16': False,
        'gen_potential': False,
        'restore_talaris_behavior': False,
    }
    if 'beta_july15' in scorefxn_name or 'beta_nov15' in scorefxn_name:
        corrections['beta_july15'] = True
    elif 'beta_nov16' in scorefxn_name:
        corrections['beta_nov16'] = True
    elif 'genpot' in scorefxn_name:
        corrections['gen_potential'] = True
        pyrosetta.rosetta.basic.options.set_boolean_option('corrections:beta_july15', True)
    elif 'talaris' in scorefxn_name:
        corrections['restore_talaris_behavior'] = True
    for corr, value in corrections.items():
        pyrosetta.rosetta.basic.options.set_boolean_option(f'corrections:{corr}', value)
    return pyrosetta.create_score_function(scorefxn_name)


class RelaxRegion(object):

    def __init__(self, scorefxn='ref2015', max_iter=1000, subset='nbrs', move_bb=True):
        super().__init__()
        self.scorefxn = get_scorefxn(scorefxn)
        self.fast_relax = FastRelax()
        self.fast_relax.set_scorefxn(self.scorefxn)
        self.fast_relax.max_iter(max_iter)
        assert subset in ('all', 'target', 'nbrs')
        self.subset = subset
        self.move_bb = move_bb

    def _region_selector(self, pose, regions):
        selector = OrResidueSelector()
        for first, last in regions:
            start = pose.pdb_info().pdb2pose(*first)
            end = pose.pdb_info().pdb2pose(*last)
            if start == 0 or end == 0:
                raise ValueError(f'Invalid relax region: {first} - {last}')
            idx_selector = ResidueIndexSelector()
            idx_selector.set_index_range(start, end)
            selector.add_residue_selector(idx_selector)
        return selector

    def __call__(self, pdb_path, flexible_residue_first=None, flexible_residue_last=None, flexible_regions=None):
        pose = pyrosetta.pose_from_pdb(pdb_path)
        original_pose = pose.clone()
        regions = _region_list(flexible_residue_first, flexible_residue_last, flexible_regions)

        tf = TaskFactory()
        tf.push_back(operation.InitializeFromCommandline())
        tf.push_back(operation.RestrictToRepacking())

        if self.subset == 'all':
            gen_selector = TrueResidueSelector()
            subset_selector = gen_selector
        else:
            if not regions:
                raise ValueError('No flexible residue region is available for Rosetta relax.')
            gen_selector = self._region_selector(pose, regions)
            if self.subset == 'nbrs':
                nbr_selector = NeighborhoodResidueSelector()
                nbr_selector.set_focus_selector(gen_selector)
                nbr_selector.set_include_focus_in_subset(True)
                subset_selector = nbr_selector
            else:
                subset_selector = gen_selector

            prevent_repacking = operation.OperateOnResidueSubset(
                operation.PreventRepackingRLT(),
                subset_selector,
                flip_subset=True,
            )
            tf.push_back(prevent_repacking)

        mmf = MoveMapFactory()
        if self.move_bb:
            mmf.add_bb_action(move_map_action.mm_enable, gen_selector)
        mmf.add_chi_action(move_map_action.mm_enable, subset_selector)
        mm = mmf.create_movemap_from_pose(pose)

        fr = self.fast_relax
        fr.set_movemap(mm)
        fr.set_task_factory(tf)
        fr.apply(pose)

        e_before = self.scorefxn(original_pose)
        e_relax = self.scorefxn(pose)
        return pose, e_before, e_relax


def run_pyrosetta(task: RelaxTask):
    if not task.can_proceed():
        return task
    if task.update_if_finished('rosetta'):
        return task

    minimizer = RelaxRegion()
    pose_min, _, _ = minimizer(
        pdb_path=task.current_path,
        flexible_residue_first=task.flexible_residue_first,
        flexible_residue_last=task.flexible_residue_last,
        flexible_regions=task.flexible_regions,
    )

    out_path = task.set_current_path_tag('rosetta')
    pose_min.dump_pdb(out_path)
    task.mark_success()
    return task


def run_pyrosetta_fixbb(task: RelaxTask):
    if not task.can_proceed():
        return task
    if task.update_if_finished('fixbb'):
        return task

    minimizer = RelaxRegion(move_bb=False)
    pose_min, _, _ = minimizer(
        pdb_path=task.current_path,
        flexible_residue_first=task.flexible_residue_first,
        flexible_residue_last=task.flexible_residue_last,
        flexible_regions=task.flexible_regions,
    )

    out_path = task.set_current_path_tag('fixbb')
    pose_min.dump_pdb(out_path)
    task.mark_success()
    return task
