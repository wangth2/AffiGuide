import os
import re
import json
import shelve
from Bio import PDB
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class EvalTask:
    in_path: str
    ref_path: str
    info: dict
    structure: str
    name: str
    method: str
    cdr: str
    ab_chains: List

    residue_first: Optional[Tuple] = None
    residue_last: Optional[Tuple] = None
    flexible_regions: List[Tuple[Tuple, Tuple]] = field(default_factory=list)

    scores: dict = field(default_factory=dict)

    def get_gen_biopython_model(self):
        parser = PDB.PDBParser(QUIET=True)
        return parser.get_structure(self.in_path, self.in_path)[0]

    def get_ref_biopython_model(self):
        parser = PDB.PDBParser(QUIET=True)
        return parser.get_structure(self.ref_path, self.ref_path)[0]

    def save_to_db(self, db: shelve.Shelf):
        db[self.in_path] = self

    def to_report_dict(self):
        return {
            'method': self.method,
            'structure': self.structure,
            'cdr': self.cdr,
            'filename': os.path.basename(self.in_path),
            **self.scores
        }


def _as_tuple(x):
    return tuple(x) if x is not None else None


def _item_regions(item):
    regions = item.get('flexible_regions')
    if regions:
        return [(_as_tuple(first), _as_tuple(last)) for first, last in regions]

    first_list = item.get('residue_first_list')
    last_list = item.get('residue_last_list')
    if first_list and last_list:
        return [(_as_tuple(first), _as_tuple(last)) for first, last in zip(first_list, last_list)]

    first = item.get('residue_first')
    last = item.get('residue_last')
    if first is not None and last is not None:
        return [(_as_tuple(first), _as_tuple(last))]
    return []


class TaskScanner:

    def __init__(self, root, postfix=None, db: Optional[shelve.Shelf]=None, ab_chains: Optional[str]=None):
        super().__init__()
        self.root = root
        self.postfix = postfix
        self.visited = set()
        self.db = db
        self.ab_chains = list(ab_chains) if ab_chains else None
        if db is not None:
            for k in db.keys():
                self.visited.add(k)

    def _get_metadata(self, fpath):
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(fpath)),
            'metadata.json'
        )
        tag_name = os.path.basename(os.path.dirname(fpath))
        method_name = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(fpath)))
        )
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            info = None
            antibody_chains = set(self.ab_chains or [])
            for item in metadata['items']:
                item_regions = _item_regions(item)
                if item['tag'] == tag_name:
                    info = dict(item)
                    info['flexible_regions'] = item_regions
                if self.ab_chains is None:
                    for first, _ in item_regions:
                        if first:
                            antibody_chains.add(first[0])

            if info is not None:
                regions = info.get('flexible_regions', [])
                if regions and ('residue_first' not in info or info.get('residue_first') is None):
                    info['residue_first'] = regions[0][0]
                    info['residue_last'] = regions[0][1]
                info['antibody_chains'] = list(antibody_chains)
                info['structure'] = metadata['identifier']
                info['method'] = method_name
            return info
        except (json.JSONDecodeError, FileNotFoundError, KeyError, TypeError, ValueError):
            return None

    def scan(self) -> List[EvalTask]:
        tasks = []
        if self.postfix is None or not self.postfix:
            input_fname_pattern = r'^\d+\.pdb$'
            ref_fname = 'REF1.pdb'
        else:
            input_fname_pattern = rf'^\d+_{self.postfix}\.pdb$'
            ref_fname = f'REF1_{self.postfix}.pdb'
        for parent, _, files in os.walk(self.root):
            for fname in files:
                fpath = os.path.join(parent, fname)
                if not re.match(input_fname_pattern, fname):
                    continue
                if os.path.getsize(fpath) == 0:
                    continue
                if fpath in self.visited:
                    continue

                ref_path = os.path.join(parent, ref_fname)
                if not os.path.exists(ref_path):
                    continue

                info = self._get_metadata(fpath)
                if info is None:
                    continue
                tasks.append(EvalTask(
                    in_path=fpath,
                    ref_path=ref_path,
                    info=info,
                    structure=info['structure'],
                    name=info['name'],
                    method=info['method'],
                    cdr=info['tag'],
                    ab_chains=info['antibody_chains'],
                    residue_first=info.get('residue_first', None),
                    residue_last=info.get('residue_last', None),
                    flexible_regions=info.get('flexible_regions', []),
                ))
                self.visited.add(fpath)
        return tasks
