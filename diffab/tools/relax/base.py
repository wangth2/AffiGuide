import os
import re
import json
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class RelaxTask:
    in_path: str
    current_path: str
    info: dict
    status: str

    flexible_residue_first: Optional[Tuple] = None
    flexible_residue_last: Optional[Tuple] = None
    flexible_regions: List[Tuple[Tuple, Tuple]] = field(default_factory=list)

    def get_in_path_with_tag(self, tag):
        name, ext = os.path.splitext(self.in_path)
        new_path = f'{name}_{tag}{ext}'
        return new_path

    def set_current_path_tag(self, tag):
        new_path = self.get_in_path_with_tag(tag)
        self.current_path = new_path
        return new_path

    def check_current_path_exists(self):
        ok = os.path.exists(self.current_path)
        if not ok:
            self.mark_failure()
            return ok
        if os.path.getsize(self.current_path) == 0:
            ok = False
            self.mark_failure()
            os.unlink(self.current_path)
        return ok

    def update_if_finished(self, tag):
        out_path = self.get_in_path_with_tag(tag)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            self.set_current_path_tag(tag)
            self.mark_success()
            return True
        return False

    def can_proceed(self):
        self.check_current_path_exists()
        return self.status != 'failed'

    def mark_success(self):
        self.status = 'success'

    def mark_failure(self):
        self.status = 'failed'


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

    def __init__(self, root, final_postfix=None):
        super().__init__()
        self.root = root
        self.visited = set()
        self.final_postfix = final_postfix

    def _get_metadata(self, fpath):
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(fpath)),
            'metadata.json'
        )
        tag_name = os.path.basename(os.path.dirname(fpath))
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            for item in metadata['items']:
                if item['tag'] == tag_name:
                    info = dict(item)
                    regions = _item_regions(info)
                    info['flexible_regions'] = regions
                    if regions and ('residue_first' not in info or info.get('residue_first') is None):
                        info['residue_first'] = regions[0][0]
                        info['residue_last'] = regions[0][1]
                    return info
        except (json.JSONDecodeError, FileNotFoundError, KeyError, TypeError, ValueError):
            return None
        return None

    def scan(self) -> List[RelaxTask]:
        tasks = []
        input_fname_pattern = r'(^\d+\.pdb$|^REF\d\.pdb$)'
        for parent, _, files in os.walk(self.root):
            for fname in files:
                fpath = os.path.join(parent, fname)
                if not re.match(input_fname_pattern, fname):
                    continue
                if os.path.getsize(fpath) == 0:
                    continue
                if fpath in self.visited:
                    continue

                if self.final_postfix is not None:
                    fpath_name, fpath_ext = os.path.splitext(fpath)
                    fpath_final = f'{fpath_name}_{self.final_postfix}{fpath_ext}'
                    if os.path.exists(fpath_final):
                        continue

                info = self._get_metadata(fpath)
                if info is None:
                    continue

                tasks.append(RelaxTask(
                    in_path=fpath,
                    current_path=fpath,
                    info=info,
                    status='created',
                    flexible_residue_first=info.get('residue_first', None),
                    flexible_residue_last=info.get('residue_last', None),
                    flexible_regions=info.get('flexible_regions', []),
                ))
                self.visited.add(fpath)
        return tasks
