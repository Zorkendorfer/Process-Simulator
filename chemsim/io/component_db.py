"""JSON component database loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from chemsim.core import Component


class ComponentDB:
    def __init__(self, db_path: str):
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"ComponentDB: cannot open '{db_path}'")
        with open(path) as f:
            self._data: dict = json.load(f)

    def get(self, component_ids: List[str]) -> List[Component]:
        result = []
        for cid in component_ids:
            if cid not in self._data:
                raise KeyError(f"ComponentDB: unknown component '{cid}'")
            result.append(Component.from_dict(cid, self._data[cid]))
        return result
