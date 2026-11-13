"""Import-path setup shared by the controller, its backends and its tests.

The p4 modules (bridge/, replay/) and the P3 simulator are flat script
directories, not packages, so we put them on sys.path. Importing this module has
no other side effect. This release ships the archived P3 simulator (simulator),
the tree described in the AICSET'26 paper.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P4 = HERE.parent
REPO = P4.parent
P3_ARCHIVED = REPO / "simulator"
P3_TREE_NAME = "archived"
P3_CODE = P3_ARCHIVED
DEFAULT_TRACE = P3_ARCHIVED / "trace_alibaba.csv"          # sha256 aab04247...
TRAIN_TRACE_SHA = "aab042475fc94db3065af4366aa059e6f9c349860c0e84d6a5690147e857e952"

for _p in (HERE, P4 / "bridge", P4 / "replay", P4 / "policies", P4 / "energy", P3_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
