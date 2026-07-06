"""Compatibility wrapper for legacy pickles that import ``models``.

New code should import model definitions from ``opsigen.models``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opsigen.models import *  # noqa: F401,F403
