"""Make project root importable (modules use flat imports like `from prompts import ...`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
