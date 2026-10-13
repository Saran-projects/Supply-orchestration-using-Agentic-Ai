"""Reconcile command script: compares logged spend with console and audits budget."""

from __future__ import annotations

import sys
from sca.llm.reconcile import main

if __name__ == "__main__":
    sys.exit(main())
