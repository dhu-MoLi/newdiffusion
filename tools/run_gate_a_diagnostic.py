from __future__ import annotations

import sys

from cadiff.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["diagnose-gate-a", *sys.argv[1:]]))
