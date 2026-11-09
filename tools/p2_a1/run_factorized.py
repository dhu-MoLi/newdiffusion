from __future__ import annotations

import sys

from cadiff.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["p2-a1-factorized", *sys.argv[1:]]))
