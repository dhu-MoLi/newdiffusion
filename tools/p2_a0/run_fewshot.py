from __future__ import annotations

import sys

from cadiff.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["p2-a0-fewshot", *sys.argv[1:]]))
