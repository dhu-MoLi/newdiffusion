from __future__ import annotations

import sys

from cadiff.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run-p0", *sys.argv[1:]]))
