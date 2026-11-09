from __future__ import annotations

import sys

from cadiff.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["p3-a0-hazelnut-spatial-response", *sys.argv[1:]]))
