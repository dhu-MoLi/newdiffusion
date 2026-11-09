#!/usr/bin/env python3
"""Thin P2-B5 command wrapper; it does not select devices or load models itself."""

from cadiff.p2_b5.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
