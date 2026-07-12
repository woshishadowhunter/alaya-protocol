#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> int:
    try:
        from alaya.cli import main as alaya_main
    except ImportError:
        print("alaya-protocol is not installed. Run: python -m pip install alaya-protocol", file=sys.stderr)
        return 2
    return alaya_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

