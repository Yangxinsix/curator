#!/usr/bin/env python3

import sys
from pathlib import Path

if __package__ in (None, ""):
    package_dir = str(Path(__file__).resolve().parent)
    repo_root = str(Path(__file__).resolve().parents[1])
    sys.path = [path for path in sys.path if path != package_dir]
    sys.path.insert(0, repo_root)

import curator.cli as cli


if __name__ == "__main__":
    cli.deploy_main()
