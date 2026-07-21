"""Compatibility shim for the Curator MCP server.

The MCP implementation lives in :mod:`curator_mcp.server` so it is separated
from the core Curator package. This module keeps older console scripts and
configs that import ``curator.mcp_server`` working.
"""

from __future__ import annotations

import sys
from pathlib import Path


try:
    from curator_mcp.server import *  # noqa: F401,F403
    from curator_mcp.server import main
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / "curator_mcp").is_dir():
        sys.path.insert(0, str(repo_root))
    from curator_mcp.server import *  # noqa: F401,F403
    from curator_mcp.server import main


if __name__ == "__main__":
    main()
