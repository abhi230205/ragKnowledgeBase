#!/usr/bin/env python3
"""PostToolUse hook: ruff-format the edited Python file.

Runs `python -m ruff format <file>` on the file just written/edited. Format-only
(no lint --fix), so it never changes semantics. No-ops for non-.py files or if
ruff isn't available. Never blocks (always exits 0).
"""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return

    ti = data.get("tool_input") or {}
    tr = data.get("tool_response") or {}
    path = tr.get("filePath") or ti.get("file_path") or ti.get("path")
    if not path or not str(path).endswith(".py"):
        return

    try:
        subprocess.run(
            [sys.executable, "-m", "ruff", "format", str(path)],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass


main()
sys.exit(0)
