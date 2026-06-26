#!/usr/bin/env python3
"""PreToolUse hook: block `git commit` if staged changes contain secrets.

Belt-and-suspenders over .gitignore for the graded "no committed credentials"
pitfall (an automatic -20). Fails OPEN on any internal error (gitignore stays the
primary guard); fails CLOSED (deny) only on a positive match.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

# Secret content patterns in the staged diff.
PATTERNS = [
    (r"sk-ant-[A-Za-z0-9_\-]{20,}", "Anthropic API key (sk-ant-...)"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "PEM private key"),
    (r'"private_key"\s*:\s*"-----BEGIN', "service-account JSON private_key"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
]

# Sensitive filenames that should never be committed (except .env.example).
SENSITIVE_NAME = re.compile(
    r"(^|/)(\.env(\..+)?|service_account.*\.json|.*credentials.*\.json|.*\.pem|.*\.key)$"
)


def _allow() -> None:
    sys.exit(0)


def _deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    sys.exit(0)


def main() -> None:
    try:
        sys.stdin.read()  # consume the hook payload (we scan git directly)
    except Exception:
        pass

    try:
        names = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--cached", "-U0"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:
        _allow()  # fail open — never block on tooling errors

    bad_files = []
    for line in names.splitlines():
        f = line.strip()
        if not f or f.endswith(".env.example"):
            continue
        if SENSITIVE_NAME.search(f):
            bad_files.append(f)
    if bad_files:
        _deny(
            "Refusing to commit sensitive file(s): "
            + ", ".join(bad_files)
            + ". Keep them gitignored (committed credentials = -20)."
        )

    for pat, label in PATTERNS:
        if re.search(pat, diff):
            _deny(
                f"Staged diff appears to contain a secret: {label}. "
                "Unstage it and store the secret outside git."
            )

    _allow()


main()
