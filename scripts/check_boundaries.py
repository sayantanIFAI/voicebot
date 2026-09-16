#!/usr/bin/env python3
"""Enforces CLAUDE.md's two non-negotiable structural rules, mechanically:

  1. Nothing under agent/ may import the orchestrator (main / main_pcm).
  2. No vendor SDK name may appear outside providers/ (which does not
     exist yet -- see docs/adr/0001 and Blueprint Part 7 / Epic E07).

Deliberately dependency-free (stdlib only) so it runs in every gate.sh
call site with nothing to install.
"""
from __future__ import annotations

import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
AGENT_DIR = REPO_ROOT / "agent"
PROVIDERS_DIR = REPO_ROOT / "providers"

FORBIDDEN_ORCHESTRATOR_IMPORTS = re.compile(
    r"^\s*(from|import)\s+(main|main_pcm)\b", re.MULTILINE
)

# Names that, if imported or referenced outside providers/, indicate the
# vendor boundary has leaked. Extend this list as real providers land
# (Epic E07); it starts narrow and explicit on purpose -- a broad regex
# here just trains everyone to ignore false positives.
VENDOR_SDK_MARKERS = [
    r"\bopenai\b", r"\banthropic\b", r"\bgoogle\.generativeai\b",
    r"\bgoogle\.cloud\.aiplatform\b", r"\bvertexai\b",
]
VENDOR_PATTERN = re.compile("|".join(VENDOR_SDK_MARKERS), re.IGNORECASE)


def check_one_import_rule() -> list[str]:
    violations = []
    if not AGENT_DIR.exists():
        return violations
    for path in AGENT_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if FORBIDDEN_ORCHESTRATOR_IMPORTS.search(text):
            violations.append(f"{path.relative_to(REPO_ROOT)}: imports the orchestrator")
    return violations


def check_vendor_boundary() -> list[str]:
    violations = []
    for path in REPO_ROOT.rglob("*.py"):
        if PROVIDERS_DIR in path.parents:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        if "site-packages" in path.parts or "venv" in path.parts or ".venv" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = VENDOR_PATTERN.search(text)
        if match:
            violations.append(
                f"{path.relative_to(REPO_ROOT)}: vendor SDK marker "
                f"'{match.group(0)}' found outside providers/"
            )
    return violations


def main() -> int:
    violations = check_one_import_rule() + check_vendor_boundary()
    if violations:
        print("Boundary violations found:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("Boundary rules OK: no orchestrator import in agent/, no vendor SDK outside providers/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
