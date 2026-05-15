#!/usr/bin/env python3
"""Compatibility wrapper for the DataEvolver Stage0 websearch framework.

Older notes referred to this as "LDR Stage0". The implementation now lives in
stage0_web_research.py and uses local-deep-research as a framework reference,
not as a runtime LLM/search backend.
"""

from __future__ import annotations

import sys

from stage0_web_research import main as stage0_main


COMMAND_ALIASES = {
    "prepare": "init",
}
KNOWN_COMMANDS = {
    "init",
    "prepare",
    "add-evidence",
    "next-iteration",
    "finalize",
    "validate",
    "status",
    "-h",
    "--help",
}


def normalize_args(argv: list[str]) -> list[str]:
    if argv and argv[0] in COMMAND_ALIASES:
        argv = [COMMAND_ALIASES[argv[0]]] + argv[1:]
    elif argv and argv[0] not in KNOWN_COMMANDS:
        argv = ["init"] + argv
    return argv


if __name__ == "__main__":
    if sys.argv[1:] in (["-h"], ["--help"]):
        print("Compatibility alias: prepare -> init\n")
    raise SystemExit(stage0_main(normalize_args(sys.argv[1:])))
