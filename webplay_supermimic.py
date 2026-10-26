#!/usr/bin/env python3
"""Use the same Dex WM Viser viewer as dex-supermimic-play.

python webplay_supermimic.py [checkpoint.pt] [--wm-horizon 32]

Select world_model/hand, world_model/object, debug/env_0/cage and
 debug/env_0/target independently in Viser's built-in scene tree.
Real meshes keep their materials; WM meshes are translucent light blue.
"""

import sys
from pathlib import Path


def main(argv=None):
    from dex_supermimic.rl.dex_supermimic.cli import play

    args = list(sys.argv[1:] if argv is None else argv)
    if args and not args[0].startswith("-"):
        args[:1] = ["--checkpoint", args[0]]
    defaults = {
        "--checkpoint": str(Path(__file__).resolve().parent /
                            "logs/dex_supermimic/2026-10-26_00-42-48/model_6000.pt"),
        "--num-envs": "1",
        "--steps": "0",
    }
    for flag, value in defaults.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in args):
            args.extend((flag, value))
    if "--viser" not in args:
        args.append("--viser")
    original = sys.argv
    try:
        sys.argv = [str(Path(__file__)), *args]
        play()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
