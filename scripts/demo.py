#!/usr/bin/env python
"""One-command demo: Warthog + Starling 2 Max in the pond digital twin, live in the Isaac Gym viewer.

    python scripts/demo.py                 # needs the Isaac Gym env (see README "Quick demo")
    python scripts/demo.py --record        # also write the videos (about 10x slower)

The robots follow a fixed loop over the lawn next to the pond; the window stays open afterwards.
Keys: 1 / 2 follow the Warthog / Starling, 0 free camera (mouse + WASD), arrows orbit / zoom, P pause,
T manual drive of the Warthog (I/K forward / back, J/L turn; T again hands back to the path follower).
Any extra argument is passed on to scripts/run_team.py (e.g. --speed 2 --follow uav --collision none).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAYPOINTS = "340 450 370 540 330 600 320 410".split()


def main():
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__); return
    missing = []
    if not (ROOT / "assets/twin/pond/full/twin_visual.urdf").exists() or not (ROOT / "configs/pond.json").exists():
        missing.append("pond twin: download pond_twin.tar.gz (link in the README) and run, in the repo root:\n"
                       "      tar xzf pond_twin.tar.gz")
    if not (ROOT / "assets/robots/warthog/warthog.urdf").exists():
        missing.append("Warthog URDF (processing env):\n      python scripts/prepare_robots.py")
    if missing:
        sys.exit("[demo] missing:\n  - " + "\n  - ".join(missing))
    if not (ROOT / "assets/robots/starling/meshes/Starling2Max_body.obj").exists():
        print("[demo] Starling meshes not found: using the box model "
              "(optional: python scripts/prepare_robots.py --starling)")
    try:
        import isaacgym  # noqa: F401
    except ImportError:
        sys.exit("[demo] Isaac Gym is not importable in this Python. Activate the Isaac Gym env first, e.g.\n"
                 "  conda activate umasstwin        (README, section Environments)")
    extra = [x for x in sys.argv[1:] if x != "--record"]
    has = lambda f: f in extra                                                  # noqa: E731
    args = ["--config", str(ROOT / "configs/pond.json"), "--waypoints", *WAYPOINTS, "--hold"]
    args += [] if has("--speed") else ["--speed", "3"]
    args += [] if has("--collision") else ["--collision", "voxel"]
    args += [] if has("--out") else ["--out", str(ROOT / "outputs/demo")]
    args += ["--no-depth-png"] if "--record" in sys.argv else ["--no-record"]
    print("[demo] loading the pond twin takes 1-2 minutes; the window opens after that", flush=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.argv = ["run_team.py", *args, *extra]
    import run_team
    try:
        run_team.main()
    except KeyboardInterrupt:
        print("\n[demo] stopped")


if __name__ == "__main__":
    main()
