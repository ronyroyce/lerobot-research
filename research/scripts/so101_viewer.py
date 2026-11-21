"""Milestone 3.1: open the SO-101 tabletop scene in MuJoCo's interactive viewer.

The viewer is given the *same* compiled model and reset state that the EnvHub env uses: we import
the pinned env.py, build SO101Env with its DEFAULTS, reset it with a seed, then hand its MjModel /
MjData to mujoco.viewer. So what you see and poke at here is exactly what policies are trained on.

    unset PYTHONPATH && uv run --no-sync python research/scripts/so101_viewer.py [--seed 0] [--check]

--check builds and resets the model, prints a summary and exits without opening a window.
"""

import argparse
import importlib.util
from pathlib import Path

import mujoco
import mujoco.viewer
from huggingface_hub import snapshot_download

REPO_ID = "lerobot/so101-tabletop-mujoco"
REVISION = "4684f500b94d07194f2c860314c4e27cb9349e8c"  # the commit reviewed in Milestone 2


def load_env_module():
    """Download the pinned repo snapshot (cached after the first time) and import its env.py."""
    root = Path(snapshot_download(REPO_ID, revision=REVISION))
    spec = importlib.util.spec_from_file_location("so101_tabletop_env", root / "env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="build and summarise only, no window")
    args = parser.parse_args()

    env_module = load_env_module()
    env = env_module.SO101Env(**env_module.DEFAULTS)
    env.reset(seed=args.seed)
    world = env._world  # private, but it is the single owner of the MjModel/MjData we want to view
    model, data = world.model, world.data

    print(f"timestep {model.opt.timestep * 1000:.3f} ms x {world.substeps} substeps = 1/{env.fps} s per env step")
    print(f"bodies {model.nbody}, joints {model.njnt}, actuators {model.nu}, cameras {model.ncam}")
    print("actuators -> ctrl range [rad]:")
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  {model.actuator(i).name:14s} [{lo:+.3f}, {hi:+.3f}]  ctrl now {data.ctrl[i]:+.3f}")

    if args.check:
        env.close()
        return

    # launch() is the managed viewer: it runs the physics loop itself and gives a GUI.
    # The Control panel's sliders write data.ctrl, i.e. the position targets of the actuators.
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
