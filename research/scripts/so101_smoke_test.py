"""Milestone 2 smoke test: build the SO-101 MuJoCo EnvHub env through LeRobot and inspect it.

Run from the repo root, in a terminal without ROS on PYTHONPATH:

    unset PYTHONPATH && uv run --no-sync python research/scripts/so101_smoke_test.py

What it checks:
  1. LeRobot's own factory (HubEnvConfig -> make_task_env) can build the env from a pinned Hub commit.
  2. The observation/action spaces are what env.py promises (2 cameras 480x640, 6-D state/action).
  3. Stepping works: hold the reset pose for N steps and the joints should barely move.
  4. Rendering works: saves the first front/wrist frames to outputs/m2_smoke_test/.
  5. Timing: how fast the sim runs relative to its 30 fps control rate.
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.envs import HubEnvConfig
from lerobot.envs.factory import make_task_env

# Pin the exact commit whose env.py we reviewed (see research_log.md, Milestone 2).
HUB_PATH = "lerobot/so101-tabletop-mujoco@4684f500b94d07194f2c860314c4e27cb9349e8c"
OUT_DIR = Path("outputs/m2_smoke_test")
N_STEPS = 60  # 2 s of sim time at 30 fps
SEED = 0


def describe(name, value, indent=0):
    """Recursively print the structure (keys, shapes, dtypes) of a nested observation."""
    pad = "  " * indent
    if isinstance(value, dict):
        print(f"{pad}{name}:")
        for key, sub in value.items():
            describe(key, sub, indent + 1)
    else:
        arr = np.asarray(value)
        print(f"{pad}{name}: shape={arr.shape} dtype={arr.dtype}")


def main():
    cfg = HubEnvConfig(hub_path=HUB_PATH, trust_remote_code=True)
    env = make_task_env(cfg, n_envs=1)  # a vector env with one sub-env
    print(f"vector env type: {type(env).__name__}, num_envs={env.num_envs}")
    print(f"action space: {env.action_space}")

    obs, info = env.reset(seed=SEED)
    print("\n--- observation structure (leading dim = n_envs) ---")
    describe("obs", obs)
    print(f"reset info: {info}")

    # LocalVectorEnv already converted env.py's raw keys (agent_pos, pixels.<cam>) into LeRobot's
    # standard feature names, the same names a real robot's dataset uses.
    state0 = np.asarray(obs["observation.state"])[0]
    print(f"\ninitial joints [deg, gripper 0-100]: {np.round(state0, 2)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key in [k for k in obs if k.startswith("observation.images.")]:
        cam, frames = key.removeprefix("observation.images."), obs[key]
        path = OUT_DIR / f"reset_seed{SEED}_{cam}.png"
        Image.fromarray(np.asarray(frames)[0]).save(path)
        print(f"saved {path}")

    # Hold the initial pose: the action is the current state, so a stable position controller
    # should keep the arm (almost) still. Large drift here would mean a units or actuator problem.
    action = state0[None, :].astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        obs, reward, terminated, truncated, info = env.step(action)
    elapsed = time.perf_counter() - t0

    state1 = np.asarray(obs["observation.state"])[0]
    print(f"\nafter {N_STEPS} hold steps joints: {np.round(state1, 2)}")
    print(f"max |drift| = {np.abs(state1 - state0).max():.3f}")
    print(f"reward={np.asarray(reward)[0]} terminated={terminated[0]} truncated={truncated[0]}")
    print(f"step rate: {N_STEPS / elapsed:.1f} steps/s (real time = 30 steps/s; includes 2-camera render)")
    env.close()


if __name__ == "__main__":
    main()
