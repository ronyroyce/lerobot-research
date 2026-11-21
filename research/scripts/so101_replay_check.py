"""Milestone 4.5: replay recorded episodes on a fresh sim robot and measure how exactly they reproduce.

`lerobot-replay` can't do this for a sim: it resets with seed=None (a random cube layout), because it was
built for real robots, whose scene is reset by hand. Here each episode is replayed on a fresh SimRobot reset
with the episode's recorded seed (from the recorder's sidecar log), sending the recorded `action` column:

    reset(seed) -> compare state[0];  for t: send action[t] -> compare state[t+1]

Checks per episode:
  - joint-state error vs the recorded observation.state (determinism: should be ~0)
  - camera frames: the replayed render vs the frame decoded from the dataset's AV1 video. Any error here is
    video compression (the dataset's images are lossy), reported as mean absolute error and PSNR
  - outcome: is the target cube placed at the end of the replay?
Control: the same actions with a different seed (--control_seed): the arm moves the same, the cube isn't there.

    unset PYTHONPATH && uv run --no-sync python research/scripts/so101_replay_check.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.envs import HubEnvConfig
from lerobot.robots.sim_robot import SimRobot, SimRobotConfig
from so101_expert import cube_placed
from so101_viewer import REVISION

HUB_PATH = f"lerobot/so101-tabletop-mujoco@{REVISION}"
CUBES = ("red_cube", "green_cube", "blue_cube", "yellow_cube")


def replay(robot, names, actions, seed):
    """Reset with `seed`, send `actions`; return the state after reset plus after each action, and frames."""
    robot.start_episode(seed=seed)
    states, fronts = [], []
    obs = robot.get_observation()
    states.append([obs[n] for n in names])
    fronts.append(obs["front"])
    for a in actions:
        robot.send_action(dict(zip(names, map(float, a), strict=True)))
        obs = robot.get_observation()
        states.append([obs[n] for n in names])
        fronts.append(obs["front"])
    return np.array(states), fronts


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="ronyroyce/so101_sim_expert_v1")
    parser.add_argument("--root", default="outputs/datasets/so101_sim_expert_v1")
    parser.add_argument("--control_seed", type=int, default=100)
    args = parser.parse_args()
    root = Path(args.root)
    side = json.loads((root.parent / f"{root.name}_expert_log.json").read_text())
    seed_of = {e["saved_as"]: e["episode_seed"] for e in side["episodes"] if e["saved_as"] is not None}
    colour_of = {e["saved_as"]: e["colour"] for e in side["episodes"] if e["saved_as"] is not None}

    ds = LeRobotDataset(args.repo_id, root=root)
    robot = SimRobot(SimRobotConfig(env=HubEnvConfig(hub_path=HUB_PATH, trust_remote_code=True)))
    names = list(robot.action_features)
    robot.connect()
    rows = []
    for ep in range(ds.num_episodes):
        lo = int(ds.meta.episodes["dataset_from_index"][ep])
        hi = int(ds.meta.episodes["dataset_to_index"][ep])
        table = ds.hf_dataset.select(range(lo, hi))
        rec_state = np.stack(table["observation.state"]).astype(np.float64)
        rec_action = np.stack(table["action"]).astype(np.float64)

        states, fronts = replay(robot, names, rec_action, seed_of[ep])
        # Recorded state[t] was observed before action[t]; replayed states[t] is the same moment.
        err = np.abs(states[:-1] - rec_state)
        world = robot._env._envs[0].unwrapped._world
        target = world.model.body(f"{colour_of[ep]}_cube").id
        placed = cube_placed(world, target, world.model.body("box").id)

        # Images: compare a few frames with the dataset's decoded video (lossy AV1).
        idx = np.linspace(0, len(rec_state) - 1, 5).round().astype(int)
        decoded = [(ds[lo + i]["observation.images.front"].numpy().transpose(1, 2, 0) * 255).round() for i in idx]
        mae = np.mean([np.abs(fronts[i].astype(np.float64) - d).mean() for i, d in zip(idx, decoded)])
        p = np.mean([psnr(fronts[i], d) for i, d in zip(idx, decoded)])
        rows.append((ep, seed_of[ep], colour_of[ep], err.max(), placed, mae, p))
        print(f"episode {ep} (seed {seed_of[ep]}, {colour_of[ep]:6s}): max |state error| {err.max():.2e} deg, "
              f"placed after replay: {placed}, video vs render: MAE {mae:.2f}/255, PSNR {p:.1f} dB")

    # Control: episode 0's actions, wrong seed.
    lo, hi = int(ds.meta.episodes["dataset_from_index"][0]), int(ds.meta.episodes["dataset_to_index"][0])
    table = ds.hf_dataset.select(range(lo, hi))
    rec_state = np.stack(table["observation.state"]).astype(np.float64)
    states, _ = replay(robot, names, np.stack(table["action"]), args.control_seed)
    err = np.abs(states[:-1] - rec_state)
    world = robot._env._envs[0].unwrapped._world
    placed = cube_placed(world, world.model.body(f"{colour_of[0]}_cube").id, world.model.body("box").id)
    print(f"\nCONTROL: episode 0's actions with seed {args.control_seed}: placed {placed}; max |state error| "
          f"per joint {np.round(err.max(0), 2)} deg ({', '.join(n.removesuffix('.pos') for n in names)})")
    robot.disconnect()

    worst = max(r[3] for r in rows)
    print(f"\n{len(rows)} episodes: worst state error {worst:.2e} deg, placed after replay "
          f"{sum(r[4] for r in rows)}/{len(rows)}, video PSNR {np.mean([r[6] for r in rows]):.1f} dB mean")


if __name__ == "__main__":
    main()
