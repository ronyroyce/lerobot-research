"""Milestone 4.3: record scripted-expert demonstrations as a LeRobotDataset.

Same dataset path as `lerobot-record` (src/lerobot/scripts/lerobot_record.py), step for step:
    robot features -> create_initial_features -> aggregate_pipeline_dataset_features (default processors)
    -> LeRobotDataset.create(robot_type=robot.robot_type) -> per step: get_observation, send_action,
    build_dataset_frame(obs) + build_dataset_frame(action) + task -> add_frame -> save_episode -> finalize
The only difference: actions come from the scripted expert instead of a teleoperator, because the expert
needs privileged sim state (cube poses) that no teleoperator can see (research_log.md, M4 4.2b).

Episode design (decided in 4.2b): one cube per episode, cycling red/green/blue/yellow, task text naming the
colour; the sim is reset with a new seed every episode. An episode ends TAIL_S after the cube is placed (in
the box, released, at rest); an episode that is not placed within MAX_STEPS is discarded and logged.

    unset PYTHONPATH && uv run --no-sync python research/scripts/so101_record_expert.py --num_episodes 10
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.envs import HubEnvConfig
from lerobot.processor import make_default_processors
from lerobot.robots.sim_robot import SimRobot, SimRobotConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from so101_expert import PickPlaceExpert, cube_placed
from so101_viewer import REVISION

HUB_PATH = f"lerobot/so101-tabletop-mujoco@{REVISION}"
COLOURS = ("red", "green", "blue", "yellow")
FPS = 30
MAX_STEPS = 300  # 10 s; the expert places at ~202
TAIL_S = 0.5  # keep recording after placement so demos show the jaws clearing the box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--repo_id", default="ronyroyce/so101_sim_expert_v0")
    parser.add_argument("--root", default="outputs/datasets/so101_sim_expert_v0")
    args = parser.parse_args()
    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"{root} exists: pick a new --root (datasets are never overwritten silently)")

    robot = SimRobot(SimRobotConfig(env=HubEnvConfig(hub_path=HUB_PATH, trust_remote_code=True)))
    # Exactly lerobot-record's feature construction (identity processors by default).
    teleop_proc, _, obs_proc = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc, initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc, initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    dataset = LeRobotDataset.create(
        args.repo_id, FPS, root=root, robot_type=robot.robot_type, features=features, use_videos=True,
        image_writer_threads=4 * len(robot.cameras),
    )
    names = list(robot.action_features)
    log = []
    robot.connect()
    t0 = time.perf_counter()
    try:
        saved = 0
        for ep in range(args.num_episodes):
            seed = args.seed_start + ep
            colour = COLOURS[ep % len(COLOURS)]
            task = f"Put the {colour} cube in the green box."
            robot.start_episode(seed=seed)
            # The SimRobot's in-process env: LocalVectorEnv -> gym env -> SO101Env -> its MuJoCo world.
            world = robot._env._envs[0].unwrapped._world
            env_module_bodies = ("red_cube", "green_cube", "blue_cube", "yellow_cube")
            expert = PickPlaceExpert(world, env_module_bodies, "box")
            target = env_module_bodies.index(f"{colour}_cube")
            others0 = {j: world.data.xpos[expert.cubes[j]][:2].copy() for j in range(4) if j != target}
            plan = expert.pick_and_place(target)

            action, placed_at, steps = None, None, 0
            while steps < MAX_STEPS:
                obs = robot.get_observation()
                action = next(plan, action)  # expert exhausted -> hold the last command
                if action is None:
                    break  # the plan was skipped (IK failure): nothing to send
                act = dict(zip(names, map(float, action), strict=True))
                robot.send_action(act)
                frame = {
                    **build_dataset_frame(dataset.features, obs, prefix=OBS_STR),
                    **build_dataset_frame(dataset.features, act, prefix=ACTION),
                    "task": task,
                }
                dataset.add_frame(frame)
                steps += 1
                if placed_at is None and cube_placed(world, expert.cubes[target], expert.box):
                    placed_at = steps
                if placed_at is not None and steps - placed_at >= round(TAIL_S * FPS):
                    break

            bump = max(np.linalg.norm(world.data.xpos[expert.cubes[j]][:2] - p) for j, p in others0.items())
            entry = {"episode_seed": seed, "colour": colour, "task": task, "steps": steps,
                     "placed_at": placed_at, "max_other_cube_disp_cm": round(100 * float(bump), 2)}
            if placed_at is None:
                dataset.clear_episode_buffer()
                entry["saved_as"] = None
                print(f"episode {ep}: NOT placed in {steps} steps, discarded ({colour}, seed {seed})")
            else:
                dataset.save_episode()
                entry["saved_as"] = saved
                print(f"episode {ep} -> dataset episode {saved}: {colour:6s} seed {seed:3d} "
                      f"placed at {placed_at}, {steps} frames, other cubes moved <= {entry['max_other_cube_disp_cm']} cm")
                saved += 1
            log.append(entry)
    finally:
        dataset.finalize()
        robot.disconnect()
    dt = time.perf_counter() - t0
    frames = sum(e["steps"] for e in log if e["saved_as"] is not None)
    print(f"\nsaved {saved}/{args.num_episodes} episodes, {frames} frames ({frames / FPS:.0f} s of demos) "
          f"in {dt:.0f} s wall clock -> {root}")
    # Expert-side metadata (seeds, placement, disturbance) next to the dataset, not inside it.
    side = root.parent / f"{root.name}_expert_log.json"
    side.write_text(json.dumps({"hub_path": HUB_PATH, "fps": FPS, "episodes": log}, indent=2) + "\n")
    print(f"expert log: {side}")


if __name__ == "__main__":
    main()
