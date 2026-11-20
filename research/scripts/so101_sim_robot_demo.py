"""Milestone 4.1: drive the SO-101 MuJoCo sim through LeRobot's hardware-agnostic `Robot` API.

This builds the same object that `--robot.type=sim --robot.env.type=envhub ...` builds on the command
line (SimRobotConfig -> SimRobot), and uses only the methods every LeRobot robot has:
connect / observation_features / action_features / get_observation / send_action / disconnect.
Swap the config for SOFollowerConfig(port=...) and the loop below would drive the real arm.

    unset PYTHONPATH && uv run --no-sync python research/scripts/so101_sim_robot_demo.py
"""

import time

import numpy as np

from lerobot.envs import HubEnvConfig
from lerobot.robots.sim_robot import SimRobot, SimRobotConfig

HUB_PATH = "lerobot/so101-tabletop-mujoco@4684f500b94d07194f2c860314c4e27cb9349e8c"
# The real follower's action/state names (so_follower.py: f"{motor}.pos" for its 6 motors).
SO_FOLLOWER_NAMES = [
    f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
]
FPS = 30
SECONDS = 4


def main():
    config = SimRobotConfig(env=HubEnvConfig(hub_path=HUB_PATH, trust_remote_code=True))
    robot = SimRobot(config)

    # --- 1. The contract, readable before connecting (scripts use it to build dataset features) ---
    print(f"robot_type        : {robot.robot_type}   (stored in a recorded dataset's metadata)")
    print(f"action_features   : {list(robot.action_features)}")
    print(f"observation_feat. : {robot.observation_features}")
    print(f"cameras           : {robot.cameras}")
    same = list(robot.action_features) == SO_FOLLOWER_NAMES
    print(f"action names == real so_follower's: {same}")

    # --- 2. connect() starts the first episode; start_episode(seed) makes it repeatable ---
    robot.connect()
    robot.start_episode(seed=0)
    obs = robot.get_observation()
    start = {name: obs[name] for name in robot.action_features}
    print("\nobservation keys  :", list(obs))
    print("start state       :", {k: round(v, 2) for k, v in start.items()})

    # get_observation() does NOT advance the simulation: two reads give identical state.
    again = robot.get_observation()
    print("re-read identical :", all(again[k] == obs[k] for k in start))

    # --- 3. Control loop at FPS: each send_action() advances the sim by exactly one 1/30 s step ---
    # Smooth targets: pan swings ±25 deg, the gripper opens and closes, the rest hold.
    n = FPS * SECONDS
    track = []
    t0 = time.perf_counter()
    for k in range(n):
        phase = 2 * np.pi * k / n
        action = dict(start)
        action["shoulder_pan.pos"] = start["shoulder_pan.pos"] + 25 * np.sin(phase)
        action["gripper.pos"] = 50 + 45 * np.sin(2 * phase)
        sent = robot.send_action(action)  # returns the action actually applied
        obs = robot.get_observation()
        track.append((sent["shoulder_pan.pos"], obs["shoulder_pan.pos"]))
    elapsed = time.perf_counter() - t0

    cmd, meas = np.array(track).T
    lag = np.abs(cmd - meas)
    print(f"\n{n} steps in {elapsed:.2f} s wall clock = {n / elapsed:.1f} steps/s (sim time {SECONDS} s)")
    print(f"pan tracking error: mean {lag.mean():.2f} deg, max {lag.max():.2f} deg")
    print(f"latest reward={robot.latest.reward[0]}, episode_ended={robot.episode_ended}")

    # --- 4. A wrong action dict is rejected with a clear message (what a mismatched teleop would see) ---
    try:
        robot.send_action({"shoulder_pan": 0.0})
    except ValueError as err:
        print(f"\nbad action rejected: {str(err)[:120]}...")

    robot.disconnect()


if __name__ == "__main__":
    main()
