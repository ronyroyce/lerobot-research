"""Milestone 4.2a: check, without hardware, that an SO-101 leader arm can drive the MuJoCo sim robot.

Builds the leader teleoperator exactly as `--teleop.type=so101_leader` does (no port is opened until
connect()), builds the sim robot, and checks the interface they must agree on:
  1. action names: leader.action_features == sim_robot.action_features
  2. units: leader arm joints in degrees (use_degrees=True), gripper 0-100, same as env.py's to_ctrl
  3. what "0 degrees" means on the real arm, using the bus's own normalisation on a made-up calibration

    unset PYTHONPATH && uv run --no-sync python research/scripts/so101_leader_readiness.py
"""

import tempfile
from pathlib import Path

from lerobot.envs import HubEnvConfig
from lerobot.motors import MotorCalibration
from lerobot.robots.sim_robot import SimRobot, SimRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

HUB_PATH = "lerobot/so101-tabletop-mujoco@4684f500b94d07194f2c860314c4e27cb9349e8c"


def main():
    # A throwaway calibration dir so this check writes nothing to ~/.cache/huggingface/lerobot.
    with tempfile.TemporaryDirectory() as calib_dir:
        leader = SOLeader(
            SOLeaderTeleopConfig(port="/dev/ttyACM0", id="research_leader", calibration_dir=Path(calib_dir))
        )
        robot = SimRobot(SimRobotConfig(env=HubEnvConfig(hub_path=HUB_PATH, trust_remote_code=True)))

        # --- 1. names ---
        teleop_names, robot_names = list(leader.action_features), list(robot.action_features)
        print(f"leader actions   : {teleop_names}")
        print(f"sim robot actions: {robot_names}")
        print(f"[{'PASS' if teleop_names == robot_names else 'FAIL'}] action names and order match")

        # --- 2. units ---
        modes = {name: motor.norm_mode.value for name, motor in leader.bus.motors.items()}
        print(f"\nleader norm modes: {modes}")
        ok = all(m == "degrees" for n, m in modes.items() if n != "gripper") and modes["gripper"] == "range_0_100"
        print(f"[{'PASS' if ok else 'FAIL'}] arm joints in degrees, gripper 0-100 (env.py's units)")
        print(f"robot_type for recordings: {robot.robot_type} (same as a real so_follower dataset)")

        # --- 3. what does 0 deg mean on hardware? ---
        # DEGREES: (raw - mid) * 360 / 4095 where mid = middle of the range recorded at calibration.
        # So 0 deg is the middle of each joint's calibrated travel, not a CAD zero. Made-up example:
        bus = leader.bus
        bus.calibration = {
            name: MotorCalibration(id=m.id, drive_mode=0, homing_offset=0, range_min=1000, range_max=3000)
            for name, m in bus.motors.items()
        }
        raw = {1: 2000, 2: 2512, 6: 1500}  # shoulder_pan at mid, shoulder_lift +512 ticks, gripper a quarter open
        norm = bus._normalize(raw)
        print("\nexample calibration range_min=1000, range_max=3000 (mid 2000), STS3215 = 4096 ticks/turn:")
        print(f"  shoulder_pan  raw 2000 -> {norm[1]:+.2f} deg   (mid of range = 0 deg)")
        print(f"  shoulder_lift raw 2512 -> {norm[2]:+.2f} deg   (512 ticks = 45 deg)")
        print(f"  gripper       raw 1500 -> {norm[6]:.1f}       (fraction of calibrated range x 100)")
        robot.disconnect()


if __name__ == "__main__":
    main()
