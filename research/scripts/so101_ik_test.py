"""Milestone 4.2b part 1: test the top-down IK on every cube, over several seeds.

For each cube: solve IK for the grasp pose (tool point at the cube centre), find the highest reachable
top-down pre-grasp (capped at 6 cm), then check the pre-grasp physically: command it through env.step and
measure where the tool point actually ends up after the servos settle (gravity, torque limits, tracking).

    unset PYTHONPATH && uv run --no-sync python research/scripts/so101_ik_test.py
"""

import mujoco
import numpy as np

from so101_expert import TOOL_OFFSET, SO101Kinematics, body_yaw, max_topdown_height
from so101_viewer import load_env_module

SEEDS = range(5)


def main():
    env_module = load_env_module()
    print(f"{'seed':>4s} {'cube':11s} {'r [m]':>6s} {'grasp':>5s} {'it':>3s} {'pos mm':>7s} {'ang deg':>7s} "
          f"{'roll deg':>8s} {'pre h cm':>8s} {'phys err mm':>11s} {'tilt deg':>8s}")
    n_ok = n = 0
    phys = []
    for seed in SEEDS:
        for name in env_module.CUBE_BODIES:
            env = env_module.SO101Env(**env_module.DEFAULTS)
            env.reset(seed=seed)
            world = env._world
            kin = SO101Kinematics(world.model)
            q_home = world.data.qpos[kin.qadr].copy()
            body = world.model.body(name).id
            cube, yaw = world.data.xpos[body].copy(), body_yaw(world.data, body)

            grasp = kin.ik_grasp(cube, yaw, q_home)
            h, pre = max_topdown_height(kin, cube, yaw, grasp.q)

            # Physical check: hold the pre-grasp for 2 s through the env's own action interface.
            err = tilt = float("nan")
            if pre is not None:
                action = np.concatenate([np.rad2deg(pre.q), [50.0]]).astype(np.float32)
                for _ in range(60):
                    env.step(action)
                R = world.data.site_xmat[kin.site].reshape(3, 3)
                actual = world.data.site_xpos[kin.site] + R @ TOOL_OFFSET
                err = 1e3 * np.linalg.norm(actual - (cube + [0, 0, h]))
                tilt = np.rad2deg(np.arccos(np.clip(-R[2, 0], -1, 1)))
                phys.append(err)
            env.close()

            n += 1
            n_ok += grasp.converged and pre is not None
            print(f"{seed:4d} {name:11s} {np.hypot(*cube[:2]):6.3f} {'ok' if grasp.converged else 'NO':>5s} "
                  f"{grasp.iters:3d} {1e3 * grasp.pos_err:7.3f} {np.rad2deg(grasp.ang_err):7.3f} "
                  f"{np.rad2deg(grasp.q[4]):8.1f} {100 * h:8.1f} {err:11.2f} {tilt:8.2f}")
    print(f"\ngrasp + pre-grasp solved: {n_ok}/{n}   physical tool-point error: "
          f"mean {np.mean(phys):.2f} mm, max {np.max(phys):.2f} mm")


if __name__ == "__main__":
    main()
