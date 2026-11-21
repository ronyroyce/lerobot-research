"""Milestone 3.3: step response of the SO-101 simulated joints.

For each test, reset to the seeded pose, then command a step of STEP degrees on ONE joint while
holding every other joint at its reset value, through the env's own unit conversion (to_ctrl).
Physics is advanced one substep at a time (same as env.step, which calls mj_step with
nstep=substeps), so we get a 210 Hz trace instead of the 30 Hz the policy sees.

    unset PYTHONPATH && uv run --no-sync python research/scripts/so101_step_response.py

Writes research/results/m3_step_response.{json,png}.

Metrics, per test (angle in the env's degrees):
  rise_time_s      time from 10 % to 90 % of the step
  peak_speed_dps   max |dq/dt| during the move
  overshoot_pct    max excursion past the target, as % of the step size
  settle_time_s    first time after which |error| stays below SETTLE_BAND_DEG
  ss_error_deg     mean error over the last 0.5 s
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from so101_viewer import REVISION, load_env_module

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED = 0
DURATION_S = 2.0
SETTLE_BAND_DEG = 0.5
# (joint, step in degrees). Signs follow the joint axes: we test both directions on the
# gravity-loaded joints to see whether gravity helps or hinders.
TESTS = [
    ("shoulder_pan", +20.0),  # rotation about the vertical axis: no gravity torque
    ("shoulder_pan", -20.0),
    ("shoulder_lift", +20.0),
    ("shoulder_lift", -20.0),
    ("elbow_flex", +20.0),
    ("elbow_flex", -20.0),
]


def run_test(env_module, joint: str, step: float) -> dict:
    env = env_module.SO101Env(**env_module.DEFAULTS)
    env.reset(seed=SEED)
    world = env._world
    model, data = world.model, world.data
    idx = env_module.JOINTS.index(joint)

    start = world.joint_state().astype(np.float64)  # degrees / gripper 0-100, the env's units
    target = start.copy()
    target[idx] += step
    data.ctrl[world.actuators] = world.to_ctrl(target)

    n = round(DURATION_S / model.opt.timestep)
    t = np.arange(1, n + 1) * model.opt.timestep
    q = np.empty(n)
    torque = np.empty(n)
    others_dev = 0.0
    for k in range(n):
        mujoco.mj_step(model, data)
        state = world.joint_state()
        q[k] = state[idx]
        torque[k] = data.actuator_force[world.actuators[idx]]
        others = np.delete(state - start, [idx, 5])  # other arm joints; gripper excluded
        others_dev = max(others_dev, float(np.abs(others).max()))
    env.close()

    progress = (q - start[idx]) / step  # 0 -> 1
    t10 = t[np.argmax(progress >= 0.1)]
    t90 = t[np.argmax(progress >= 0.9)]
    speed = np.abs(np.gradient(q, t))
    err = q - target[idx]
    outside = np.flatnonzero(np.abs(err) > SETTLE_BAND_DEG)
    settle = float(t[outside[-1] + 1]) if outside.size and outside[-1] + 1 < n else (0.0 if not outside.size else None)
    tail = t >= DURATION_S - 0.5
    sat = np.abs(torque) >= 0.99 * model.actuator_forcerange[world.actuators[idx], 1]
    return {
        "joint": joint,
        "step_deg": step,
        "start_deg": round(float(start[idx]), 3),
        "target_deg": round(float(target[idx]), 3),
        "rise_time_s": round(float(t90 - t10), 4),
        "peak_speed_dps": round(float(speed.max()), 2),
        "overshoot_pct": round(float(max(0.0, (progress.max() - 1.0) * 100)), 2),
        "settle_time_s": None if settle is None else round(settle, 4),
        "ss_error_deg": round(float(err[tail].mean()), 4),
        "torque_saturated_s": round(float(sat.sum() * model.opt.timestep), 4),
        "max_other_joint_dev_deg": round(others_dev, 3),
        "_trace": {"t": t, "q": q, "torque": torque, "target": target[idx], "start": start[idx]},
    }


def main():
    env_module = load_env_module()
    results = [run_test(env_module, joint, step) for joint, step in TESTS]

    print(f"{'joint':14s} {'step':>5s} {'rise s':>7s} {'peak °/s':>9s} {'over %':>7s} {'settle s':>9s} "
          f"{'ss err °':>9s} {'sat s':>6s} {'others °':>8s}")
    for r in results:
        print(f"{r['joint']:14s} {r['step_deg']:+5.0f} {r['rise_time_s']:7.3f} {r['peak_speed_dps']:9.1f} "
              f"{r['overshoot_pct']:7.2f} {str(r['settle_time_s']):>9s} {r['ss_error_deg']:+9.4f} "
              f"{r['torque_saturated_s']:6.3f} {r['max_other_joint_dev_deg']:8.3f}")

    # ---- plot: angle (top) and actuator torque (bottom) for every test ----
    fig, (ax_q, ax_tau) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for r in results:
        tr = r["_trace"]
        label = f"{r['joint']} {r['step_deg']:+.0f}°"
        ax_q.plot(tr["t"], (tr["q"] - tr["start"]) / r["step_deg"], label=label)
        ax_tau.plot(tr["t"], tr["torque"], label=label)
    ax_q.axhline(1.0, color="k", lw=0.8, ls="--")
    ax_q.set_ylabel("normalised angle (0 = start, 1 = target)")
    ax_q.legend(fontsize=8, ncol=2)
    ax_q.set_title(f"SO-101 sim step response (seed {SEED}, dt {1e3 * DURATION_S / len(tr['t']):.3f} ms)")
    for lim in (2.94, -2.94):
        ax_tau.axhline(lim, color="k", lw=0.8, ls="--")
    ax_tau.set_ylabel("actuator torque [N·m]")
    ax_tau.set_xlabel("time [s]")
    for ax in (ax_q, ax_tau):
        ax.grid(alpha=0.3)
    ax_q.set_xlim(0, 1.0)
    fig.tight_layout()

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "m3_step_response.png", dpi=120)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    meta = {"revision": REVISION, "mujoco": mujoco.__version__, "seed": SEED, "duration_s": DURATION_S,
            "settle_band_deg": SETTLE_BAND_DEG}
    (RESULTS / "m3_step_response.json").write_text(json.dumps({"meta": meta, "results": clean}, indent=2) + "\n")
    print(f"\nwrote {RESULTS / 'm3_step_response.png'} and .json")


if __name__ == "__main__":
    main()
