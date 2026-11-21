"""Milestone 3.2: dump the physics parameters of the SO-101 tabletop model.

Builds the model exactly as the EnvHub env does (pinned env.py + DEFAULTS), prints the parameters
that matter for control, grasping and sim-to-real, and writes them all to a JSON baseline that later
experiments (Milestone 5) are diffed against.

    unset PYTHONPATH && uv run --no-sync python research/scripts/so101_inspect_model.py

Units: MuJoCo SI (m, kg, s, rad, N, N*m). Body poses are the XML defaults (pre-randomization).
"""

import json
from pathlib import Path

import mujoco
import numpy as np

from so101_viewer import REVISION, load_env_module

# Anchored to the repo, so the script can be run from any directory.
OUT = Path(__file__).resolve().parents[1] / "baselines" / "so101_tabletop_model.json"


def name(model, obj, i):
    return mujoco.mj_id2name(model, obj, i) or f"<unnamed {i}>"


def r(x, nd=5):
    """Round numpy scalars/arrays into plain JSON-friendly lists/floats."""
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def main():
    env_module = load_env_module()
    env = env_module.SO101Env(**env_module.DEFAULTS)
    env.reset(seed=0)
    world = env._world
    m = world.model
    J = mujoco.mjtJoint
    joint_types = {J.mjJNT_FREE: "free", J.mjJNT_BALL: "ball", J.mjJNT_SLIDE: "slide", J.mjJNT_HINGE: "hinge"}

    opt = m.opt
    report = {
        "source": {"repo": "lerobot/so101-tabletop-mujoco", "revision": REVISION, "mujoco": mujoco.__version__},
        "option": {
            "timestep": float(opt.timestep),
            "substeps_per_env_step": int(world.substeps),
            "integrator": mujoco.mjtIntegrator(opt.integrator).name,
            "solver": mujoco.mjtSolver(opt.solver).name,
            "iterations": int(opt.iterations),
            "tolerance": float(opt.tolerance),
            "cone": mujoco.mjtCone(opt.cone).name,
            "impratio": float(opt.impratio),
            "gravity": r(opt.gravity),
            "noslip_iterations": int(opt.noslip_iterations),
        },
        "joints": [],
        "actuators": [],
        "bodies": [],
        "geoms": [],
        "cameras": [],
    }

    for j in range(m.njnt):
        dof = m.jnt_dofadr[j]
        report["joints"].append(
            {
                "name": name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                "type": joint_types[m.jnt_type[j]],
                "limited": bool(m.jnt_limited[j]),
                "range_rad": r(m.jnt_range[j]),
                "range_deg": r(np.rad2deg(m.jnt_range[j]), 2),
                "damping": float(m.dof_damping[dof]),
                "armature": float(m.dof_armature[dof]),
                "frictionloss": float(m.dof_frictionloss[dof]),
            }
        )

    for a in range(m.nu):
        # Position actuator: force = gain*ctrl + bias0 + bias1*qpos + bias2*qvel
        #                          = kp*(ctrl - qpos) - kv*qvel   with bias = [0, -kp, -kv]
        report["actuators"].append(
            {
                "name": name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a),
                "joint": name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0]),
                "kp": float(m.actuator_gainprm[a, 0]),
                "kv": float(-m.actuator_biasprm[a, 2]),
                "ctrlrange_rad": r(m.actuator_ctrlrange[a]),
                "forcelimited": bool(m.actuator_forcelimited[a]),
                "forcerange": r(m.actuator_forcerange[a]),
            }
        )

    for b in range(1, m.nbody):
        report["bodies"].append(
            {
                "name": name(m, mujoco.mjtObj.mjOBJ_BODY, b),
                "parent": name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b]),
                "mass_kg": round(float(m.body_mass[b]), 5),
                "inertia_diag": r(m.body_inertia[b], 7),
            }
        )

    for g in range(m.ngeom):
        # Only collision geoms take part in contact: contype/conaffinity non-zero.
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        report["geoms"].append(
            {
                "name": name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
                "body": name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]),
                "type": mujoco.mjtGeom(m.geom_type[g]).name.removeprefix("mjGEOM_"),
                "size": r(m.geom_size[g]),
                "friction": r(m.geom_friction[g]),  # sliding, torsional, rolling
                "condim": int(m.geom_condim[g]),
                "solref": r(m.geom_solref[g]),
                "solimp": r(m.geom_solimp[g]),
            }
        )

    for c in range(m.ncam):
        report["cameras"].append(
            {
                "name": name(m, mujoco.mjtObj.mjOBJ_CAMERA, c),
                "body": name(m, mujoco.mjtObj.mjOBJ_BODY, m.cam_bodyid[c]),
                "fovy_deg": float(m.cam_fovy[c]),
                "pos": r(world.cam_pos0[c]),
            }
        )
    env.close()

    # ---- human-readable summary ----
    o = report["option"]
    print(f"== solver: {o['integrator']} integrator, {o['solver']} solver, {o['cone']} cone, "
          f"dt={o['timestep'] * 1e3:.3f} ms x {o['substeps_per_env_step']}, gravity {o['gravity']}")
    print("\n== arm joints (hinge) and their position actuators")
    print(f"{'joint':14s} {'range deg':>18s} {'damp':>6s} {'armat':>6s} {'fric':>6s} {'kp':>6s} {'kv':>5s} {'force±':>7s}")
    acts = {a["joint"]: a for a in report["actuators"]}
    for jt in report["joints"]:
        if jt["type"] != "hinge":
            continue
        a = acts.get(jt["name"], {})
        print(f"{jt['name']:14s} {str(jt['range_deg']):>18s} {jt['damping']:6.2f} {jt['armature']:6.3f} "
              f"{jt['frictionloss']:6.3f} {a.get('kp', 0):6.2f} {a.get('kv', 0):5.2f} {a.get('forcerange', [0, 0])[1]:7.2f}")
    print("\n== bodies (mass kg)")
    for b in report["bodies"]:
        print(f"  {b['name']:22s} {b['mass_kg']:8.4f}   parent={b['parent']}")
    print("\n== collision geoms on cubes, box and gripper (friction = sliding, torsional, rolling)")
    for g in report["geoms"]:
        if any(k in g["body"] for k in ("cube", "box", "jaw", "gripper")):
            print(f"  {g['body']:18s} {g['name']:24s} {g['type']:8s} size={g['size']} friction={g['friction']} condim={g['condim']}")
    print("\n== cameras")
    for c in report["cameras"]:
        print(f"  {c['name']:10s} on {c['body']:16s} fovy={c['fovy_deg']:.1f} deg")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nfull report: {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT} ({len(report['geoms'])} collision geoms, {len(report['bodies'])} bodies)")


if __name__ == "__main__":
    main()
