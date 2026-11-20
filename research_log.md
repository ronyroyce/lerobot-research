# Research Log — LeRobot Research Fork

Milestone structure follows `motive.md`. Each entry records the date, what was decided or done, why, and how to reproduce it.

---

## Machine snapshot (verified 2026-11-19)

| Item | Value |
| --- | --- |
| OS | Ubuntu 24.04.3 LTS |
| System Python | 3.12.3 (`/usr/bin/python3`) |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB VRAM, driver 595.99.02 |
| uv | 0.13.9 (`~/.local/bin`) |
| Project venv (`.venv`) | created by `uv sync --locked` (Milestone 2) |
| Git | `origin` = ronyroyce/lerobot-research, `upstream` = huggingface/lerobot (push DISABLED), on `main` |

---

## Git workflow (set 2026-11-20)

```
main ────────────── mirror of huggingface/lerobot (never commit research work here)
  └─ research/main ─ integration branch; each verified milestone is merged in
       └─ research/mN-<topic> ─ one branch per milestone, merged back by a PR inside the fork
```

- `origin` = `git@github.com:ronyroyce/lerobot-research.git` (SSH, key `~/.ssh/id_ed25519`). `upstream` = huggingface/lerobot with push DISABLED.
- PRs are opened through `https://github.com/ronyroyce/lerobot-research/compare/research/main...<branch>` and merged with **Create a merge commit**, which keeps each milestone boundary visible in the history.
- ⚠️ The fork's "Compare & pull request" banner defaults its base to `huggingface/lerobot`. Always check the **base repository** before clicking Create.
- Pulling upstream updates: `git fetch upstream` → on `main`: `git merge upstream/main` → on `research/main`: `git merge main`. This stays conflict-free because research work lives only in `research/` and `research_log.md`.
- History: M2 was merged through PR #1 (merge commit `9f437d0c9`).

---

## Milestone 1 — Understand LeRobot's Architecture
Status: in progress (studied in a separate learning chat). Simulator located and reviewed.

- 2026-11-19 — **Found the SO-101 simulator.** It is not in `src/lerobot/sim/`. It lives on the Hub as the EnvHub repo [`lerobot/so101-tabletop-mujoco`](https://huggingface.co/lerobot/so101-tabletop-mujoco) (see `docs/source/simulation.mdx`). LeRobot can load it in two ways:
  - as a **batched env** for evaluation: `--env.type=envhub --env.hub_path=lerobot/so101-tabletop-mujoco`
  - as a **sim robot**, which sits behind the same `Robot` interface as the real follower: `--robot.type=sim --robot.env.type=envhub --robot.env.hub_path=... --robot.env.trust_remote_code=true`. A `so101_leader` teleoperator can drive it, and recordings get `robot_type=so_follower`.
  - Its code is downloaded and run locally (`trust_remote_code`), so **read its source before running it**.
- Alternative: LeIsaac (IsaacLab) SO-101 tasks via EnvHub (`docs/source/envhub_leisaac.mdx`). It is heavier (needs Isaac Sim), so it is deferred.

## Milestone 2 — Establish a Working Development Environment
Status: done (2026-11-19).

- 2026-11-19 — **ROS 2 / venv isolation.** `~/.bashrc` sources ROS 2 Jazzy, which exports a `PYTHONPATH` pointing to `/opt/ros/jazzy/...` site-packages. `PYTHONPATH` overrides venv isolation, and ROS also uses Python 3.12, so the clash would go unnoticed. **Rule:** use a dedicated LeRobot terminal and run `unset PYTHONPATH` first. Integrate with ROS later through explicit interfaces, not shared site-packages.
- 2026-11-19 — **Installed uv 0.13.9** (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- 2026-11-19 — **Base env:** `unset PYTHONPATH && uv sync --locked`. Verified: venv Python 3.12.3, lerobot 0.7.1 (editable, `src/lerobot`), torch 2.15.0+cu132 (CUDA 13.2 runtime shipped as pip wheels, no system CUDA toolkit needed), `torch.cuda.is_available()` = True on the RTX 4060, GPU matmul OK, no `/opt/ros` entries in `sys.path`.
- 2026-11-19 — **SO-101 env source reviewed** (`lerobot/so101-tabletop-mujoco`):
  - Pinned commit `4684f500b94d07194f2c860314c4e27cb9349e8c`. `env.py` and `requirements.txt` are byte-identical to HEAD `e68145d` (later commits changed only the README).
  - Code review: plain Gymnasium + MuJoCo. The only network access is `snapshot_download` of the same repo. No subprocess calls and no file writes.
  - Interface: observations are `pixels.front` and `pixels.wrist` (480×640 RGB) plus `agent_pos` (6 joint values). Actions are 6 absolute joint targets: degrees for the arm and 0–100 for the gripper, matching `so101_follower` with `use_degrees=True`. 30 fps, and the physics timestep is adjusted so each control period is a whole number of physics steps. Reward = fraction of the 4 cubes in the box. Truncated at 3600 steps. Seeded randomization of cubes, box, front camera and joints.
  - Declares `LEROBOT_SIM` (fps, control_mode, semantics `so101-joint-degrees-v1`, joint names, `robot_type=so_follower`). LeRobot uses this to check that the env matches the robot or teleoperator.
  - Requires `mujoco==3.3.7`. **Conflict:** LeRobot's lockfile has `mujoco 3.8.1` (pulled in through the aloha/libero/metaworld extras). We use the env's pin so the physics matches the author's, for reproducibility.
- 2026-11-19 — **Decision: run in-process (local), not served.** Docker isn't installed, and in-process is easier to debug and modify (needed for Milestone 5). Serving through Docker (`lerobot-env-server`) can be revisited later for isolation.
- 2026-11-19 — **Decision: fork-specific dependencies go in `research/requirements-so101-sim.txt`**, not `pyproject.toml`/`uv.lock`, so upstream merges stay conflict-free. Caveat: a plain `uv sync` removes these packages, so reinstall them after every sync.
- 2026-11-19 — **Smoke test passed** (`research/scripts/so101_smoke_test.py`, seed 0):
  - `HubEnvConfig` → `make_task_env` gives a `LocalVectorEnv`. LeRobot renames the raw keys: `agent_pos` → `observation.state` (1, 6) float32, and `pixels.{front,wrist}` → `observation.images.{front,wrist}` (1, 480, 640, 3) uint8. `info['task']` = "Put all the cubes in the green box."
  - Reset joints `[0.55, -60.92, 58.16, 33.07, 1.25, 31.65]` vs HOME `[0, -60, 60, 35, 0, 30]`, i.e. within the ±2° joint noise.
  - Holding the state as the action for 60 steps: max drift 0.027°. This confirms absolute position control and consistent units.
  - Speed: 44.5 steps/s with both cameras rendered at 480×640, about 1.5× real time (30 fps).
  - Front and wrist frames saved to `outputs/m2_smoke_test/` (gitignored) and inspected visually: OK.

**Reproduce Milestone 2 from a fresh clone:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once
cd ~/projects/lerobot-research
unset PYTHONPATH                                   # ROS 2 isolation, in every LeRobot terminal
uv sync --locked
uv pip install -r research/requirements-so101-sim.txt
uv run --no-sync python research/scripts/so101_smoke_test.py
```

**Milestone 2: DONE (base env).** Extras for datasets, visualisation and teleoperation (`dataset`, `viz`, `feetech`) will be added in Milestone 4 when they're actually needed.

## Milestone 3 — Run an Existing Simulation Baseline
Status: in progress (branch `research/m3-sim-baseline`).

- 2026-11-20 — **3.1 Interactive viewer** (`research/scripts/so101_viewer.py`): opens `mujoco.viewer.launch` on the env's own MjModel/MjData after a seeded reset (`--check` prints a summary without a window). The physics step is 4.762 ms × 7 substeps = 1/30 s: Menagerie's 5 ms step, adjusted by env.py to divide the control period exactly. 10 joints = 6 arm hinges + 4 cube free joints. Viewer sliders write raw actuator `ctrl` in **radians**, below the env's degree/0–100 interface (`to_ctrl`). Checked visually by the user: arm, gripper, perturbations, contacts and cameras all behave plausibly.
- 2026-11-20 — **3.2 Model inspection** (`research/scripts/so101_inspect_model.py`) → baseline file `research/baselines/so101_tabletop_model.json` (all joints, actuators, bodies, collision geoms, cameras and solver options; diff against it in M5).
  - Solver: implicitfast integrator, Newton solver, elliptic friction cone, gravity −9.81.
  - **All 6 joints share one STS3215 servo model** (Menagerie class `sts3215`): damping 0.60, armature 0.028, frictionloss 0.052. Position actuator kp = 998.22 N·m/rad, kv = 2.731, **forcerange ±2.94 N·m**.
  - ⇒ Saturation: kp·error reaches 2.94 N·m once the error exceeds 2.94/998 rad ≈ **0.17°**. For any real move the actuator is **torque-saturated** (bang-bang-like), and kp only matters in the final ~0.17°. *Prediction to test in 3.3:* the unloaded top speed is set by torque/(damping + kv) ≈ 2.94/3.33 ≈ 0.88 rad/s ≈ **50°/s**, modified by gravity on the shoulder and elbow.
  - Masses: cubes 10 g (3 cm, friction 1.0/0.03/0.003, condim 4), box 93 g (5 static geoms), arm links 79–147 g. Gripper collision is approximated by primitive boxes, spheres and capsules (condim 6, solref 0.01).
  - Cameras: `front` (world, fovy 60°), `top` (world, 60°, not used in observations), `wrist_cam` (on `camera_mount`, 61°).
  - **Sim-to-real notes (for M8):**
    1. Menagerie's comment: the gains "are not a 1-to-1 mapping of the servo gains used in LeRobot". They were derived assuming servo P = 16. LeRobot's `so_follower` does write P = 16, but also **D = 32** and I = 0 (`config_so_follower.py:45-47`). The sim's kv was not derived from D = 32.
    2. The real follower limits the **gripper to 50 % torque** (`Max_Torque_Limit=500`, `so_follower.py:155`). The sim gripper has the full ±2.94 N·m, so **sim grasps are stronger than real ones**.
    3. Menagerie defines a ±0.5° backlash class but applies it to no joint. The real servos have backlash; the sim has none.
- 2026-11-20 — **3.3 Step response** (`research/scripts/so101_step_response.py` → `research/results/m3_step_response.{json,png}`). Seed 0; ±20° steps through the env's `to_ctrl`; 210 Hz trace (one physics substep per sample); 2 s per test.

  | joint | step | rise 10–90 % | peak speed | overshoot | settle (±0.5°) | ss error | others moved |
  |---|---|---|---|---|---|---|---|
  | shoulder_pan | ±20° | 0.090 s | 223.5 °/s | 18.6 % | 0.31 s | 0.000° | 0.04° |
  | shoulder_lift | +20° | 0.090 s | 233.6 °/s | 22.0 % | 0.33 s | +0.013° | 0.05° |
  | shoulder_lift | −20° | 0.095 s | 218.1 °/s | 18.3 % | 0.32 s | +0.003° | 0.06° |
  | elbow_flex | +20° | 0.086 s | 242.2 °/s | 10.1 % | 0.22 s | +0.002° | 0.32° |
  | elbow_flex | −20° | 0.105 s | 193.9 °/s | 16.7 % | 0.34 s | +0.026° | 0.04° |

  - **My 3.2 prediction (~50 °/s) was wrong by about 4×.** Wrong assumption: kv brakes the motion. In MuJoCo, `forcerange` clips the *whole* actuator force, kp·(ctrl−q) − kv·q̇, and kp·error (≈350 N·m for 20°) is far above 2.94 N·m. So while saturated, **kv has no effect**, and only the passive joint damping (0.60) and frictionloss (0.052) resist. Corrected terminal speed = (2.94 − 0.052)/0.60 = 4.81 rad/s = **275.8 °/s**. Checked with a 90° shoulder_pan move: **274.1 °/s** measured (0.6 % off). The 20° moves are too short to reach top speed, hence ~220 °/s.
  - **Saturated (bang-bang) behaviour, visible in the torque plot:** full +2.94 N·m to accelerate, then full −2.94 N·m to brake. Because braking torque is also capped, the joint can't stop in time → **10–22 % overshoot** and 1–2 ringing cycles, settling in ~0.3 s. This is far from the critically damped response that kp = 998 might suggest.
  - **From the policy's point of view (30 Hz):** a 20° step reaches 90 % in under 3 frames, and the overshoot and ringing last about 10 frames. A policy that sends large jumps in target will see oscillation in `observation.state`. Smooth trajectories (action chunking, as ACT does) avoid this.
  - Gravity changes the response only slightly: shoulder_lift ±20° differ by ~7 % in peak speed, and elbow_flex is the most asymmetric (+20°: faster, less overshoot; −20°: slower, more ringing). Coupling: an elbow_flex +20° step moves the other joints by up to 0.32°.
  - Steady-state error is almost zero (≤ 0.026°): kp = 998 holds against gravity once the joint leaves saturation.
  - **Sim-to-real (M8):** the real STS3215 at P=16/D=32 almost certainly does NOT ring like this (its D term and internal speed and acceleration limits damp it). Running this same test on hardware will be the first sim-vs-real measurement.

## Milestone 4 — Understand the Robot-Learning Pipeline
Status: not started.

## Milestone 5 — First Controlled Customisation
Status: not started.

## Milestone 6 — Train and Evaluate a Robot-Learning Policy
Status: not started.

## Milestone 7 — Advanced Robot Learning Research
Status: not started.

## Milestone 8 — Hardware and Sim-to-Real
Status: not started.

## Milestone 9 — Research Documentation and Publication
Status: not started.
