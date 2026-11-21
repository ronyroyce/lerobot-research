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
uv sync --locked --extra feetech --extra core_scripts --inexact   # extras added in M4
uv pip install -r research/requirements-so101-sim.txt
uv run --no-sync python research/scripts/so101_smoke_test.py
```

**Milestone 2: DONE (base env).** Extras for datasets, visualisation and teleoperation (`dataset`, `viz`, `feetech`) will be added in Milestone 4 when they're actually needed.

## Milestone 3 — Run an Existing Simulation Baseline
Status: **complete** (branch `research/m3-sim-baseline`, merged into `research/main` by PR #2).

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
Status: **complete** 2026-11-21 (branch `research/m4-data-pipeline`).

Plan: 4.1 sim robot via the `Robot` API → 4.2 action sources (a: leader-arm readiness, b: scripted IK expert as a `Teleoperator`) → 4.3 record a dataset → 4.4 inspect it → 4.5 replay and determinism → 4.6 policy inference path.
**Decision (2026-11-20):** an SO-101 leader arm exists but isn't available yet. We record with a scripted expert (option B) now, and keep the leader-arm route (option A) ready and checked without hardware.

- 2026-11-20 — **4.1 Sim robot** (`research/scripts/so101_sim_robot_demo.py`). `SimRobotConfig(env=HubEnvConfig(...))` → `SimRobot`, the same object `--robot.type=sim --robot.env.type=envhub` builds.
  - `robot_type=so_follower`. `action_features` = the 6 `<motor>.pos` names, **identical to the real `so_follower` and `so_leader`** (both `use_degrees=True`). A leader arm's `get_action()` can therefore feed `send_action()` directly.
  - `get_observation()` does not advance time (two reads are identical). `send_action()` advances exactly one 1/30 s step. Simulated time is decoupled from wall clock (44 steps/s here), so `--fps` must equal the env's 30.
  - Smooth ±25° pan sine over 4 s (peak ~39 °/s): tracking error mean 0.22°, max 0.66°, with no ringing, unlike the 3.3 steps. ⇒ The scripted expert must send smooth, interpolated targets.
  - Seed 0 start state identical to M2's through a different code path. Wrong action names are rejected at the first `send_action()`.
- 2026-11-20 — **4.2a Leader-arm readiness (option A), checked without hardware.**
  - Installed `uv sync --locked --extra feetech --extra core_scripts --inexact` (22 packages: rustypot, pyserial, datasets, pyarrow, av, torchcodec, pynput, rerun-sdk…). **`--inexact` is essential**: a plain sync removes the mujoco 3.3.7 / matplotlib packages that live outside the lock.
  - Added the user to `dialout` (serial ports are `root:dialout`; without it you get `Permission denied`, which is easy to misread as a cable or driver fault).
  - `research/scripts/so101_leader_readiness.py`: PASS, `SOLeader` action names and order == `SimRobot`'s. PASS, leader joints in degrees, gripper 0–100.
  - Real degree mode = `(raw − mid)·360/4095`, with `mid` the middle of the **calibrated** range. So the leader's 0° = mid of the recorded travel, while the sim's 0° = MJCF zero. The sim's joint ranges are symmetric (so101_new_calib), so they should roughly agree; **measure the per-joint offset when the arm arrives** (M8 input).
  - `lerobot-teleoperate --robot.type=sim … --teleop.type=so101_leader …` parsed, built the sim robot, and failed only at `bus.connect()` on the missing port, as expected.
  - Default settings to watch: `lerobot-teleoperate --fps` defaults to **60** (must be 30 for this sim); `lerobot-record --dataset.push_to_hub` defaults to **true**.
  - Runbook: `research/runbooks/so101_leader_to_sim.md`.
- 2026-11-20 — **4.2b design correction:** the scripted expert can't sit behind the `Teleoperator` interface. A teleoperator only sees what a real robot exposes (joints and images), while the expert needs **privileged sim state** (cube and box poses). Decision: write our own recorder on the same `LeRobotDataset` API that `lerobot-record` uses internally, so the dataset format is identical. The expert reads sim state directly.
- 2026-11-20 — **4.2b part 1: top-down IK** (`research/scripts/so101_expert.py`, test `so101_ik_test.py`).
  - Gripper geometry: in the `gripperframe` site frame, +x = approach direction (toward the tips), z = jaw closing axis (fixed jaw at z = −2 cm, moving jaw at +3 cm when half open). Tool point = site + R·(−1.5, 0, −0.5) cm, i.e. 1.5 cm into the jaws and shifted toward the fixed jaw.
  - 5 arm joints for 6 top-down constraints is still solvable: after the pan joint the arm is planar, so "approach straight down" costs one constraint. Solved with damped least squares on MuJoCo's Jacobian (λ = 0.01) on a private `MjData`. Cube 90° symmetry: try 4 jaw headings and keep the one that converges with the least wrist roll.
  - **Bug 1 (0/20 converged): `mj_jac` returned an all-zero Jacobian.** It needs `mj_comPos` (cdof, subtree_com), and `mj_kinematics` alone doesn't compute them. No error is raised; the giveaway was that wrist roll never moved. After the fix, MuJoCo's Jacobian matches a central finite difference to 3 decimal places.
  - **Finding: a top-down pre-grasp at 8 cm is often unreachable** (9/20 failed). The cause is the **wrist_flex joint limit (±95°)** every time, not arm length. Max top-down height above a cube falls with distance from the base: 9 cm at r = 0.18 m, 4 cm at r = 0.30 m. ⇒ `max_topdown_height()` picks the pre-grasp per cube (≤ 6 cm, 5 mm steps).
  - Result over 5 seeds × 4 cubes: **20/20 grasp and pre-grasp solved** (4–8 iterations, ≤ 0.2 mm / 0.55°). Physical check (hold the pre-grasp 2 s through `env.step`): tool point within **0.29 mm mean, 0.54 mm max** of the target, approach-axis tilt ≤ 0.06°. Gravity sag is negligible at these gains. The smallest margin is seed 0 red (r = 0.302 m), where only a 3.5 cm pre-grasp is possible.
- 2026-11-20 — **4.2b part 2: pick-and-place, 4 cubes. Several attempts, all measured** (`so101_expert.py` `PickPlaceExpert`, `so101_pick_place_test.py`).
  - Gripper command vs jaw gap: 0 → 0.3 cm, 18.5 → 3 cm (one cube width), 35 → 5.4 cm, 100 → 10.5 cm. OPEN = 35, CLOSE = 0 (the cube stops the jaw near 18.5, so the servo squeezes at its 2.94 N·m limit).
  - Motion: minimum-jerk joint segments for free-space moves; straight vertical Cartesian segments (IK every step) for descending and lifting.
  - **Single cube (seed 0): placed, `cubes_in_box: 1`.** The grip is stable: the cube's position inside the gripper doesn't change from closing to release (no slip).
  - Tool depth 1.5 cm → the fingertips hit the table and the tool stops 0.7 cm short. Changed to 0.8 cm.
  - **Attempt A, lower the cube onto the box floor:** "10/10 success", but **misleading**. (1) env.py counts a held cube once it's below 6 cm inside the footprint, so success fired **before the last release**. (2) The closed jaws land on the 3.5 cm walls or on placed cubes (0.8–1.4 cm of room in the 9 cm box), so cubes ended **stacked** (z 4.9 cm = 1.9 + 3.0). The env allows stacked cubes ("resting on another cube").
  - **Attempt B, release above 6 cm (cube centre 6.8 cm):** fixes counting while held, but far-side slots become unreachable top-down (wrist_flex) and 4 cm falls tumble the cubes → 1/10.
  - **Attempt C, low drop (cube bottom 1 mm above the walls):** 3/20. Diagnosis: (1) the far corner slot at r ≥ 0.29 m is unreachable even at 5.1 cm; (2) a systematic +x landing error of 1–3 cm (the in-gripper offset correction was measured with the old tool depth); (3) cubes are still knocked or stacked (seed 2: one cube knocked onto the table, 7.4 cm off).
  - **Conclusion:** 4 cubes in this 9 cm box is a hard placement problem for this gripper (jaw width, wrist reach). Not a good first dataset. **Known env flaw** worth an M5 change: success should require the cubes to be released (not held).
- 2026-11-21 — **4.2b decision (user): one cube per episode, colour varies per episode.** The task text names the colour ("Put the blue cube in the green box."), and the scene is reset with a new seed between episodes. The user's first idea (remove each placed cube and spawn the next) was rejected for two reasons: (1) the box would look empty in every training frame but not at evaluation, a train/test mismatch; (2) cubes vanishing between frames are physically impossible. One cube per episode gives the same simplicity with physically real frames. **M6 consequence:** plain ACT ignores task text, so the plan is ACT on a single-colour subset as a baseline plus a language-conditioned policy (SmolVLA) on all colours. The 4-cube task stays as an M7 generalisation question.
- 2026-11-21 — **4.2b result: single-cube expert** (`PickPlaceExpert.pick_and_place(i)` drops at the box centre; `cube_placed()` = in the box (env.py geometry) + jaws open + at rest < 1 cm/s).
  - **40/40 placed** (10 seeds × 4 colours), 0 skipped, placed at step 201–203 (6.7 s).
  - Cube in gripper after the lift (24 grasps): (0.07, −0.01, −0.78) cm, std ≤ 0.04 mm. The correction was right; the +x landing error is not from carrying.
  - Error by stage: held cube vs target just before release, mean (0.00, −0.02) cm, std ≤ 0.06 cm. Landed vs target, mean (0.71, 0.63) cm, max 2.1 cm. ⇒ **all placement error comes from the release** (jaw swing plus a 1.7 cm fall), well inside the centre drop's 3 cm clearance.
  - Side effect: **3/40 episodes bump a non-target cube** by 0.7–2.1 cm (open jaws touching a neighbour; distance alone doesn't predict it). Kept as a realistic imperfection; tracked as a dataset quality metric.
- 2026-11-21 — **4.3 Recorder** (`research/scripts/so101_record_expert.py`). Mirrors `lerobot-record`'s data path: `create_initial_features` → `aggregate_pipeline_dataset_features` (default processors) → `LeRobotDataset.create(robot_type=…)` → per step `get_observation` → `send_action` → `build_dataset_frame` ×2 + task → `add_frame` → `save_episode` → `finalize`. Actions come from the expert, which reaches its privileged state through `robot._env._envs[0].unwrapped._world`. Each frame = (observation_t, action_t sent in response). No wall-clock pacing (the sim only advances on `send_action`). Episodes that aren't placed within 300 steps are discarded (`clear_episode_buffer`) and logged. 0.5 s tail after placement. Expert metadata goes in a sidecar `<root>_expert_log.json` (seed, colour, placed_at, disturbance of other cubes).
  - **Dataset `so101_sim_expert_v0`**: `outputs/datasets/so101_sim_expert_v0` (gitignored, 28 MB). Recreate with `uv run --no-sync python research/scripts/so101_record_expert.py --num_episodes 10` (seeds 0–9). **10/10 saved, 2172 frames (72 s)**, colours red/green/blue/yellow 3/3/2/2, placed at step 202–203, one episode (seed 8) moved another cube by 1.36 cm.
- 2026-11-21 — **4.4 Dataset anatomy (LeRobotDataset v3.0)**:
  - `meta/info.json`: schema (features with dtype, shape and names), fps 30, `robot_type: so_follower`, totals, path templates, splits. Videos: **AV1**, yuv420p, CRF 30, **GOP g = 2** (keyframe every 2 frames: random access is cheap for training dataloaders, at a cost in file size).
  - `data/chunk-000/file-000.parquet`: one row per frame. `action` (6), `observation.state` (6), `timestamp`, `frame_index` (within the episode), `episode_index`, `index` (global), `task_index`. **No images**: those live in the MP4s.
  - `videos/<camera>/chunk-000/file-000.mp4`: **all episodes concatenated into one video per camera.** `meta/episodes` stores each episode's `from_timestamp` / `to_timestamp` in that file.
  - `meta/tasks.parquet`: task string → `task_index` (4 tasks). `meta/episodes/…parquet`: per-episode length, data row range (`dataset_from_index` / `to_index`), video time range, and per-episode stats.
  - `meta/stats.json`: min/max/mean/std/quantiles per feature, used for policy normalisation. Image stats are per channel, on 0–1 floats.
  - **Finding 1: the expert commands wrist_flex at its +95° limit most of the time** (action q90 = 94.8°, max 95.0°), because top-down grasps need full wrist bend (see 4.2b). A policy will learn to command the hard limit. On hardware (M8) with a calibration offset, that could press into the mechanical stop.
  - **Finding 2: low diversity.** Every episode is 217–218 frames with the same phase timing; only cube and box positions differ. The gripper action is binary (0 or 35). Imitation learning from such uniform demos tends to be brittle (one timing, one strategy). Possible remedies for later datasets: randomised segment durations and speeds, noise injection (DART-style).
  - Task text quirk: "Put the green cube in the green box." uses "green" for both objects, a small ambiguity for language-conditioned policies.
- 2026-11-21 — **4.2b fix: neighbour bumps** (spotted by the user in Rerun: picking red in episode 8 nudged blue).
  - Diagnosis over 40 episodes: 3 bumps (blue ×2, yellow ×1), **all during the descent**.
  - Attempt 1, a hand-estimated open-jaw footprint used to choose among the 4 equivalent headings: bumps 3 → 1, but **placed 40 → 39** (that heading's drop plan was infeasible and there was no fallback) and one bump remained (contact with the `gripper` body, outside the estimated footprint).
  - Attempt 2 (kept): **measure clearance on MuJoCo's real collision geometry** with `mj_geomDistance` (all 24 hand geoms vs the neighbouring cubes' geoms, jaws open, at 4 poses along the descent, on a private synced `MjData`). Choose the heading with clearance ≥ 5 mm and the least wrist roll; **if its drop plan is infeasible, fall back to the next heading**. Result: **40/40 placed, 0 bumps > 5 mm** (max neighbour displacement 1.58 mm).
  - A "0.0 mm" clearance (9/40) means no clear heading had a feasible drop, so the planner fell back to a tight one. Traced deterministically to **`camera_box2`, the wrist-camera housing**, reaching down to a neighbour's height on one side of the gripper. The real SO-101 has the same camera mount, so it's a collision hazard for real grasps too.
  - Spacing is set by the seed (env.py's `randomize`: footprints only need to not overlap; there's no minimum gap). It could be changed through `make_env_kwargs` without editing env.py, but that changes the task distribution. **Kept at the default**; clutter/spacing is a candidate variable for an M5/M7 experiment.
  - v0 dataset (recorded before this fix) is kept as-is.
- 2026-11-21 — **Dataset `so101_sim_expert_v1`** (fixed expert): `uv run --no-sync python research/scripts/so101_record_expert.py --num_episodes 10 --repo_id ronyroyce/so101_sim_expert_v1 --root outputs/datasets/so101_sim_expert_v1` (seeds 0–9). **10/10 saved, 2171 frames**; max displacement of a non-target cube **0.16 cm** (v0: 1.36 cm). User checked episode 8 in Rerun: blue is no longer nudged. **v1 is the reference dataset for 4.5 onward.**
- 2026-11-21 — **4.5 Replay and determinism** (`research/scripts/so101_replay_check.py`).
  - **`lerobot-replay` can't reproduce a sim episode**: it resets the sim robot with `seed=None` (a random layout), since it was designed for real robots whose scene is reset by hand. Our script resets `SimRobot.start_episode(seed)` with the recorded seed (from the sidecar log; **the dataset itself doesn't store seeds**, so the sidecar is essential) and sends the recorded `action` column.
  - **Joint states: bit-exact** (max |error| = 0.00 over all 10 episodes × ~217 frames). **10/10 placed again.** ⇒ seed + action sequence fully determine an episode on this machine (MuJoCo 3.3.7, CPU, single thread). Reproducible experiments are possible; any future non-zero replay error means something changed (MuJoCo version, env commit, threading).
  - **Video vs render: MAE ≈ 1.9/255, PSNR ≈ 38.5 dB** (AV1, CRF 30). The images a policy trains on are slightly lossy, while the sim robot at evaluation gives exact renders: a small train/eval image gap, and video codec noise in general is part of the sim-to-real gap.
  - **Control** (episode 0's actions, seed 100): placed **False**. The arm joints stay within 0.1–1.3° of the recording (differences from the seed's ±2° initial-joint noise), while the **gripper differs by 15°**: it closes to ~0 on empty air instead of stopping at ~15 on a cube. Recorded actions are **open-loop**; without perception they don't transfer to a new layout. This is what the policy's camera inputs are for.
- 2026-11-21 — **4.6 Policy inference path.** Goal: test the **pipeline**, not task success. Plan: a short ACT run on v1, inspect the checkpoint, then roll it out on the sim robot with `lerobot-rollout`.
  - Installed the `training` extra: `uv sync --locked --extra feetech --extra core_scripts --extra training --inexact` (adds accelerate and trackio; mujoco stays at 3.3.7 thanks to `--inexact`). ACT needs no extra of its own (PyTorch plus torchvision ResNet-18).
  - **Short ACT run** (`outputs/train/m4_act_smoke`, gitignored, 1.2 GB):
    `unset PYTHONPATH && uv run --no-sync lerobot-train --dataset.repo_id=ronyroyce/so101_sim_expert_v1 --dataset.root=outputs/datasets/so101_sim_expert_v1 --policy.type=act --policy.device=cuda --policy.push_to_hub=false --output_dir=outputs/train/m4_act_smoke --job_name=m4_act_smoke --steps=500 --batch_size=8 --log_freq=50 --save_freq=250 --trackio.enable=true --trackio.project=so101-research`
    - Default trap: **`policy.push_to_hub` defaults to `true`** (`configs/policies.py:73`), so set it to false for local runs.
    - **Full record (config + every logged metric + versions + checkpoint sizes): `research/results/m4_act_smoke_train.json`**, written by `research/scripts/export_train_run.py --output_dir outputs/train/m4_act_smoke --trackio_project so101-research --out research/results/m4_act_smoke_train.json` (reads `train_config.json` from the checkpoint and the run from Trackio's SQLite db). Use it to export every M6 run too.
    - Training params (ACT defaults unless noted):

      | group | value |
      |---|---|
      | model | ACT, **51,597,190** learnable params; ResNet-18 (ImageNet1K_V1, downloaded once to `~/.cache/torch/hub`), one pass per camera, full 480×640; dim_model 512, 8 heads, 4 enc / 1 dec layers, ffn 3200, dropout 0.1 |
      | action head | chunk_size 100, n_action_steps 100, no temporal ensembling; CVAE latent 32, kl_weight 10 |
      | inputs | `observation.state` (6) + `images.front` + `images.wrist`; n_obs_steps 1 |
      | normalisation | MEAN_STD for state/action/visual; visual stats = ImageNet (`use_imagenet_stats`) |
      | optimiser | AdamW lr 1e-5 (backbone 1e-5), betas 0.9/0.999, wd 1e-4, grad clip 10, **no scheduler** |
      | run | steps 500 (overridden), batch 8, seed 1000, fp32 (`mixed_precision: no`, TF32 on), no compile, **image augmentation off** (`image_transforms.enable=false`) |
      | data loading | torchcodec **0.17.0+cpu** (CPU decoding), 4 workers, prefetch 4, spawn context |
      | versions | lerobot 0.7.1, torch 2.15.0, torchvision 0.29.2, accelerate 1.15.0, trackio 0.45.0, mujoco 3.3.7; RTX 4060 Laptop 8 GB |
    - Performance: 500 steps × batch 8 = 4000 samples = **1.84 epochs**, training loop **2 min 47 s** (166.5 s), plus ~4 s of setup. Steady state **0.317 s/step, 25 samples/s, 3.13 it/s**. The first 50-step window is slower (0.447 s/step, data 0.073 s) because of worker spawn and warm-up.
      **GPU-bound:** dataloading 0.002 s vs update 0.317 s per step, so CPU AV1 decoding with 4 workers keeps up easily. GPU utilisation median 99 %. Memory: 3.72 GB allocated by torch (4.89 GB on the device) of 8 GB ⇒ room for roughly batch 16 at this resolution, or more with bf16 or smaller images.
      ⇒ **Cost model for M6 planning:** ~0.32 s/step at batch 8 ⇒ 10k steps ≈ 53 min, 50k ≈ 4.4 h, 100k (the default) ≈ 8.8 h on this laptop.
      Loss per log step (l1 / kld / total / grad norm):

      | step | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 |
      |---|---|---|---|---|---|---|---|---|---|---|
      | l1 | 0.699 | 0.549 | 0.475 | 0.454 | 0.429 | 0.404 | 0.391 | 0.374 | 0.345 | 0.328 |
      | kld | 1.348 | 0.417 | 0.341 | 0.306 | 0.270 | 0.263 | 0.232 | 0.227 | 0.219 | 0.211 |
      | loss | 14.18 | 4.72 | 3.88 | 3.51 | 3.13 | 3.03 | 2.71 | 2.64 | 2.54 | 2.44 |
      | grad norm | 287 | 132 | 112 | 104 | 93 | 92 | 85 | 81 | 81 | 80 |
    - Loss (= l1 + 10·kld): 14.18 → 2.44. **L1 0.70 → 0.33** (normalised units), KLD 1.35 → 0.21; still falling at the end, as expected for a smoke run. Grad norm 287 → 80, always above the clip of 10, so every update is clipped.
    - **Checkpoint check** (`checkpoints/000500/pretrained_model`, `last → 000500`): `model.safetensors` 207 MB plus `config.json`, `train_config.json`, and the pre/post-processor JSON and safetensors files. `training_state/` (394 MB: optimizer, RNG, step) is only needed to resume. Preprocessor = rename → to_batch → device → normalizer. Postprocessor = unnormalizer.
    - **Normaliser stats = `stats.json` exactly** (max |diff| 0.0) for state and action. Camera mean/std were replaced by **ImageNet** values (0.485/0.456/0.406, 0.229/0.224/0.225) because `dataset.use_imagenet_stats=true`. The checkpoint carries its own stats, so inference doesn't need the dataset.
    - Trackio stores the run locally in `~/.cache/huggingface/trackio/so101-research.db` (view with `trackio show --project so101-research`). It also logs GPU system metrics.
  - **Rollout on the sim robot (pipeline check passed)**:
    `unset PYTHONPATH && uv run --no-sync lerobot-rollout --strategy.type=base --policy.path=outputs/train/m4_act_smoke/checkpoints/last/pretrained_model --robot.type=sim --robot.env.type=envhub --robot.env.hub_path=lerobot/so101-tabletop-mujoco@4684f500b94d07194f2c860314c4e27cb9349e8c --robot.env.trust_remote_code=true --task="Put the red cube in the green box." --fps=30 --duration=20 --play_sounds=false --display_data=true`
    - Chain verified end to end: checkpoint load → `validate_checkpoint` (policy `input_features` vs the sim robot's declared features) → env processors + checkpoint pre/post-processors → `SyncInferenceEngine` (inline, no thread) → 590 ticks → return to the initial pose (7 s) → clean disconnect.
    - Facts about `lerobot-rollout` with a sim robot: only the `base` and `episodic` strategies accept one. Episodes restart via `start_episode()` **without a seed** (random layout). The env only ends an episode when **all 4 cubes** are in the box or after 3600 steps, so **rollout reports no single-cube success**. The loop is paced to **wall clock** (`CycleTimer`), unlike our recorder.
    - Cadence: 29.46 Hz effective, 6/589 ticks over the 33.3 ms budget, 6.3 ms mean headroom.

      | section | mean | worst | share |
      |---|---|---|---|
      | send (physics step + render of 2 cameras) | 23.8 ms | 30.7 ms | 86 % |
      | infer | 2.3 ms | 247 ms | 8 % |
      | telemetry (Rerun) | 1.2 ms | 5.6 ms | 4 % |
      | observe + process_obs | 0.08 ms | — | <1 % |

    - **The sim, not the policy, is the bottleneck**: `send` ≈ 24 ms matches 4.1's 44 steps/s.
    - **ACT inference pattern**: one full forward pass per 100 ticks (n_action_steps 100) takes **17.5–18.8 ms**. The first pass takes **247 ms** (CUDA/cuDNN warm-up). The other ticks just pop the action queue. **All 6 "loop slower than 30 Hz" warnings fall on the chunk ticks**: 3.3 s apart, 24 + 18 ≈ 42–45 ms per tick (the warning's "22.5 Hz" is that single tick's rate).
    - Harmless in the sim (physics only advances on `send_action`), but the margin is thin: temporal ensembling (inference every tick) or a heavier VLA would overrun on every tick. ⇒ **M6 success-rate evaluation should use a non-real-time loop** (`lerobot-eval` or our own), with `lerobot-rollout` kept for deployment-style tests. On hardware, `send` is a serial-bus write, much cheaper than physics plus rendering.
    - Inference footprint: GPU 0.76 GiB, process 2.12 GiB.
  - **Behaviour (user, in Rerun):** the arm moved toward the cube but missed and tipped it. The gripper closed on air, and the arm then headed toward the green box anyway. Motion very jerky, with a visible jump at each chunk boundary (every 3.3 s). This is the expected result for 500 steps / 1.8 epochs: the **coarse phase structure was learned** (reach → close → transport), the **precision was not**, and the policy runs the expert's script **open-loop**. That is the same failure as the 4.5 control (gripper closes on empty air), only now with cameras that it doesn't yet use well.
  - **Rerun can't replay rollout images:** `rerun_visualization.py:157` logs camera frames with `static=True` (no timeline, each frame replaces the last; viewer memory capped by `LEROBOT_RERUN_MEMORY_LIMIT=10%`). Only the scalar joint plots can be scrubbed. ⇒ `--display_data` is a live view, not a recording.
  - **Hypothesis for M6 (user's suggestion): lower `n_action_steps` (100 → 50/30).** It's inference-only (≤ chunk_size, no retraining). It trades one big chunk join every 3.3 s for more frequent joins plus more closed-loop reactivity. ACT's own remedy for joins is temporal ensembling (`n_action_steps=1`, `temporal_ensemble_coeff=0.01`), at a cost of ~18 ms of inference per tick, which is right at the cadence limit above. Test {100, 50, 30, ensembling} on the **same converged checkpoint** (on this 500-step model, under-training would confound the result). Metrics: success rate, action jerk, cadence.
  - **Tooling decision:** the "record inference and replay it" need goes into **M6.1, an evaluation harness**: seeded layouts, our single-cube `cube_placed` success check (the env's own success is the 4-cube one), a non-real-time loop, and one MP4 per episode plus a metrics JSON. A GUI control panel (start/record buttons) is deferred, since it would be a thin layer over the harness. The MuJoCo viewer (3.1) covers live viewing until then.
  - **4.6 verdict: the policy inference path works end to end** (train → checkpoint → processors → rollout on the sim robot). Task success was not the goal and is 0 here.

**M4 status: complete** (4.1–4.6 verified). Reference dataset `so101_sim_expert_v1`, reference smoke checkpoint `outputs/train/m4_act_smoke`.

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
