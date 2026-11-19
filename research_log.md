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
Status: not started.

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
