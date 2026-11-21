# Runbook: drive and record the SO-101 MuJoCo sim with a physical leader arm (option A)

Status: checked without hardware on 2026-11-20 (`research/scripts/so101_leader_readiness.py`). The commands below
parse and build the sim robot; the only thing that fails is opening the missing serial port. Verify every step
when the arm is connected, and update this file with what actually happened.

## 0. One-time machine setup (done 2026-11-20)

```bash
unset PYTHONPATH
uv sync --locked --extra feetech --extra core_scripts --inexact   # serial bus + record/teleop scripts
uv pip install -r research/requirements-so101-sim.txt             # in case a sync removed them
sudo usermod -aG dialout $USER                                    # serial port access; log out and back in
id -nG | grep dialout                                             # must print dialout
```

## 1. Find the port

```bash
lerobot-find-port        # unplug the USB cable when it asks; the port that disappears is the arm's
```
Expected: something like `/dev/ttyACM0`. Use it as `PORT` below.

## 2. Calibrate the leader (once per arm; stored on disk)

```bash
uv run --no-sync lerobot-calibrate --teleop.type=so101_leader --teleop.port=$PORT --teleop.id=research_leader
```
Saved to `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/research_leader.json`.
**Why it matters for sim:** in degree mode a joint reads `(raw − mid) · 360/4095`, where `mid` is the middle
of the range you moved it through during calibration. So the leader's 0° is "middle of the recorded travel",
while the sim's 0° is the MJCF joint zero. Menagerie's model derives from `so101_new_calib.xml`, and its
joint ranges are symmetric around 0 (pan ±110°, lift ±100°, elbow ±96.8°, wrist_flex ±95°), so the two
should roughly agree. Any per-joint offset comes from calibration quality. **Measure it** (step 3 check)
and record it in research_log.md.

## 3. Teleoperate the sim (no recording)

Hold the leader near the sim's home pose first, `[0, −60, 60, 35, 0, 30]`: the sim jumps to the leader's pose on
the first action, and a large jump saturates the actuators (overshoot and ringing, see M3 step 3.3).

```bash
uv run --no-sync lerobot-teleoperate \
  --robot.type=sim --robot.env.type=envhub \
  --robot.env.hub_path=lerobot/so101-tabletop-mujoco@4684f500b94d07194f2c860314c4e27cb9349e8c \
  --robot.env.trust_remote_code=true \
  --teleop.type=so101_leader --teleop.port=$PORT --teleop.id=research_leader \
  --fps=30 --display_data=true
```
- `--fps=30` is **required**: the default is 60, and each action advances the sim exactly 1/30 s.
- Check: put the leader in a known pose (e.g. everything at 0°) and compare its numbers with the sim's state
  in the Rerun view. That gives the calibration offset per joint.

## 4. Record a dataset

```bash
uv run --no-sync lerobot-record \
  --robot.type=sim --robot.env.type=envhub \
  --robot.env.hub_path=lerobot/so101-tabletop-mujoco@4684f500b94d07194f2c860314c4e27cb9349e8c \
  --robot.env.trust_remote_code=true \
  --teleop.type=so101_leader --teleop.port=$PORT --teleop.id=research_leader \
  --dataset.repo_id=ronyroyce/so101_sim_leader_v0 \
  --dataset.root=outputs/datasets/so101_sim_leader_v0 \
  --dataset.single_task="Put all the cubes in the green box." \
  --dataset.fps=30 --dataset.num_episodes=5 --dataset.episode_time_s=120 --dataset.reset_time_s=0 \
  --dataset.push_to_hub=false --display_data=true
```
- `--dataset.push_to_hub=false` is **required** unless you mean to publish: the default is `true`.
- `--dataset.reset_time_s=0`: the sim resets itself at each episode start, so there's nothing to reset by hand.
- An episode ends on success (all 4 cubes in the box), at the env's 3600-step limit, or after
  `episode_time_s`, whichever comes first.
