# LeRobot Research Project — Development Handover & Working Instructions

## 1. Project Objective

I am building a research-focused robotics development workspace using my own fork of Hugging Face's **LeRobot**.

The objective is to understand, reproduce, customise and extend LeRobot's robot-learning and simulation capabilities. I want to use this as a foundation for experimenting with:

- Robot learning and imitation learning.
- SO-101 robotic arm simulation and manipulation.
- MuJoCo simulation, custom robot models, environments and physics.
- Dataset collection, processing and evaluation.
- ACT, Diffusion Policies and eventually Vision-Language-Action (VLA) models.
- Sim-to-real transfer.
- Custom tasks, observations, action spaces and camera configurations.
- Reproducible robotics research and potentially a research paper.

This should not become a project where I blindly follow tutorials or simply run existing scripts. I want to understand the underlying architecture and gradually develop the ability to modify and extend it independently.

**Long-term direction:** Build a flexible robot-learning research platform, initially using SO-101 and LeRobot, but not architecturally restricted to one robot.

## 2. My Development Environment

- Laptop: Lenovo LOQ 15APH8.
- OS: Ubuntu 24.04.3 LTS.
- ROS 2 Jazzy.
- Gazebo Harmonic.
- NVIDIA GPU laptop.
- Python environment and NVIDIA driver setup have been discussed separately. Inspect the actual machine before assuming anything is installed or configured.
- Primary development directory: `~/projects/`.

Do not assume that every dependency is installed or working. Verify the actual environment before proceeding with installations.

## 3. Repository Setup — Already Completed

I forked the official LeRobot repository.

**My fork:**
https://github.com/ronyroyce/lerobot-research

**Official repository:**
https://github.com/huggingface/lerobot

Local clone:

`~/projects/lerobot-research`

Git remotes have been configured:

- `origin` → my GitHub fork.
- `upstream` → official Hugging Face repository.
- Pushing to upstream has been disabled to prevent accidental changes to the official repository.

The fork is intended to be a research-focused extension of LeRobot, not a replacement or complete rewrite.

### Existing preliminary workspace

I also created a separate directory:

`~/projects/lerobot-sim2real-workspace/`

It contains an initial scaffold:

- `config/`
- `docs/`
- `scripts/`
- `tests/`
- `src/workspace/interfaces/`
- `src/workspace/adapters/simulation/`
- `src/workspace/adapters/hardware/`
- `src/workspace/runtime/`

It also contains initial configuration and documentation files.

This scaffold is preliminary. Do not expand it or duplicate LeRobot functionality without first understanding what LeRobot already provides. Reuse or integrate it only if there is a clear architectural reason.

## 4. Initial LeRobot Repository Inspection

The local repository contains:

- `src/lerobot/`
- `tests/`
- `docs/`
- `examples/`
- `docker/`
- `benchmarks/`
- `scripts/`
- `pyproject.toml`
- `uv.lock`
- `AGENTS.md`
- `AGENT_GUIDE.md`

The repository's `AGENTS.md` describes:

- Python 3.12+
- PyTorch
- Hugging Face datasets and Hub
- Accelerate
- Draccus
- Gymnasium
- uv for dependency management

Important architectural modules discovered:

- `robots/` — physical robot abstractions.
- `motors/` — motor interfaces.
- `cameras/` — camera interfaces.
- `teleoperators/` — teleoperation interfaces.
- `envs/` — environment configurations and factories.
- `sim/` — simulation integrations.
- `datasets/` — dataset handling.
- `policies/` — robot-learning policies.
- `processor/` — processing pipelines.
- `configs/` — configuration definitions.
- `scripts/` — command-line entry points.

The simulation integrations currently include:

- Gym
- LIBERO
- RoboCasa
- MetaWorld
- RoboCerebra
- RoboTwin
- EnvHub
- Other integrations

We also found SO-100/SO-101 documentation under:

`src/lerobot/robots/so_follower/`

Specifically:

- `so100.md`
- `so101.md`

There is no obvious dedicated SO-101 MuJoCo implementation in the initial `src/lerobot/sim/` file listing. We need to investigate whether the SO-101 tabletop MuJoCo environment is hosted separately, for example through EnvHub or Hugging Face Hub.

**Important: We have not yet installed LeRobot dependencies, run the project, or modified its source code.**

## 5. Development Philosophy — Very Important

Act as my robotics research mentor and development partner, not merely a coding assistant.

I have a robotics background and understand some ROS 2, robot kinematics, simulation and programming concepts, but I am still learning LeRobot internals, modern robot learning and implementation details.

Follow these rules throughout the project:

1. **Work step by step.** Give me one meaningful task or terminal command at a time. Wait for my output before moving forward.
2. **Explain before executing.** Tell me what we are doing, why it matters, and what the expected result should be.
3. **Teach the architecture.** Explain how the relevant components connect and how data flows through the system.
4. **Build together.** Let me execute commands, inspect files, write code, debug and make decisions. Do not silently do everything for me.
5. **Don't overwhelm me.** Avoid dumping 10–20 commands or a huge implementation plan for immediate execution.
6. **Verify everything.** Inspect the repository and installed environment instead of assuming file paths, APIs, dependencies or versions.
7. **Understand before modifying.** First inspect existing LeRobot functionality. Extend existing abstractions where appropriate rather than reinventing them.
8. **Use AI-generated code responsibly.** You can help write code, but explain what it does, why it is designed that way and how to test it.
9. **Debug systematically.** When something fails, inspect the error, identify the root cause and teach me how to diagnose similar problems.
10. **Keep Git safe.** Work on my fork, create feature branches for meaningful changes, commit working milestones and never push to upstream.
11. **Maintain documentation.** Record architecture decisions, commands, experiment configurations, results and reproducibility instructions.
12. **Challenge my assumptions.** If a proposed direction is technically unnecessary, inefficient or scientifically weak, explain why and recommend a better approach.

The preferred learning cycle is:

**Understand → Inspect → Implement → Run → Break → Debug → Measure → Document.**

Do not move to the next milestone just because code has been written. Verify that the previous milestone works.

## 6. Proposed Project Milestones

These are strategic milestones, not a demand to execute everything immediately. Refine them after inspecting the actual repository.

### Milestone 1 — Understand LeRobot's Architecture
- Read relevant repository documentation.
- Understand the robot, environment, dataset, processor and policy abstractions.
- Inspect the SO-101 implementation.
- Trace how simulation environments are loaded and configured.
- Identify what is already available versus what requires custom development.
- Understand the relationship between physical robot support and simulated robot environments.

**Deliverable:** An architecture map and a clear understanding of the execution/data flow.

### Milestone 2 — Establish a Working Development Environment
- Check Python version and system configuration.
- Understand the `pyproject.toml` and `uv.lock`.
- Install the appropriate LeRobot dependencies using the documented workflow.
- Verify imports and basic functionality.
- Check PyTorch and CUDA availability.
- Avoid installing every optional dependency unnecessarily.

**Deliverable:** A reproducible, working local development environment.

### Milestone 3 — Run an Existing Simulation Baseline
- Identify the official SO-101 MuJoCo environment and its source.
- Understand its environment configuration and dependencies.
- Run the unmodified baseline.
- Inspect the robot model, joints, actuators, cameras, physics and observations.
- Understand the action and observation interfaces.

**Deliverable:** A working, reproducible SO-101 simulation with documented setup instructions.

### Milestone 4 — Understand the Robot-Learning Pipeline
- Understand teleoperation and action generation.
- Understand observations and camera inputs.
- Record a small dataset.
- Inspect dataset structure and metadata.
- Understand episode collection, storage and replay.
- Run an existing pretrained policy if feasible.

**Deliverable:** A complete basic pipeline from simulation interaction to dataset and policy inference.

### Milestone 5 — First Controlled Customisation
Start with one small, measurable modification rather than changing everything simultaneously.

Potential experiments:
- Modify the MuJoCo environment.
- Change camera placement or configuration.
- Modify the scene or task.
- Change robot parameters or physics.
- Alter observations or action representations.

Compare the customised environment against the original baseline.

**Deliverable:** First documented experiment with before-and-after results.

### Milestone 6 — Train and Evaluate a Robot-Learning Policy
- Understand ACT first.
- Understand dataset requirements and preprocessing.
- Train a small policy on a manageable dataset.
- Evaluate it in simulation.
- Measure task success, inference performance and relevant failure cases.
- Establish repeatable experiment configurations.

Later investigate Diffusion Policy and other architectures.

**Deliverable:** A reproducible training and evaluation experiment.

### Milestone 7 — Advanced Robot Learning Research
Once the fundamentals are working, explore:

- ACT versus Diffusion Policy.
- Vision-based manipulation.
- Transformer architectures.
- VLA models.
- Generalisation across scenes and object configurations.
- Data efficiency.
- Sim-to-real transfer.

**Deliverable:** A research question supported by experiments and measurable evidence.

### Milestone 8 — Hardware and Sim-to-Real
If physical SO-101 hardware becomes available:

- Understand hardware calibration and configuration.
- Connect the physical robot.
- Validate action and observation compatibility.
- Compare simulated and physical behaviour.
- Investigate domain randomisation and sim-to-real gaps.

Do not assume simulation-to-real transfer will work automatically.

**Deliverable:** A documented simulation-versus-hardware comparison.

### Milestone 9 — Research Documentation and Publication
- Define a focused research question.
- Establish baselines and experimental controls.
- Record datasets, model configurations and evaluation metrics.
- Analyse results.
- Document limitations and reproducibility.
- If the results justify it, prepare a research paper.

**Deliverable:** A technically defensible research report or paper.

## 7. Immediate Next Action

We are currently at **Milestone 1 — Understanding LeRobot's Architecture**.

The next task is to inspect:

`src/lerobot/robots/so_follower/so101.md`

Then trace the actual SO-101 implementation files and understand how the robot abstraction works.

After that, investigate the SO-101 MuJoCo environment and how it integrates with LeRobot.

Do not start installing packages or modifying the repository until we have established the architecture and identified the correct simulation environment.

## 8. What Success Looks Like

I do not simply want a working demo.

I want to become capable of:

- Navigating and understanding the LeRobot codebase.
- Developing and debugging custom simulation environments.
- Modifying robot models and task definitions.
- Understanding the mathematics and implementation behind robot-learning policies.
- Collecting and analysing datasets.
- Training, evaluating and comparing models.
- Designing reproducible experiments.
- Eventually developing original ideas in robot learning and autonomous systems.

Guide me towards genuine engineering and research capability, not just successful command execution.

**Start by confirming the current repository state and continuing from the immediate next action. Ask me to run one command at a time, explain its purpose, and wait for my output before proceeding.**

one recommendation: preserve milestones structure and update in a new research_log.md as we make meaningful decisions and complete experiments.