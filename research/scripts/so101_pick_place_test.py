"""Milestone 4.2b part 2: one scripted single-cube episode, measured.

Runs the expert on ONE cube (default red), dropped at the box centre, which is the episode design chosen in
4.2b. Records the cube height at every env step and reports:
  - grasp held?   how high the cube rose during the lift
  - placed?       our per-cube check: in the box, jaws open, at rest (env.py's own count also includes held
                  cubes, so it isn't enough on its own)
Saves a contact sheet of front-camera frames to research/results/m4_pick_place_seed<N>_<cube>.png.

    unset PYTHONPATH && uv run --no-sync python research/scripts/so101_pick_place_test.py [--seed 0] [--cube red]

--all runs the old 4-cube slot mode instead (kept for the record: it fails often, see research_log.md).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from so101_expert import TOOL_OFFSET, PickPlaceExpert, cube_placed
from so101_viewer import load_env_module

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cube", default="red", choices=["red", "green", "blue", "yellow"])
    parser.add_argument("--all", action="store_true", help="old 4-cube slot mode")
    args = parser.parse_args()

    env_module = load_env_module()
    env = env_module.SO101Env(**env_module.DEFAULTS)
    env.reset(seed=args.seed)
    world = env._world
    expert = PickPlaceExpert(world, env_module.CUBE_BODIES, env_module.BOX_BODY)

    if args.all:
        actions = expert.actions()
    else:
        target = list(env_module.CUBE_BODIES).index(f"{args.cube}_cube")
        actions = expert.pick_and_place(target)  # k=None: drop at the box centre

    tool_z, cube_z, frames = [], [], []
    info, steps, placed_at = {"cubes_in_box": 0}, 0, None
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        R = world.data.site_xmat[expert.kin.site].reshape(3, 3)
        tool_z.append(float((world.data.site_xpos[expert.kin.site] + R @ TOOL_OFFSET)[2]))
        cube_z.append(float(world.data.xpos[world.model.body(expert.log[-1]["cube"]).id][2]))
        if not args.all and placed_at is None and cube_placed(world, expert.cubes[target], expert.box):
            placed_at = steps
        if steps % 15 == 0:
            frames.append((steps, obs["pixels"]["front"]))
        if terminated:
            break

    for e in expert.log:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()})
    tool_z, cube_z = np.array(tool_z), np.array(cube_z)
    lifted = cube_z.max() - cube_z[0]
    print(f"\nsteps: {steps} ({steps / 30:.1f} s sim)   cubes_in_box: {info['cubes_in_box']}   "
          f"success (all 4): {bool(info['is_success'])}")
    print(f"tracked cube ({expert.log[-1]['cube']}): rose {100 * lifted:.1f} cm max, "
          f"final z {100 * cube_z[-1]:.1f} cm")
    if not args.all:
        print(f"placed (in box, released, at rest): {placed_at is not None}"
              + (f" at step {placed_at} ({placed_at / 30:.1f} s)" if placed_at else ""))

    pick = [frames[j] for j in np.linspace(0, len(frames) - 1, min(12, len(frames))).round().astype(int)]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, (k, img) in zip(axes.flat, pick):
        ax.imshow(img)
        ax.set_title(f"step {k}  t={k / 30:.1f}s", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"m4_pick_place_seed{args.seed}_{'all' if args.all else args.cube}.png"
    fig.savefig(out, dpi=80)
    print(f"contact sheet: {out}")
    env.close()


if __name__ == "__main__":
    main()
