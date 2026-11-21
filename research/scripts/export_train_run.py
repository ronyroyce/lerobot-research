"""Export a finished `lerobot-train` run to a small, git-tracked JSON record.

`outputs/` is gitignored and checkpoints are large, so this keeps what later
milestones need to compare runs: the full resolved train config, the per-log-step
metrics from Trackio, a GPU summary, checkpoint sizes and the software versions.

Usage:
    uv run --no-sync python research/scripts/export_train_run.py \
        --output_dir outputs/train/m4_act_smoke --trackio_project so101-research \
        --out research/results/m4_act_smoke_train.json
"""

import argparse
import json
import math
import sqlite3
import statistics
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch
from safetensors import safe_open

TRACKIO_DIR = Path.home() / ".cache/huggingface/trackio"


def read_trackio(project: str, job_name: str, run_name: str | None) -> dict:
    db = sqlite3.connect(f"file:{TRACKIO_DIR / f'{project}.db'}?mode=ro", uri=True)
    if run_name is None:  # latest run whose name starts with the job name
        rows = db.execute(
            "SELECT run_name FROM configs WHERE run_name LIKE ? ORDER BY created_at DESC LIMIT 1",
            (f"{job_name}-%",),
        ).fetchall()
        if not rows:
            raise SystemExit(f"no Trackio run named {job_name}-* in project {project}")
        run_name = rows[0][0]
    (created_at,) = db.execute("SELECT created_at FROM configs WHERE run_name = ?", (run_name,)).fetchone()
    metrics = [
        {"step": step, "time": ts, **json.loads(m)}
        for ts, step, m in db.execute(
            "SELECT timestamp, step, metrics FROM metrics WHERE run_name = ? ORDER BY id", (run_name,)
        )
    ]
    system = [json.loads(m) for (m,) in db.execute(
        "SELECT metrics FROM system_metrics WHERE run_name = ? ORDER BY id", (run_name,)
    )]
    util = [s["gpu/0/utilization"] for s in system if "gpu/0/utilization" in s]
    alloc = [s["gpu/0/allocated_memory"] for s in system if "gpu/0/allocated_memory" in s]
    gpu = {
        "samples": len(util),
        "utilization_pct_median": statistics.median(util) if util else None,
        "device_memory_gb_max": max(alloc) if alloc else None,
        "device_memory_gb_total": next((s["gpu/0/total_memory"] for s in system if "gpu/0/total_memory" in s), None),
    }
    return {"run_name": run_name, "created_at": created_at, "metrics": metrics, "gpu": gpu}


def checkpoint_info(output_dir: Path) -> dict:
    ckpts = {}
    for d in sorted((output_dir / "checkpoints").iterdir()):
        if d.is_symlink():
            ckpts[d.name] = f"-> {d.readlink()}"
            continue
        ckpts[d.name] = {
            sub.name: round(sum(f.stat().st_size for f in sub.rglob("*") if f.is_file()) / 1e6, 1)
            for sub in d.iterdir()
        }
    last = output_dir / "checkpoints/last/pretrained_model/model.safetensors"
    with safe_open(last, "pt") as f:
        n_tensor_elems = sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())
    return {"sizes_mb": ckpts, "model_safetensors_elements": n_tensor_elems}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--trackio_project", required=True)
    p.add_argument("--run_name", default=None, help="Trackio run name; default = latest <job_name>-*")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = json.loads((args.output_dir / "checkpoints/last/pretrained_model/train_config.json").read_text())
    ds_root = Path(cfg["dataset"]["root"])
    info = json.loads((ds_root / "meta/info.json").read_text())
    tr = read_trackio(args.trackio_project, cfg["job_name"], args.run_name)

    m = tr["metrics"]
    # Each log line averages step_s over the log_freq steps since the previous one.
    # (Trackio's run created_at is not a reliable start time: it gave 144 s vs the console's 167 s.)
    log_freq = cfg["log_freq"]
    train_loop_s = sum(r["train/step_s"] for r in m) * log_freq
    steady = m[1:]  # first log window includes warm-up (dataloader spawn, cudnn autotune)
    summary = {
        "steps": m[-1]["step"],
        "epochs": round(m[-1]["train/epochs"], 3),
        "train_loop_s": round(train_loop_s, 1),
        "loss_first_last": [m[0]["train/loss"], m[-1]["train/loss"]],
        "l1_first_last": [m[0]["train/l1_loss"], m[-1]["train/l1_loss"]],
        "kld_first_last": [m[0]["train/kld_loss"], m[-1]["train/kld_loss"]] if "train/kld_loss" in m[0] else None,
        "steady_update_s_median": statistics.median(r["train/update_s"] for r in steady) if steady else None,
        "steady_dataloading_s_median": statistics.median(r["train/dataloading_s"] for r in steady) if steady else None,
        "steady_samples_per_s_median": statistics.median(r["train/samples_per_s"] for r in steady) if steady else None,
        "torch_mem_gb_max": max(r["train/gpu_mem_gb"] for r in m),
        "gpu": tr["gpu"],
    }
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    record = {
        "output_dir": str(args.output_dir),
        "trackio": {"project": args.trackio_project, "run_name": tr["run_name"]},
        "git_commit_at_export": git,
        "versions": {pkg: version(pkg) for pkg in ["lerobot", "torch", "torchvision", "torchcodec", "accelerate", "trackio", "mujoco"]},
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset": {"repo_id": cfg["dataset"]["repo_id"], "root": str(ds_root),
                    "total_frames": info["total_frames"], "total_episodes": info["total_episodes"], "fps": info["fps"]},
        "summary": summary,
        "checkpoints": checkpoint_info(args.output_dir),
        "train_config": cfg,
        "metrics": m,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ["trackio", "versions", "gpu_name", "dataset", "summary"]}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
