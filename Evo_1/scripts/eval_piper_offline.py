"""Replay Piper dataset observations through Evo-1 and plot open-loop action errors.

This compares each predicted action chunk with the same next-pose targets used
during training. It does not move the robot or measure closed-loop success.
"""

import argparse
import asyncio
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import websockets
from scipy.spatial.transform import Rotation

from Evo1_client_piper import make_payload


def decode_bgr(value):
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("Expected embedded JPEG bytes in Parquet image column")
    frame = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode Parquet image")
    return frame


def next_pose_targets(states, start, horizon):
    """Match PiperEEFNextPoseSuite: shift within the padded horizon window."""
    if states.ndim != 2 or states.shape[1] != 7 or not 0 <= start < len(states):
        raise ValueError("Expected nonempty 7D states and a valid start frame")
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    last = min(start + horizon - 1, len(states) - 1)
    indices = np.minimum(start + np.arange(1, horizon + 1), last)
    return states[indices]


def measure_errors(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 7:
        raise ValueError("Prediction and target must both be [horizon, 7]")
    if not np.isfinite(predicted).all():
        raise ValueError("Model returned nonfinite actions")
    pos_mm = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=1) * 1000
    relative_rotation = Rotation.from_rotvec(target[:, 3:6]).inv() * Rotation.from_rotvec(predicted[:, 3:6])
    rot_deg = np.degrees(relative_rotation.magnitude())
    gripper_mm = np.abs(predicted[:, 6] - target[:, 6])
    return pos_mm, rot_deg, gripper_mm


def plot_sample(path, base_bgr, wrist_bgr, predicted, target, errors, title):
    pos_mm, rot_deg, gripper_mm = errors
    ticks = np.arange(1, len(predicted) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axes[0, 0].imshow(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Global camera")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Wrist camera")
    axes[0, 1].axis("off")
    for axis, name in enumerate("xyz"):
        axes[0, 2].plot(ticks, target[:, axis], label=f"{name} GT", linestyle="--")
        axes[0, 2].plot(ticks, predicted[:, axis], label=f"{name} pred")
    axes[0, 2].set_title("Absolute TCP position (m)")
    axes[0, 2].legend(ncol=2, fontsize=8)
    axes[1, 0].plot(ticks, target[:, 6], label="GT")
    axes[1, 0].plot(ticks, predicted[:, 6], label="pred")
    axes[1, 0].set_title("Gripper width (mm)")
    axes[1, 0].legend()
    axes[1, 1].plot(ticks, pos_mm, label="position (mm)")
    axes[1, 1].plot(ticks, gripper_mm, label="gripper (mm)")
    axes[1, 1].set_title("Absolute error")
    axes[1, 1].legend()
    axes[1, 2].plot(ticks, rot_deg)
    axes[1, 2].set_title("Orientation error (deg)")
    for ax in axes[1]:
        ax.set_xlabel("Predicted action step")
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


async def run(args):
    info = json.loads((args.dataset / "meta/info.json").read_text())
    if info.get("robot_type") != "piper_eef" or info.get("pose_coordinate_frame") != "robot_base_tcp":
        raise ValueError("Expected the raw Piper EEF dataset")
    if not (args.dataset / "meta/stats.json").is_file():
        raise FileNotFoundError("Prepared dataset meta/stats.json is missing")
    tasks = {int(task["task_index"]): task["task"] for task in
             (json.loads(line) for line in (args.dataset / "meta/tasks.jsonl").read_text().splitlines())}
    total_episodes = int(info["total_episodes"])
    episodes = (list(range(total_episodes)) if args.episodes == "all"
                else [int(value) for value in args.episodes.split(",")])
    if not episodes or any(ep < 0 or ep >= total_episodes for ep in episodes):
        raise ValueError(f"Choose episodes in 0..{total_episodes - 1}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    async with websockets.connect(args.server, max_size=100_000_000) as ws:
        for episode in episodes:
            parquet = args.dataset / info["data_path"].format(
                episode_chunk=episode // int(info["chunks_size"]), episode_index=episode)
            frame_table = pd.read_parquet(parquet, columns=["state", "image", "wrist_image", "task_index"])
            states = np.stack(frame_table["state"].to_list()).astype(np.float64)
            if len(states) < 2:
                print(f"Skip episode {episode}: fewer than two frames")
                continue
            # Keep a complete horizon for each sample; the final target repeats
            # the last state, exactly as in the training window.
            latest_start = max(0, len(states) - args.horizon)
            starts = np.unique(np.linspace(0, latest_start, args.samples_per_episode, dtype=int))
            for start in starts:
                row = frame_table.iloc[int(start)]
                base = decode_bgr(row["image"])
                wrist = decode_bgr(row["wrist_image"])
                prompt = tasks[int(row["task_index"])]
                await ws.send(json.dumps(make_payload(base, wrist, states[start], prompt)))
                chunk = np.asarray(json.loads(await ws.recv()), dtype=np.float64)
                if chunk.ndim != 2 or chunk.shape[0] < args.horizon or chunk.shape[1] < 7:
                    raise RuntimeError(f"Server returned {chunk.shape}; expected at least [{args.horizon}, 7]")
                predicted = chunk[:args.horizon, :7]
                target = next_pose_targets(states, int(start), args.horizon)
                errors = measure_errors(predicted, target)
                for step, (pos, rot, grip) in enumerate(zip(*errors), start=1):
                    rows.append({"episode": episode, "start_frame": int(start), "step": step,
                                 "position_error_mm": float(pos), "rotation_error_deg": float(rot),
                                 "gripper_error_mm": float(grip)})
                plot_sample(args.output / f"episode_{episode:06d}_frame_{start:05d}.png",
                            base, wrist, predicted, target, errors,
                            f"Episode {episode}, frame {start}, open-loop prediction")
                print(f"episode={episode} frame={start} first-3 position error="
                      f"{np.mean(errors[0][:3]):.1f} mm")

    if not rows:
        raise RuntimeError("No usable samples found")
    with (args.output / "errors.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"dataset": str(args.dataset.resolve()), "episodes": episodes,
               "samples": len({(r["episode"], r["start_frame"]) for r in rows}),
               "note": "Training-set open-loop replay if these episodes were used for training; not task success."}
    for count in (1, min(3, args.horizon), args.horizon):
        subset = [r for r in rows if r["step"] <= count]
        summary[f"first_{count}_steps"] = {
            key: float(np.mean([r[key] for r in subset]))
            for key in ("position_error_mm", "rotation_error_deg", "gripper_error_mm")}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("piper_data"))
    parser.add_argument("--server", default="ws://localhost:9000")
    parser.add_argument("--episodes", default="0,10,20,30,40", help="Comma-separated IDs or all")
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("piper_offline_eval"))
    args = parser.parse_args()
    if args.samples_per_episode < 1 or args.horizon < 1:
        parser.error("samples-per-episode and horizon must be positive")
    asyncio.run(run(args))
