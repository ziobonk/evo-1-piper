"""Make a writable Evo-1 view of a Piper same-frame LeRobot v2.1 dataset.

The large Parquet data stays in the source directory. Only metadata is copied,
and Evo-1-specific normalization statistics are written in the output view.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def vector_stats(array: np.ndarray) -> dict:
    return {
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not (source / "meta/info.json").is_file():
        parser.error(f"Missing LeRobot metadata under {source}")
    info = json.loads((source / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1" or "state" not in info.get("features", {}):
        parser.error("Expected a LeRobot v2.1 dataset with a state column")
    files = sorted((source / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        parser.error("No episode Parquet files found")
    if output == source or source in output.parents:
        parser.error("Output must be outside the source dataset")

    states, targets = [], []
    same_frame_count = 0
    frame_count = 0
    for file in files:
        df = pd.read_parquet(file, columns=["state", "actions"])
        state = np.stack(df["state"].to_list()).astype(np.float64)
        action = np.stack(df["actions"].to_list()).astype(np.float64)
        if state.shape[1] != 7 or action.shape != state.shape:
            parser.error(f"Expected matched 7D state/actions in {file}")
        same_frame_count += int(np.count_nonzero(np.all(np.isclose(state, action, atol=1e-6), axis=1)))
        frame_count += len(state)
        states.append(state)
        targets.append(np.concatenate((state[1:], state[-1:]), axis=0))
    if same_frame_count != frame_count:
        parser.error(
            f"Only {same_frame_count}/{frame_count} frames have actions == state; "
            "the next-pose processing suite is not valid for this dataset"
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(exist_ok=True)
    data_link = output / "data"
    if data_link.is_symlink():
        if data_link.resolve() != (source / "data").resolve():
            parser.error(f"{data_link} points at a different dataset")
    elif data_link.exists():
        parser.error(f"{data_link} exists and is not a symlink")
    else:
        data_link.symlink_to(source / "data", target_is_directory=True)

    for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
        src = source / "meta" / name
        if src.exists():
            shutil.copy2(src, output / "meta" / name)
    stats = {
        "observation.state": vector_stats(np.concatenate(states)),
        "action": vector_stats(np.concatenate(targets)),
    }
    (output / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Prepared {len(files)} episodes and {frame_count} frames at {output}")
    print("Action targets: next measured absolute TCP pose; last pose repeats at episode end")


if __name__ == "__main__":
    main()
