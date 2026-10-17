"""Compare request codecs and CPU preprocessing on saved Piper images.

This benchmark doesn't connect to a robot, open a server, or load a model.
"""

import argparse
import json
from pathlib import Path
from statistics import mean
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torchvision.transforms.functional import to_tensor

from Evo1_client_piper import make_payload
from Evo1_server import prepare_image_batch
from inference_transport import encode_request, decode_request


def measure(frames, transport):
    started = time.perf_counter()
    payload = make_payload(*frames, [0] * 7, "benchmark", image_transport=transport)
    payload.update({"return_timing": True, "request_id": "benchmark-request"})
    built = time.perf_counter()
    message = encode_request(payload, transport)
    encoded = time.perf_counter()
    decoded = decode_request(message)
    parsed = time.perf_counter()
    if transport == "json":
        # The previous server path: list -> NumPy -> resize -> RGB -> tensor,
        # repeated for all three images.
        images = [to_tensor(cv2.cvtColor(cv2.resize(np.array(image, dtype=np.uint8), (448, 448)),
                                        cv2.COLOR_BGR2RGB)) for image in decoded["image"]]
    else:
        images = prepare_image_batch(decoded["image"], device="cpu")
    prepared = time.perf_counter()
    sizes = len(message) if isinstance(message, bytes) else len(message.encode("utf-8"))
    return images, {
        "payload_build_ms": (built - started) * 1000,
        "request_encode_ms": (encoded - built) * 1000,
        "server_request_decode_ms": (parsed - encoded) * 1000,
        "server_preprocess_cpu_ms": (prepared - parsed) * 1000,
        "total_ms": (prepared - started) * 1000, "request_bytes": sizes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("piper_data"))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("logs/inference_transport_benchmark.json"))
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    source = sorted((args.dataset / "data").rglob("*.parquet"))[0]
    row = next(pq.ParquetFile(source).iter_batches(batch_size=1, columns=["image", "wrist_image"])).to_pylist()[0]
    frames = []
    for key in ("image", "wrist_image"):
        value = row[key]
        encoded_image = value["bytes"] if isinstance(value, dict) else value
        frame = cv2.imdecode(np.frombuffer(encoded_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot decode {key} from {source}")
        frames.append(frame)
    torch.set_num_threads(2)
    reference, _ = measure(frames, "json")
    optimized, _ = measure(frames, "binary")
    if not all(torch.equal(left, right) for left, right in zip(reference, optimized)):
        raise AssertionError("Optimized inputs differ from the previous pipeline")
    samples = {"json": [], "binary": []}
    for _ in range(args.iterations):
        for transport in samples:
            _, sample = measure(frames, transport)
            samples[transport].append(sample)
    means = {transport: {key: mean(sample[key] for sample in values) for key in values[0]}
             for transport, values in samples.items()}
    report = {
        "scope": "Offline request build, encoding, decoding and CPU preprocessing; excludes WebSocket, compression, CUDA and model inference",
        "source": str(source.resolve()), "iterations": args.iterations,
        "inputs_equal": True, "mean": means,
        "codec_and_cpu_preprocess_speedup": means["json"]["total_ms"] / means["binary"]["total_ms"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
