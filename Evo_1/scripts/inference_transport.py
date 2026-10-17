"""Lossless BGR image requests: a small JSON header followed by uint8 pixels."""

import json
import struct
import time

import numpy as np


MAGIC = b"EVO1IMG1"
PREFIX_SIZE = len(MAGIC) + 4
MAX_HEADER_SIZE = 65536
IMAGE_SHAPE = (448, 448, 3)
IMAGE_BYTES = int(np.prod(IMAGE_SHAPE))
ZERO_IMAGE = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
ZERO_IMAGE.setflags(write=False)


def encode_request(data, transport="binary"):
    if transport == "json":
        return json.dumps(data)
    if transport != "binary":
        raise ValueError(f"Unknown image transport: {transport}")
    images = data["image"]
    if len(images) not in (2, 3):
        raise ValueError("Binary requests require two or three images")
    if len(images) == 2 and data.get("image_mask") != [1, 1, 0]:
        raise ValueError("Omitting the third image requires image_mask=[1, 1, 0]")
    for image in images:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != IMAGE_SHAPE:
            raise ValueError("Binary images must be uint8 BGR arrays with shape (448, 448, 3)")
    header = {key: value for key, value in data.items() if key != "image"}
    header.update({"image_count": len(images), "image_encoding": "uint8_bgr_448"})
    encoded_header = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded_header) > MAX_HEADER_SIZE:
        raise ValueError("Binary request metadata is too large")
    return b"".join((MAGIC, struct.pack(">I", len(encoded_header)), encoded_header,
                     *(image.tobytes(order="C") for image in images)))


def decode_request(message, metrics=None):
    started = time.perf_counter() if metrics is not None else None
    if isinstance(message, str):
        result = json.loads(message)
        if not isinstance(result, dict):
            raise ValueError("Inference request must be a JSON object")
        if metrics is not None:
            metrics.update({"server_json_decode_ms": (time.perf_counter() - started) * 1000,
                            "server_unpack_ms": 0.0, "image_transport": "json"})
        return result
    if not isinstance(message, bytes) or not message.startswith(MAGIC) or len(message) < PREFIX_SIZE:
        raise ValueError("Invalid binary inference request prefix")
    header_size = struct.unpack_from(">I", message, len(MAGIC))[0]
    offset = PREFIX_SIZE + header_size
    if not 0 < header_size <= MAX_HEADER_SIZE or offset > len(message):
        raise ValueError("Invalid binary inference request header length")
    decode_started = time.perf_counter() if metrics is not None else None
    header = json.loads(message[PREFIX_SIZE:offset])
    decoded_at = time.perf_counter() if metrics is not None else None
    if not isinstance(header, dict) or header.get("image_encoding") != "uint8_bgr_448":
        raise ValueError("Unsupported binary image encoding")
    count = header.get("image_count")
    if type(count) is not int or count not in (2, 3):
        raise ValueError("Invalid binary image count")
    if len(message) != offset + count * IMAGE_BYTES:
        raise ValueError("Binary image byte length does not match the header")
    if count == 2 and header.get("image_mask") != [1, 1, 0]:
        raise ValueError("Omitting the third image requires image_mask=[1, 1, 0]")
    images = np.frombuffer(message, dtype=np.uint8, offset=offset).reshape(count, *IMAGE_SHAPE)
    header["image"] = tuple(images) + ((ZERO_IMAGE,) if count == 2 else ())
    if metrics is not None:
        json_ms = (decoded_at - decode_started) * 1000
        metrics.update({"server_json_decode_ms": json_ms,
                        "server_unpack_ms": (time.perf_counter() - started) * 1000 - json_ms,
                        "image_transport": "binary"})
    return header
