"""Verify lossless image transport and model input parity."""

import json
from pathlib import Path
import struct
import sys
import unittest

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.Evo1_client_piper import make_payload
from scripts.Evo1_server import prepare_image_batch
from scripts.inference_transport import MAGIC, PREFIX_SIZE, encode_request, decode_request


class ImageTransportTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(17)
        self.frames = [generator.integers(0, 256, (h, w, 3), dtype=np.uint8)
                       for h, w in ((720, 1280), (480, 640))]

    def payload(self):
        return make_payload(*self.frames, [0] * 7, "抓取 cube", image_transport="binary")

    def test_roundtrip_preserves_resized_pixels_and_zero_image_exactly(self):
        payload = self.payload()
        message = encode_request(payload)
        metrics = {}
        decoded = decode_request(message, metrics)
        for actual, source in zip(decoded["image"], self.frames):
            np.testing.assert_array_equal(actual, cv2.resize(source, (448, 448), interpolation=cv2.INTER_CUBIC))
        self.assertFalse(decoded["image"][2].any())
        self.assertEqual(decoded["prompt"], "抓取 cube")
        self.assertEqual(decoded["image_mask"], [1, 1, 0])
        self.assertEqual(decoded["action_mask"], [1] * 7 + [0] * 17)
        self.assertTrue(np.shares_memory(decoded["image"][0], np.frombuffer(message, dtype=np.uint8)))
        self.assertEqual(metrics["image_transport"], "binary")

    def test_model_inputs_match_legacy_conversion_exactly_on_cpu(self):
        data = decode_request(encode_request(self.payload()))
        actual = prepare_image_batch(data["image"], device="cpu")
        for result, bgr in zip(actual, data["image"]):
            reference = to_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            self.assertTrue(torch.equal(result, reference))
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(tuple(result.shape), (3, 448, 448))

    def test_legacy_json_remains_readable_and_produces_same_images(self):
        legacy = make_payload(*self.frames, [0] * 7, "test")
        decoded = decode_request(encode_request(legacy, "json"))
        for legacy_image, binary_image in zip(decoded["image"], decode_request(encode_request(self.payload()))["image"]):
            np.testing.assert_array_equal(legacy_image, binary_image)

    def test_existing_448_image_is_not_resized(self):
        frame = np.zeros((448, 448, 3), dtype=np.uint8)
        self.assertIs(make_payload(frame, frame, [0] * 7, "test", image_transport="binary")["image"][0], frame)

    def test_malformed_prefix_and_byte_lengths_are_rejected(self):
        message = encode_request(self.payload())
        for value in (b"wrong", message[:5], message[:-1], message + b"x",
                      MAGIC + struct.pack(">I", 70000), MAGIC + struct.pack(">I", 10) + b"{}"):
            with self.subTest(size=len(value)), self.assertRaises(ValueError):
                decode_request(value)

    def test_omitted_active_image_is_rejected(self):
        payload = self.payload()
        payload["image_mask"] = [1, 1, 1]
        with self.assertRaisesRegex(ValueError, "Omitting"):
            encode_request(payload)
        message = encode_request(self.payload())
        length = struct.unpack_from(">I", message, len(MAGIC))[0]
        header = json.loads(message[PREFIX_SIZE:PREFIX_SIZE + length])
        header["image_mask"] = [1, 1, 1]
        encoded_header = json.dumps(header).encode()
        changed = MAGIC + struct.pack(">I", len(encoded_header)) + encoded_header + message[PREFIX_SIZE + length:]
        with self.assertRaisesRegex(ValueError, "Omitting"):
            decode_request(changed)

    def test_wrong_dtype_and_image_shape_are_rejected(self):
        for image in (np.zeros((448, 448, 3), dtype=np.float32), np.zeros((32, 32, 3), dtype=np.uint8)):
            payload = self.payload()
            payload["image"][0] = image
            with self.assertRaisesRegex(ValueError, "uint8 BGR"):
                encode_request(payload)

    def test_legacy_different_resolution_still_uses_same_resize(self):
        images = [*self.frames, np.zeros((24, 24, 3), dtype=np.uint8)]
        actual = prepare_image_batch(images, device="cpu")
        for result, bgr in zip(actual, images):
            reference = to_tensor(cv2.cvtColor(cv2.resize(bgr, (448, 448)), cv2.COLOR_BGR2RGB))
            self.assertTrue(torch.equal(result, reference))


if __name__ == "__main__":
    unittest.main()
