"""Check model timing boundaries without requiring CUDA or a model download."""

import asyncio
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import Evo1_server as server
from scripts.inference_transport import encode_request


class InferenceTimingTests(unittest.TestCase):
    def test_inference_excludes_input_and_output_processing(self):
        clock = [0.0]
        synchronizations = []
        image = torch.zeros(3, 448, 448)
        actions = torch.zeros(1, 50, 24)

        def prepare(images, device):
            clock[0] += 0.3
            return [image] * 3

        def normalize(state, *keys):
            clock[0] += 0.2
            return state

        def infer(**inputs):
            clock[0] += 0.02
            return actions

        def synchronize(device):
            synchronizations.append(device)
            # Pending preprocessing before the start boundary is excluded;
            # unfinished model kernels at the end boundary are included.
            clock[0] += 1.0 if len(synchronizations) == 1 else 0.03

        def denormalize(action, *keys):
            clock[0] += 0.4
            return action

        model = MagicMock()
        model.run_inference.side_effect = infer
        normalizer = MagicMock()
        normalizer.normalize_state.side_effect = normalize
        normalizer.denormalize_action.side_effect = denormalize
        original_tensor, original_as_tensor = torch.tensor, torch.as_tensor

        def cpu_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return original_tensor(*args, **kwargs)

        def cpu_as_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return original_as_tensor(*args, **kwargs)

        data = {"image": [0, 0, 0], "state": [0] * 7, "prompt": "test",
                "image_mask": [1, 1, 0], "action_mask": [1] * 7 + [0] * 17}
        metrics = {}
        with patch.object(server, "prepare_image_batch", side_effect=prepare), \
                patch.object(server.torch, "tensor", side_effect=cpu_tensor), \
                patch.object(server.torch, "as_tensor", side_effect=cpu_as_tensor), \
                patch.object(server.torch.cuda, "synchronize", side_effect=synchronize), \
                patch.object(server.torch.amp, "autocast", side_effect=lambda **kwargs: nullcontext()), \
                patch.object(server.time, "perf_counter", side_effect=lambda: clock[0]), \
                redirect_stdout(io.StringIO()):
            result = server.infer_from_json_dict(data, model, normalizer, "arm", "dataset", metrics=metrics)
        self.assertEqual(synchronizations, ["cuda", "cuda", "cuda"])
        self.assertAlmostEqual(metrics["model_inference_ms"], 50.0)
        self.assertAlmostEqual(metrics["server_preprocess_ms"], 1500.0)
        self.assertAlmostEqual(metrics["server_postprocess_ms"], 400.0)
        self.assertEqual(len(result), 50)
        self.assertEqual(len(result[0]), 24)
        model.run_inference.assert_called_once()

    def test_model_error_still_records_elapsed_time(self):
        model = MagicMock()
        model.run_inference.side_effect = RuntimeError("model failed")
        metrics = {}
        with patch.object(server.torch.cuda, "synchronize") as synchronize, \
                patch.object(server.time, "perf_counter", side_effect=[1.0, 1.025]):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                server.timed_model_inference(model, metrics)
        self.assertEqual(synchronize.call_count, 2)
        self.assertAlmostEqual(metrics["model_inference_ms"], 25.0)

    def run_handler(self, *, fail=False, return_timing=False, binary=False):
        log = io.StringIO()
        self.stats = server.InferenceTimingStats()
        self.latency_stats = server.LatencyStats(server.SERVER_FIELDS)
        actions = [[0] * 24 for _ in range(50)]

        class WebSocket:
            async def messages(self):
                data = {"arm_key": "request_arm", "return_timing": return_timing,
                        "request_id": "client-request"}
                if binary:
                    data.update({"image": [np.zeros((448, 448, 3), dtype=np.uint8)] * 2,
                                 "image_mask": [1, 1, 0]})
                    yield encode_request(data)
                else:
                    yield json.dumps(data)

            def __aiter__(self):
                return self.messages()

        websocket = WebSocket()

        async def send(message):
            # The timing record must be persisted before transmitting actions.
            self.assertEqual(len(log.getvalue().splitlines()), 1)
            response = json.loads(message)
            if return_timing:
                self.assertEqual(response["actions"], actions)
                self.assertEqual(response["request_id"], "client-request")
                self.assertEqual(response["timings"]["model_inference_ms"], 12.5)
                self.assertIn("server_processing_ms", response["timings"])
            else:
                self.assertEqual(response, actions)

        websocket.send = AsyncMock(side_effect=send)

        def infer(*args, metrics):
            if binary:
                self.assertEqual(len(args[0]["image"]), 3)
                self.assertEqual(args[0]["image"][0].shape, (448, 448, 3))
                self.assertFalse(args[0]["image"][2].any())
            metrics["model_inference_ms"] = 12.5
            if fail:
                raise RuntimeError("model failed")
            return actions

        with patch.object(server, "infer_from_json_dict", side_effect=infer), redirect_stdout(io.StringIO()):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "model failed"):
                    asyncio.run(server.handle_request(websocket, None, None, "arm", "dataset",
                                                      log_file=log, run_id="test", stats=self.stats,
                                                      latency_stats=self.latency_stats))
            else:
                asyncio.run(server.handle_request(websocket, None, None, "arm", "dataset",
                                                  log_file=log, run_id="test", stats=self.stats,
                                                  latency_stats=self.latency_stats))
        return websocket, json.loads(log.getvalue())

    def test_server_logs_model_time_without_changing_response(self):
        websocket, record = self.run_handler()
        self.assertEqual(record["model_inference_ms"], 12.5)
        self.assertEqual(record["chunk_length"], 50)
        self.assertEqual(record["arm_key"], "request_arm")
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["run_id"], "test")
        websocket.send.assert_awaited_once()
        self.assertEqual(self.stats.summary()["inference_count"], 1)
        self.assertEqual(self.stats.summary()["mean_model_inference_ms"], 12.5)

    def test_server_logs_inference_failure_without_sending_actions(self):
        websocket, record = self.run_handler(fail=True)
        self.assertEqual(record["status"], "error")
        self.assertIn("model failed", record["error"])
        self.assertEqual(record["model_inference_ms"], 12.5)
        websocket.send.assert_not_awaited()
        self.assertEqual(self.stats.summary()["failed_requests"], 1)
        self.assertIsNone(self.stats.summary()["mean_model_inference_ms"])
        self.assertEqual(self.latency_stats.sample_count, 0)

    def test_timing_response_is_opt_in_and_includes_request_identity(self):
        websocket, record = self.run_handler(return_timing=True)
        self.assertEqual(record["request_id"], "client-request")
        self.assertGreaterEqual(record["server_response_encode_ms"], 0)
        self.assertEqual(self.latency_stats.summary()["metrics"]["model_inference_ms"]["mean_ms"], 12.5)
        websocket.send.assert_awaited_once()

    def test_binary_request_preserves_timing_and_response_protocol(self):
        websocket, record = self.run_handler(return_timing=True, binary=True)
        self.assertEqual(record["image_transport"], "binary")
        self.assertGreaterEqual(record["server_unpack_ms"], 0)
        self.assertEqual(record["request_id"], "client-request")
        websocket.send.assert_awaited_once()

    def test_exit_summary_averages_only_successful_inference(self):
        stats = server.InferenceTimingStats()
        for status, latency in (("ok", 25.0), ("error", 1000.0),
                                ("ok", 75.0), ("interrupted", None)):
            stats.observe({"status": status, "model_inference_ms": latency})
        log, output = io.StringIO(), io.StringIO()
        with redirect_stdout(output):
            server.save_server_summary(log, "session", stats)
        record = json.loads(log.getvalue())
        self.assertEqual(record["event"], "session_end")
        self.assertEqual(record["run_id"], "session")
        self.assertEqual(record["request_count"], 4)
        self.assertEqual(record["inference_count"], 2)
        self.assertEqual(record["mean_model_inference_ms"], 50.0)
        self.assertEqual(record["total_model_inference_ms"], 100.0)
        self.assertEqual(record["failed_requests"], 1)
        self.assertEqual(record["interrupted_requests"], 1)
        self.assertIn("50.000 ms", output.getvalue())

    def test_exit_without_requests_records_null_average(self):
        log, output = io.StringIO(), io.StringIO()
        with redirect_stdout(output):
            server.save_server_summary(log, "empty", server.InferenceTimingStats())
        record = json.loads(log.getvalue())
        self.assertEqual(record["inference_count"], 0)
        self.assertIsNone(record["mean_model_inference_ms"])
        self.assertIn("no successful inference", output.getvalue())


if __name__ == "__main__":
    unittest.main()
