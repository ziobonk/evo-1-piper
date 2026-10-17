"""Exercise target rejection without connecting to cameras or a robot."""

import asyncio
from contextlib import redirect_stdout
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import Evo1_client_piper as client
from inference_transport import decode_request


class PiperClientTests(unittest.TestCase):
    def run_client(self, *, execute, grip, position_delta=0, cycles=2,
                   chunk_length=1, steps_per_chunk=1, reject_keyframe=None,
                   fail_pose_command=None, server_timing=False, mismatched_request=False,
                   image_transport="binary"):
        current = np.array([0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
        target = current.copy()
        target[0] += position_delta
        target[6] = grip
        piper = MagicMock()
        piper.EnablePiper.return_value = True
        if fail_pose_command is not None:
            def send_pose(*args):
                if piper.EndPoseCtrl.call_count == fail_pose_command:
                    raise RuntimeError("Simulated CAN send failure")
            piper.EndPoseCtrl.side_effect = send_pose
        sdk = ModuleType("piper_sdk")
        sdk.C_PiperInterface_V2 = MagicMock(return_value=piper)
        camera = MagicMock()
        camera.read.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
        chunk = np.repeat(target[None], chunk_length, axis=0)
        if reject_keyframe is not None:
            chunk[reject_keyframe, 0] += 0.08
        response = json.dumps(chunk.tolist())
        ws = SimpleNamespace(send=AsyncMock(), recv=AsyncMock(return_value=response))
        if server_timing:
            async def timed_response():
                request_id = decode_request(ws.send.call_args.args[0])["request_id"]
                return json.dumps({
                    "actions": chunk.tolist(), "request_id": "wrong" if mismatched_request else request_id,
                    "server_run_id": "server-test",
                    "timings": {"model_inference_ms": 0.4, "server_preprocess_ms": 0.1,
                                "server_json_decode_ms": 0.1, "server_postprocess_ms": 0.1,
                                "server_processing_ms": 0.7},
                })
            ws.recv.side_effect = timed_response
        if cycles == 0:
            ws.recv.side_effect = [response] * 2 + [KeyboardInterrupt()]
        connection = MagicMock()
        connection.__aenter__ = AsyncMock(return_value=ws)
        connection.__aexit__ = AsyncMock(return_value=False)
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            meta = dataset / "meta"
            meta.mkdir()
            (meta / "info.json").write_text(json.dumps({
                "robot_type": "piper_eef", "pose_coordinate_frame": "robot_base_tcp",
                "tcp_offset_m": [0, 0, 0], "gripper_mapping": {"robot_open_width_mm": 70},
                "fps": 20, "features": {key: {"shape": [2, 2, 3]} for key in ("image", "wrist_image")},
            }))
            (meta / "tasks.jsonl").write_text(json.dumps({"task": "test"}))
            (meta / "stats.json").write_text(json.dumps({
                "observation.state": {"min": [-1] * 7, "max": [1] * 7},
            }))
            args = SimpleNamespace(
                dataset=dataset, control_hz=100, camera_backend="opencv",
                base_camera="0", wrist_camera="1", camera_fps=30, camera_warmup_frames=0,
                can_name="can0", execute=execute, speed_pct=20, server="ws://test",
                cycles=cycles, steps_per_chunk=steps_per_chunk, max_step_m=0.02, max_step_rad=0.15,
                log_path=dataset / "logs" / "inference.jsonl",
                image_transport=image_transport,
            )
            ticks = itertools.count()
            with patch.dict(sys.modules, {"piper_sdk": sdk}), \
                    patch.object(client, "OpenCVCamera", return_value=camera), \
                    patch.object(client, "read_state", side_effect=lambda *args: current.copy()), \
                    patch.object(client.asyncio, "to_thread", new=AsyncMock(side_effect=lambda fn: fn())), \
                    patch.object(client.asyncio, "sleep", new=AsyncMock()), \
                    patch.object(client.time, "perf_counter", side_effect=lambda: next(ticks) * 0.001), \
                    patch.object(client.websockets, "connect", return_value=connection) as connect, \
                    redirect_stdout(output):
                error = None
                try:
                    asyncio.run(client.run(args))
                except (RuntimeError, KeyboardInterrupt) as exc:
                    error = exc
            self.log_events = [json.loads(line) for line in args.log_path.read_text().splitlines()]
            self.latency_report = json.loads(args.log_path.with_suffix(".latency.json").read_text())
            self.assertIsNone(connect.call_args.kwargs["compression"])

        if not execute:
            piper.EndPoseCtrl.assert_not_called()
            piper.GripperCtrl.assert_not_called()
        piper.DisconnectPort.assert_called_once()
        piper.DisableArm.assert_not_called()
        self.assertEqual(camera.close.call_count, 2)
        return piper, ws, output.getvalue(), error

    def test_dry_run_clips_negative_width_and_continues(self):
        piper, ws, output, error = self.run_client(execute=False, grip=-0.25)
        self.assertIsNone(error)
        self.assertEqual(ws.send.await_count, 2)
        self.assertNotIn("rejected:", output)
        self.assertEqual(output.count("gripper clipped: -0.25 -> 0 mm"), 2)
        piper.EnablePiper.assert_not_called()

    def test_execute_sends_clipped_width(self):
        for grip, raw_grip in ((-0.25, 0), (70.25, 70000)):
            with self.subTest(grip=grip):
                piper, ws, output, error = self.run_client(execute=True, grip=grip)
                self.assertIsNone(error)
                self.assertEqual(ws.send.await_count, 2)
                self.assertEqual(piper.GripperCtrl.call_count, 10)
                self.assertEqual(piper.GripperCtrl.call_args.args[0], raw_grip)
                self.assertTrue(all(0 <= call.args[0] <= 70000 for call in piper.GripperCtrl.call_args_list))
                self.assertIn("gripper clipped:", output)
                self.assertIn("Completed 2 inference cycles", output)

    def test_execute_rejects_nonfinite_width_without_disabling_arm(self):
        for grip in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(grip=grip):
                piper, ws, output, error = self.run_client(execute=True, grip=grip)
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("invalid 7D target", str(error))
                self.assertEqual(ws.send.await_count, 1)
                self.assertIn("target=", output)
                piper.EndPoseCtrl.assert_not_called()
                self.assertIsNone(self.log_events[1]["action_chunk"][0][6])
                self.assertEqual(self.log_events[1]["status"], "rejected")

    def test_clipping_does_not_allow_large_pose_step(self):
        for execute in (False, True):
            with self.subTest(execute=execute):
                piper, ws, output, error = self.run_client(execute=execute, grip=-0.25, position_delta=0.08)
                self.assertIn("gripper clipped:", output)
                if execute:
                    self.assertIn("Target step too large", str(error))
                    self.assertEqual(ws.send.await_count, 1)
                    piper.EndPoseCtrl.assert_not_called()
                else:
                    self.assertIsNone(error)
                    self.assertEqual(output.count("rejected: Target step too large"), 2)
                    self.assertEqual(ws.send.await_count, 2)

    def test_valid_dry_run_still_sends_no_commands(self):
        _, ws, output, error = self.run_client(execute=False, grip=0)
        self.assertIsNone(error)
        self.assertEqual(ws.send.await_count, 2)
        self.assertNotIn("rejected:", output)

    def test_continuous_mode_cleans_up_on_interrupt_without_disabling_arm(self):
        piper, ws, output, error = self.run_client(execute=True, grip=0, cycles=0)
        self.assertIsInstance(error, KeyboardInterrupt)
        self.assertEqual(ws.send.await_count, 3)
        self.assertEqual(piper.EndPoseCtrl.call_count, 10)
        self.assertNotIn("Completed", output)
        self.assertEqual(self.log_events[-2]["status"], "interrupted")
        self.assertIsNone(self.log_events[-2]["inference_latency_ms"])
        self.assertGreater(self.log_events[-2]["request_elapsed_ms"], 0)
        self.assertEqual(self.log_events[-1]["status"], "interrupted")
        self.assertEqual(self.log_events[-1]["executed_steps"], 2)
        self.assertEqual(self.log_events[-1]["sent_control_steps"], 10)

    def test_log_separates_returned_chunk_selected_steps_and_control_steps(self):
        self.run_client(execute=True, grip=-0.25, chunk_length=50, steps_per_chunk=3)
        self.assertEqual([item["event"] for item in self.log_events],
                         ["session_start", "cycle", "cycle", "session_end"])
        record = self.log_events[1]
        self.assertEqual(record["chunk_length"], 50)
        self.assertEqual(record["requested_steps"], 3)
        self.assertEqual(record["executed_steps"], 3)
        self.assertEqual(record["sent_control_steps"], 15)
        self.assertAlmostEqual(record["inference_latency_ms"], 1.0)
        self.assertEqual(len(record["action_chunk"]), 50)
        self.assertEqual(len(record["accepted_targets"]), 3)
        self.assertEqual(record["action_chunk"][0][6], -0.25)
        self.assertEqual(record["accepted_targets"][0][6], 0)
        self.assertEqual(self.log_events[-1]["executed_steps"], 6)
        self.assertEqual(self.log_events[-1]["sent_control_steps"], 30)

    def test_dry_run_records_zero_actual_execution(self):
        self.run_client(execute=False, grip=0, chunk_length=50, steps_per_chunk=3)
        record = self.log_events[1]
        self.assertEqual(record["status"], "dry_run")
        self.assertEqual(record["processed_steps"], 3)
        self.assertEqual(record["executed_steps"], 0)
        self.assertEqual(record["sent_control_steps"], 0)

    def test_rejection_logs_preceding_executed_steps(self):
        piper, _, _, error = self.run_client(
            execute=True, grip=0, chunk_length=3, steps_per_chunk=3, reject_keyframe=1)
        self.assertIsInstance(error, RuntimeError)
        record = self.log_events[1]
        self.assertEqual(record["status"], "rejected")
        self.assertEqual(record["keyframe_index"], 1)
        self.assertEqual(record["executed_steps"], 1)
        self.assertEqual(record["sent_control_steps"], 5)
        self.assertEqual(piper.EndPoseCtrl.call_count, 5)
        self.assertEqual(self.log_events[-1]["status"], "error")

    def test_partial_control_failure_preserves_completed_counts(self):
        _, _, _, error = self.run_client(execute=True, grip=0, fail_pose_command=2)
        self.assertIsInstance(error, RuntimeError)
        record = self.log_events[1]
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["executed_steps"], 0)
        self.assertEqual(record["sent_control_steps"], 1)
        self.assertIn("CAN send failure", record["error"])

    def test_separate_json_contains_both_server_and_client_stages(self):
        _, _, _, error = self.run_client(execute=False, grip=0, server_timing=True)
        self.assertIsNone(error)
        record = self.log_events[1]
        self.assertEqual(record["server_model_inference_ms"], 0.4)
        self.assertEqual(record["server_run_id"], "server-test")
        self.assertAlmostEqual(record["communication_and_other_ms"], 0.3)
        metrics = self.latency_report["metrics"]
        self.assertEqual(self.latency_report["sample_count"], 2)
        self.assertEqual(metrics["server_model_inference_ms"]["count"], 2)
        self.assertEqual(metrics["server_model_inference_ms"]["mean_ms"], 0.4)
        self.assertGreater(metrics["client_data_processing_ms"]["mean_ms"], 0)
        self.assertGreater(metrics["payload_build_ms"]["mean_ms"], 0)
        self.assertGreater(metrics["request_encode_ms"]["mean_ms"], 0)

    def test_old_server_response_keeps_unknown_stages_null(self):
        self.run_client(execute=False, grip=0)
        metric = self.latency_report["metrics"]["server_model_inference_ms"]
        self.assertEqual(metric["count"], 0)
        self.assertIsNone(metric["mean_ms"])

    def test_mismatched_timing_response_is_rejected(self):
        _, _, _, error = self.run_client(execute=True, grip=0, server_timing=True, mismatched_request=True)
        self.assertIn("request_id", str(error))
        self.assertEqual(self.latency_report["sample_count"], 0)

    def test_binary_requests_are_default_and_keep_only_two_images_on_wire(self):
        _, ws, _, error = self.run_client(execute=False, grip=0, server_timing=True)
        self.assertIsNone(error)
        message = ws.send.call_args.args[0]
        self.assertIsInstance(message, bytes)
        self.assertLess(len(message), 1_205_000)
        data = decode_request(message)
        self.assertEqual(data["image_count"], 2)
        self.assertEqual(data["image_mask"], [1, 1, 0])
        self.assertEqual(self.log_events[1]["image_transport"], "binary")

    def test_json_transport_remains_available(self):
        _, ws, _, error = self.run_client(execute=False, grip=0, image_transport="json")
        self.assertIsNone(error)
        self.assertIsInstance(ws.send.call_args.args[0], str)
        self.assertEqual(self.log_events[1]["image_transport"], "json")


if __name__ == "__main__":
    unittest.main()
