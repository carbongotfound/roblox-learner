"""Regression checks for evidence handling; generated fixtures are not gameplay."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from roblox_learner.evaluate import EvaluationError, evaluate, wilson_interval


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_hash = "a" * 64

    def write_jsonl(self, path, records):
        path.write_text("".join(json.dumps(row) + "\n" for row in records))

    def episode(self, episode_id="one", complete=True):
        folder = self.root / episode_id
        folder.mkdir()
        Image.new("RGB", (32, 32), "red").save(folder / "start.png")
        Image.new("RGB", (32, 32), "green").save(folder / "finish.png")
        rows = [
            {"type": "start", "episode_id": episode_id, "timestamp": 10,
             "frame_path": "start.png", "model_sha256": self.model_hash},
            {"type": "step", "episode_id": episode_id, "timestamp": 11,
             "frame_path": "finish.png", "inference_ms": 8, "rss_mb": 300},
        ]
        if complete:
            rows.append({"type": "stop", "episode_id": episode_id, "timestamp": 12,
                         "frame_path": "finish.png", "duration": 2, "terminal_frame_current": True})
        path = folder / "episode.jsonl"
        self.write_jsonl(path, rows)
        return path

    def annotation(self, episode_id="one", **changes):
        terminal = self.root / episode_id / "finish.png"
        result = {"episode_id": episode_id, "outcome": "win", "criterion_id": "fixture_only",
                  "reviewed_by": "unit-test fixture", "success_visible": True,
                  "unassisted": True, "interventions": 0,
                  "checkpoint_sha256": self.model_hash, "terminal_frame": "finish.png",
                  "terminal_sha256": hashlib.sha256(terminal.read_bytes()).hexdigest()}
        result.update(changes)
        path = self.root / "outcomes.jsonl"
        self.write_jsonl(path, [result])
        return path

    def test_no_trials_is_not_zero_percent_or_success(self):
        report = evaluate([])
        self.assertEqual(report["status"], "no_trials")
        self.assertIsNone(report["verified_success_rate"])
        self.assertIsNone(report["verified_success_wilson_95"])

    def test_wilson_accounts_for_small_samples(self):
        lo, hi = wilson_interval(1, 1)
        self.assertAlmostEqual(lo, 0.20654931437723745)
        self.assertAlmostEqual(hi, 1)
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)

    def test_unreviewed_trial_stays_in_denominator(self):
        log = self.episode()
        other = self.episode("two")
        report = evaluate([log, other], self.annotation())
        self.assertEqual(report["trials"], 2)
        self.assertEqual(report["verified_wins"], 1)
        self.assertEqual(report["verified_success_rate"], 0.5)
        self.assertEqual(report["unannotated_trials"], 1)

    def test_reported_win_without_terminal_evidence_is_not_verified(self):
        log = self.episode()
        outcomes = self.annotation(terminal_frame="missing.png")
        report = evaluate([log], outcomes)
        self.assertEqual(report["reported_wins"], 1)
        self.assertEqual(report["verified_wins"], 0)

    def test_matching_hash_required(self):
        log = self.episode()
        report = evaluate([log], self.annotation(terminal_sha256="b" * 64))
        self.assertEqual(report["verified_wins"], 0)
        self.assertIn("annotation lacks matching terminal_sha256", report["episodes"][0]["reasons"])

    def test_human_intervention_cannot_be_autonomous_win(self):
        log = self.episode()
        report = evaluate([log], self.annotation(interventions=1))
        self.assertEqual(report["verified_wins"], 0)

    def test_checkpoint_mismatch_rejected(self):
        log = self.episode()
        report = evaluate([log], self.annotation(checkpoint_sha256="b" * 64))
        self.assertEqual(report["verified_wins"], 0)

    def test_cached_stop_screenshot_cannot_verify_current_terminal_state(self):
        log = self.episode()
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        rows[-1]["terminal_frame_current"] = False
        self.write_jsonl(log, rows)
        report = evaluate([log], self.annotation())
        self.assertEqual(report["verified_wins"], 0)

    def test_crashed_episode_cannot_be_win(self):
        log = self.episode(complete=False)
        report = evaluate([log], self.annotation())
        self.assertEqual(report["trials"], 1)
        self.assertEqual(report["verified_wins"], 0)

    def test_different_criterion_does_not_count(self):
        log = self.episode()
        report = evaluate([log], self.annotation(), criterion_id="shipment_complete")
        self.assertEqual(report["verified_wins"], 0)

    def test_duplicate_trials_fail_instead_of_inflating_rate(self):
        log = self.episode()
        with self.assertRaises(EvaluationError):
            evaluate([log, log])

    def test_duplicate_outcomes_fail(self):
        log = self.episode()
        annotation = self.annotation()
        row = json.loads(annotation.read_text())
        self.write_jsonl(annotation, [row, row])
        with self.assertRaises(EvaluationError):
            evaluate([log], annotation)

    def test_unknown_episode_outcomes_fail(self):
        log = self.episode()
        annotation = self.annotation()
        row = json.loads(annotation.read_text())
        row["episode_id"] = "invented"
        self.write_jsonl(annotation, [row])
        with self.assertRaises(EvaluationError):
            evaluate([log], annotation)

    def test_invalid_json_is_not_silently_skipped(self):
        path = self.root / "corrupt.jsonl"
        path.write_text('{"type":"start"}\nnot json\n')
        with self.assertRaises(EvaluationError):
            evaluate([path])

    def test_performance_missing_samples_stay_unknown(self):
        log = self.episode()
        report = evaluate([log])
        self.assertEqual(report["performance"]["inference_ms"]["p95"], 8)
        self.assertIsNone(report["performance"]["end_to_end_ms"]["p95"])
        self.assertEqual(report["performance"]["rss_mb"]["max"], 300)


if __name__ == "__main__":
    unittest.main()
