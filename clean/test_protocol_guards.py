"""Isolated authorization tests; never read real data or the real final ledger."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from clean import dataset, run


class ProtocolGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "results").mkdir()
        (self.root / "results/full_audit.json").write_text('{"synthetic_test_only": true}')
        self.split = {"train": [0], "test": [200], "out_of_range": [230]}
        self.ledger = self.root / "results/final_evaluations.jsonl"

    def write_ledger(self, events):
        self.ledger.write_text("".join(json.dumps(event) + "\n" for event in events))

    def assert_loader_denied(self, group, episodes, purpose, candidate=None, match=None):
        with mock.patch.object(dataset, "ROOT", self.root), \
             mock.patch.object(dataset, "verify", return_value=self.split), \
             mock.patch.object(dataset, "record_access") as record, \
             mock.patch.object(dataset.np, "load") as load:
            with self.assertRaisesRegex(PermissionError, match or ".+"):
                list(dataset.episode_arrays(group, episodes, purpose, candidate))
            load.assert_not_called()
            record.assert_not_called()

    def test_heldout_data_requires_an_active_matching_final(self):
        for group in ("test", "out_of_range"):
            self.assert_loader_denied(group, range(5), "final_fit")
        self.write_ledger([{"event": "begin", "candidate": "chosen"}])
        self.assert_loader_denied("test", range(5), "final_fit", "other")
        self.assert_loader_denied("test", range(5), "train", "chosen", "nonfinal purpose")
        self.write_ledger([{"event": "begin", "candidate": "chosen"},
                           {"event": "complete", "candidate": "chosen"}])
        self.assert_loader_denied("test", range(5), "final_fit", "chosen")
        self.write_ledger([{"event": "begin", "candidate": "chosen"}] * 2)
        self.assert_loader_denied("out_of_range", range(5), "final_fit", "chosen")

    def test_final_fit_and_score_episodes_cannot_overlap(self):
        self.write_ledger([{"event": "begin", "candidate": "chosen"}])
        self.assert_loader_denied("test", [0, 5], "final_fit", "chosen", "role mismatch")
        self.assert_loader_denied("out_of_range", [4, 9], "final_score", "chosen", "role mismatch")
        self.assert_loader_denied("train", [0, 8], "train", match="Validation episodes")

    def test_valid_final_roles_pass_only_the_guard_without_opening_data(self):
        self.write_ledger([{"event": "begin", "candidate": "chosen"}])
        empty_split = {name: [] for name in self.split}
        with mock.patch.object(dataset, "ROOT", self.root), \
             mock.patch.object(dataset, "verify", return_value=empty_split), \
             mock.patch.object(dataset, "record_access") as record, \
             mock.patch.object(dataset.np, "load") as load:
            self.assertEqual(list(dataset.episode_arrays("test", range(5), "final_fit", "chosen")), [])
            self.assertEqual(list(dataset.episode_arrays("out_of_range", range(5, 10), "final_score", "chosen")), [])
            self.assertEqual(record.call_count, 2)
            load.assert_not_called()

    def test_resume_seal_mismatch_stops_before_loading_or_training(self):
        out = self.root / "runs/iteration_01"
        out.mkdir(parents=True)
        (out / "seal.json").write_text(json.dumps({"old_source": "different"}))
        with mock.patch.object(run, "ROOT", self.root), \
             mock.patch.object(run, "require_audit", return_value=({}, {})), \
             mock.patch.object(run, "fit_normalizer", return_value={}), \
             mock.patch.object(run, "digest", side_effect=lambda path: str(path)), \
             mock.patch.object(run, "load_pack") as load, \
             mock.patch.object(run, "train_model") as train:
            with self.assertRaisesRegex(RuntimeError, "cannot mix seed runs"):
                run.candidate(1, copy.deepcopy(run.DEFAULT), "hypothesis", "change", "cpu")
            load.assert_not_called()
            train.assert_not_called()

    def write_candidate(self, success):
        out = self.root / "runs/iteration_01"
        out.mkdir(parents=True)
        (out / "result.json").write_text(json.dumps({
            "config": dict(run.DEFAULT), "success": {f"S{i}": success for i in range(1, 5)}, "runs": [],
            "audit_sha256": run.digest(self.root / "results/full_audit.json")}))

    def test_candidate_from_old_data_version_cannot_open_final_data(self):
        self.write_candidate(True)
        (self.root / "results/full_audit.json").write_text('{"synthetic_test_only": "new data version"}')
        with mock.patch.object(run, "ROOT", self.root), \
             mock.patch.object(run, "require_audit", return_value=({}, {})), \
             mock.patch.object(run, "verify"), \
             mock.patch("clean.recovery.prepare_recovery") as prepare, \
             mock.patch.object(run, "load_pack") as load:
            with self.assertRaisesRegex(RuntimeError, "different data version"):
                run.final(1, "cpu")
            load.assert_not_called()
            prepare.assert_not_called()
        self.assertFalse(self.ledger.exists())

    def test_stale_pilot_generator_cannot_authorize_full_collection(self):
        from clean import collect
        for name in ("schema.py", "native.py", "script.py"):
            (self.root / name).write_text("synthetic current generator")
        report = {"status": "pass", "static_provenance": {"source_sha256": {
            name: run.digest(self.root / name) for name in ("schema.py", "native.py", "script.py")}},
            "episode_sha256": {}}
        report["static_provenance"]["source_sha256"]["script.py"] = "previous_generator"
        (self.root / "results/pilot_audit.json").write_text(json.dumps(report))
        with mock.patch.object(collect, "ROOT", self.root), \
             mock.patch.object(collect, "initialize"), \
             mock.patch("sys.argv", ["collect", "--stage", "full"]), \
             mock.patch.object(collect, "make_env") as make_env:
            with self.assertRaisesRegex(RuntimeError, "different generator"):
                collect.main()
            make_env.assert_not_called()

    def test_changed_pilot_data_cannot_authorize_full_collection(self):
        from clean import collect
        for name in ("schema.py", "native.py", "script.py"):
            (self.root / name).write_text("synthetic current generator")
        episode = self.root / "data/train/cube_0000_episode_00.npz"
        episode.parent.mkdir(parents=True)
        episode.write_bytes(b"synthetic changed episode; not an actual archive")
        report = {"status": "pass", "static_provenance": {"source_sha256": {
            name: run.digest(self.root / name) for name in ("schema.py", "native.py", "script.py")}},
            "episode_sha256": {"train/cube_0000_episode_00.npz": "previous_episode"}}
        (self.root / "results/pilot_audit.json").write_text(json.dumps(report))
        with mock.patch.object(collect, "ROOT", self.root), \
             mock.patch.object(collect, "initialize"), \
             mock.patch("sys.argv", ["collect", "--stage", "full"]), \
             mock.patch.object(collect, "make_env") as make_env:
            with self.assertRaisesRegex(RuntimeError, "Pilot gate data changed"):
                collect.main()
            make_env.assert_not_called()

    def test_failing_early_candidate_cannot_spend_final_access(self):
        self.write_candidate(False)
        with mock.patch.object(run, "ROOT", self.root), \
             mock.patch.object(run, "require_audit", return_value=({}, {})), \
             mock.patch.object(run, "verify"), \
             mock.patch("clean.recovery.prepare_recovery") as prepare, \
             mock.patch.object(run, "load_pack") as load:
            with self.assertRaisesRegex(RuntimeError, "validation success or all20"):
                run.final(1, "cpu")
            load.assert_not_called()
            prepare.assert_not_called()
        self.assertFalse(self.ledger.exists())

    def test_existing_final_ledger_blocks_retry_and_further_tuning(self):
        self.write_candidate(True)
        self.write_ledger([{"event": "begin", "candidate": "previous"}])
        with mock.patch.object(run, "ROOT", self.root), \
             mock.patch.object(run, "require_audit", return_value=({}, {})), \
             mock.patch.object(run, "verify"), \
             mock.patch("clean.recovery.prepare_recovery") as prepare, \
             mock.patch.object(run, "load_pack") as load, \
             mock.patch.object(run, "fit_normalizer") as norm:
            with self.assertRaisesRegex(RuntimeError, "already invoked"):
                run.final(1, "cpu")
            with self.assertRaisesRegex(RuntimeError, "no further tuning"):
                run.candidate(2, copy.deepcopy(run.DEFAULT), "hypothesis", "change", "cpu")
            load.assert_not_called()
            norm.assert_not_called()
            prepare.assert_not_called()

    def test_final_begin_is_saved_before_any_heldout_loader_call(self):
        self.write_candidate(True)
        class StopBeforeData(Exception):
            pass
        def prepare(candidate_dir, normalizer, device):
            self.assertFalse(self.ledger.exists())
            self.assertEqual(candidate_dir, self.root / "runs/iteration_01")
            self.assertEqual(normalizer, {})
            self.assertEqual(device, "cpu")
            recovery = candidate_dir / "recovery"
            recovery.mkdir()
            (recovery / "preparation.json").write_text('{"synthetic_test_only": true}')
        def loader(group, episodes, normalizer, device, purpose, final_candidate):
            events = [json.loads(line) for line in self.ledger.read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event"], "begin")
            self.assertEqual(events[0]["candidate"], final_candidate)
            self.assertEqual(group, "test")
            self.assertEqual(list(episodes), list(range(5)))
            self.assertEqual(purpose, "final_fit")
            seal = json.loads((self.root / "results/final/seal.json").read_text())
            self.assertIn("runs/iteration_01/recovery/preparation.json", seal["recovery_sha256"])
            raise StopBeforeData()
        with mock.patch.object(run, "ROOT", self.root), \
             mock.patch.object(run, "require_audit", return_value=({}, {})), \
             mock.patch.object(run, "verify"), \
             mock.patch.object(run, "fit_normalizer", return_value={}), \
             mock.patch("clean.recovery.prepare_recovery", side_effect=prepare) as preparation, \
             mock.patch.object(run, "load_pack", side_effect=loader):
            with self.assertRaises(StopBeforeData):
                run.final(1, "cpu")
            preparation.assert_called_once()

    def test_cache_key_tracks_scaling_and_keeps_unaffected_baselines(self):
        for path in ["data/protocol.json", "data/normalizer.json", "results/full_audit.json",
                     "models.py", "train.py", "dataset.py"]:
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("initial")
        with mock.patch.object(run, "ROOT", self.root):
            baseline = run.key_for("plain", run.DEFAULT, 101)
            two_part = run.key_for("two_part", run.DEFAULT, 101)
            config = dict(run.DEFAULT, z_dim=4, fit_steps=400)
            self.assertEqual(run.key_for("plain", config, 101), baseline)
            self.assertNotEqual(run.key_for("two_part", config, 101), two_part)
            (self.root / "data/normalizer.json").write_text("changed")
            self.assertNotEqual(run.key_for("plain", run.DEFAULT, 101), baseline)


if __name__ == "__main__":
    unittest.main()
