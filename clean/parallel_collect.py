"""Parallelize the already-audited full collector without changing its script.

Source/reference: ``clean/collect.py``. The reset, property setter, fixed plan,
action execution, measured-state constructor, and episode schema are identical
to that serial collector. Each worker task owns one cube and its environment;
the ten immutable pilot plans are read only. This is collection infrastructure,
not a scientific candidate or a change to script settings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from clean.protocol import ROOT, digest, verify
from clean.script import sample_settings


CRITICAL_SOURCES = ("schema.py", "native.py", "script.py")
COLLECTION_SOURCES = ("parallel_collect.py", "collect.py", *CRITICAL_SOURCES)


def validate_pilot_gate(root=ROOT):
    """Reject stale pilot evidence before starting any worker or simulator."""
    root = Path(root)
    path = root / "results/pilot_audit.json"
    if not path.exists():
        raise RuntimeError("Full collection requires passing saved ten-training-cube audit")
    report = json.loads(path.read_text())
    if report.get("status") != "pass":
        raise RuntimeError("Full collection requires passing saved ten-training-cube audit")
    for name in CRITICAL_SOURCES:
        if report["static_provenance"]["source_sha256"][name] != digest(root / name):
            raise RuntimeError("Pilot gate belongs to a different generator")
    for name, expected in report["episode_sha256"].items():
        if digest(root / "data" / name) != expected:
            raise RuntimeError("Pilot gate data changed")
    settings = [sample_settings(episode) for episode in range(10)]
    saved = json.loads((root / "data/episode_settings.json").read_text())
    if saved != settings:
        raise RuntimeError("Episode settings changed; explicitly version data before recollecting")
    plans = {}
    for episode in range(10):
        plan = root / "data/plans" / f"episode_{episode:02d}.npz"
        if not plan.is_file():
            raise RuntimeError(f"Missing read-only pilot plan: {plan.name}")
        plans[episode] = digest(plan)
    return settings, plans


def verify_receipt(path, cube_id, episode_id, source_hashes):
    """Verify existing bytes and identity without reading trajectory arrays."""
    path = Path(path)
    receipt_path = path.with_suffix(".json")
    if not path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"Existing episode failed integrity: {path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("cube_id") != cube_id or receipt.get("episode_id") != episode_id:
        raise RuntimeError(f"Existing episode identity mismatch: {path}")
    if receipt.get("sha256") != digest(path):
        raise RuntimeError(f"Existing episode failed integrity: {path}")
    for name in CRITICAL_SOURCES:
        if receipt.get("source_sha256", {}).get(name) != source_hashes[name]:
            raise RuntimeError(f"Existing episode has a different generator: {path}")
    return receipt


def _publish_exclusive(temporary, destination):
    """Atomic publication that cannot replace another process's final file."""
    os.link(temporary, destination)
    Path(temporary).unlink()


def _write_json_exclusive(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        _publish_exclusive(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def collect_cube(cube, settings, plan_hashes, source_hashes, root_string):
    """One task owns exactly one cube; properties are passed only to its setter."""
    # A spawned worker imports its simulator locally. Import creates no scene.
    from clean.native import make_env, reset_env, set_cube_properties, execute
    from clean.schema import observe

    root = Path(root_string)
    data = root / "data"
    cube_id = int(cube["cube_id"])
    folder = data / cube["split"]
    folder.mkdir(exist_ok=True)
    env = None
    completed = 0
    skipped = 0
    started = time.monotonic()
    try:
        for setting in settings:
            episode = setting["episode_id"]
            destination = folder / f"cube_{cube_id:04d}_episode_{episode:02d}.npz"
            if destination.exists() or destination.with_suffix(".json").exists():
                verify_receipt(destination, cube_id, episode, source_hashes)
                skipped += 1
                continue
            if env is None:
                env = make_env(setting["scene_seed"])
            else:
                reset_env(env, setting["scene_seed"])
            set_cube_properties(env, cube)
            pose = env.cube.pose.raw_pose[0].cpu().numpy().copy()
            plan_path = data / "plans" / f"episode_{episode:02d}.npz"
            if digest(plan_path) != plan_hashes[episode]:
                raise RuntimeError(f"Read-only pilot plan changed: {plan_path.name}")
            with np.load(plan_path, allow_pickle=False) as plan:
                if not np.array_equal(pose, plan["pose"]):
                    raise RuntimeError("Matched initial pose differs")
                actions = plan["action"].copy()
            states = [observe(env, first=True)]
            for action in actions:
                execute(env, action)
                states.append(observe(env))
            states = np.asarray(states)
            count = len(actions)
            temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
            try:
                with temporary.open("xb") as handle:
                    np.savez_compressed(handle,
                        cube_id=np.full(count, cube_id, np.int32),
                        episode_id=np.full(count, episode, np.int16),
                        step=np.arange(count, dtype=np.int32),
                        state=states[:-1], action=actions, next_state=states[1:])
                _publish_exclusive(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
            receipt = {"cube_id": cube_id, "episode_id": episode, "steps": count,
                "path": str(destination.relative_to(root)), "sha256": digest(destination),
                "source_sha256": source_hashes, "plan_sha256": plan_hashes[episode],
                "collector": "parallel_collect.py", "environment_scope": "one cube per worker task"}
            _write_json_exclusive(destination.with_suffix(".json"), receipt)
            completed += 1
    finally:
        if env is not None:
            env.close()
    return {"cube_id": cube_id, "completed_this_run": completed, "skipped_verified": skipped,
            "elapsed_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between one and eight")
    split = verify()
    settings, plans = validate_pilot_gate(ROOT)
    sources = {name: digest(ROOT / name) for name in COLLECTION_SOURCES}
    cubes = json.loads((ROOT / "data/cubes.json").read_text())["cubes"]
    expected_ids = set(split["train"] + split["test"] + split["out_of_range"])
    if len(cubes) != 260 or {cube["cube_id"] for cube in cubes} != expected_ids:
        raise RuntimeError("Cube registry does not match the immutable split")
    for cube in cubes:
        if cube["cube_id"] not in split[cube["split"]]:
            raise RuntimeError("Cube group does not match the immutable split")
    # Bound native math thread pools in every newly spawned worker.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    started = time.monotonic()
    completed = 0
    skipped = 0
    futures = []
    executor = ProcessPoolExecutor(max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"))
    try:
        futures = [executor.submit(collect_cube, cube, settings, plans, sources, str(ROOT))
                   for cube in cubes]
        for future in as_completed(futures):
            event = future.result()
            completed += event["completed_this_run"]
            skipped += event["skipped_verified"]
            print(json.dumps(dict(stage="full", workers=args.workers, **event)), flush=True)
        executor.shutdown(wait=True)
        verify()
        if any(digest(ROOT / name) != value for name, value in sources.items()):
            raise RuntimeError("Collection source changed during parallel collection")
        if any(digest(ROOT / "data/plans" / f"episode_{episode:02d}.npz") != value
               for episode, value in plans.items()):
            raise RuntimeError("Read-only pilot plan changed during collection")
        verified = 0
        for cube in cubes:
            for episode in range(10):
                path = ROOT / "data" / cube["split"] / f"cube_{cube['cube_id']:04d}_episode_{episode:02d}.npz"
                verify_receipt(path, cube["cube_id"], episode, sources)
                verified += 1
        if verified != 2600 or completed + skipped != verified:
            raise RuntimeError("Parallel full collection did not verify exactly 2600 episodes")
        manifest = {"stage": "full", "collector": "parallel_collect.py", "workers": args.workers,
            "cube_count": len(cubes), "episode_count": verified, "verified_receipts": verified,
            "completed_this_run": completed, "skipped_verified": skipped,
            "elapsed_seconds": time.monotonic() - started,
            "versions": {name: importlib.metadata.version(name)
                         for name in ["mani_skill", "sapien", "mplib", "torch", "numpy"]},
            "split_sha256": digest(ROOT / "data/split.json"), "source_sha256": sources,
            "plan_sha256": plans, "reference": "clean/collect.py"}
        manifest_path = ROOT / "data/full_manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing.get("split_sha256") != manifest["split_sha256"] or existing.get("episode_count") != verified:
                raise RuntimeError("Existing full manifest conflicts with verified collection")
        else:
            _write_json_exclusive(manifest_path, manifest)
        with (ROOT / "results/collection_events.jsonl").open("a") as handle:
            handle.write(json.dumps(manifest, allow_nan=False) + "\n")
        print(json.dumps(manifest, allow_nan=False), flush=True)
    except BaseException as error:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        with (ROOT / "results/collection_events.jsonl").open("a") as handle:
            handle.write(json.dumps({"stage": "full", "collector": "parallel_collect.py",
                "event": "failure", "error": repr(error), "completed_this_run": completed,
                "skipped_verified": skipped, "elapsed_seconds": time.monotonic() - started}) + "\n")
        raise


if __name__ == "__main__":
    main()
