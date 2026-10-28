"""Separate saved policy progress from a run's originally planned budget.

The pinned PPO loop writes ckpt_N before learning iteration N. That checkpoint
therefore contains N-1 completed rollout/learning cycles. A final checkpoint
inherits the planned count only when its training manifest confirms completion.
"""
from __future__ import annotations

from pathlib import Path
import re


def checkpoint_progress(checkpoint, manifest):
    settings = manifest.get("settings", {})
    batch = settings.get("num_envs", 0) * settings.get("num_steps", 0)
    batch = int(batch) if batch and batch > 0 else None
    planned = manifest.get("planned_iterations")
    if planned is None and batch and settings.get("total_timesteps") is not None:
        total = settings["total_timesteps"]
        if total % batch == 0:
            planned = total // batch
    if planned is not None:
        if isinstance(planned, bool) or int(planned) != planned or planned < 0:
            raise ValueError("Planned cycle count must be a nonnegative integer")
        planned = int(planned)
    name = Path(checkpoint).name
    match = re.fullmatch(r"ckpt_(\d+)\.pt", name)
    counter = int(match.group(1)) if match else None
    if match:
        if counter < 1:
            raise ValueError("Numbered PPO checkpoints start at one")
        completed, basis = counter - 1, "Numbered checkpoint is saved before iteration N; completed cycles=N-1"
    elif name == "final_ckpt.pt" and manifest.get("status") == "complete" and planned is not None:
        completed, basis = planned, "Completed training manifest confirms final checkpoint's planned cycle count"
    else:
        completed, basis = None, "Checkpoint filename/manifest do not establish completed learning cycles"
    return dict(checkpoint_filename=name, checkpoint_iteration=counter,
                completed_learning_cycles=completed,
                checkpoint_transitions=completed * batch if completed is not None and batch else None,
                transitions_per_cycle=batch, original_planned_iterations=planned,
                original_planned_transitions=planned * batch if planned is not None and batch else None,
                training_manifest_status=manifest.get("status"), accounting_basis=basis,
                full_planned_run_completed=bool(manifest.get("status") == "complete" and planned is not None
                                                and completed is not None and completed >= planned))


def stopped_training_progress(record, manifest=None):
    """Validate explicit user-stop accounting without inferring unsaved weights."""
    if record.get("status") != "stopped_by_user" or record.get("requested_50000_completed") is not False:
        raise ValueError("Early-stop record must confirm the user stop and incomplete requested run")
    executed, saved = record.get("completed_rollouts"), record.get("saved_checkpoint_completed_cycles")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (executed, saved)) or saved > executed:
        raise ValueError("Saved cycles must be nonnegative integers no greater than executed rollouts")
    match = re.fullmatch(r"ckpt_(\d+)\.pt", Path(record["latest_saved_checkpoint"]).name)
    if not match or int(match.group(1)) - 1 != saved:
        raise ValueError("Early-stop saved cycle count disagrees with numbered checkpoint accounting")
    if manifest is not None:
        progress = checkpoint_progress(record["latest_saved_checkpoint"], manifest)
        if progress["checkpoint_transitions"] is not None and progress["checkpoint_transitions"] != record.get("saved_checkpoint_transitions"):
            raise ValueError("Early-stop saved transitions disagree with the checkpoint's rollout size")
        if progress["transitions_per_cycle"] and executed * progress["transitions_per_cycle"] != record.get("collected_transitions"):
            raise ValueError("Early-stop collected transitions disagree with executed rollout count")
        if progress["original_planned_iterations"] != record.get("original_planned_iterations"):
            raise ValueError("Early-stop original budget disagrees with the training manifest")
    return dict(status="stopped_by_user", original_planned_iterations=record.get("original_planned_iterations"),
                executed_rollout_cycles=executed, collected_transitions=record.get("collected_transitions"),
                saved_checkpoint_completed_cycles=saved,
                saved_checkpoint_transitions=record.get("saved_checkpoint_transitions"),
                latest_saved_checkpoint=record["latest_saved_checkpoint"],
                latest_saved_checkpoint_sha256=record.get("latest_saved_checkpoint_sha256"),
                unsaved_rollout_cycles=executed - saved,
                requested_50000_iterations_completed=False,
                training_restart_authorized=False,
                accounting_note="Executed rollouts after the last saved checkpoint are not part of the evaluated saved policy.")
