"""Produce a conservative evidence report from completed evaluation artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robot_adaptation.policy.checkpoint_metadata import stopped_training_progress


def assess(policy=None, estimator=None, retry=None, training=None, transfer=None, early_stop=None):
    if retry is not None and transfer is not None and retry.get("task", "vertical") != transfer.get("task", "vertical"):
        raise ValueError("Retry and estimator-transfer reports must use the same declared task")
    report = {"assessment_spec": "configs/adaptation_assessment.json"}
    if policy is None:
        report["policy"] = {"status": "pending", "reason": "No completed policy comparison supplied."}
    else:
        comparisons = policy["paired_comparisons"]
        required = ("fixed_both", "wrong_mass", "wrong_friction")
        passed = {name: comparisons.get(name, {}).get(
            "paired_bootstrap_95_interval", [-1, 1])[0] > 0 for name in required}
        report["policy"] = {
            "status": "supported_on_tested_cases" if all(passed.values()) else
                      "partially_supported" if any(passed.values()) else "not_demonstrated",
            "correct_inputs_benefit": passed,
            "paired_comparisons": {name: comparisons.get(name) for name in required},
            "metrics": policy["metrics"], "checkpoint": policy["checkpoint"],
            "scope": "Matched held-out starts in simulation, static=dynamic object friction and fixed finger/table materials. Seed-block intervals describe starting-state variation conditional on the tested physical grid.",
            "input_sensitivity_is_sufficient_evidence": False}
    if estimator is None:
        report["estimator"] = {"status": "pending", "reason": "No completed held-condition test supplied."}
    else:
        at_five = estimator["attempts"]["5"]
        confidence = at_five["improvement_condition_bootstrap_95ci"]
        passed = {name: confidence[i][0] > 0 for i, name in enumerate(("mass", "friction"))}
        report["estimator"] = {
            "status": "supported_on_historical_held_out_conditions" if all(passed.values()) else
                      "partially_supported" if any(passed.values()) else "not_demonstrated",
            "targets_improve_over_training_median": passed,
            "held_out_conditions": estimator["held_conditions"],
            "constant_prior_mae": estimator["prior_median_mae"],
            "attempts": estimator["attempts"], "checkpoint": estimator["checkpoint"],
            "scope": "Independent attempts from the historical fixed policy pooled under identical physics. Physical settings are held out, while starting seeds recur across settings; condition-bootstrap intervals are conditional on those shared starts. Real robot sensing and broad new-policy transfer are unproven."}
    if retry is None:
        report["closed_loop"] = {"status": "pending"}
    else:
        difference = retry["methods"].get("adaptive_vs_fixed", {})
        report["closed_loop"] = {
            "status": "supported_on_tested_cases" if difference.get("supported_on_this_sample") else "not_demonstrated",
            "comparison": difference,
            "success_rates_within_budget": {method: values["success_rate_within_budget"]
                for method, values in retry["methods"].items()
                if "success_rate_within_budget" in values},
            "attempt_budget": retry["attempt_budget"],
            "task": retry.get("task", "vertical"),
            "task_description": retry.get("task_description", "Legacy vertical pick-and-hold protocol"),
            "policy_checkpoint_progress": retry.get("policy_checkpoint_progress"),
            "scope": "Adaptive-versus-fixed retry success can reflect a useful estimator prior or parameter shifts as well as information recovered from episodes. Interpret any gain alongside held-out transfer errors; retry improvement alone does not establish accurate physical inference as its cause."}
    if training is not None:
        report["training"] = {
            "status": training.get("status"), "planned_iterations": training.get("planned_iterations"),
            "requested_50000_iterations_completed": bool(training.get("status") == "complete" and
                training.get("planned_iterations", 0) >= 50000)}
    if transfer is not None:
        confidence=transfer["attempts"]["5"]["improvement_condition_bootstrap_95ci"]
        passed={name:confidence[i][0]>0 for i,name in enumerate(("mass","friction"))}
        report["estimator_transfer"]={
            "status":"supported_on_tested_grid" if all(passed.values()) else
                     "partially_supported" if any(passed.values()) else "not_demonstrated",
            "targets_improve_over_training_median":passed,
            "held_conditions":transfer["held_conditions"],
            "constant_prior_mae":transfer["prior_median_mae"],
            "attempts":transfer["attempts"],
            "scope":"Frozen estimator, new policy, full independent mass/friction grid, fixed supplied guesses. Confidence is conditional on the shared starting seeds; historical recovery and task benefit remain separate checks."}
        report["estimator_transfer"].update(task=transfer.get("task", "vertical"),
            task_description=transfer.get("task_description", "Legacy vertical pick-and-hold protocol"),
            policy_checkpoint_progress=transfer.get("policy_checkpoint_progress"))
    if early_stop is not None:
        progress = stopped_training_progress(early_stop, training)
        pinned = early_stop.get("latest_saved_checkpoint_sha256")
        evaluated_hashes = []
        if policy is not None and policy.get("checkpoint_sha256"):
            evaluated_hashes.append(policy["checkpoint_sha256"])
        if retry is not None and isinstance(retry.get("checkpoint_sha256"), dict):
            evaluated_hashes.append(retry["checkpoint_sha256"].get("policy"))
        if transfer is not None and transfer.get("policy_checkpoint_sha256"):
            evaluated_hashes.append(transfer["policy_checkpoint_sha256"])
        if pinned and any(value != pinned for value in evaluated_hashes):
            raise ValueError("Evaluation policy checkpoint differs from the early-stop pinned checkpoint")
        report["training"] = dict(progress, planned_iterations=progress["original_planned_iterations"],
                                  original_manifest_status=training.get("status") if training else None)
    return report


def markdown(report):
    policy, estimator, loop = (report[key] for key in ("policy", "estimator", "closed_loop"))
    lines = ["# Physics conditioning and interaction estimation assessment", "",
             "Results are judged against `configs/adaptation_assessment.json`. Code checks and a completed training process do not by themselves establish a learned capability.", "",
             "| Check | Assessment |", "| --- | --- |",
             f"| Accurate physical inputs improve the policy | {policy['status']} |",
             f"| Interaction evidence recovers mass and friction | {estimator['status']} |",
             f"| Updated estimates improve actual retry success | {loop['status']} |", ""]
    if "estimator_transfer" in report:
        transfer=report["estimator_transfer"]
        lines.insert(-1,f"| Estimator transfers to new policy and full crossed grid | {transfer['status']} |")
    if "training" in report:
        value = report["training"]
        if value["status"] == "stopped_by_user":
            lines += [f"Training was **stopped by the user** after {value['executed_rollout_cycles']:,} executed rollouts "
                      f"({value['collected_transitions']:,} collected transitions). The evaluated saved checkpoint contains "
                      f"**{value['saved_checkpoint_completed_cycles']:,} completed learning cycles / "
                      f"{value['saved_checkpoint_transitions']:,} transitions**. The original {value['planned_iterations']:,}-cycle "
                      "budget was not completed; the unsaved later rollouts are not represented in this checkpoint. "
                      "Training remains stopped.", ""]
        else:
            lines += [f"Training status: **{value['status']}**. Planned cycles: {value['planned_iterations']}. The requested 50,000-cycle run is " +
                      ("complete." if value["requested_50000_iterations_completed"] else "not complete."), ""]
    if "attempts" in estimator:
        lines += ["## Estimator held-condition errors", "",
                  "| Evidence | Mass MAE (g) | Friction MAE |", "| --- | ---: | ---: |"]
        prior = estimator["constant_prior_mae"]
        lines.append(f"| Training-median constant | {prior[0]*1000:.1f} | {prior[1]:.4f} |")
        for count, value in sorted(estimator["attempts"].items(), key=lambda row: int(row[0])):
            mass, friction = value["mae"]
            lines.append(f"| {count} independent attempts | {mass*1000:.1f} | {friction:.4f} |")
        lines += ["", estimator["scope"], ""]
        if "robustness_caveat" in estimator:
            lines += [estimator["robustness_caveat"], ""]
    if "metrics" in policy:
        lines += ["## Policy matched comparisons", "",
                  f"Frozen policy checkpoint: `{policy['checkpoint']}`.", "",
                  "These rows compare supplied physics inputs for this one checkpoint on identical actual physics and starts. They do not compare training durations.", "",
                  "| Supplied inputs | Final success | Final 0.5 s continuous success |",
                  "| --- | ---: | ---: |"]
        for mode, value in policy["metrics"].items():
            lines.append(f"| {mode} | {value['final_success']:.1%} | {value['sustained_success']:.1%} |")
        lines += ["", "Required paired comparisons (true minus altered-input final held-goal success):", ""]
        for name, value in policy["paired_comparisons"].items():
            if value:
                interval = value["paired_bootstrap_95_interval"]
                lines.append(f"- {name}: {value['true_minus_comparison_success']:+.1%}, 95% interval [{interval[0]:+.1%}, {interval[1]:+.1%}].")
        lines += ["", policy["scope"], ""]
    if "success_rates_within_budget" in loop:
        lines += ["## Actual retry comparison", "", "Task: " + loop.get("task_description", "Unspecified task") + ".", ""]
        for name, value in loop["success_rates_within_budget"].items():
            lines.append(f"- {name}: {value:.1%} success within {loop['attempt_budget']} attempts.")
        lines += ["", "A small diagnostic may establish that the loop runs while leaving a performance benefit unproven.", "",
                  loop.get("scope", ""), ""]
    if "estimator_transfer" in report:
        transfer=report["estimator_transfer"]
        lines += ["## Estimator transfer falsification", "",
                  "Task: " + transfer.get("task_description", "Unspecified task") + ".", "",
                  "| Evidence | Mass MAE (g) | Friction MAE |", "| --- | ---: | ---: |"]
        prior=transfer["constant_prior_mae"]
        lines.append(f"| Constant training median | {prior[0]*1000:.1f} | {prior[1]:.4f} |")
        for count,value in sorted(transfer["attempts"].items(),key=lambda row:int(row[0])):
            mass,friction=value["mae"]
            lines.append(f"| {count} independent attempts | {mass*1000:.1f} | {friction:.4f} |")
        lines += ["",transfer["scope"],""]
    return "\n".join(lines)


def plot_estimator(estimator, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = sorted(map(int, estimator["attempts"]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for index, (axis, title, scale) in enumerate(zip(axes, ["Mass error (g)", "Friction error"], [1000, 1])):
        means = np.array([estimator["attempts"][str(k)]["mae"][index] for k in counts]) * scale
        ci = np.array([estimator["attempts"][str(k)]["mae_condition_bootstrap_95ci"][index] for k in counts]) * scale
        axis.errorbar(counts, means, yerr=np.maximum(np.array([means-ci[:, 0], ci[:, 1]-means]), 0),
                      fmt="o-", color="#245b91", capsize=3, label="Interaction estimator")
        axis.axhline(estimator["prior_median_mae"][index]*scale, linestyle="--", color="#9a4b30", label="Constant training median")
        axis.set(xlabel="Independent attempts pooled", ylabel=title, xticks=counts, ylim=(0, None))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Historical policy: {estimator['held_conditions']} held-out physical settings; 95% condition-bootstrap intervals", fontsize=10)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-evaluation", type=Path)
    parser.add_argument("--estimator-evaluation", type=Path)
    parser.add_argument("--retry-evaluation", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--estimator-robustness", type=Path)
    parser.add_argument("--transfer-evaluation", type=Path)
    parser.add_argument("--early-stop-record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(path.read_text()) if path else None
    policy, estimator, retry, training = map(read, (args.policy_evaluation, args.estimator_evaluation,
                                                   args.retry_evaluation, args.training_manifest))
    report = assess(policy, estimator, retry, training, read(args.transfer_evaluation), read(args.early_stop_record))
    if args.estimator_robustness:
        robustness=read(args.estimator_robustness)
        results=robustness.get("results",[])
        report["estimator"]["training_seed_robustness"] = robustness
        counts={name:[sum(row[field][i][0]>0 for row in results) for i in range(2)]
                for name,field in (("more_evidence","five_attempt_vs_one_ci"),
                                   ("constant_prior","five_attempt_vs_prior_ci"))}
        report["estimator"]["robustness_caveat"] = (
            f"Across {len(results)} fixed training seeds, five-versus-one improvement has positive "
            f"95% condition-bootstrap intervals for mass in {counts['more_evidence'][0]} seeds "
            f"and friction in {counts['more_evidence'][1]}. Improvement over the constant prior "
            f"has positive intervals for mass in {counts['constant_prior'][0]} seeds and friction "
            f"in {counts['constant_prior'][1]}. Primary-seed support alone does not establish robustness.")
    report["sources"] = {key: str(value.resolve()) for key, value in vars(args).items() if value and key != "output"}
    report["source_sha256"] = {key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
                                for key, value in report["sources"].items()}
    report["assessment_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "assessment.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "assessment.md").write_text(markdown(report) + "\n")
    if estimator:
        plot_estimator(estimator, args.output / "estimator_errors.png")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
