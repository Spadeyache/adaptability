"""Report the stronger-shove pilot exclusively from saved evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
TARGETS=('mass_kg','combined_friction')
FAMILIES=('polynomial_ridge','extra_trees')


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def link(root,name,label):return f'[{label}]({root/name})'
def score(value):return f"{value['mean_r2']:.4f} ± {value['seed_std_r2']:.4f}"


def generate(root=ROOT):
    root=Path(root).resolve()
    paths={name:root/filename for name,filename in {
        'timing':'results/timing.json','metrics':'results/pilot_metrics.json',
        'audit':'results/pilot_audit.json','probe':'results/action_probe.json',
        'protocol':'data/protocol.json','plans':'data/plans.json',
        'previous_metrics':'versions/v3/results/pilot_metrics.json',
        'previous_audit':'versions/v3/results/pilot_audit.json',
    }.items()}
    values={name:read(path) for name,path in paths.items()}
    timing,metrics,audit,probe,protocol=(values[name] for name in ['timing','metrics','audit','probe','protocol'])
    if audit['status']!='passed':raise RuntimeError('Data/physics audit has not passed')
    if len(metrics)!=100 or {r['cube_id'] for r in metrics}!=set(range(10)):
        raise RuntimeError('Expected only the same ten pilot cubes, ten episodes each')
    if timing['plain_oracle_training_started'] or timing['belief_model_started']:
        raise RuntimeError('Pilot-only authorization boundary changed')
    if timing['approval_status']!='AWAITING_USER_GO':raise RuntimeError('Expected explicit waiting status')
    if protocol['planned_evaluation']['score_groups']!=['push','moving_coast','all_steps']:
        raise RuntimeError('Three required score groups are not present')
    old=values['previous_metrics'];old_audit=values['previous_audit']
    previous_launch=sum(r['launch_speed_m_s']>.02 for r in old)
    old_moving=old_audit['moving_coast_transitions']/old_audit['coast_transitions']
    target=protocol['launch_fraction_target'];threshold=protocol['launch_speed_threshold_m_s']
    passed=audit['launch_fraction']>=target
    physics_threshold=protocol['mass_range_kg'][1]*9.8*(protocol['cube_friction_range'][1]+protocol['floor_material_friction'])/2
    cap_min,cap_max=protocol['cap_range_n']
    margin=cap_min/physics_threshold-1
    commanded_margin=.95*cap_min/physics_threshold-1
    no_launch=[r for r in metrics if r['launch_speed_m_s']<=threshold]
    launch=[r for r in metrics if r['launch_speed_m_s']>threshold]
    plans={(p['cube_id'],p['episode_id']):p for p in values['plans']}
    largest=max(metrics,key=lambda r:r['max_contact_force_n'])
    largest_cap=plans[(largest['cube_id'],largest['episode_id'])]['cap_n']
    line_status=(f"**Launch target passed: {audit['launches']}/100 shoves launched.**" if passed else
                 f"**Launch target not met: {audit['launches']}/100 shoves launched; the target is at least {target:.0%}.**")
    lines=[
        '# Stronger-shove ten-cube pilot', '',line_status,'',
        '**Stopped here, awaiting your go. No Plain, Oracle, or belief model has been trained; the prediction gate is untested.**','',
        'The same ten cube identities and all non-force randomized settings were retained. Only force caps changed '
        'from uniform 2–4 N to uniform 4–7 N using the same random quantiles. No episodes or cubes were discarded or '
        'resampled because of their physical outcomes. The previous pilot, source, probes, and data are archived in `versions/v3/`.', '',
        '## Required pilot measurements','',
        '| Measurement | Previous 2–4 N pilot | New 4–7 N pilot |','|---|---:|---:|',
        f'| Launches above 2 cm/s at release | {previous_launch}/100 ({previous_launch:.0f}%) | {audit["launches"]}/100 ({audit["launch_fraction"]:.1%}) |',
        f'| Launches above 1 mm/s, secondary threshold | {sum(r["launch_speed_m_s"]>.001 for r in old)}/100 | {audit["launches_over_1mm_s"]}/100 |',
        f'| Moving coast fraction before trimming | {old_moving:.2%} | {audit["untrimmed_moving_coast_fraction"]:.2%} |',
        f'| Moving coast fraction in retained data | {old_moving:.2%} | {audit["moving_coast_fraction"]:.2%} |',
        f'| Maximum coast distance | {100*max(r["stopping_distance_m"] for r in old):.3f} cm | {100*audit["maximum_coast_distance_m"]:.3f} cm |',
        f'| Retained transitions | {old_audit["transitions"]:,} | {audit["transitions"]:,} |',
        f'| Transitions trimmed | 0 | {audit["trimmed_transitions"]:,} |','',
        f'The primary launch definition is horizontal release speed > {threshold:.3f} m/s. The looser 1 mm/s result is '
        'shown separately so numerical movement is not quietly substituted for a useful launch. Moving coast means '
        'a coast transition whose observed current full linear speed exceeds 1 mm/s or whose angular speed exceeds 0.05 rad/s.', '',
        f'Of the retained {audit["coast_transitions"]:,} coast transitions, **{audit["moving_coast_transitions"]:,} '
        f'are moving ({audit["moving_coast_fraction"]:.2%})**. Before trimming, the same moving steps were '
        f'{audit["untrimmed_moving_coast_transitions"]:,}/{audit["untrimmed_coast_transitions"]:,} '
        f'({audit["untrimmed_moving_coast_fraction"]:.2%}). No moving transition was removed.', '',
        '## Force margin and observed contact','',
        f'The specified maximum static-friction threshold is 0.50 × 9.8 × 0.65 = **{physics_threshold:.3f} N**. '
        f'The minimum cap of {cap_min:g} N exceeds it by **{margin:.1%}**. The fixed 95% drive fraction means '
        f'the minimum applied pusher command is {.95*cap_min:.1f} N, giving **{commanded_margin:.1%}** margin. '
        'Caps remain independent of cube properties. This comparison concerns the configured force magnitude; '
        'it does not guarantee that an oblique, freely moving pusher maintains contact for the full pulse.', '',
        f'The full 3D contact-force audit passed. Maximum measured contact/cap ratio was '
        f'**{audit["maximum_actual_contact_force_over_cap"]:.3%}**; the largest absolute measured force was '
        f'**{largest["max_contact_force_n"]:.4f} N** under its **{largest_cap:.4f} N** cap. '
        f'All 100 episodes stopped, and maximum contact force after release was {audit["maximum_coast_contact_force_n"]:g} N.', '',
    ]
    if launch and no_launch:
        a=float(np.mean([r['push_contact_fraction'] for r in launch]))
        b=float(np.mean([r['push_contact_fraction'] for r in no_launch]))
        lines += [f'Launched episodes had contact during an average **{a:.2%}** of the push interval; '
                  f'nonlaunching episodes averaged **{b:.2%}**. This saved-trace contrast indicates that '
                  'command magnitude alone has not resolved the short contact durations. It does not prove a '
                  'unique failure mechanism. No contact-geometry changes or extra tuning runs were made in this pilot.', '']
    contact=audit.get('contact_loss_diagnostic')
    if contact:
        failed=contact['groups']['not_launched_at_release']
        successful=contact['groups']['launched_at_release']
        lines += [f'**{contact["moved_during_push"]}/100 moved above 2 cm/s during the push, but only '
                  f'{contact["release_launches"]}/100 still exceeded that speed at release.** '
                  f'Among the {failed["episodes"]} failed release launches, {failed["moved_during_push"]} '
                  'moved earlier. Their median last contact above '
                  f'{contact["forceful_threshold_n"]:g} N occurred at '
                  f'**{failed["median_last_forceful_push_time_ms"]:g} ms** of the '
                  f'{contact["push_duration_ms"]:g} ms pulse; the successful group reached '
                  f'**{successful["median_last_forceful_push_time_ms"]:g} ms**. '
                  'The traces show early contact loss followed by stopping before scheduled release. '
                  'Movement during the push is not counted as passing the release-launch target.', '']
    lines += [
        '## Tail trimming and leakage controls','',
        'The full one-second coast remains in raw diagnostics to verify stop confirmation and force compliance. '
        'Stop onset is the first state in 100 ms continuously below 1 mm/s linear and 0.05 rad/s angular speed. '
        'The saved model episode ends at the first 5 ms boundary at or after that onset, plus **five steps (25 ms)**. '
        'Confirmation duration itself is not counted as extra stationary training data.', '',
        f'**{audit["trimmed_transitions"]:,}/{audit["untrimmed_transitions"]:,} transitions were removed '
        f'({audit["trimmed_fraction"]:.2%})**, all stationary coast steps. Every episode retains all 16 push steps. '
        f'The final dataset has {audit["push_transitions"]:,} push and {audit["coast_transitions"]:,} coast steps; '
        f'episode lengths span {audit["episode_length_min"]}–{audit["episode_length_max"]} transitions.', '',
        '**Trimmed length is an outcome.** It may reveal properties even when every command was generated '
        'independently. Neither length, stop time, padding masks, IDs, nor termination indicators may be model '
        'features. Future stopping-position rollouts use the fixed physical horizon, extending the preplanned '
        'zero-force coast command rather than stopping at the observed file length. The probes below distinguish '
        'command-generation predictability from this separate length side channel.', '',
        '## Actions-only probe rerun on the new pilot','',
        'The actual 100 saved episodes are evaluated by leave-one-cube-out prediction: nine cubes train, one cube '
        'is held out, repeated for all ten cubes. R² is computed from all pooled out-of-fold predictions, not '
        'separately on each constant-label cube. Fixed hyperparameters and five seeds avoid fitting choices to '
        'the held-out pilot predictions. Command-only inputs use the fixed 16-step push prefix; no observed '
        'state, force, sequence length, or stop information enters that input.', '',
        '| Pilot input | Probe | Mass R² ± seed SD | Combined-friction R² ± seed SD |',
        '|---|---|---:|---:|',
    ]
    pilot=probe['pilot'];generator=probe['generator']
    feature_labels={'commands_only':'Command prefix only','retained_length_only':'Trimmed length only','commands_plus_retained_length':'Commands + trimmed length'}
    for feature,label in feature_labels.items():
        for family in FAMILIES:
            entry=pilot['aggregate'][feature][family]
            lines.append(f'| {label} | {family} | {score(entry["mass_kg"])} | {score(entry["combined_friction"])} |')
    baseline=pilot['constant_fold_training_mean_r2']
    command_improvements=[pilot['aggregate']['commands_only'][family][target]['mean_skill_vs_fold_training_mean']
                          for family in FAMILIES for target in TARGETS]
    length_improvements=[pilot['aggregate']['retained_length_only'][family][target]['mean_skill_vs_fold_training_mean']
                         for family in FAMILIES for target in TARGETS]
    lines += ['',f'The fold-training-mean baseline gives raw R² **{baseline["mass_kg"]:.4f}** for mass '
              f'and **{baseline["combined_friction"]:.4f}** for combined friction. With ten equal cube groups, '
              'leave-one-cube-out averaging itself makes this baseline negative. Scores are reported literally; '
              'a small-pilot negative score is not relabeled as zero. Cube friction and combined friction are '
              'affine targets, so their R² values agree; both are saved in the JSON.', '',
              ('The command-only probes perform worse than the fold-mean baseline for both properties; '
               'no useful command-to-property predictive signal was found. The raw pilot R² values are not '
               'numerically near zero, and ten held-out identities provide limited precision.' if max(command_improvements)<0 else
               'At least one command-only probe improves on the fold-mean baseline; inspect the saved raw scores before claiming no predictive signal.'), '',
              f'Length-only probes improve mean-squared error over the fold-mean baseline by '
              f'{min(length_improvements):.2%}–{max(length_improvements):.2%} in this pilot. '
              'This is a baseline-relative skill value, not raw R². It does not establish that the outcome-dependent length is safe to expose.', '',
              'The length-only and commands-plus-length rows are explicit side-channel diagnostics. Their inputs '
              'are forbidden for the prediction models. A positive score there does not mean that force caps '
              'were generated from labels, but it prevents treating arbitrary variable-length action sequences '
              'as automatically leak-free.', '',
              '### Separate full planned-command generator check','',
              'The updated 200-train/30-test preplanned schedules are also checked without collecting additional '
              'motion. This is supplementary to the pilot probe, not a replacement for it. It uses independent '
              'cube splits and training-only hyperparameter selection.', '',
              '| Probe | Mass R² ± seed SD | Combined-friction R² ± seed SD |','|---|---:|---:|']
    for family in FAMILIES:
        entry=generator['aggregate'][family]
        lines.append(f'| {family} | {score(entry["mass_kg"])} | {score(entry["combined_friction"])} |')
    lines += [
        '', '## Timing and required future comparisons','',
        f'The revised 100-episode collection took **{timing["this_invocation_wall_seconds"]:.3f} s** including '
        f'startup; per-episode simulation and saving summed to **{timing["episode_wall_seconds_sum"]:.3f} s**. '
        f'The actions-only probe rerun took **{probe["elapsed_seconds"]:.3f} s** separately. '
        'These are measured runtimes, not the total development/debugging wall time.', '',
        'Every later comparison must report three groups separately: **push**, **moving coast**, and '
        '**all retained steps**. This applies to one-step error, ten-step recursive rollout error, and '
        'stopping-position error. Recursive rollouts never consume observed intermediate states. The exact '
        'group definitions and fixed-horizon stopping-position evaluation are saved in `data/protocol.json`.', '',
        'The primary prediction gate is now **moving-coast one-step NMSE**: at least 20% lower mean Oracle error '
        'than Plain, and `Oracle mean + sample SD < Plain mean − sample SD` across five paired seeds. '
        'All-step error cannot replace the moving-coast score. Plain and Oracle remain 3×256 MLPs with train-only '
        'normalization; Oracle context is true mass plus combined friction. This gate has not been trained or evaluated.', '',
        '**No further cubes or prediction models will run before your go.**','',
        '## Saved evidence','',
    ]
    for label,name in [('Timing','results/timing.json'),('Episode metrics','results/pilot_metrics.json'),
                       ('Independent audit','results/pilot_audit.json'),('Action-probe results','results/action_probe.json'),
                       ('Updated protocol','data/protocol.json'),('Previous pilot','versions/v3/PILOT_REPORT.md')]:
        lines.append('- '+link(root,name,label))
    lines += ['', 'Rebuild from saved runs only:', '', '```bash', 'cd /home/ubuntu/faliure',
              'PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove_gate.report','```','',
              'Input SHA-256 hashes:','']
    lines.extend(f'- `{path.relative_to(root)}`: `{digest(path)}`' for path in paths.values())
    output=root/'PILOT_REPORT.md';output.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(output),'episodes':100,'launch_target_passed':passed,'status':'AWAITING_USER_GO'}))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    generate(parser.parse_args().root)
