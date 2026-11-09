"""Build the supervisor's summary from completed checks and matched physical replays."""
import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .check import write_json


def load(path):
    return json.loads(path.read_text())


def recovery(check_dir):
    with np.load(check_dir/'sweeps.npz') as data:
        losses, offsets = data['loss'],data['offsets']
        centers = offsets[np.argmin(losses, axis=1)]
        success = (np.abs(centers)<=.5) & (losses[:,8]<np.minimum(losses[:,4],losses[:,12]))
        with (check_dir/'episode_errors.csv').open() as fp:
            rows = list(csv.DictReader(fp))
        assert np.array_equal(data['episode_ids'],[int(row['episode']) for row in rows])
        all_axes = success.all(0)
        tasks = np.array([row['task'] for row in rows])
        return dict(episodes=len(tasks),all_axes=int(all_axes.sum()),per_axis=success.sum(1).tolist(),
                    by_task={t:dict(passed=int(all_axes[tasks==t].sum()),episodes=int((tasks==t).sum()))
                             for t in np.unique(tasks)})


def report(root, check_dirs):
    original, extra = [load(root/name/'audit.json') for name in ('data','grip_pulses')]
    pairs = load(root/'fresh_physics'/'summary.json')
    checks, counts = [load(d/'results.json') for d in check_dirs], [recovery(d) for d in check_dirs]
    names = [{'z8':'8D original','z4_grip':'4D grip','z4_matched':'4D matched'}.get(d.parent.name,d.parent.name)
             + (' (fresh)' if r.get('external') else '') for d,r in zip(check_dirs,checks)]
    fig,(a,b) = plt.subplots(1,2,figsize=(12.3,5.4),layout='constrained')
    colors = ['#1e6470','#be663e','#70639d','#487da5']
    labels = ['Swap across episodes','Training mean','Zero vector']
    x = np.arange(len(labels))
    for i,(r,s,name) in enumerate(zip(checks,counts,names)):
        with (check_dirs[i]/'episode_errors.csv').open() as fp:
            episodes = list(csv.DictReader(fp))
        draws = np.random.default_rng(701).integers(len(episodes),size=(4000,len(episodes)))
        ref = np.asarray([float(row['Encoded']) for row in episodes])
        reference_draws = ref[draws].mean(1)
        y, intervals = [], []
        for k in ('Swapped','Mean','Zero'):
            values = np.asarray([float(row[k]) for row in episodes])
            effect = 100*(values.mean()/ref.mean()-1)
            lower,upper = np.quantile(100*(values[draws].mean(1)/reference_draws-1),[.025,.975])
            y.append(effect)
            intervals.append([effect-lower,upper-effect])
        width = .82/len(checks)
        offset = (i-(len(checks)-1)/2)*width
        a.bar(x+offset,y,.93*width,color=colors[i],label=name,yerr=np.transpose(intervals),capsize=3)
        for pos,change,interval in zip(x+offset,y,intervals):
            a.text(pos,change+interval[1]+.7,f'{change:.0f}%',ha='center',fontsize=9)
        percent = 100*s['all_axes']/s['episodes']
        b.bar(i,percent,.55,color=colors[i])
        b.text(i,percent+2,f"{s['all_axes']} / {s['episodes']}",ha='center')
    a.set(xticks=x,xticklabels=labels,ylabel='Increase in mean rollout loss (%)',title='Forecast error after replacing the encoded context')
    a.margins(y=.25)
    a.legend(fontsize=9,loc='upper left')
    b.set(xticks=np.arange(len(names)),xticklabels=[n.replace(' (','\n(') for n in names],ylim=(0,100),
          ylabel='Held-out episodes passing every axis (%)',title='Single-episode coordinate sweeps')
    if checks[-1].get('external'):
        last = checks[-1]
        k = sum(axis['passes'] for axis in last['coordinate_sweep'])
        fig.suptitle('Hidden physical context in Panda manipulation',weight='bold')
        fig.supxlabel(f"On the designated fresh Pick grasp, {k}/{last['latent']} coordinate sweeps passed. "
                      f"{counts[-1]['all_axes']:,}/{last['episodes']:,} episodes passed every sweep.\n"
                      "Error bars give paired 95% episode-bootstrap intervals. Each sweep tests whether the forecast loss dips near the encoded coordinate.",
                      fontsize=9)
    fig.savefig(root/'latent_recovery.png',dpi=180)
    with PdfPages(root/'context_diagnostics.pdf',metadata={'Title':'Inferring hidden physics from Panda motion'}) as pdf:
        if checks[-1].get('external'):
            last = checks[-1]
            ab = last['ablation']
            pick = counts[-1]['by_task']['PickCube-v1']
            grip = next(row for row in pairs if row['source']=='Policy pick' and row['change']=='Gripper strength')
            cover = plt.figure(figsize=(10,7.5))
            cover.text(.08,.91,'Inferring hidden physics from Panda motion',fontsize=24,weight='bold',color='#164a51')
            cover.text(.08,.86,'ManiSkill3  ·  Franka Panda  ·  November supervisor review',fontsize=11)
            cover.text(.08,.75,'Correct physical context improves ten-step forecasts',fontsize=16,weight='bold')
            cover.text(.08,.65,
                f"Across {last['episodes']:,} fresh episodes, normalized loss was {ab['Encoded']['loss']:.3f} with the episode’s encoded context.\n"
                f"Swapping contexts within the same task raised it to {ab['Swapped']['loss']:.3f}. The training mean gave "
                f"{ab['Mean']['loss']:.3f} and zero gave {ab['Zero']['loss']:.3f}.",fontsize=11,linespacing=1.6)
            cover.text(.08,.55,'Recovery from an individual grasp remains uneven',fontsize=16,weight='bold')
            cover.text(.08,.44,
                f"On the designated fresh Pick grasp, {k} of {last['latent']} coordinate sweeps passed. Across the fresh set, "
                f"{counts[-1]['all_axes']} of {last['episodes']:,} episodes\n"
                f"passed every coordinate, including {pick['passed']} of {pick['episodes']} Pick grasps. "
                "The sweep tolerance was half a training SD.",fontsize=11,linespacing=1.6)
            cover.text(.08,.34,'Gripper strength physically changes the motion',fontsize=16,weight='bold')
            cover.text(.08,.22,
                f"Changing only the finger force cap produced {grip['median_peak_cube_position_mm']:.1f} mm median peak cube separation "
                f"over {grip['n']} paired Pick replays.\nThe gripper’s linear readout from context had R² of "
                f"{last['linear_probe_r2']['grip_scale']:.3f}. The damping and mass readouts were "
                f"{last['linear_probe_r2']['damping_scale']:.3f} and {last['linear_probe_r2']['mass_kg']:.3f}.",
                fontsize=11,linespacing=1.6)
            decision = ('Fresh Step 4 checks passed.' if last['passed'] else
                        'Step 4 remains open. Deployment inference and 50-trial confidence tracking were withheld.')
            cover.text(.08,.10,decision,fontsize=12,weight='bold',color='#164a51')
            pdf.savefig(cover)
            plt.close(cover)
        pdf.savefig(fig)
        for directory,name in zip(check_dirs,names):
            image = plt.imread(directory/'checks.png')
            page=plt.figure(figsize=(11.5,7.7))
            page.suptitle(name+' context model',fontsize=14,weight='bold')
            axis=page.add_axes([0,0,1,.95])
            axis.imshow(image)
            axis.axis('off')
            pdf.savefig(page)
            plt.close(page)
    plt.close(fig)
    collection = f"The original collection contains {original['transitions']:,} transitions and the corrected grip-probe supplement contains {extra['transitions']:,}. Both use a 10% episode holdout. The original model had 36,000 training episodes. The first revision had 41,400. The first two checks used separate 1,000-episode folds on the original task mix."
    matched = any(d.parent.name=='z4_matched' for d in check_dirs)
    if matched:
        pairs_audit = load(root/'matched_train/audit.json')
        confirmation = load(root/'confirmation/audit.json')
        collection += f" The matched-pair repair adds {pairs_audit['transitions']:,} recorded transitions, keeping both members of each pair together. It uses 48,600 training episodes. Its checks revisit the previously failed fold and evaluate all {confirmation['episodes']:,} episodes of a separate confirmation set."
    lines = ['# Hidden physical context from Panda motion','',collection,'',
             '| Model and evaluation | Encoded loss | Swapped | Mean | Zero | Sweeps passed on designated Pick | Episodes passing all axes |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r,s in zip(names,checks,counts):
        ab = r['ablation']
        k = sum(x['passes'] for x in r['coordinate_sweep'])
        lines.append(f"| {name} | {ab['Encoded']['loss']:.3f} | +{100*ab['Swapped']['relative_increase']:.1f}% | +{100*ab['Mean']['relative_increase']:.1f}% | +{100*ab['Zero']['relative_increase']:.1f}% | {k}/{r['latent']} (episode {r['example_id']}) | {s['all_axes']}/{s['episodes']} |")
    lines += ['', 'The held-out swap keeps the motion, issued commands and task fixed. The bars show paired 95% episode-bootstrap intervals. The sweep changes one latent coordinate while keeping the others at their encoder values. Recovery requires a loss minimum within half a training standard deviation and lower loss at the encoder value than at both one-standard-deviation neighbors. Horizons differ between models, so compare ablations within each row.', '',
              '| Constant | '+' | '.join(names)+' |','| --- |'+' ---: |'*len(names)]
    for prop in checks[0]['linear_probe_r2']:
        lines.append('| '+prop+' | '+' | '.join(f"{r['linear_probe_r2'][prop]:.3f}" for r in checks)+' |')
    lines += ['', 'Table entries are held-out linear probe R² from privileged encodings to the raw physical constants. They describe what the encoder retains. Single-episode recovery is a separate test.', '',
              '## Matched physical replays','',
              'Each pair started at the same recorded state and replayed exactly the same 50 issued commands. We changed one setting across its training extremes. Deltas below are medians over 48 starts of the largest separation during each replay. The repeated-control column reruns the same physical setting.', '',
              '| Motion | Setting changed | Finger width (mm) | Cube position (mm) | Repeated control, cube (mm) |',
              '| --- | --- | ---: | ---: | ---: |']
    for p in pairs:
        lines.append(f"| {p['source']} | {p['change']} | {p['median_peak_width_mm']:.1f} | {p['median_peak_cube_position_mm']:.1f} | {p['repeat_median_peak_cube_mm']:.3f} |")
    lines += ['', '## Decision','']
    if checks[-1]['passed']:
        lines.append('The revised forecast model passed the specified checks on the designated fold. Coordinate recovery frequencies across the fold quantify how far that result generalizes.')
    else:
        lines.append('The final model did not pass every required individual-episode sweep. Deployment inference and the reset/carry experiment remain gated. The available evidence establishes context-dependent forecasting. Confidence about fixed versus changing hidden properties remains an open hypothesis.')
    last = checks[-1]
    if last.get('external'):
        bad = [str(axis['dim']) for axis in last['coordinate_sweep'] if not axis['passes']]
        if bad:
            lines.append(f"The untouched Pick grasp missed the sweep criterion on coordinate {', '.join(bad)}. Its best offsets, in training SD units, were "
                         f"{[axis['best_offset_sd'] for axis in last['coordinate_sweep']]}. The corresponding minima after averaging across the entire confirmation set were "
                         f"{[axis['population_best_offset_sd'] for axis in last['coordinate_sweep']]}.")
        pick = counts[-1]['by_task']['PickCube-v1']
        lines.append(f"In the 250 confirmation Pick grasps, {pick['passed']} passed all coordinate sweeps. This is a local, one-coordinate-at-a-time test with the other coordinates held at their encoder values.")
    lines += ['', 'A hidden command queue makes delayed dynamics partially observed. Learned latent axes can also mix several physical constants. These issues affect interpretation of per-coordinate confidence.',
              '', 'The gripper and vertical COM interventions produced visible motion differences in physical replay. A weak linear readout of either variable only establishes that the learned encoding made it difficult to read out linearly. Whether a robot can identify it from one trajectory remains a separate question.',
              '', '[Reproducible methods](../../../experiments/context_dynamics/README.md) · [Supervisor figures](context_diagnostics.pdf) · [Physical replay measurements](fresh_physics/paired_physics.csv)']
    for d in check_dirs:
        lines.append(f" · [{d.parent.name} diagnostics]({d.relative_to(root)}/results.json)")
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'recovery.json',dict(zip(names,counts)))
    artifacts = [root/'SUMMARY.md',root/'context_diagnostics.pdf',root/'latent_recovery.png',
                 root/'recovery.json',root/'fresh_physics/paired_physics.csv',root/'fresh_physics/summary.json']
    datasets = ['data','grip_pulses'] + (['matched_train','confirmation'] if matched else [])
    for directory in datasets:
        data = root/directory
        artifacts += [data/'audit.json',data/'dataset.json',*sorted(data.glob('shard_*.npz'))]
    for d in check_dirs:
        artifacts += [d/'results.json',d/'sweeps.npz',d/'readout.npz',d.parent/'model.pt',
                      d.parent/'settings.json',d.parent/'training.jsonl']
    for d in check_dirs:
        artifacts.extend((d.parent/'source').glob('*.py'))
    for directory in datasets:
        artifacts.extend((root/directory/'source').glob('*.py'))
    write_json(root/'manifest.json',dict(artifact_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
               for f in artifacts}, versions={p:version(p) for p in ('mani_skill','sapien','torch','numpy','matplotlib')},
               excluded_runs=['grip_data','z4'],reason='Supplementary scene dispatch was corrected before evaluating the revised model.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('checks',nargs='+',type=Path)
    a=p.parse_args()
    report(a.root,a.checks)
