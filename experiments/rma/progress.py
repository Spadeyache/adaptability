"""Render training curves from TensorBoard records and adapter logs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def render(root):
    root = Path(root)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    for directory in sorted(root.glob('base_*')):
        if not list(directory.glob('events.out.tfevents.*')):
            continue
        events = EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()
        tags = events.Tags()['scalars']
        tag = 'eval/success_at_end'
        if tag in tags:
            values = events.Scalars(tag)
            axes[0].plot([v.step for v in values], [v.value for v in values], label=directory.name)
    for path in sorted(root.glob('adapter_*/learning.jsonl')):
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A writer may still be completing the last record.
        if rows:
            axes[1].plot([r['transitions'] for r in rows], [r['loss'] for r in rows], label=path.parent.name)
    axes[0].set(title='Privileged policy: GPU training-evaluation success', ylabel='Final-step success', ylim=(0,1.05))
    axes[1].set(title='Adapter regression on its own rollouts', ylabel='Latent MSE', xlabel='Transitions within each block')
    for ax in axes:
        if ax.lines:
            ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(root/'training_progress.png', dpi=140); plt.close(fig)
    state = json.loads((root/'manifest.json').read_text())
    import html
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">'
        '<title>RMA training progress</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto}img{max-width:100%}</style>'
        '<h1>RMA training progress</h1><p>Development measurements; not paper reproduction or final OOD evidence.</p>'
        '<p>Status: '+html.escape(state['status'])+' · '+html.escape(state.get('current_step',''))+'</p>'
        '<img src="training_progress.png"><ul>'+''.join(f'<li><a href="{p.parent.name}/index.html">{p.parent.name}</a></li>' for p in sorted(root.glob('*/index.html')))+'</ul>')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('output'); render(p.parse_args().output)
