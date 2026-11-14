"""Show the paired pilot's directly measured cube-to-gripper gap."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    pilot = root / 'pilot_step45'
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout='constrained', sharey=True)
    cases = [
        ('Mass comparison: friction 0.4, grip 1 N/finger',
         [(.04, .4, 1., '40 g'), (.5, .4, 1., '500 g')]),
        ('Friction comparison: 500 g, grip 8 N/finger',
         [(.5, 1., 8., 'Friction 1.0'), (.5, .4, 8., 'Friction 0.4')]),
    ]
    for ax, (title, groups) in zip(axes, cases):
        for color, (mass, friction, cap, label) in zip(('#187d88', '#b7503a'), groups):
            curves = []
            for seed in (6101, 6102, 6103):
                path = pilot / f'task_0_m_{mass:.2f}_mu_{friction:.1f}_seed_{seed}_cap_{cap:g}.npz'
                with np.load(path, allow_pickle=False) as rows:
                    gap = np.r_[rows['state'][0, 21] - rows['tcp_pose'][0, 2],
                                rows['next_state'][:, 21] - rows['next_tcp_pose'][:, 2]]
                drift = (gap[45] - gap) * 1000
                time = np.arange(len(gap)) / 20 - 2.25
                curves.append(drift)
                ax.plot(time, drift, color=color, alpha=.3, linewidth=.8)
            ax.plot(time, np.median(curves, axis=0), color=color, linewidth=2, label=label)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=.9)
        ax.axhline(0, color='#999999', linewidth=.5)
        ax.set(title=title, xlabel='Time since grip reduction (s)', xlim=(-.5, 3.75))
        ax.grid(alpha=.15)
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Downward cube-to-gripper drift since grip change (mm)')
    fig.suptitle('Firm pickup followed by weaker grip exposes different responses')
    fig.supxlabel('Thin lines: three matched reset seeds. Thick lines: median. '
                  'The low-friction group includes one rotating loss.', fontsize=8)
    path = root / 'pilot_checks.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root', type=Path)
    print(plot(cli.parse_args().root))
