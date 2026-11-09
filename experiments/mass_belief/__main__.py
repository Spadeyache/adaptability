"""Run random-mass collection, training, paired replay, policy side check and plots."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import shutil

from robot_adaptation.collection.collect import write_json
from robot_adaptation.estimators.interaction.data import sha256
from .collect import collect
from .evaluate import comparison, side
from .report import report
from .train import train


def provenance(run):
    c = json.loads((run/'config.json').read_text())
    historical = Path(c['source'])
    source_manifest = json.loads((historical/'manifest.json').read_text())
    source_files = [
        *Path('experiments/mass_belief').glob('*.py'),
        *Path('robot_adaptation/estimators/mass_belief').glob('*.py'),
        Path('robot_adaptation/policy/frozen_v1.py'),
        Path('robot_adaptation/simulation/collection_lift.py'),
        Path('robot_adaptation/simulation/environment.py'),
        Path('robot_adaptation/simulation/grip_controller.py'),
    ]
    for path in source_files:
        dest = run/'source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,dest)
    keys = ['config.json','estimator.pt','calibration.json','fit.json','trials.csv','steps.csv',
            'side_check.csv','summary.json','RESULTS.md','mass_across_trials.png','mass_across_trials.pdf','within_trial.png']
    write_json(run/'manifest.json', dict(status='completed', simulator='physx_cpu',
        policy_checkpoint=source_manifest['checkpoint'], policy_sha256=source_manifest['checkpoint_sha256'],
        historical_index_sha256=sha256(historical/'episodes.json'),
        versions={name:version(name) for name in ['torch','numpy','mani_skill','sapien']},
        source_sha256={str(path):sha256(path) for path in source_files},
        artifacts_sha256={key:sha256(run/key) for key in keys},
        episode_sha256={str(path.relative_to(run)):sha256(path) for path in sorted(run.glob('*/*.npz'))}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/mass_belief.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    collect(a.config,a.output)
    train(a.output)
    comparison(a.output)
    side(a.output)
    report(a.output)
    provenance(a.output)
