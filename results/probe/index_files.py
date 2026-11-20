"""Index candidate data containers; does not treat checkpoints as datasets."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/probe'
EXCLUDED = ['.git', '.venv', '.cache', '.maniskill', 'node_modules', '.aws', '.codex']
SUFFIXES = {'.npz','.npy','.h5','.hdf5','.parquet','.pkl','.pickle','.csv','.jsonl',
            '.json','.pt','.pth','.zip','.tar','.gz','.zarr'}

if __name__ == '__main__':
    cmd = ['rg','--files','--hidden','--no-ignore']
    for name in EXCLUDED:
        cmd += ['-g',f'!{name}/**','-g',f'!**/{name}/**']
    names = subprocess.check_output(cmd,cwd=ROOT,text=True).splitlines()
    records, manifests = [], []
    for name in sorted(names):
        p = Path(name)
        if name.startswith('results/probe/') or p.suffix.lower() not in SUFFIXES:
            continue
        records.append(dict(path=name,bytes=(ROOT/p).stat().st_size,suffix=p.suffix.lower()))
        if p.name in ('manifest.json','dataset_manifest.json','completion.json'):
            manifests.append(name)
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'file_inventory.json').write_text(json.dumps(dict(excluded_infrastructure=EXCLUDED,
        candidate_files=records,manifest_paths=manifests),indent=2)+'\n')
    print(f'{len(records)} candidate files indexed; images inventoried separately by dataset manifests.')
