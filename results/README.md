# Object-belief experiment and advisor packet

[Research note, two pages](advisor/research_note.pdf) · [Slides](advisor/slides.pdf) · [Editable slides](advisor/slides.pptx) · [Stage-by-stage research log](../RESEARCH_LOG.md)

| Figure | Saved measurements |
| --- | --- |
| [Supplied physical context and forecast error](forecast/forecast.png) | [Three means, intervals, paired differences and source hashes](forecast/forecast.json) |
| [Same-cube mass estimates carried or reset](retention/retention.png) | [Five trials, intervals, paired endpoint difference and source hashes](retention/retention.json) |

## Reproduce today's results from the recordings

Run from the repository root on the machine holding the archived `runs/` directory. It contains the local trajectories and checkpoints and is excluded from Git. These commands read existing files and generate the audit results, publication figures and note. They use the repository's installed Python environment. Each JSON names its source files or records.

```bash
cd /home/ubuntu/faliure
source scripts/activate_grip.sh
python results/stage0/audit_lifts.py
python results/stage0/audit_push_coverage.py
python results/stage0/audit_observability.py
python results/stage0/audit_retention.py
python results/render_figures.py
python results/advisor/render_note.py
```

The four audits reproduce the [condition and trial inventory](stage0/lift_inventory.json), [commanded and incidental push coverage](stage0/push_coverage.json), [one-attempt estimator versus the training average](stage0/observability_check.json), and [within-trial settling comparison](stage0/retention_check.json).

The PowerPoint contains the editable slide text and both figures. To publish an edited copy, open `results/advisor/slides.pptx` in PowerPoint and select **File → Export → Create PDF/XPS**. When changing either figure, replace its embedded picture before exporting.

## Reproduce the source experiments

The commands below create separate run directories for the earlier experiments that supplied the evidence. The source is in this repository, and the historical configurations record the fixed seeds. The retrospective packet can be reproduced with the shorter commands above. The policy checkpoint used by the old sweep is [final_ckpt.pt](../runs/learning/20261009T045047-c852bfc9/final_ckpt.pt).

### Supervised mass and friction diagnostic

The original corrected estimator used the prepared 440-episode lift dataset. The splits and normalization live in `runs/interaction_estimator/contact_dataset_v1`. Use the saved prepared data to reproduce the original fit. The split contains 26 training, nine validation and nine test physical settings. The new audit additionally reports the eight test settings inside the proposed bounds.

```bash
prepared=runs/interaction_estimator/contact_dataset_v1
for seed in 101 102 103
do
  estimator="runs/reproduction/interaction_seed_${seed}"
  python -m robot_adaptation.estimators.interaction.train --dataset "$prepared" --output "$estimator" --epochs 300 --seed "$seed" --threads 2
  python -m robot_adaptation.estimators.interaction.evaluate --dataset "$prepared" --checkpoint "$estimator/best.pt" --output "$estimator/test_metrics.json"
done
```

The [dataset guide](../robot_adaptation/estimators/interaction/README.md) documents preparation from the dense raw trajectories. The [original fit manifest](../runs/interaction_estimator/gru_v2_corrected/manifest.json) fingerprints the archived fit code and prepared dataset.

### PickCube forecasting with ten supplied simulator settings

The original collector and training run used a CUDA GPU. The saved forecasting check ran on CPU. In the audited shell PyTorch reported zero CUDA devices. To reproduce on a compatible GPU host using the existing WSL setup, run these commands from the activated repository environment.

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
forecast_run=runs/reproduction/pick_forecast
study="$forecast_run/world_model_z8"
python -m experiments.context_dynamics.collect_gpu "$forecast_run" --pilot
python -m experiments.context_dynamics.collect_gpu "$forecast_run"
python -m experiments.context_dynamics.verify_gpu "$forecast_run/data"
python -m experiments.context_dynamics.gpu_data_report "$forecast_run"
python -m experiments.context_dynamics.pick_train "$forecast_run/data" "$study" --device cuda
python -m experiments.context_dynamics.pick_evaluate compare "$forecast_run/data" "$study" --device cuda
python -m experiments.context_dynamics.pick_evaluate checks "$forecast_run/data" "$study" --device cpu
```

To refit the forecast model on the exact archived trajectories, set `forecast_run=runs/context_dynamics/pick_gpu_2m` and `study=runs/reproduction/forecast_on_archived_data`, then run the final three commands. [Run settings](../runs/context_dynamics/pick_gpu_2m/world_model_z8/settings.json) · [Source experiment commands including the separate four-task confirmation](../experiments/context_dynamics/README.md)

### Five attempts with one cube and a mass filter

This CPU workflow collects the original splits, trains the supervised mass estimator, calibrates the filter, replays the paired trials, checks the policy in simulation, and exports raw logs and figures. The output directory must be new.

```bash
python -m experiments.mass_belief --config configs/mass_belief.json --output runs/reproduction/mass_retention
```

The historical switch comparison can be reproduced separately from the saved fifth-trial beliefs.

```bash
python -m experiments.mass_belief.switch runs/mass_belief/nov17_v1 runs/reproduction/switch_80g_mu1 --mass-g 80 --friction 1.0
```

[Five-trial config](../configs/mass_belief.json) · [Run and source fingerprints](../runs/mass_belief/nov17_v1/manifest.json)

## Outstanding stages

The [research log](../RESEARCH_LOG.md) records the pass condition for each of the eight proposed stages. Commands above reproduce completed work. The named two-constant world model on contact steps, probability-map inference, ten-trial reuse of that map, population starting guess and label-free per-object vectors have no completed run to reproduce.
