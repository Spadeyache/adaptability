# Does a cube belief learned from lifts transfer to sliding?

This experiment fits a shared dynamics model and a four-number vector per training cube. The test asks whether a new cube’s lift transitions can identify a vector that improves prediction on that cube’s slides. The simulator’s physical labels enter only the final diagnostic readout.

## Results

All five seeds passed the validation check. On 20,036 moving validation steps, each cube’s own z improved on zero by 4.35–8.58% and on a shuffled z by 2.20–3.51% across seeds. Each paired 95% interval excluded zero. The best checkpoints fell between epochs 41 and 55, and all five runs stopped on validation patience.

The 30 held-out cubes supplied 150 prove slides and 2,296 moving steps, with 47–100 steps per cube. Scores below average the MSE over moving steps within each cube, then give each cube equal weight.

| Condition | Mean slide MSE | Standard error across model seeds |
| --- | ---: | ---: |
| Zero z | 0.02548 | 0.00046 |
| Same cube’s five lifts | 0.02430 | 0.00054 |
| Another cube’s five lifts | 0.02484 | 0.00054 |
| Separate fit on five slides | 0.02195 | 0.00032 |

Five lifts reduced slide error by 4.65% against zero z. Averaging the five model scores within each cube, 22 of 30 cubes improved. The mean paired error reduction was 0.00118 (95% interval 0.00006–0.00223). The advantage over another cube’s lifts was smaller, 0.00054 (95% interval −0.00090 to 0.00193). These post-run intervals resample the 30 complete cubes in pairs across conditions, holding the five trained models fixed (10,000 draws, seed 5801). The cube-specific transfer comparison remains inconclusive.

The mean slide MSE after 0–5 lifts was **0.02548, 0.02262, 0.02340, 0.02389, 0.02405, 0.02430**. The first lift gave the lowest error. Each later lift increased it, while the five-lift error remained below the zero-lift baseline. Lift count also increases the cumulative number of optimization steps in this protocol.

Five-lift z supported a linear mass prediction with mean held-out R² of 0.624 and mean absolute error of 61 g. The friction decoder’s R² was negative for every seed (mean −0.115, mean absolute error 0.168). A fresh replay from disk reproduced all 150 cube-by-seed five-lift scores using only prove slides.

[Results CSV](../../runs/lift_slide_belief/nov13/study/results.csv) · [Transfer plot](../../runs/lift_slide_belief/nov13/study/transfer.png) · [Formation plot](../../runs/lift_slide_belief/nov13/study/formation.png) · [Meaning plot](../../runs/lift_slide_belief/nov13/study/meaning.png)

## Data and splits

The collector sampled 200 training cubes and 30 new test cubes. Each received an independent uniform mass in 0.04–0.50 kg and an independent uniform friction coefficient in 0.4–1.0. These intervals span the frozen lift policy’s training settings. The original policy saw discrete masses with correlated friction. The lift policy receives the same 40 g guess on every attempt. A scripted push runs in the same PickCube scene. Both command sources receive zero-mean Gaussian noise with SD 0.05 in normalized action units before clipping to [−1, 1].

Every cube has ten lifts and ten slides of 50 environment steps. Each recorded row spans one issued command at 20 Hz and the resulting observation after five 100 Hz PhysX substeps. The state stores arm/finger joint positions and velocities, gripper opening, cube pose (including two rotation-matrix columns), cube linear and angular velocity, and net joint effort. Effort comes from PhysX incoming joint wrenches and starts at zero on reset. The five action components control translation, finger opening, and grip strength. A row counts as moving when the pre-action cube linear speed exceeds 0.01 m/s. All attempts are retained.

Of 200,000 training-cube rows, 160,000 fit the world model and 40,000 select the checkpoint. Split seed 721 selects two complete lifts and two complete slides per cube for validation. Model seeds 0–4 share this split and the same 30 test cubes. [Dataset fields and checkpoint](../../runs/lift_slide_belief/nov13/data/dataset.json)

## Training and validation

The network has three 256-unit ReLU hidden layers. It predicts the standardized one-step state difference from standardized state/action measurements and the cube’s four-dimensional z. Scaling statistics use the fitting episodes only. All 200 table entries start at zero. Adam minimizes mean squared error over rows and state coordinates plus 0.0001 times the mean squared latent coordinate. The network learning rate is 0.001, the table learning rate is 0.01, and each batch has up to 1,024 rows.

The best validation checkpoint includes both the network and its latent table. Patience is 12 epochs without a 0.1% relative improvement in validation MSE. On moving validation rows, the correct z must reduce MSE by at least 1% against each of two controls, zero and another cube’s z. Each paired improvement must also have a positive lower 95% bootstrap bound across training cubes (2,000 draws). The runner reports failed seeds and proceeds to test evaluation only for seeds meeting both criteria. [Validation details by seed](../../runs/lift_slide_belief/nov13/study/validation.json)

## Held-out evaluation

For each test cube, the model is frozen. The first five lift episodes are revealed one at a time. Starting at zero, the procedure takes 100 Adam steps on all lifts seen so far after each reveal, warm-starting at the previous z. It saves each snapshot by cube ID. Slides 0–4 supply identical moving-step predictions for every condition. A shuffled control uses a different cube’s five-lift z. A slide-fitted reference starts at zero and receives the same 500 total Adam steps on slides 5–9. Later, the saved five-lift z is reloaded by ID and scored again on the recorded prove slides. Test lifts 5–9 are kept in the archive.

The CSV records normalized slide MSE per cube. Each seed contributes eight rows per cube (zero, the five successive lift snapshots, another cube’s five-lift z, and the slide-fitted reference). The figures average cubes equally, with error bars of one standard error across model seeds. These bars measure training-seed variability on this fixed set of cubes. Each seed’s linear mass/friction decoder fits the 200 training table entries with an intercept. The meaning plot compares its predictions from the five-lift test vectors to the true properties.

## Reproduce

Run from the repository root in the project environment. Data collection uses CPU PhysX. World-model fitting and inference support CPU or CUDA.

```bash
source scripts/activate_grip.sh
run=runs/lift_slide_belief/nov13
OPENBLAS_NUM_THREADS=1 python -m experiments.lift_slide_belief.collect "$run/data" --workers 8
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    python -m experiments.lift_slide_belief "$run/data" "$run/study" --device cuda
```

The collector resumes at missing cube files. The study resumes from completed seed checkpoints in the same protocol. Each seed directory contains the frozen checkpoint, validation split and diagnostics, and, after a pass, the saved test-cube beliefs. To replay just the saved beliefs with a fresh process, run from the repository root with the project environment active.

```python
from pathlib import Path
from experiments.lift_slide_belief.collect import load_rows
from experiments.lift_slide_belief.evaluation import retrieve
from experiments.lift_slide_belief.model import load_world_model

run = Path("runs/lift_slide_belief/nov13")
seed = run / "study/seed_0"
rows = load_rows(run / "data", "test")
prove = (rows["cube_id"] == 200) & (rows["task"] == 1) & (rows["episode_id"] < 5)
slides = {k: v[prove] for k, v in rows.items()}
errors = retrieve(load_world_model(seed / "model.pt"), slides, seed / "beliefs.npz")
```
