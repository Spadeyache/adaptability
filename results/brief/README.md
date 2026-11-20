# Saved-results brief

Start with [the one-page results section](/home/ubuntu/faliure/results/brief/results_section.pdf); its editable text is [results_section.md](/home/ubuntu/faliure/results/brief/results_section.md).
All materials use saved runs only. No training, collection, checkpoint loading, prediction evaluation, or decoder fitting was performed to create this brief.

| Material | PNG | SVG |
|---|---|---|
| Training and deployment setup | [setup.png](/home/ubuntu/faliure/results/brief/setup.png) | [setup.svg](/home/ubuntu/faliure/results/brief/setup.svg) |
| Actions-only leak checks | [fig_leak.png](/home/ubuntu/faliure/results/brief/fig_leak.png) | [fig_leak.svg](/home/ubuntu/faliure/results/brief/fig_leak.svg) |
| Plain versus Oracle | [fig_physics.png](/home/ubuntu/faliure/results/brief/fig_physics.png) | [fig_physics.svg](/home/ubuntu/faliure/results/brief/fig_physics.svg) |
| Belief controls and fitting-shove curve | [fig_belief.png](/home/ubuntu/faliure/results/brief/fig_belief.png) | [fig_belief.svg](/home/ubuntu/faliure/results/brief/fig_belief.svg) |
| Mass readout on unseen cubes | [fig_mass.png](/home/ubuntu/faliure/results/brief/fig_mass.png) | [fig_mass.svg](/home/ubuntu/faliure/results/brief/fig_mass.svg) |
| Sequential full-weight adaptation | [fig_drift.png](/home/ubuntu/faliure/results/brief/fig_drift.png) | [fig_drift.svg](/home/ubuntu/faliure/results/brief/fig_drift.svg) |

- [numbers.md](/home/ubuntu/faliure/results/brief/numbers.md): supplied claims checked against saved sources, with qualifications and failed criteria.
- [captions.md](/home/ubuntu/faliure/results/brief/captions.md): two sentences per figure.
- [limits.md](/home/ubuntu/faliure/results/brief/limits.md): four evidence-backed limitations.
- [primary_sources.md](/home/ubuntu/faliure/results/brief/primary_sources.md) and [drift_sources.md](/home/ubuntu/faliure/results/brief/drift_sources.md): exact populations, aggregation methods and source keys.
- `fig_*_data.csv` and `drift_numbers.csv`: every plotted measured value, with individual seeds and source pointers.
- `primary_sources_manifest.json` and `drift_source_hashes.json`: hashes of saved inputs.
- `validation.json`: artifact-format, source-hash, and link checks.

The setup diagram is a schematic of the procedure, not measured data; the number of boxes drawn in a belief row is illustrative.
Step 2 push/belief figures and Step 3 drift figures use distinct test populations and must not be combined as one matched comparison; the source notes identify both.

## Reproduce the materials

From `/home/ubuntu/faliure`, run the following plotting/typesetting commands, which do not run models:

```bash
.venv/bin/python results/brief/setup_diagram.py
.venv/bin/python results/brief/primary_figures.py
.venv/bin/python results/brief/drift_figure.py
.venv/bin/python results/brief/build_results_section.py
```
