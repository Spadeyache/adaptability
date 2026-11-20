# Step 2: push confirmation and object-latent adaptation

**C1 passed; C2 failed; C3 passed.** See [RESULTS.md](RESULTS.md) for the saved scores, validation and prefix curves, mass readout, secondary forecasts, and limitations.

The [protocol and reasons](PROTOCOL.md) preceded all Step 2 runs. Original training traces remain byte-identical; IDs 200–259 are retired and unchanged. Fresh test IDs 260–289 use seed 2026111519.

IDs 0–179 shoves 0–7 fit weights and normalization; shoves 8–9 choose checkpoints and budgets. IDs 180–199 only select latent dimension and z-fitting settings. Test support is shoves 0–4; query is shoves 5–9.

No test result changes a checkpoint, fitting choice, gate, or sample. A protocol STOP is final for this run. Report generation reads saved outputs only; it never simulates, trains, adapts, or scores.

Rebuild the report from `/home/ubuntu/faliure`:

```bash
PYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.push_latent.report
```
