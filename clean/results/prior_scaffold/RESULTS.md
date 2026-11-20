# Pilot stopped before collection

The installed motion planner crashes before it can generate the scripted lift. No episodes were collected, no probes were trained, and no world-model results exist. This is an infrastructure failure, not evidence that mass or friction is unobservable.

| Stage | Outcome |
|---|---|
| Verify old ranges and installed version | Mass 0.04–0.50 kg; cube friction 0.4–1.0. ManiSkill3.0.1, mplib0.1.1, SAPIEN3.0.3. |
| CPU simulator and explicit state reader | Reached successfully in host preflight. |
| Motion-planner construction | Native segmentation fault, exit139, in `mplib/planner.py:65`, constructing `ArticulatedModel`. |
| Ten-cube collection | Not started. |
| Data checks1–5 | Not completed. No correlation flags, action-probe R², signal-probe R², or measured height plot. |
| Full collection; models; checks A–E | Not started; still require the user's pilot approval. |

The first sandboxed preflight crashed. Host access allowed the simulator and state reader to initialize but did not fix planner construction. A further diagnostic captured the native crash stack in [preflight_diagnostic.log](/home/ubuntu/faliure/clean/results/preflight_diagnostic.log). The exact cause inside the native constructor is not yet established. NumPy2.2.6 is installed; compatibility has not been established as the cause, and no version was changed.

The new source lives entirely under `clean/` and has no old-project imports. It adapts the installed PickCube reach/grasp/lift solution into a precomputed command sequence that receives only observed initial cube pose and random settings. Physics setters and sensor construction are separate. Provenance and intended matched settings are documented in README.md and data/STATE_COLUMNS.md. These are implementation descriptions, not a completed runtime leakage certification.

Exact diagnostic rerun from the repo root (expected to reproduce the current native crash):

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/clean-mpl \
LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONFAULTHANDLER=1 \
.venv/bin/python -m clean.preflight review
```

Use host library access; the sandbox also failed. A successful Python-level completion writes a new `clean/results/preflight_review.json`. A native crash cannot write that JSON, so capture stderr in a new log file if rerunning. No installed dependency changes, new collection, or alternative controller will be attempted until the user reviews this stop.

The actions-only “near zero” criterion is also awaiting the user's answer. The pilot uses eight train and two test cubes, so its R² will be preliminary. A future GRU probe measures what that fitted probe recovers; it is not a mathematical upper bound on every possible belief estimator.
