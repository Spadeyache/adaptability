# STOP report — the question remains unanswered

Iteration 001 reached the user’s stop rule about earlier reported results. No probes were trained. This is not an all-lenses-pass final report, and “untested” does not mean “no information.”

| Task | Mass R² from one episode | Friction R² from one episode |
|---|---|---|
| Lift (fixed or varied grip) | Not run | Not run |
| Slow slide | Not run | Not run |
| Shove | Not run | Not run |
| Other archived tasks | Not run | Not run |

**What leaked.** Some historical observations contain the answers: column 42 stores mass × gravity; explicit friction conditioning adds friction at column 45. In `experiments/interaction_adaptation/evaluate.py:175`, true mode sets the supplied estimates to actual mass/friction before the policy acts. `robot_adaptation/simulation/collection_lift.py:53` converts supplied mass into weight. `robot_adaptation/simulation/training_lift.py:248` directly reads true mass; lines 252–259 append friction. `experiments/physics_conditioning/evaluate.py:25` preserves or replaces these coordinates. Wrong-value interventions can remain invertible functions of the truth. Removing these columns later would not remove privileged information already used to choose actions.

**Since when.** Git records show weight in input construction by `3c716ec`, October 2, 2026; weight at index 42 by `482a4d4`, October 7; explicit friction and true-mode retry injection by `730435a`, October 28. The privileged context encoder is present by `89ca5db`, November 9. These are verified commit dates, not exact first-run dates.

**Earlier results affected.** The policy-conditioning report’s true-input success and retry oracle references assume supplied physics. The older world-model report’s 6.4% forecast improvement and mass/friction readout R² of 0.6822/0.5513 use an encoder fed true constants (`experiments/context_dynamics/train.py:66`). These values come from its saved comparison.json/probe.csv. They remain valid privileged-input results, but cannot demonstrate inference from motion; the older report explicitly acknowledges this. This does not establish invalidity of adaptive/fixed retry arms, or B0/B1 results on the main datasets. B2 was explicitly an oracle. Main lift/slide/shove state/action builders have no direct label injection identified by the inventory.

**Review.** Leakage fails for unrestricted archives. Confounds and splits have unresolved dataset-specific problems. Robustness and physics sanity are untested. The skeptical-PI test confirms that earlier privileged readouts cannot answer the present question. No lens is claimed to pass; detailed verdicts are in `results/loop/001/review.json`.

**Weakest remaining point.** No leakage-free episode-to-property probe has run under all required controls. Absence of direct label columns alone does not prove that action generation is unconfounded.

**Fix/status.** No fix applied and no older output changed. Awaiting the user's decision about retaining privileged results only for their original oracle/conditioning claims and excluding those recordings from identification probes. No thresholds or success criteria have been relaxed.

Reproduce the audit into a new directory:

```bash
cd /home/ubuntu/faliure
.venv/bin/python results/loop/001/audit.py --output results/loop/reproduce_001
.venv/bin/python results/loop/001/review.py --output results/loop/reproduce_001
```

All iteration evidence and source snapshots: `results/loop/001/`. No training command exists yet because the stop rule fired first.
