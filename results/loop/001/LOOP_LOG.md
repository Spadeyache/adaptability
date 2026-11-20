# Research loop log

## Iteration 001 — provenance audit; STOP requires user decision

Plan:
1. Trace explicit physics inputs through saved observations, owning code, and Git history.
2. Match those paths to earlier reported claims before training any probe.
3. Stop for user review if earlier reported results rely on those inputs.

Ran: read-only NumPy checks on saved interaction and physics-conditioning recordings; Git history/blame; source and saved-result inspection. Evidence and source snapshots are under `results/loop/001/`. No model was fitted or changed and no data was collected.

Confirmed: interaction `sim__policy_observation[:,42]` contains true mass × gravity in true mode. Physics-conditioned observation columns 42 and 45 contain true weight and friction. The context encoder is explicitly supplied true constants, including mass and friction.

Git provenance: weight in the general input builder is present by commit `3c716ec` (2026-10-02); the 42-index weight interface is present by `482a4d4` (2026-10-07). True-parameter retry injection and explicit friction conditioning are present by `730435a` (2026-10-28). Privileged context-model source is present by `89ca5db` (2026-11-09). These are repository-history dates, not asserted first execution dates. One historical lookup used an obsolete directory name; its failure is retained in audit.json and the correct path is saved in review.json.

| Lens | Verdict |
|---|---|
| Leakage | FAIL for unrestricted historical inputs; direct truth verified. Shuffled control not run. |
| Confounds | NOT PASSED. Some historical friction was mass-correlated; actions-only control not run. No statistical tolerance silently chosen. |
| Splits | NOT PASSED globally. Main split is cube-disjoint within dataset; smoke identities overlap full training data; legacy cube splits unresolved. No normalization fitted. |
| Robustness | NOT TESTED. No probe/seed runs before STOP. |
| Skeptical PI | FAIL if privileged readouts are used as motion-identification evidence. Tested the weakest point by tracing `encode(data.constants)` to the true-label encoder and saved readout scores. |
| Physics sanity | NOT TESTED. No new physical-observability result exists to assess. |

Earlier claims affected in interpretation: supplied-input policy success, true-input retry references, privileged-context world-model gains and latent readouts. These were documented as privileged comparisons; their numerical results are not thereby invalidated. They cannot answer this loop's label-free question. The current evidence does not establish contamination of the main lift/slide/shove state/action vectors or B0/B1 baselines.

Changed: added this log, FINDINGS.md (explicit STOP report), and iteration evidence only. No earlier results overwritten. No success criteria changed. User decision required before any exclusion, reclassification of prior claims, or training.

Reproduce to a new directory (commands intentionally refuse to overwrite evidence):

```bash
cd /home/ubuntu/faliure
.venv/bin/python results/loop/001/audit.py --output results/loop/reproduce_001
.venv/bin/python results/loop/001/review.py --output results/loop/reproduce_001
```
