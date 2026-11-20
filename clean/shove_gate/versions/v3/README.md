# Varied-shove Oracle gate: ten-cube pilot

The current authorized stage is **ten newly sampled training cubes, ten shoves
each, then wait for the user's go**. Read [PILOT_REPORT.md](PILOT_REPORT.md) for
saved measurements and timing. No Plain, Oracle or belief model has been trained.

The independent split defines 200 training cubes and 30 test cubes. Only cube
IDs 0–9 have active simulator trajectories. Mass is sampled independently from
cube friction, over **0.18–0.50 kg** and **0.4–1.0**, respectively. The raised
mass minimum replaces the earlier 0.04 kg minimum. Every cube/episode has a
separate property-independent random plan: cap uniform 2–4 N, world direction
and cube yaw each uniform over a full turn, face contact offset uniform ±6 mm,
and cube XY start uniform in a ±10 cm square. No plan is rejected based on labels.

This remains an idealized physical pusher on an infinite plane, not a Panda
controller. The 4 cm cube has all six degrees of freedom. The 5 g pusher has
free XYZ translation, locked rotation, no gravity or damping. The upstream
cube face and sphere approach position are calculated solely from the randomized
geometry. Approach and preload use the previous durations; the pulse is 80 ms
at 95% of cap. The release parking target is computed from the initial plan,
never from the resulting cube/pusher motion.

## Combined friction and actual force

The floor material is fixed at 0.3. SAPIEN 3.0.3 creates its native material
without overriding the combine mode ([material source](https://raw.githubusercontent.com/haosulab/SAPIEN/3.0.3/src/physx/material.cpp)).
The installed PhysX 5.3.1 default is average, giving
`combined_friction = (cube_friction + 0.3)/2`
([default](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/_api_build/class_px_material.html),
[combination rule](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/_api_build/struct_px_combine_mode.html)).
This is the configured material combination, not an estimate from deceleration.
All three per-cube true properties occur only in `cubes.json`.

An independent audit of oblique shoves found that PhysX's `extractContacts`
exports **normal impulse only**, despite exposing a vector:
[`dst.impulse = dst.normal * impulse`](https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/105.1-physx-5.3.1/physx/include/PxSimulationEventCallback.h).
[SAPIEN copies that field](https://raw.githubusercontent.com/haosulab/SAPIEN/3.0.3/include/sapien/physx/simulation_callback.hpp).
It cannot validate total oblique contact force by itself.

The corrected collector instead measures full 3D force on the cube from
`F_contact = F_command − m_pusher × Δv_pusher / dt` at every 1 ms physics step.
The pusher has no other translational forces or constraints. Other-body normal
impulses must be zero; friction cannot act at those pairs without normal force.
Zero-impulse proximity points from PhysX's contact margin are recorded separately
from force-bearing contacts. The normal-only API is retained for an independent
projection cross-check. Full force magnitude must remain below cap; after
withdrawal, there must be no cube–pusher contact force.

`versions/v1/` preserves the original 100-episode pilot with incomplete force
instrumentation. `versions/v2/` preserves a one-episode attempt stopped by a
guard that incorrectly treated zero-impulse proximity points as force-bearing
contacts. Active revision 3 repeats the **same ten cube identities and exactly
the same randomized plans** with corrected instrumentation. Archived runs are
excluded from the active training dataset; no inconvenient outcomes were removed.

## Data and planned gate

Each active episode saves exactly 216 transitions: 16 push and 200 coast,
at 5 ms intervals. Coast duration is fixed at one second, including the rest
tail, so sequence length and masks do not reveal stopping time. Required fields
are `cube_id`, `episode_id`, `step`, `phase`, `state`, `action`, `next_state`.
IDs and phase labels are metadata; no IDs enter model features.

The state whitelist contains cube position, 6D rotation, linear/angular velocity,
and pusher position/velocity. Actions contain commanded force, cap, engaged flag,
and the preplanned release target. Measured contact force is diagnostic only.
True properties are accessed only by the simulator setup setter, explicit
Oracle context, and separate offline audits/reporting.

The actions-only diagnostic uses the **preplanned command schedules** for all
230 cubes, with cube-disjoint fitting and evaluation. It does not collect test
motion. Testing ten collected cubes alone would not establish probe generalization.

After go, the requested Plain and Oracle models will use five paired seeds and
three hidden layers of 256 units. All normalization will use training data only.
The sealed prospective protocol names coast one-step normalized error as the
primary gate: at least 20% mean reduction and
`Oracle mean + sample SD < Plain mean − sample SD`.
Ten-step rollouts, stopping-position errors, cube-only errors and moving-coast
errors are also reported; none replaces the primary metric after looking at test
results. Ten-step windows remain entirely within their named phase. Rollouts
use their own predicted states, not intermediate observed states.

## Commands

From `/home/ubuntu/faliure`, collection is deliberately limited to the pilot:

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.collect
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove_gate.audit
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove_gate.report
```

Completed traces are never overwritten. A `STOP.json` blocks automatic resume.
Original `clean/` and `clean/shove/` data seals remain unchanged. The earlier
shove report has a force-measurement correction note; its motion data remain intact.
