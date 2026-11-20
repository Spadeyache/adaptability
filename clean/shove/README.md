# Capped-force shove sensitivity check

**Force-measurement correction from the subsequent varied-shove audit:** the
PhysX contact API used below exports normal impulses only. Its reported values
do not bound full 3D contact force. The original momentum audit checked the
pushing-axis component; the locked transverse axes prevent reconstructing the
remaining components. The saved stopping-distance and speed measurements remain
valid. The corrected full-force measurement is documented in
[shove_gate/README.md](../shove_gate/README.md).

This follow-up performs no world-model training. The measured results are in
[RESULTS.md](RESULTS.md), with every final run saved in `data/trajectories/`.
The original `clean/data/` split, property table, protocol and seals are unchanged.

The probe reuses the original cube mass/inertia/material setter, 4 cm cube,
surface material coefficient 0.3, and CPU SAPIEN/PhysX. It replaces the Panda
with an instrumented dynamic spherical pusher on a horizontal rail and replaces
the finite table with an infinite plane. This is an explicit object-physics
probe, not a validation of a Panda shove controller. The cube retains all six
degrees of freedom. Its position and velocity are never set after initialization.

The pusher weighs 5 g, has an 8 mm radius, starts 3 mm from the cube face, and
coasts toward the cube at 5 mm/s for 0.8 s. A fixed 0.03 N preload lasts 0.1 s.
The push then lasts 80 ms at a force command of 95% of the declared 2, 3 or 4 N
cap. This headroom is fixed across objects. At release the pusher is parked
10 cm back without changing cube state; this is an ideal instantaneous tool
withdrawal. The cube then slides freely to confirmed rest.

At every 1 ms physics step we sum the magnitudes of all pusher–cube contact
impulses and divide by 1 ms. This is a conservative bound on the resultant
contact-force magnitude at that simulation resolution. The collector rejects a
force-cap exceedance, contact after release, failure to stop, or excessive cube
tilt/liftoff. No contact impulses are clipped to manufacture the measured cap.
The actual delivered force is lower for lighter cubes because the pusher itself
accelerates; measured force ranges are reported.

`script.py` accepts only public settings and emits identical force/time
sequences. It cannot inspect cube IDs, simulator properties, or labels. The
new true property table lives only in `data/cubes.json`; it is consumed by the
setup setter and offline reporting. Sensor arrays contain no mass/friction
labels or numerical cube IDs. No original held-out motion is read.

Both seven-point sweeps use three identical fresh scene resets at each setting.
No artificial noise is added. Thus zero repeat spread means deterministic
reproduction, not robustness to sensor noise, alignment errors, or hardware.
The 3 N sweeps run first; if their stopping-distance effect does not exceed the
largest within-setting repeat range, collection stops. The same check is applied
after each remaining strength. The definitions are sealed in `data/protocol.json`.

## Reproduce saved analysis

From `/home/ubuntu/faliure`:

```bash
PYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove.analyze
```

The collector is resumable and will not overwrite completed raw traces:

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove.collect
```

To reproduce physics from scratch, copy the complete `clean/` package to a
separate workspace, preserving the package path `clean/shove/`. Remove only
`clean/shove/data/` and `clean/shove/results/` in that copy. Change to the copied
workspace and use the original environment's absolute interpreter path
`/home/ubuntu/faliure/.venv/bin/python` with `PYTHONPATH=clean/vendor:.` to run the
copied collector. A differently named probe directory alone will not work
because imports use `clean.shove`. Do not delete or regenerate this experiment's
sealed evidence. If a collection creates `results/STOP.json`, inspect that
failure and stop; do not resume past it.

Independent saved-trace validation:

```bash
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove.test_shove --expected-trials 126
```

This recomputes the reported metrics, checks stop-window onset and unchanged
seals, checks identical commands, and verifies that measured contact impulse
balances applied actuator impulse minus the pusher's change in momentum.

`development/` holds nine preliminary actuator checks. Their approach used a
speed servo; their provisional stopping-time field includes a confirmation tail
and is **not used for final figures or conclusions**. The final collection uses
the fixed open-loop approach and corrected stop-onset measurement. Two initial
development attempts crashed when applying force to a disabled SAPIEN body;
that API error was fixed by parking a kinematic pusher on release. Those attempts
produced no result traces and are not counted as completed trials.
