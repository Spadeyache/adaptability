# Isolated NumPy compatibility dependency

Copied unchanged from `/home/ubuntu/.local/lib/python3.10/site-packages/`:
- `numpy/`
- `numpy.libs/`
- `numpy-1.26.4.dist-info/`

This pins NumPy 1.26.4 for the installed mplib 0.1.1 planner. No installed package was changed. The same minimal mplib Planner constructor crashed with the repository venv NumPy 2.2.6 and passed after importing NumPy 1.26.4; saved logs are `clean/results/planner_minimal.log` and `clean/results/planner_numpy1_venv.log`. This local compatibility result does not establish a general defect in either library.

From repository root, run `PYTHONPATH=clean/vendor:. .venv/bin/python ...`.
