# HPC Dashboard

Standalone Python terminal dashboard. The installed commands are `hpc-dashboard` and `hpcd`; do not restore the retired `sj` / `slurmjobs` launchers. Supports Slurm and Run:AI/Kubernetes. Run:AI is read-only and excludes finished workloads by default.

- `hpc_dashboard.py` owns the Slurm backend, shared caching/rendering, CLI selection, and responsive fullscreen input.
- `hpc_dashboard_model.py` owns scheduler-neutral display records and terminal text sanitization.
- `hpc_dashboard_runai.py` owns the read-only Run:AI backend: pinned contexts, scoped discovery, safe field projections, UID-based grouping, and terminal filtering. Never read raw Pod documents/credentials or add exec/collector side effects for usage. Keep RCP-specific configuration outside this repository.
- `hpc_dashboard_telemetry.py` owns persistent collectors, shared-cache state, launch locks, and bounded restart budgets.
- `README.md` is the portable installation, operation, and migration guide. Keep machine-specific operational state and secrets out of this repository.
- Run `python3 -m unittest -v test_hpc_dashboard test_telemetry test_watch_pty test_runai` and `git diff --check` before committing. Tests are offline and must not submit or cancel real jobs.
- For live read-only verification, use `hpcd --me --no-watch --no-usage --no-color --timeout 2`. Usage is enabled by default and can launch monitoring steps; do not casually exercise it against live jobs.
- On Run:AI clusters, verify `hpcd --backend runai --me --no-watch --no-usage --no-color --timeout 5` and repeat with `--include-finished`; do not submit a test workload without permission.
- Preserve legacy telemetry cache/step recognition so migration cannot reset lifetime budgets, move active lock inodes, or duplicate collectors. Never delete live launch state.
- Keep Markdown paragraphs and list items on one physical line.
