# HPC Dashboard

Standalone Python terminal dashboard. The installed commands are `hpc-dashboard` and `hpcd`; do not restore the retired `sj` / `slurmjobs` launchers. Only Slurm is supported; do not implement RunAI without a separate request.

- `hpc_dashboard.py` owns queue parsing, rendering, and responsive fullscreen input.
- `hpc_dashboard_telemetry.py` owns persistent collectors, shared-cache state, launch locks, and bounded restart budgets.
- `README.md` is the portable installation, operation, and migration guide. Keep machine-specific operational state and secrets out of this repository.
- Run `python3 -m unittest -v test_hpc_dashboard test_telemetry test_watch_pty` and `git diff --check` before committing. Tests are offline and must not submit or cancel real jobs.
- For live read-only verification, use `hpcd --me --no-watch --no-usage --no-color --timeout 2`. Usage is enabled by default and can launch monitoring steps; do not casually exercise it against live jobs.
- Preserve legacy telemetry cache/step recognition so migration cannot reset lifetime budgets, move active lock inodes, or duplicate collectors. Never delete live launch state.
- Keep Markdown paragraphs and list items on one physical line.
