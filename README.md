# HPC Dashboard

A standalone terminal dashboard, available as `hpc-dashboard` or `hpcd`, with Slurm and Run:AI/Kubernetes backends. Finished Run:AI workloads are hidden by default.

An indented workload viewer for hosts with Python 3 and Rich. Backend auto-detection selects Slurm when only `squeue` is installed or Run:AI when only `kubectl` is installed; if both clients are installed, select `--backend slurm` or `--backend runai` explicitly. It opens a fullscreen watch dashboard on terminals. The Run:AI backend is strictly read-only and never launches collectors or executes commands inside containers. On Slurm, usage is enabled by default, starting/reusing persistent telemetry steps in your own running allocations. Use `--no-usage` for read-only Slurm operation and `--no-watch` for a snapshot. Your jobs appear first, followed by other users grouped by username. Section and username headers use white text on black backgrounds; detail labels and empty-state text use darker grey (`#707070`). There are no blank spacer lines within either section; a single blank line separates your jobs from the other-users section.

## Install

Requires Python 3.10+ and Rich on POSIX. Slurm needs `squeue` (`srun` and `sstat` for telemetry); GPU readings additionally need `nvidia-smi` on the allocated nodes. Run:AI needs an authenticated `kubectl` context and namespace read permissions; `runai` is needed only for automatic accessible-project discovery. No Kubernetes Python package or Run:AI CLI upgrade is required. Source and tests live at the root of this repository, not in dotfiles.

Clone and install into an isolated user-local environment with uv:

```sh
git clone https://github.com/jrhuebers/hpc-dashboard.git ~/hpc-dashboard
uv tool install --editable ~/hpc-dashboard
uv tool update-shell
hpcd --help
```

Alternatively, if your system Python already has Rich, install checkout symlinks without a package manager:

```sh
python3 -c 'import rich'
command -v squeue
mkdir -p ~/.local/bin
chmod +x ~/hpc-dashboard/hpc_dashboard.py
ln -s ~/hpc-dashboard/hpc_dashboard.py ~/.local/bin/hpc-dashboard
ln -s hpc-dashboard ~/.local/bin/hpcd
```

Keep `~/.local/bin` on PATH. Choose one installation method, and inspect existing destinations before installing; do not replace unrelated commands. New invocations use editable-checkout updates immediately. Restart already-running dashboards after pulling updates; they do not reload themselves.

## Migration from sj / slurmjobs

The old command names are retired, not aliases. After installing and verifying `hpcd`, remove the old launcher symlinks if they still point to the former dotfiles installation:

```sh
ls -l ~/.local/bin/sj ~/.local/bin/slurmjobs
rm ~/.local/bin/sj ~/.local/bin/slurmjobs
hash -r
```

Fresh installations store telemetry in `~/.cache/hpc-dashboard/telemetry/`. If `~/.cache/slurmjobs/telemetry/` already exists, it is deliberately reused instead: moving live collectors' locks or resetting their two-start lifetime budgets could create duplicate monitoring steps. Existing `sj-telemetry` steps are recognized; only new collectors use `hpcd-telemetry`. Do not rename or delete live cache state. Old collectors can continue with their loaded code until their allocations end, but cannot hot-reload from a deleted source checkout. New collectors load this repository's source. No workload or collector restart is required for the command rename.

The source history was extracted from `jrhuebers/dotfiles` using `git subtree split --prefix=tools/slurmjobs`; unrelated dotfiles history is not included.

## Run:AI / Kubernetes

```sh
hpcd --backend runai
hpcd --backend runai --project my-project
hpcd --backend runai --project my-project --project another-project
hpcd --backend runai --namespace my-custom-namespace
hpcd --backend runai --context my-cluster --project my-project
hpcd --backend runai --me --no-watch --no-usage
hpcd --backend runai --state running
hpcd --backend runai --state running,pending
hpcd --backend runai --include-finished
hpcd --backend runai --state failed --include-finished
```

The default view excludes retained succeeded, failed, cancelled, stopped, and deleted workloads. `--include-finished` includes retained terminal objects, not an accounting archive; deleted Kubernetes objects cannot be recovered. An explicit `--state` filter takes precedence over the default and `--include-finished`, so `--state failed` also works by itself. State names are case-insensitive; canonical terminal names are `succeeded`, `failed`, `cancelled`, `stopped`, and `deleted`.

Without project/namespace flags, accessible projects are discovered through `runai list projects` and cached for five minutes. Explicit projects conventionally map to `runai-<project>` namespaces; use `--namespace` for nonstandard mappings. Kubernetes's cluster-wide Project catalog is not treated as an authorization list. No cluster-wide Pod listing is attempted. Explicit `--context` requires explicit project/namespace flags to avoid mixing the Kubernetes context with a different Run:AI CLI cluster. The resolved Kubernetes context is pinned for the lifetime of the dashboard; it never changes your CLI configuration. Own-job identity comes from `kubectl auth whoami`, not the local Unix login. Use `--user EMAIL` when identity discovery is unavailable or the workload owner uses a different identity.

The backend reads available training, interactive, distributed, inference, and distributed-inference workload types, plus legacy `RunaiJob` objects and Pods. Higher-level workload objects take precedence over child controllers; owner UIDs, owned-resource UIDs, and top-owner labels join Pod attempts without matching stale objects solely by a reused name. One row represents a workload, with running/pending Pod counts and failed Pod-attempt counts. Missing owner information may require a legacy job or orphan Pod row. Queue refreshes use batched queries per namespace, not per-workload queries. Unknown and inaccessible fields remain unknown, not zero. Partial query failures are visible warnings; previous complete namespace snapshots are retained where available, and cached values are not presented as new measurements.

The header shows the Kubernetes context and selected projects. Detail lines show project/node location, assigned GPU models where node metadata is readable, requested and allocated GPU quantities, fractional CPU requests, RAM requests, purpose, and research tag. The `purpose` annotation (with `comment` fallback) supplies explanatory text; a custom `project` label supplies the research tag when distinct from the scheduling project. Only running/nonterminal Pod resource requests contribute to current CPU/RAM totals, so retained failed attempts do not inflate allocations. GPU fractions and GPU memory limits are not measurements of compute utilization or billing. Time is elapsed wall time from available start/completion timestamps, not GPU-active time; absent deadlines remain unknown.

Run:AI utilization monitoring is not implemented in this release. With usage enabled, own running jobs explicitly show that metrics are unavailable; `--no-usage` suppresses this line. The backend never starts Slurm collectors, injects containers, uses `kubectl exec`, submits jobs, or modifies workloads. Node metadata is optional. Credentials remain with the native CLI; the backend does not read kubeconfig files or Kubernetes Secrets. Kubernetes output is narrowly projected before entering the dashboard: only allowlisted identity/resource/status/metadata fields and the single `RUNAI_GPU_MEMORY_LIMIT` variable are returned, not arbitrary environment values, commands, annotations, volume definitions, or raw Pod documents.

## Slurm usage

```sh
hpc-dashboard
hpcd --me
hpcd --watch
hpcd --me --watch --usage
hpcd --no-watch
hpcd --no-watch --no-usage
hpcd --watch --interval 30
hpcd --compact
hpcd --no-color
hpcd --timeout 5
```

Each job shows its complete ID and name, state, running step count in the title, optional submission `--comment`, partition and nodes, GPUs, total RAM, CPU count, and elapsed/limit wall time. Pending jobs also show their waiting reason. Job state and running-step count are right-aligned at the terminal edge, with ID/name on the left. Long names wrap in the reserved left column without ellipses; narrow terminals place the status on a separate right-aligned line when it fits. Comments wrap without ellipses, and detail continuations align under their values. Resource amounts are allocations where reported for active jobs, otherwise requests, not measured utilization. GPU models appear when Slurm provides them; unspecified types and unknown representations are labeled explicitly.

Time uses `M:SS`, `H:MM:SS`, or `D-HH:MM:SS`; unlimited and unknown limits are explicit. Pending jobs show zero elapsed time. `--compact` combines resources and time. Piped output and `--no-color` disable colors; `NO_COLOR` is also honored.

A snapshot uses one bounded `squeue --json` call plus one batched `squeue --steps --json` call, without per-job step queries. Titles show `N running steps`, counting only steps whose own Slurm state includes `RUNNING` (`batch`, `extern`, telemetry, and workload steps included). Completing, finished, cancelled, suspended, or pending steps are excluded. This is not historical step count or the lifetime step-ID budget, and 'running' does not mean CPU/GPU activity is continuous. A failed step query or a client without step-state JSON shows `running steps ?` rather than guessing from a state-less step listing. Both queue and step queries are cached for 10 seconds in watch mode. Clients rejecting JSON use a unit-separator text fallback, which has less GPU detail; multi-node GPU requests are labeled per node. Legacy names/comments containing embedded newlines or separator characters cannot be reliably parsed. Queue visibility matches normal `squeue` defaults.

Watch mode is the default when stdout is a terminal. It refreshes fullscreen in the alternate screen every second and exits with `q`, Escape, or Ctrl-C, restoring the previous screen contents. Interactive input temporarily uses cbreak mode with echo and software flow control disabled, so accidental Ctrl-S cannot freeze output; the exact previous terminal settings are restored on normal exit and handled errors. When content exceeds the terminal height, scroll with Up/Down or `k`/`j` (one line), `u`/`d` (half a page up/down), Page Up/Down, Home/End, or the mouse wheel. Half-page motion is half the content viewport height, rounded down with a one-line minimum. A footer always appears at the bottom of the window, showing the visible line range and controls even when everything fits. Scrolling repaints immediately without triggering extra scheduler/telemetry queries, and the scroll offset is retained across refreshes and clamped on resize or queue shrink. Mouse reporting is enabled only during interactive watch and disabled on exit. Unrecognized function-key sequences are ignored. With redirected stdin, only Ctrl-C is available. Queue queries remain limited to once every 10 seconds; elapsed times advance locally between queries. `--interval` controls display refresh, not accounting frequency. On query errors it preserves the previous queue and displays a warning until a successful queue query. Retry intervals are measured from query completion, so a slow timeout does not immediately trigger another query. Content exceeding the viewport is scrollable; `--me`, `--compact`, and `--no-watch` remain available for smaller views or full scrollback snapshots. Piped/redirected stdout automatically uses a snapshot; explicit `--watch` requires a terminal. Usage remains enabled for snapshots/pipes unless `--no-usage` is supplied.

## Slow or unavailable scheduler/filesystem

Fullscreen watch acquires frames in one daemon background worker, with at most one in-flight refresh and one result slot. Keyboard handling, cached-content scrolling, terminal repainting, and q/Escape/Ctrl-C exit remain responsive even during the first query or a stalled scheduler/shared-cache operation. After a refresh has taken a second, the footer shows its duration and, when available, the age of the last displayed frame. Old job/usage values are not new measurements; no replacement workers or queued refreshes accumulate while a fetch is blocked. Exit does not wait for stalled background work.

Scheduler queries honor `--timeout`, including dashboard expiry-cleanup checks. Expiry cleanup checks at most two historical jobs per scan (at most four scheduler calls, including fallbacks), rotates through candidates to avoid starvation, and backs off from scan completion. Collector liveness probes and accounting similarly back off after completion. These safeguards limit scheduler traffic; they cannot impose a deadline on kernel/shared-filesystem calls. A permanently stalled filesystem can therefore leave data stale, but watch input remains available. Snapshot mode still performs synchronous acquisition. To isolate telemetry problems, use `hpcd --no-usage --timeout 2`.

## Persistent utilization collectors

`hpcd` (or explicit `hpcd --usage`) starts or reuses a named `hpcd-telemetry` step in each of your running jobs; it never monitors other users' allocations or starts a new allocation. One detached supervisor owns each step, so closing `hpcd` or its terminal does not stop the collector. Concurrent dashboard invocations share readings and launch locks. Pending jobs have no collector. Sampling is once per second, and readings are published atomically in a user-private shared-home cache rather than an unbounded log. Under a single `Usage` label, CPU/RAM appear on the first line and GPU/VRAM on an aligned second line; jobs without GPUs omit the GPU usage line. Both lines wrap with the same value-column alignment.

CPU readings come from changes in job-level cgroup CPU time, expressed as busy cores and percentage of allocated CPUs. RAM is job-level cgroup-charged memory, which includes cache and collector overhead; it is not process RSS. If cgroup counters are inaccessible, a batched `sstat --allsteps` query supplies slower CPU/RSS fallback readings, at most every 10 seconds. Multi-node cgroup readings must be complete before a job-wide total is shown. Watch mode retains the last successful reading for each node and metric across transient shared-filesystem reads, failed GPU queries, or collector warm-up. Retained values remain visible with an explicit stale/unconfirmed marker and the age of the underlying measurement; they are never silently presented as current or converted to zero. A measurement is unavailable only if no valid value has been seen. Retention is in-memory for that dashboard instance and ends when the job leaves its running-job list.

GPU readings require `nvidia-smi` on the compute nodes. The GPU line shows `GPU device %` (`utilization.gpu`, time with a kernel executing), `Mem activity %` (`utilization.memory`, time during which global device memory was read or written), and VRAM used/total. Memory activity is not access latency, percentage of maximum bandwidth, or memory occupancy. Percentages are averaged over verified devices; partial memory-activity coverage is labeled. Unsupported memory-activity counters remain unavailable without suppressing GPU/VRAM values; last-good memory activity is retained independently with its own stale age. These are device-wide measurements, not per-process GPU compute usage; sharing a device can include another workload's activity. Slurm step GPU UUIDs are used directly; numeric GRES IDs require GPU UUID verification against same-user processes in the exact job cgroup, avoiding assumptions about GRES/index ordering. Thus an idle allocation with no GPU context may show unavailable until ownership can be verified. Totals describe the verified devices; node coverage is displayed, and unverified devices are not guessed. The collector does not change Slurm accounting configuration or require root.

Collector launches use an overlapping step with one lightweight task per node. A persistent collector consumes one step ID, not one per refresh. Failed launches can also consume step IDs; launch locking, startup timeouts, backoff, and a bounded restart budget prevent unbounded step creation. Collector work is charged to the existing job, and the job's end terminates its step. This feature requires a shared cache location and Slurm configurations permitting overlapping steps. Collectors re-exec themselves when their source changes, preserving the same Slurm step rather than consuming another step ID.

After confirmed job completion, the detached supervisor waits one hour locally (without retaining a Slurm step) and removes expired samples and metadata. Subsequent `hpcd --usage` invocations also reclaim expired files if the supervisor was lost. Tiny lock files/directories remain deliberately to avoid lock-inode races. No system cleanup service is installed. The supervisor survives closing `hpcd`, but termination of its launch host or originating allocation can still kill it; later invocations reattach/recover where possible. Do not remove launch-state files for a running job: they preserve duplicate-launch protection and the two-start restart budget.

## Verify

```sh
hpcd --help
hpcd --me --no-watch --no-usage --no-color
cd ~/hpc-dashboard
python3 -m unittest -v test_hpc_dashboard test_telemetry test_watch_pty test_runai
```

For Run:AI read-only verification, use `hpcd --backend runai --me --no-watch --no-usage --no-color --timeout 5`, then repeat with `--include-finished` to inspect retained jobs.

Tests use offline fixtures and mocked commands, so they do not submit or alter jobs. Run:AI coverage includes default terminal filtering, SSO identity, context selection, namespace permissions, fractional resource quantities, owner graph deduplication, distributed/retry Pods, unavailable data, safe output projection, and backend isolation. They cover rendering, scrolling and navigation, queue-query caching/backoff, JSON/legacy parsing, scheduler failures, telemetry parsing, cache lifecycle, and collector launch safeguards. Isolated pseudo-terminal regressions cover stalled initial/subsequent refreshes, cached-content scrolling, q/Escape/Ctrl-C exit, Ctrl-S immunity, alternate-screen/mouse cleanup, and exact terminal-setting restoration.

## Remove

```sh
rm ~/.local/bin/hpcd ~/.local/bin/hpc-dashboard
```

For a uv-managed installation, use `uv tool uninstall hpc-dashboard` instead of removing its symlinks manually. Retain repository source/tests. Existing collectors are independent of the command symlinks and continue until their allocations end; retain `hpc_dashboard_telemetry.py` until then. Expired samples/metadata are cleaned automatically, while tiny lock tombstones remain. No shell aliases, system services, or scheduler configuration changes need removal.
