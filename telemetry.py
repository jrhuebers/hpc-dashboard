#!/usr/bin/env python3
"""Persistent, best-effort telemetry for the caller's running allocations.

The monitoring step shares allocation GRES via --overlap. GPU readings are
restricted to SLURM_JOB_GPUS (device-wide, not process attribution). CPU and
charged RAM come from job cgroups; sstat is a throttled fallback. Files older
than one hour are reclaimed after confirmed completion by the supervisor or
on the next invocation (fallback if the launch host terminates the supervisor).
No scancel is ever used.
"""
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import pwd
import signal
import socket
import subprocess
import sys
import time

NAME = 'sj-telemetry'
STALE = 45
SAMPLE_STALE = 5
TTL = 3600
MAX_STARTS = 2
ACCOUNTING_INTERVAL = 10
_CLEANUP_AT = {}
JOB_ID = re.compile(r'\d+(?:_\d+)?(?:\+\d+)?\Z')


def cache_root():
    root = Path.home() / '.cache' / 'slurmjobs' / 'telemetry'
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise OSError('unsafe telemetry cache')
    root.chmod(0o700)
    return root


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_meta(path):
    meta = read_json(path)
    # Missing metadata is a first launch; an existing but unreadable/corrupt
    # record must never silently reset the lifetime step budget.
    if path.exists() and 'starts' not in meta:
        meta['starts'] = MAX_STARTS
    for field in ('heartbeat', 'last_seen', 'ended_at'):
        if not finite_number(meta.get(field, 0)):
            meta[field] = 0
    starts = meta.get('starts', 0)
    if isinstance(starts, bool) or not isinstance(starts, int) or not 0 <= starts <= MAX_STARTS:
        meta['starts'] = MAX_STARTS  # Fail closed on a corrupt restart budget.
    return meta


def read_sample(path):
    sample = read_json(path)
    if not finite_number(sample.get('time')):
        return {}
    for field in ('cpu_cores', 'ram_mib'):
        value = sample.get(field)
        if value is not None and (not finite_number(value) or value < 0):
            return {}
    gpus = sample.get('gpus', [])
    if not isinstance(gpus, list):
        return {}
    for gpu in gpus:
        if not isinstance(gpu, dict) or not isinstance(gpu.get('uuid'), str):
            return {}
        if any(not finite_number(gpu.get(key)) or gpu[key] < 0
               for key in ('util', 'used', 'total')):
            return {}
    return sample


def atomic_json(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, allow_nan=False)
    os.replace(tmp, path)


@contextlib.contextmanager
def lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            yield True
    finally:
        os.close(fd)


def run(args, timeout=10):
    try:
        result = subprocess.run(args, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def live_step(jobid, timeout=10):
    output = run(['squeue', '--steps', '--jobs', jobid, '--noheader',
                  '--format=%j'], timeout)
    return None if output is None else NAME in output.splitlines()


def job_ended(jobid):
    states = run(['squeue', '--jobs', jobid, '--states=all', '--noheader', '--format=%T'])
    if states is not None:
        return not states.strip()
    # Some releases return an error for an ID that has left the queue. A
    # successful owner-wide listing distinguishes this from scheduler failure.
    ids = run(['squeue', '--user', pwd.getpwuid(os.getuid()).pw_name,
               '--array', '--states=all', '--noheader', '--format=%i'])
    return ids is not None and jobid not in {line.strip() for line in ids.splitlines()}


def node_count(nodes):
    """Count Slurm compressed hostlists, including multiple bracket groups."""
    parts = re.split(r',(?=[^\]]*(?:\[|$))', nodes)
    total = 0
    for part in parts:
        count = 1
        for group in re.findall(r'\[([^]]+)\]', part):
            size = 0
            for entry in group.split(','):
                ends = entry.split('-')
                size += int(ends[1]) - int(ends[0]) + 1 if len(ends) == 2 else 1
            count *= size
        total += count
    return max(1, total)


def seconds(value):
    days, _, clock = value.rpartition('-')
    pieces = clock.split(':')
    total = 0.0
    for piece in pieces:
        total = total * 60 + float(piece)
    return total + (int(days) * 86400 if days else 0)


def mib(value):
    match = re.fullmatch(r'([\d.]+)([KMGTP]?)', value.strip(), re.I)
    if not match:
        raise ValueError(value)
    return float(match[1]) * {'': 1 / 1024**2, 'K': 1 / 1024, 'M': 1,
                            'G': 1024, 'T': 1024**2, 'P': 1024**3}[match[2].upper()]


class Telemetry:
    def __init__(self, user, timeout=10):
        self.user = user
        self.timeout = timeout
        self.previous = {}
        self.accounting = {}
        self.accounting_at = 0

    def _launch(self, job, directory):
        with lock(directory / 'launch.lock') as acquired:
            if not acquired:
                return
            meta = read_meta(directory / 'meta.json')
            now = time.time()
            if now - meta.get('heartbeat', 0) < STALE:
                return
            # A supervisor may be alive despite an old heartbeat (slow filesystem).
            with lock(directory / 'run.lock') as idle:
                if not idle:
                    return
                if meta.get('starts', 0) >= MAX_STARTS:
                    return
                # Fail closed: scheduler errors must never create duplicate steps.
                if live_step(job.id, self.timeout) is not False:
                    return
                meta.update(starts=meta.get('starts', 0) + 1,
                            heartbeat=now, last_seen=now, ended_at=0)
                atomic_json(directory / 'meta.json', meta)
            try:
                subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                  'supervisor', job.id, str(node_count(job.nodes)),
                                  str(job.cpus), str(job.memory or 0)],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True,
                                 close_fds=True, env=launch_env())
            except OSError:
                # Failed attempts also consume the bounded restart budget.
                meta['heartbeat'] = 0
                atomic_json(directory / 'meta.json', meta)

    def _accounting(self, jobs):
        output = run(['sstat', '--allsteps', '--jobs', ','.join(j.id for j in jobs),
                      '--noheader', '--parsable2',
                      '--format=JobID,NTasks,AveCPU,AveRSS'], self.timeout)
        totals = {}
        if output is None:
            return totals
        for line in output.splitlines():
            fields = line.split('|')
            if len(fields) < 4:
                continue
            step, tasks, cpu, rss = fields[:4]
            jobid, _, suffix = step.partition('.')
            if suffix == 'extern':
                continue
            try:
                n = int(tasks)
                c, r = seconds(cpu) * n, mib(rss) * n
            except ValueError:
                continue
            totals.setdefault(jobid, {})[step] = (c, r)
        return totals

    def fetch(self, jobs):
        jobs = [j for j in jobs if j.user == self.user and j.state == 'RUNNING'
                and JOB_ID.fullmatch(j.id)]
        try:
            root = cache_root()
            cleanup(root)
        except OSError:
            return {j.id: 'utilization unavailable (cache)' for j in jobs}
        if not jobs:
            return {}
        now = time.time()
        if now - self.accounting_at >= ACCOUNTING_INTERVAL:
            self.accounting = self._accounting(jobs)
            self.accounting_at = now
        accounting = self.accounting
        result = {}
        for job in jobs:
            directory = root / job.id
            try:
                directory.mkdir(mode=0o700, exist_ok=True)
                if directory.is_symlink():
                    raise OSError('unsafe job directory')
                self._launch(job, directory)
                # Touch without modifying supervisor-owned metadata.
                (directory / 'seen').touch(mode=0o600)
                samples = [read_sample(p) for p in directory.glob('sample-*.json')]
                samples = [s for s in samples if s]
            except OSError:
                result[job.id] = 'utilization unavailable (cache)'
                continue
            steps = accounting.get(job.id, {})
            previous = self.previous.get(job.id)
            cpu_label = 'CPU warming up' if steps else 'CPU unavailable'
            if steps and previous and self.accounting_at > previous[0]:
                # Only compare surviving steps: completed steps cannot cause negatives.
                common = steps.keys() & previous[1].keys()
                if common:
                    delta = sum(max(0, steps[s][0] - previous[1][s][0]) for s in common)
                    busy = delta / (self.accounting_at - previous[0])
                    cpu_label = f'CPU {busy:.1f}/{job.cpus} cores ({100 * busy / max(1, job.cpus):.0f}%)'
            if steps and (not previous or self.accounting_at > previous[0]):
                self.previous[job.id] = (self.accounting_at, steps, cpu_label)
            elif previous:
                cpu_label = previous[2]
            rss = sum(v[1] for v in steps.values())
            ram_label = (f'RSS {rss / 1024:.1f} GiB' if steps else 'RSS unavailable')
            fresh = [s for s in samples if 0 <= now - s.get('time', 0) < SAMPLE_STALE]
            if len(fresh) == node_count(job.nodes) and all(s.get('cpu_cores') is not None for s in fresh):
                busy = sum(s['cpu_cores'] for s in fresh)
                cpu_label = f'CPU {busy:.1f}/{job.cpus} cores ({100 * busy / max(1, job.cpus):.0f}%)'
            if len(fresh) == node_count(job.nodes) and all(s.get('ram_mib') is not None for s in fresh):
                ram_label = f'cgroup RAM {sum(s["ram_mib"] for s in fresh) / 1024:.1f} GiB'
            gpu = [g for s in fresh for g in s.get('gpus', [])]
            if job.gpu == 'no GPU':
                gpu_label = 'no GPU'
            elif gpu:
                gpu_label = (f'GPU device {sum(g["util"] for g in gpu) / len(gpu):.0f}% · '
                             f'VRAM {sum(g["used"] for g in gpu) / 1024:.1f}/'
                             f'{sum(g["total"] for g in gpu) / 1024:.1f} GiB '
                             f'({len([s for s in fresh if s.get("gpus")])}/{node_count(job.nodes)} nodes)')
            else:
                gpu_label = 'GPU unavailable'
            result[job.id] = ' · '.join((cpu_label, ram_label, gpu_label))
            if samples and not fresh:
                result[job.id] += ' · samples stale'
        self.previous = {k: v for k, v in self.previous.items() if k in result}
        return result


def cleanup(root, now=None):
    """Remove old files, not lock inodes/directories (avoids lock unlink races)."""
    if now is None:
        now = time.time()
        if now - _CLEANUP_AT.get(str(root), 0) < ACCOUNTING_INTERVAL:
            return
        _CLEANUP_AT[str(root)] = now
    for directory in root.iterdir():
        if not directory.is_dir() or directory.is_symlink() or not JOB_ID.fullmatch(directory.name):
            continue
        try:
            files = [p for p in directory.iterdir() if p.name not in ('launch.lock', 'run.lock')]
            if not files:
                continue
            meta = read_meta(directory / 'meta.json')
            # Cached queue frames may touch 'seen' briefly after job completion.
            # A confirmed end timestamp prevents those readers extending expiry.
            latest = meta.get('ended_at', 0) or max(
                [directory.stat().st_mtime] + [p.stat().st_mtime for p in files])
            if now - latest < TTL:
                continue
            # Do not reset a failed collector's start budget in a live allocation.
            if not job_ended(directory.name):
                continue
            with lock(directory / 'launch.lock') as acquired:
                if not acquired:
                    continue
                with lock(directory / 'run.lock') as idle:
                    if idle:
                        for p in directory.iterdir():
                            if p.name not in ('launch.lock', 'run.lock') and p.is_file():
                                p.unlink()
        except OSError:
            continue


def launch_env():
    # An sj invocation may itself run inside a different allocation.
    return {k: v for k, v in os.environ.items()
            if not k.startswith(('SLURM_', 'SRUN_'))}


def cgroup_sample(jobid):
    """Read enclosing job counters only; hybrid non-job v2 roots are ignored.

    Charged RAM includes cache and collector overhead, not merely process RSS.
    """
    cpu = ram = None
    try:
        entries = Path('/proc/self/cgroup').read_text().splitlines()
        for entry in entries:
            _, controllers, relative = entry.split(':', 2)
            parts = Path(relative).parts
            matches = [i for i, part in enumerate(parts) if part == 'job_' + jobid]
            if not matches:
                continue
            jobpath = Path(*parts[1:matches[-1] + 1])
            if not controllers:
                base = Path('/sys/fs/cgroup') / jobpath
                stats = dict(line.split() for line in (base / 'cpu.stat').read_text().splitlines())
                cpu = int(stats['usage_usec']) / 1e6
                ram = int((base / 'memory.current').read_text()) / 1024**2
            else:
                names = controllers.split(',')
                candidates = [Path('/sys/fs/cgroup') / controllers / jobpath]
                candidates += [Path('/sys/fs/cgroup') / n / jobpath for n in names]
                for base in candidates:
                    if 'cpuacct' in names and (base / 'cpuacct.usage').exists():
                        cpu = int((base / 'cpuacct.usage').read_text()) / 1e9
                    if 'memory' in names and (base / 'memory.usage_in_bytes').exists():
                        ram = int((base / 'memory.usage_in_bytes').read_text()) / 1024**2
    except (OSError, ValueError, KeyError):
        pass
    return cpu, ram


def gpu_ids(value):
    """Slurm global device indices/UUIDs, never CUDA_VISIBLE_DEVICES ordinals."""
    ids = set()
    for token in value.split(','):
        token = token.strip()
        if re.fullmatch(r'\d+', token):
            ids.add(str(int(token)))
        elif re.fullmatch(r'\d+-\d+', token):
            first, last = map(int, token.split('-'))
            if 0 <= last - first <= 1024:
                ids.update(str(i) for i in range(first, last + 1))
        elif re.fullmatch(r'(?:GPU|MIG)-[A-Za-z0-9/-]+', token):
            ids.add(token)
    return ids


def nvidia_query(kind='gpu'):
    """Return classified diagnostics, not arbitrary stderr/environment dumps."""
    query = ('--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total'
             if kind == 'gpu' else '--query-compute-apps=pid,gpu_uuid')
    command = ['nvidia-smi', query, '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=2, check=False)
    except FileNotFoundError:
        return None, {'status': 'nvidia-smi missing'}
    except subprocess.TimeoutExpired:
        return None, {'status': 'nvidia-smi timeout'}
    except OSError:
        return None, {'status': 'nvidia-smi execution error'}
    if result.returncode:
        return None, {'status': 'nvidia-smi failed', 'returncode': result.returncode}
    return result.stdout, {'status': 'ok'}


def process_gpu_uuids():
    """Prove GPU UUID ownership using same-UID PIDs in the actual job cgroup.

    Numeric GRES ordering is site-dependent, so it is not an authorization map.
    Idle GPUs without UUID Slurm IDs or a verified context remain unavailable.
    """
    jobid = os.environ.get('SLURM_JOB_ID', '')
    if not JOB_ID.fullmatch(jobid):
        return set(), 'no actual Slurm job ID'
    output, detail = nvidia_query('compute')
    if output is None:
        return set(), detail['status']
    uuids = set()
    for line in output.splitlines():
        fields = [v.strip() for v in line.split(',')]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].startswith('GPU-'):
            continue
        proc = Path('/proc') / fields[0]
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            cgroups = (proc / 'cgroup').read_text().splitlines()
            if any('job_' + jobid in entry.split(':', 2)[-1].split('/') for entry in cgroups):
                uuids.add(fields[1])
        except OSError:
            continue  # Process exit and inaccessible proc entries are normal.
    return uuids, 'verified process contexts' if uuids else 'no verified process contexts'


def gpu_sample(diagnostics=None):
    # STEP_GPUS exists in srun --jobid steps even when JOB_GPUS is not set.
    source = 'SLURM_STEP_GPUS' if os.environ.get('SLURM_STEP_GPUS', '').strip() else 'SLURM_JOB_GPUS'
    allowed = gpu_ids(os.environ.get(source, ''))
    detail = {'source': source, 'allowed_ids': sorted(allowed), 'status': 'no Slurm GPU IDs'}
    if not allowed:
        if diagnostics is not None:
            diagnostics.update(detail)
        return []
    authorized = {identifier for identifier in allowed if identifier.startswith('GPU-')}
    if any(identifier.isdigit() for identifier in allowed):
        process_uuids, identity_status = process_gpu_uuids()
        authorized.update(process_uuids)
        detail['identity_status'] = identity_status
    detail['authorized_uuids'] = sorted(authorized)
    if not authorized:
        detail['status'] = 'GPU identity unverified'
        if diagnostics is not None:
            diagnostics.update(detail)
        return []
    output, query_detail = nvidia_query()
    detail.update(query_detail)
    readings = []
    matched = 0
    for line in (output or '').splitlines():
        values = [v.strip() for v in line.split(',')]
        if len(values) != 5 or values[1] not in authorized:
            continue
        matched += 1
        try:
            util, used, total = map(float, values[2:])
            if all(math.isfinite(v) for v in (util, used, total)):
                readings.append({'uuid': values[1], 'util': util, 'used': used, 'total': total})
        except ValueError:
            continue
    if output is not None and not readings:
        detail['status'] = 'invalid GPU metrics' if matched else 'no matching GPU devices'
    detail['matched_devices'] = matched
    if diagnostics is not None:
        diagnostics.update(detail)
    return readings


def source_version():
    stat = Path(__file__).resolve().stat()
    return stat.st_mtime_ns, stat.st_size


def reload_collector(version, jobid):
    """Exec in-place on source changes, preserving the same Slurm step/PID."""
    try:
        changed = source_version() != version
    except OSError:
        return
    if changed:
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                                 'collector', jobid])


def collector(jobid):
    directory = cache_root() / jobid
    host = re.sub(r'[^\w.-]', '_', socket.gethostname())
    version = source_version()
    previous = None
    while True:
        start = time.monotonic()
        reload_collector(version, jobid)
        cpu, ram = cgroup_sample(os.environ.get('SLURM_JOB_ID', jobid))
        now = time.monotonic()
        cores = None
        if cpu is not None and previous is not None:
            cores = max(0, (cpu - previous[1]) / (now - previous[0]))
        previous = (now, cpu) if cpu is not None else None
        diagnostics = {}
        gpus = gpu_sample(diagnostics)
        try:
            atomic_json(directory / f'sample-{host}.json',
                        {'time': time.time(), 'gpus': gpus, 'gpu_diagnostic': diagnostics,
                         'cpu_cores': cores, 'ram_mib': ram})
        except OSError:
            # Transient shared-filesystem failures must not spend a step restart.
            pass
        time.sleep(max(0, 1 - (time.monotonic() - start)))


def supervisor(jobid, nodes, cpus, memory):
    root = cache_root()
    directory = root / jobid
    directory.mkdir(mode=0o700, exist_ok=True)
    with lock(directory / 'run.lock') as acquired:
        if not acquired or live_step(jobid) is not False:
            return
        # --overlap shares CPUs, memory and inherited allocation GRES. No GPUs
        # outside this allocation are requested. Never scancel the allocation.
        command = ['srun', '--overlap', '--exact', '--immediate=10', f'--nodes={nodes}',
                   '--ntasks-per-node=1', '--cpus-per-task=1', '--jobid', jobid,
                   '--job-name=' + NAME, sys.executable,
                   str(Path(__file__).resolve()), 'collector', jobid]
        proc = None
        stopping = False
        def stop(signum, frame):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        next_check = 0
        ended = False
        try:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=True, close_fds=True, env=launch_env())
            while proc.poll() is None and not stopping:
                now = time.time()
                with lock(directory / 'launch.lock') as writable:
                    if writable:
                        meta = read_meta(directory / 'meta.json')
                        meta.update(heartbeat=now, last_seen=now)
                        atomic_json(directory / 'meta.json', meta)
                if now >= next_check:
                    if job_ended(jobid):
                        ended = True
                        break
                    next_check = now + 30
                time.sleep(1)
        finally:
            if proc is not None and proc.poll() is None:
                # Terminate only our srun process, never a job or unrelated step.
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            with lock(directory / 'launch.lock') as writable:
                if writable:
                    meta = read_meta(directory / 'meta.json')
                    meta['heartbeat'] = 0
                    atomic_json(directory / 'meta.json', meta)
    if not ended:
        ended = job_ended(jobid)
    if ended:
        with lock(directory / 'launch.lock') as writable:
            if writable:
                meta = read_meta(directory / 'meta.json')
                meta['ended_at'] = time.time()
                atomic_json(directory / 'meta.json', meta)
    if ended and not stopping:
        # Local-only sleeper: the Slurm monitoring step has already exited.
        # Keeping the budget until GC also avoids a final refresh launching again.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        time.sleep(TTL + 1)
    cleanup(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['supervisor', 'collector'])
    parser.add_argument('jobid')
    parser.add_argument('resources', nargs='*')
    args = parser.parse_args()
    if not JOB_ID.fullmatch(args.jobid):
        parser.error('invalid job ID')
    if args.mode == 'collector':
        collector(args.jobid)
    elif len(args.resources) == 3:
        nodes, cpus, memory = args.resources
        supervisor(args.jobid, int(nodes), int(cpus), float(memory))
    else:
        parser.error('supervisor requires nodes cpus memory')


if __name__ == '__main__':
    main()
