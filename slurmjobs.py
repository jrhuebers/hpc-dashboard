#!/usr/bin/env python3
"""Vertically formatted Slurm queue viewer with optional job telemetry."""
import argparse
import getpass
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text


def number(value, default=0):
    if isinstance(value, dict):
        if value.get('infinite'):
            return math.inf
        return value.get('number', default) if value.get('set', True) else default
    return default if value is None else value


def clean(value):
    # Job names/comments are untrusted terminal input, not Rich markup.
    return ''.join(c if c.isprintable() or c == '\n' else ' ' for c in str(value or ''))


def duration(seconds):
    if seconds == math.inf:
        return 'unlimited'
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f'{days}-{hours:02}:{minutes:02}:{seconds:02}'
    if hours:
        return f'{hours}:{minutes:02}:{seconds:02}'
    return f'{minutes}:{seconds:02}'


def parse_duration(value):
    if value.upper() in ('UNLIMITED', 'INFINITE'):
        return math.inf
    if value.upper() in ('N/A', 'NOT_SET', 'INVALID', ''):
        return None
    days = 0
    if '-' in value:
        day, value = value.split('-', 1)
        days = int(day)
    parts = [int(p) for p in value.split(':')]
    if len(parts) == 1:
        return days * 86400 + parts[0] * 60
    if len(parts) == 2:
        return days * 86400 + parts[0] * 60 + parts[1]
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def memory_mib(value):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGTPE]?)(?:[cn])?', str(value), re.I)
    if not match:
        return None
    amount, unit = match.groups()
    return float(amount) * {'': 1, 'K': 1 / 1024, 'M': 1, 'G': 1024,
                           'T': 1024**2, 'P': 1024**3, 'E': 1024**4}[unit.upper()]


def memory_label(mib):
    if mib is None:
        return 'RAM unknown'
    if mib == 0:
        return 'all node RAM'
    for divisor, unit in [(1024**2, 'TiB'), (1024, 'GiB'), (1, 'MiB')]:
        if mib >= divisor:
            return f'{mib / divisor:g} {unit} RAM'
    return f'{mib:g} MiB RAM'


def tres(value):
    return dict(item.split('=', 1) for item in (value or '').split(',') if '=' in item)


def gpu_label(resources, requested=''):
    typed = []
    total = resources.get('gres/gpu')
    for key, count in resources.items():
        if key.startswith('gres/gpu:'):
            typed.append(f'{count} × {key.split(":", 1)[1]}')
    if typed:
        return ', '.join(typed)
    if total and total != '0':
        return f'{total} × GPU (type unspecified)'
    # Older releases may expose GRES only as gpu:type:count.
    found = re.findall(r'(?:^|,)(?:gres/)?gpu(?::([^,:()]+))?:(\d+)', requested)
    if found:
        return ', '.join(f'{count} × {kind or "GPU (type unspecified)"}' for kind, count in found)
    if requested in ('N/A', 'Unknown', 'UNKNOWN') or 'gpu' in requested.lower():
        return 'GPU unknown'
    return 'no GPU'


@dataclass
class Job:
    id: str
    name: str
    user: str
    state: str
    partition: str
    nodes: str
    cpus: int
    memory: float | None
    gpu: str
    elapsed: float
    limit: float | None
    comment: str = ''
    reason: str = ''


def from_json(raw, now):
    states = raw.get('job_state', [])
    state = ', '.join(states) if isinstance(states, list) else str(states)
    pending = 'PENDING' in state
    resources = tres(raw.get('tres_alloc_str') if not pending else '')
    resources = resources or tres(raw.get('tres_req_str'))
    cpus = int(resources.get('cpu', number(raw.get('cpus'))))
    nodes = int(resources.get('node', number(raw.get('node_count'), 1)))
    memory = memory_mib(resources['mem']) if 'mem' in resources else None
    if memory is None:
        per_cpu = number(raw.get('memory_per_cpu'))
        per_node = number(raw.get('memory_per_node'))
        if per_cpu or (isinstance(raw.get('memory_per_cpu'), dict) and raw['memory_per_cpu'].get('set')):
            memory = per_cpu * cpus
        elif per_node or (isinstance(raw.get('memory_per_node'), dict) and raw['memory_per_node'].get('set')):
            memory = per_node * nodes
    start = number(raw.get('start_time'))
    end = number(raw.get('end_time'))
    suspend = number(raw.get('suspend_time'))
    before_suspend = number(raw.get('pre_sus_time'))
    elapsed = 0
    if not pending and start and start <= now:
        if 'SUSPENDED' in state:
            elapsed = before_suspend
        elif suspend:
            elapsed = before_suspend + max(0, (min(now, end) if end else now) - suspend)
        else:
            elapsed = max(0, (min(now, end) if end else now) - start)
    limit = number(raw.get('time_limit'), None)
    if limit is not None:
        limit *= 60
    job_id = str(raw.get('job_id', '?'))
    array_id = number(raw.get('array_job_id'))
    task_id = number(raw.get('array_task_id'), None)
    task_string = raw.get('array_task_string')
    if array_id and task_string:
        job_id = f'{array_id}_[{task_string}]'
    elif array_id and task_id is not None and task_id != 4294967294:
        job_id = f'{array_id}_{task_id}'
    if number(raw.get('het_job_id')):
        job_id = f'{number(raw["het_job_id"])}+{number(raw.get("het_job_offset"))}'
    reason = raw.get('state_description') or raw.get('state_reason') or ''
    return Job(job_id, clean(raw.get('name')), clean(raw.get('user_name')), state,
               clean(raw.get('partition')), clean(raw.get('nodes')), cpus, memory,
               gpu_label(resources, raw.get('tres_per_node', '')), elapsed, limit,
               clean(raw.get('comment')), clean(reason))


# ASCII unit separator avoids collisions with ordinary names and comments.
SEPARATOR = '\x1f'
FIELDS = ['%i', '%j', '%u', '%T', '%P', '%N', '%C', '%m', '%D', '%M', '%l', '%b', '%k', '%r']


def from_text(line):
    values = line.split(SEPARATOR)
    if len(values) != len(FIELDS):
        raise ValueError('Cannot parse legacy squeue output; use a Slurm version with --json')
    jid, name, user, state, partition, nodes, cpu, mem, nnode, elapsed, limit, gres, comment, reason = values
    ram = memory_mib(mem)
    if ram is not None:
        ram *= int(cpu) if mem.endswith('c') else int(nnode)
    gpu = gpu_label({}, gres)
    # Legacy %b is a per-node request, not an allocated job-wide GPU total.
    if int(nnode) > 1 and ' × ' in gpu:
        gpu += ' per node'
    return Job(jid, clean(name), clean(user), state, partition, nodes, int(cpu), ram,
               gpu, parse_duration(elapsed) or 0, parse_duration(limit),
               '' if comment in ('(null)', 'N/A') else clean(comment), clean(reason))


class Queue:
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.legacy = False

    def run(self, args):
        try:
            return subprocess.run(['squeue', *args], capture_output=True, text=True,
                                  timeout=self.timeout, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError('squeue is not installed or not on PATH') from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'Slurm did not respond within {self.timeout:g} seconds') from exc

    def fetch(self, me=None):
        filters = ['--user', me] if me else []
        if not self.legacy:
            result = self.run(['--json', *filters])
            if result.returncode:
                error = result.stderr.lower()
                if 'json' in error and any(s in error for s in ('unrecognized', 'unknown', 'invalid option', 'not supported')):
                    self.legacy = True
                else:
                    raise RuntimeError(clean(result.stderr.strip()) or 'squeue failed')
            else:
                data = json.loads(result.stdout)
                if data.get('errors'):
                    raise RuntimeError('Slurm returned errors: ' + clean(str(data['errors'])))
                return [from_json(job, time.time()) for job in data.get('jobs', [])]
        result = self.run(['--noheader', '--noconvert', '--format', SEPARATOR.join(FIELDS), *filters])
        if result.returncode:
            raise RuntimeError(clean(result.stderr.strip()) or 'squeue failed')
        return [from_text(line) for line in result.stdout.splitlines() if line.strip()]


def job_lines(job, indent='', compact=False, usage=None):
    state_color = 'green' if job.state == 'RUNNING' else 'yellow' if job.state == 'PENDING' else 'cyan'
    # Continuation indentation is preserved even for very long names/comments.
    title = Text(indent + job.id + ' · ' + job.name + '  ', style='bold', overflow='fold')
    title.append(job.state, style=state_color)
    yield title
    def detail(label, value):
        text = Text(indent + '  ' + f'{label:<11}', style='dim', overflow='fold')
        text.append(value, style='default')
        return text
    if job.comment.strip():
        yield detail('Comment', job.comment)
    yield detail('Location', job.partition + ' · ' + (job.nodes or 'not allocated'))
    resources = f'{job.gpu} · {memory_label(job.memory)} · {job.cpus} CPUs'
    timing = duration(job.elapsed) + '/' + (duration(job.limit) if job.limit is not None else 'unknown')
    if compact:
        yield detail('Resources', resources + ' · Time ' + timing)
    else:
        yield detail('Resources', resources)
        yield detail('Time', timing)
    if usage is not None:
        yield detail('Usage', clean(usage))
    if job.state == 'PENDING' and job.reason not in ('', 'None', '(null)'):
        yield detail('Waiting', job.reason)


def wrap_lines(lines, width):
    """Wrap without ellipses, aligning detail continuations under their values."""
    console = Console(width=width)
    result = []
    for line in lines:
        plain = line.plain
        indent = len(plain) - len(plain.lstrip(' '))
        detail = plain.lstrip().startswith(('Comment ', 'Location ', 'Resources ', 'Time ', 'Waiting ', 'Usage '))
        prefix = indent + 13 if detail else indent + 2
        if not plain or len(plain) <= width and '\n' not in plain:
            result.append(line)
            continue
        if detail:
            head = line[:prefix]
            chunks = line[prefix:].wrap(console, max(1, width - prefix), overflow='fold')
            for index, chunk in enumerate(chunks):
                result.append((head if index == 0 else Text(' ' * prefix)) + chunk)
        else:
            chunks = line.wrap(console, max(1, width - prefix), overflow='fold')
            for index, chunk in enumerate(chunks):
                result.append((Text('') if index == 0 else Text(' ' * prefix)) + chunk)
    return result


def render(jobs, user, compact=False, width=80, usage=None):
    own = [job for job in jobs if job.user == user]
    others = defaultdict(list)
    for job in jobs:
        if job.user != user:
            others[job.user].append(job)
    def sort(jobs):
        return sorted(jobs, key=lambda j: (0 if j.state == 'RUNNING' else 1 if j.state == 'PENDING' else 2, j.id))
    lines = [Text(f'MY JOBS · {user} · {len(own)} jobs', style='bold #ffffff on #000000')]
    for job in sort(own):
        lines.extend(job_lines(job, compact=compact, usage=(usage or {}).get(job.id)))
    if not own:
        lines.append(Text('  No active jobs.', style='dim'))
    if others:
        lines.extend([Text(''), Text(f'OTHER USERS · {sum(map(len, others.values()))} jobs', style='bold #ffffff on #000000')])
        for name, group in sorted(others.items()):
            lines.append(Text(name, style='bold #ffffff on #000000'))
            for job in sort(group):
                lines.extend(job_lines(job, indent='  ', compact=compact))
    return Group(*wrap_lines(lines, width))


class QueueCache:
    """Render at 1 Hz without polling the scheduler at 1 Hz."""
    def __init__(self, queue, user=None, interval=10):
        self.queue = queue
        self.user = user
        self.interval = interval
        self.jobs = None
        self.fetched = 0
        self.attempted = -math.inf

    def fetch(self):
        now = time.monotonic()
        if self.jobs is None or now - self.attempted >= self.interval:
            # Retain the last good queue on failure, and retry at the usual cadence.
            self.attempted = now
            self.jobs = self.queue.fetch(self.user)
            now = self.fetched = time.monotonic()
        age = max(0, now - self.fetched)
        return [replace(job, elapsed=job.elapsed + age)
                if job.state == 'RUNNING' else job for job in self.jobs]


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be a finite positive number')
    return result


def main():
    parser = argparse.ArgumentParser(description='Readable Slurm jobs. --usage starts persistent collectors in your jobs. Time is elapsed/limit.')
    parser.add_argument('--me', action='store_true', help='show only your jobs')
    parser.add_argument('--watch', action='store_true', help='refresh in place; Ctrl-C to stop')
    parser.add_argument('--interval', type=positive, default=1, help='display refresh in seconds (default: 1; queue queried every 10s)')
    parser.add_argument('--usage', action='store_true', help='start/reuse persistent CPU/RAM/GPU collectors for your running jobs')
    parser.add_argument('--timeout', type=positive, default=10, help='scheduler query timeout (default: 10)')
    parser.add_argument('--compact', action='store_true', help='combine resources and time')
    parser.add_argument('--no-color', action='store_true', help='disable colors')
    args = parser.parse_args()
    console = Console(no_color=args.no_color or 'NO_COLOR' in os.environ, highlight=False)
    user = getpass.getuser()
    if args.watch and not console.is_terminal:
        parser.error('--watch requires a terminal; omit it for piped output')
    queue = QueueCache(Queue(args.timeout), user if args.me else None)
    telemetry = None
    if args.usage:
        # Resolve sibling module even when invoked through ~/.local/bin/sj.
        sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
        from telemetry import Telemetry
        telemetry = Telemetry(user, timeout=args.timeout)
    def frame():
        jobs = queue.fetch()
        usage = None
        if telemetry is not None:
            try:
                usage = telemetry.fetch(jobs)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                usage = {job.id: 'Telemetry unavailable: ' + clean(exc)
                         for job in jobs if job.user == user and job.state == 'RUNNING'}
        return render(jobs, user, args.compact, console.width, usage=usage)
    try:
        initial = frame()
        if not args.watch:
            console.print(initial)
            return 0
        # No alternate screen: output remains accessible in terminal scrollback.
        with Live(initial, console=console, auto_refresh=False, vertical_overflow='visible') as live:
            while True:
                time.sleep(args.interval)
                try:
                    live.update(frame(), refresh=True)
                except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                    live.update(Group(initial, Text('Refresh failed: ' + clean(exc), style='red')), refresh=True)
                else:
                    initial = live.renderable
    except KeyboardInterrupt:
        return 0
    except BrokenPipeError:
        return 0
    except (RuntimeError, ValueError, TypeError, KeyError) as exc:
        Console(stderr=True, no_color=args.no_color).print(Text('slurmjobs: ' + clean(exc), style='red'))
        return 1


if __name__ == '__main__':
    sys.exit(main())
