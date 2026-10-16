"""Read-only Run:AI adapter using projected Kubernetes objects, never Pod exec.

Only allowlisted fields leave kubectl: no commands, secret references, arbitrary
annotations, or container environments. Go's %q quotes are parsed as literals
(not evaluated). This supports control-character escapes that are not JSON.
"""
import ast
from collections import Counter, defaultdict
from datetime import datetime
import math
import re
import shutil
import subprocess
import time

from hpc_dashboard_model import Job, clean

MODERN = ('trainingworkloads', 'interactiveworkloads', 'distributedworkloads',
          'inferenceworkloads', 'distributedinferenceworkloads')
TERMINAL = {'SUCCEEDED', 'FAILED', 'CANCELLED', 'DELETED', 'STOPPED'}
NAMESPACE = re.compile(r'[a-z0-9](?:[-a-z0-9]*[a-z0-9])?\Z')
ANNOTATIONS = ('purpose', 'comment', 'user', 'gpu-fraction', 'gpu-memory',
               'gpu-fraction-num-devices', 'runai-nodepools',
               'runai-calculated-status', 'runai-total-requested-gpus',
               'runai-current-requested-gpus', 'runai-current-allocated-gpus')
LABELS = ('project', 'workloadName', 'workloadKind', 'run.ai/top-owner-uid')


def quoted(expression):
    return '{{template "q" ' + expression + '}}'


def fields(expressions):
    return '{' + ''.join(repr(k) + ':' + quoted(v) + ',' for k, v in expressions.items()) + '}'


def metadata_map(expression, keys):
    allowed = 'or ' + ' '.join('(eq $k "' + k + '")' for k in keys)
    return ('{ ' + '{{range $k,$v := ' + expression + '}}{{if ' + allowed + '}}'
            '{{printf "%q" $k}}:{{template "q" $v}},{{end}}{{end}}' + '}')


def owner_list(expression, modern=False):
    return ('[{{range ' + expression + '}}' + fields({
        'uid': '.uid', 'name': '.name', 'kind': '.kind.kind' if modern else '.kind',
    }) + ',{{end}}]')


def containers(expression):
    resources = '{'
    for typ in ('requests', 'limits'):
        resources += repr(typ) + ':{ ' + '{{range $k,$v := .resources.' + typ + '}}'
        resources += '{{printf "%q" $k}}:{{template "q" $v}},{{end}}},'
    resources += '}'
    return ('[{{range ' + expression + '}}{' + "'name':" + quoted('.name') +
            ",'restart_policy':" + quoted('.restartPolicy') +
            ",'resources':" + resources + ",'gpu_limit':" +
            '[{{range .env}}{{if eq .name "RUNAI_GPU_MEMORY_LIMIT"}}' + quoted('.value') +
            ',{{end}}{{end}}]},{{end}}]')


def projection():
    """A narrow output projection, shared by CRDs, RunaiJobs and Pods."""
    result = '{{define "q"}}{{if eq . nil}}None{{else}}{{printf "%q" (printf "%v" .)}}{{end}}{{end}}'
    result += '[{{range .items}}' + fields({
        'kind': '.kind', 'name': '.metadata.name', 'uid': '.metadata.uid',
        'namespace': '.metadata.namespace', 'created': '.metadata.creationTimestamp',
        'phase': '.status.phase', 'reason': '.status.reason', 'phase_message': '.status.phaseMessage', 'start': '.status.startTime',
        'end': '.status.completionTime', 'node': '.spec.nodeName',
        'deadline': '.spec.activeDeadlineSeconds', 'parallelism': '.spec.parallelism',
        'suspend': '.spec.suspend', 'deleting': '.metadata.deletionTimestamp',
        'gpu': '.spec.gpu.value', 'gpu_devices': '.spec.gpuDevices.value',
        'cpu': '.spec.cpu.value', 'memory': '.spec.memory.value', 'user': '.spec.username.value',
        'allocated_gpu': '.status.totalAllocatedGPU',
        'running': '.status.runningPods', 'pending': '.status.pendingPods',
    })[:-1]
    result += "'annotations':" + metadata_map('.metadata.annotations', ANNOTATIONS) + ','
    result += "'labels':" + metadata_map('.metadata.labels', LABELS) + ','
    result += "'template_annotations':" + metadata_map('.spec.template.metadata.annotations', ANNOTATIONS) + ','
    result += "'template_labels':" + metadata_map('.spec.template.metadata.labels', LABELS) + ','
    for category, keys in (('annotations', ANNOTATIONS), ('labels', LABELS)):
        # Modern workload assets wrap values in .items.<key>.value.
        result += repr('spec_' + category) + ':{ '
        result += '{{range $k,$v := .spec.' + category + '.items}}{{if or '
        result += ' '.join('(eq $k "' + k + '")' for k in keys)
        result += '}}{{if not $v.deleted}}{{printf "%q" $k}}:{{template "q" $v.value}},{{end}}{{end}}{{end}}},'
    result += "'owners':" + owner_list('.metadata.ownerReferences') + ','
    result += "'owned':" + owner_list('.status.ownedResources', modern=True) + ','
    result += "'containers':" + containers('.spec.containers') + ','
    result += "'init_containers':" + containers('.spec.initContainers') + ','
    result += "'overhead':{ {{range $k,$v := .spec.overhead}}{{printf \"%q\" $k}}:{{template \"q\" $v}},{{end}}},'pod_requests':{ {{range $k,$v := .spec.resources.requests}}{{printf \"%q\" $k}}:{{template \"q\" $v}},{{end}}},"
    result += "'template_containers':" + containers('.spec.template.spec.containers') + ','
    result += "'conditions':[{{range .status.conditions}}" + fields({
        'type': '.type', 'status': '.status', 'reason': '.reason', 'message': '.message',
    }) + ',{{end}}],'
    result += "'container_statuses':[{{range .status.containerStatuses}}" + fields({
        'waiting': '.state.waiting.reason', 'terminated': '.state.terminated.reason',
        'finished': '.state.terminated.finishedAt', 'restarts': '.restartCount',
    }) + ',{{end}}]},{{end}}]'
    return result


PROJECTION = projection()


def scalar(value):
    if value in (None, '<nil>', '<no value>', ''):
        return None
    return value


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def quantity(value, cpu=False):
    """Kubernetes quantities: CPU cores or memory MiB (decimal and binary SI)."""
    match = re.fullmatch(r'([+]?\d+(?:\.\d*)?|[+]?\.\d+)([eE][+-]?\d+|[EPTGMKk]i?|[mun])?', str(value))
    if not match:
        return None
    amount = float(match[1])
    suffix = match[2] or ''
    if re.fullmatch(r'[eE][+-]?\d+', suffix):
        result = float(str(amount) + suffix.lower()) / (1 if cpu else 1024**2)
        return result if math.isfinite(result) else None
    elif suffix in ('m', 'u', 'n'):
        factor = {'m': 1e-3, 'u': 1e-6, 'n': 1e-9}[suffix]
    elif suffix:
        power = 'KMGTPE'.index(suffix[0].upper()) + 1
        factor = (1024 if suffix.endswith('i') else 1000) ** power
    else:
        factor = 1
    result = amount * factor / (1 if cpu else 1024**2)
    return result if math.isfinite(result) else None


def timestamp(value):
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def maps(obj):
    annotations, labels = {}, {}
    for key in ('annotations', 'template_annotations', 'spec_annotations'):
        annotations.update({k: scalar(v) for k, v in obj.get(key, {}).items()})
    for key in ('labels', 'template_labels', 'spec_labels'):
        labels.update({k: scalar(v) for k, v in obj.get(key, {}).items()})
    return annotations, labels


def state_of(obj):
    annotations, _ = maps(obj)
    raw = scalar(obj.get('phase')) or annotations.get('runai-calculated-status')
    if not raw:
        for condition in obj.get('conditions', []):
            if condition.get('status') == 'True' and condition.get('type') in ('Complete', 'Failed'):
                raw = condition['type']
    raw = raw or 'Unknown'
    state = str(raw).upper().replace('-', '').replace('_', '')
    state = {'COMPLETED': 'SUCCEEDED', 'COMPLETE': 'SUCCEEDED', 'SUCCESS': 'SUCCEEDED',
             'QUEUED': 'PENDING', 'CREATING': 'PENDING', 'INITIALIZING': 'PENDING',
             'CANCELED': 'CANCELLED'}.get(state, state)
    if state not in TERMINAL:
        if scalar(obj.get('deleting')):
            state = 'TERMINATING'
        elif obj.get('suspend') in ('true', True):
            state = 'SUSPENDED'
    return state, str(raw)


def resource_request(containers_, key):
    values = []
    for container in containers_:
        resources = container.get('resources', {})
        value = resources.get('requests', {}).get(key)
        if value is None:
            value = resources.get('limits', {}).get(key)
        values.append(quantity(value, cpu=key == 'cpu') if key in ('cpu', 'memory') else numeric(value))
    if not values or any(v is None for v in values):
        return None
    return sum(values)


def gpu_request(obj, containers_):
    a, _ = maps(obj)
    for key in ('runai-total-requested-gpus', 'runai-current-requested-gpus', 'gpu-fraction'):
        value = numeric(a.get(key))
        if value is not None:
            if key == 'gpu-fraction':
                value *= numeric(a.get('gpu-fraction-num-devices')) or 1
            return value
    value = numeric(obj.get('gpu'))
    if value is not None:
        return value * (numeric(obj.get('gpu_devices')) or 1) if value <= 1 else value
    value = resource_request(containers_, 'nvidia.com/gpu')
    if value is not None:
        return value
    if a.get('gpu-memory') is not None:
        return None  # Memory-based fraction needs a verified GPU model/capacity.
    # Absence of a GPU request is zero only for actual Pod/container specs.
    if containers_ and not any(any('gpu' in k or '/mig-' in k for k in c.get('resources', {}).get('limits', {})) for c in containers_):
        return 0.0
    return None


def effective_resource(obj, key):
    if key in obj.get('pod_requests', {}):
        value = quantity(obj['pod_requests'][key], cpu=key == 'cpu')
    else:
        regular = resource_request(obj.get('containers', []), key)
        if regular is None:
            return None
        sidecars, peak = 0, 0
        for container in obj.get('init_containers', []):
            request = resource_request([container], key)
            if request is None:
                return None
            if container.get('restart_policy') == 'Always':
                sidecars += request
                peak = max(peak, sidecars)
            else:
                peak = max(peak, sidecars + request)
        value = max(regular + sidecars, peak)
    overhead = quantity(obj.get('overhead', {}).get(key, '0'), cpu=key == 'cpu')
    return value + overhead if value is not None and overhead is not None else None


def total(values):
    values = list(values)
    return sum(values) if values and all(v is not None for v in values) else None


def amount(value):
    return 'unknown' if value is None else f'{value:g}'


def ram(value):
    if value is None:
        return 'unknown'
    return f'{value / 1024:g} GiB' if value >= 1024 else f'{value:g} MiB'


def normalize(objects, context, namespace, project, models=None, now=None):
    """Aggregate by owner UID; old retry Pods never inflate live resources."""
    now = time.time() if now is None else now
    models = models or {}
    by_uid = {o['uid']: o for o in objects if scalar(o.get('uid'))}
    # Modern workloads list descendants even when controller ownerRefs do not.
    parents = {}
    for obj in objects:
        for ref in obj.get('owned', []):
            if scalar(ref.get('uid')):
                parents[ref['uid']] = obj.get('uid')
        for ref in obj.get('owners', []):
            if ref.get('uid') in by_uid:
                parents.setdefault(obj.get('uid'), ref['uid'])
    modern = {o['uid']: o for o in objects if o.get('kind') not in ('Pod', 'RunaiJob') and scalar(o.get('uid'))}
    for obj in objects:
        _, labels = maps(obj)
        top = labels.get('run.ai/top-owner-uid')
        target = top if top in modern else None
        if target and target != obj.get('uid'):
            parents.setdefault(obj['uid'], target)

    def root(uid):
        seen = set()
        while uid in parents and parents[uid] in by_uid and uid not in seen:
            seen.add(uid)
            uid = parents[uid]
        return uid

    groups = defaultdict(list)
    for obj in objects:
        if not scalar(obj.get('uid')) or not scalar(obj.get('name')):
            continue
        groups[root(obj['uid'])].append(obj)
    jobs = []
    for uid, members in groups.items():
        top = by_uid[uid]
        pods = [o for o in members if o.get('kind') == 'Pod']
        children = [o for o in members if o.get('kind') == 'RunaiJob']
        state, raw_state = state_of(top)
        if top.get('kind') == 'Pod':
            state, raw_state = state_of(top)
        annotations, labels = {}, {}
        for obj in [*pods, *children, top]:
            a, l = maps(obj)
            annotations.update(a)
            labels.update(l)
        owner = scalar(top.get('user')) or annotations.get('user') or 'unknown'
        live = [p for p in pods if state_of(p)[0] not in TERMINAL and not scalar(p.get('deleting'))]
        running = [p for p in live if state_of(p)[0] == 'RUNNING']
        pending = [p for p in live if state_of(p)[0] == 'PENDING']
        counts = Counter(state_of(p)[0] for p in pods)
        summary = f'{len(running)} running / {len(pending)} pending Pods'
        if counts['FAILED']:
            summary += f' · {counts["FAILED"]} failed attempts'
        if not pods:
            summary = f'{int(numeric(top.get("running")) or 0)} running / {int(numeric(top.get("pending")) or 0)} pending Pods'
        containers_ = top.get('template_containers', [])
        replicas = numeric(top.get('parallelism')) or 1
        requested = gpu_request(top, containers_)
        if children and top.get('kind') != 'RunaiJob':
            child_gpu = total(gpu_request(c, c.get('template_containers', [])) for c in children)
            if child_gpu is not None:
                requested = child_gpu
        elif top.get('kind') == 'RunaiJob' and 'runai-total-requested-gpus' not in maps(top)[0]:
            requested = requested * replicas if requested is not None else None
        if top.get('kind') == 'Pod':
            requested = gpu_request(top, top.get('containers', []))
        allocated = numeric(top.get('allocated_gpu'))
        if allocated is None:
            allocated = numeric(maps(top)[0].get('runai-current-allocated-gpus'))
        if allocated is None:
            allocated = 0.0 if state in TERMINAL or state in ('PENDING', 'SUSPENDED') else total(gpu_request(p, p.get('containers', [])) for p in running)
        cpu = quantity(top.get('cpu'), cpu=True)
        memory = quantity(top.get('memory'))
        if cpu is None and containers_:
            cpu = resource_request(containers_, 'cpu')
            cpu = cpu * replicas if cpu is not None else None
        if memory is None and containers_:
            memory = resource_request(containers_, 'memory')
            memory = memory * replicas if memory is not None else None
        if live:
            # Live container requests, not measured CPU/RAM utilization.
            cpu = total(effective_resource(p, 'cpu') for p in live)
            memory = total(effective_resource(p, 'memory') for p in live)
        elif children and not containers_:
            cpu = total(resource_request(c.get('template_containers', []), 'cpu') * (numeric(c.get('parallelism')) or 1)
                        if resource_request(c.get('template_containers', []), 'cpu') is not None else None for c in children)
            memory = total(resource_request(c.get('template_containers', []), 'memory') * (numeric(c.get('parallelism')) or 1)
                           if resource_request(c.get('template_containers', []), 'memory') is not None else None for c in children)
        # Assigned model only: never claim a pending node selector was allocated.
        nodes = sorted({p['node'] for p in (live or pods) if scalar(p.get('node'))})
        types = sorted({models[n] for n in nodes if n in models})
        gpu = 'no GPU' if requested == 0 else f'GPU requested {amount(requested)} · allocated {amount(allocated)}'
        if types and requested != 0:
            gpu += ' · ' + ', '.join(types)
        if annotations.get('gpu-memory'):
            gpu += f' · memory request {annotations["gpu-memory"]} MiB/device'
        limits = sorted({str(v) for p in (live or pods) for c in p.get('containers', []) for v in c.get('gpu_limit', []) if scalar(v)})
        if limits:
            gpu += ' · GPU memory limit ' + ', '.join(limits)
        starts = [timestamp(o.get('start')) for o in [top, *children, *pods]]
        starts = [v for v in starts if v is not None]
        ends = [timestamp(o.get('end')) for o in [top, *children]]
        ends += [timestamp(c.get('finished')) for p in pods for c in p.get('container_statuses', [])]
        ends = [v for v in ends if v is not None]
        elapsed = 0.0 if state == 'PENDING' else None
        if starts and state != 'PENDING':
            finish = max(ends) if state in TERMINAL and ends else now if state not in TERMINAL else None
            elapsed = max(0, finish - min(starts)) if finish is not None else None
        reason = scalar(top.get('reason')) or scalar(top.get('phase_message')) or ''
        for condition in top.get('conditions', []):
            if condition.get('status') == 'True' and condition.get('type') in ('Failed', 'Pending', 'Degraded'):
                reason = scalar(condition.get('message')) or scalar(condition.get('reason')) or reason
        for p in live:
            for c in p.get('container_statuses', []):
                if scalar(c.get('waiting')):
                    reason = c['waiting']
            for c in p.get('conditions', []):
                if c.get('type') == 'PodScheduled' and c.get('status') == 'False':
                    reason = c.get('message') or c.get('reason') or reason
        deadline = numeric(top.get('deadline'))
        jobs.append(Job(
            id=f'{namespace}/{top["name"]}', name=clean(top['name']), user=clean(owner),
            state=state, partition=clean(project), nodes=clean(', '.join(nodes)),
            cpus=cpu, memory=memory, gpu=clean(gpu), elapsed=elapsed, limit=deadline,
            comment=clean(annotations.get('purpose') or annotations.get('comment')),
            reason=clean(reason), backend='runai', context=context, namespace=namespace,
            uid=uid, pod_summary=summary,
            tag=clean(labels.get('project') if labels.get('project') != project else ''),
            resource_details=f'{gpu} · CPU request {amount(cpu)} cores · RAM request {ram(memory)}',
            raw_state=clean(raw_state),
        ))
    return jobs


class RunaiQueue:
    supports_steps = False

    def __init__(self, timeout=10, projects=None, namespaces=None, context=None,
                 include_finished=False, states=None, user=None):
        self.timeout = timeout
        self.projects = projects or []
        self.namespaces = namespaces or []
        self.context = context
        self.explicit_context = context is not None
        self.include_finished = include_finished
        self.states = {s.upper() for s in states or []}
        self.user = user
        self.warnings = []
        self.scopes = None
        self.resources = None
        self.models = {}
        self.models_at = -math.inf
        self.discovered_at = -math.inf
        self.namespace_jobs = {}
        self.namespace_fetched = {}
        self.initialized = False

    def run(self, args):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=self.timeout, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(f'{args[0]} is not installed or not on PATH') from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'{args[0]} did not respond within {self.timeout:g} seconds') from exc
        if result.returncode:
            # kubectl errors may quote request details; do not print arbitrary stderr.
            error = result.stderr.lower()
            reason = 'permission denied' if 'forbidden' in error else 'authentication required' if any(s in error for s in ('unauthorized', 'logged in', 'credentials')) else 'resource unavailable' if any(s in error for s in ('not found', 'resource type')) else 'invalid output projection' if 'template' in error else 'request failed'
            operation = next((a for a in args[1:] if a in ('get', 'auth', 'api-resources', 'list', 'config')), 'query')
            raise RuntimeError(f'{args[0]} {operation}: {reason}')
        return result.stdout

    def kubectl(self, args):
        prefix = ['kubectl', f'--request-timeout={self.timeout:g}s']
        if self.context:
            prefix += ['--context', self.context]
        return self.run(prefix + args)

    def initialize(self):
        if self.initialized:
            return
        if not self.context:
            self.context = self.run(['kubectl', 'config', 'current-context']).strip()
            if not self.context:
                raise RuntimeError('No Kubernetes context; configure access or use --context')
        if not self.user:
            self.user = self.kubectl(['auth', 'whoami', '-o', 'jsonpath={.status.userInfo.username}']).strip()
            if not self.user:
                raise RuntimeError('Cannot determine Kubernetes identity; use --user EMAIL')
        if self.explicit_context and not self.projects and not self.namespaces:
            raise RuntimeError('--context requires --project or --namespace (avoids cross-cluster Run:AI discovery)')
        available = self.kubectl(['api-resources', '--api-group=run.ai', '-o', 'name'])
        self.resources = {s.split('.')[0] for s in available.split()}
        self.initialized = True

    def discover(self):
        if self.scopes is not None and time.monotonic() - self.discovered_at < 300:
            return
        if self.projects or self.namespaces:
            scopes = [(p, 'runai-' + p) for p in self.projects]
            scopes += [(n.removeprefix('runai-'), n) for n in self.namespaces]
        else:
            if self.scopes is not None and self.run(['kubectl', 'config', 'current-context']).strip() != self.context:
                raise RuntimeError('Current Kubernetes context changed; restart the dashboard before project rediscovery')
            if not shutil.which('runai'):
                raise RuntimeError('Project discovery needs runai; specify --project or --namespace')
            output = self.run(['runai', '--suppress-deprecation-message', 'list', 'projects'])
            scopes = []
            for line in output.splitlines():
                cols = line.split()
                if len(cols) >= 3 and NAMESPACE.fullmatch(cols[0]) and cols[-1].startswith('runai-') and NAMESPACE.fullmatch(cols[-1]):
                    scopes.append((cols[0], cols[-1]))
            if not scopes:
                raise RuntimeError('No accessible Run:AI projects discovered; specify --project or --namespace')
        if any(not NAMESPACE.fullmatch(n) or len(n) > 63 for _, n in scopes):
            raise RuntimeError('Invalid project/namespace name')
        self.scopes = list(dict.fromkeys(scopes))
        self.discovered_at = time.monotonic()

    def objects(self, resources, namespace):
        if not resources:
            return []
        output = self.kubectl(['get', ','.join(resources), '-n', namespace, '-o', 'go-template=' + PROJECTION])
        try:
            values = ast.literal_eval(output)
            if not isinstance(values, list) or not all(isinstance(o, dict) for o in values):
                raise ValueError('not an object list')
            return values
        except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
            raise RuntimeError('Invalid projected Kubernetes response') from exc

    def fetch(self, me=None):
        self.warnings = []
        self.initialize()
        self.discover()
        if time.monotonic() - self.models_at >= 300:
            try:
                output = self.kubectl(['get', 'nodes', '-o', 'go-template={{range .items}}{{.metadata.name}}{{"\\t"}}{{index .metadata.labels "nvidia.com/gpu.product"}}{{"\\n"}}{{end}}'])
                self.models = {parts[0]: parts[1] for line in output.splitlines() if len(parts := line.split('\t')) == 2 and scalar(parts[1])}
            except RuntimeError:
                self.warnings.append('GPU models unavailable (node metadata not accessible)')
            self.models_at = time.monotonic()
        result, failures = [], []
        for project, namespace in self.scopes:
            objects, successful = [], 0
            groups = [('workloads', [r for r in MODERN if r in self.resources]),
                      ('jobs', ['runaijobs'] if 'runaijobs' in self.resources else []),
                      ('pods', ['pods'])]
            failed = False
            for label, resources in groups:
                if not resources:
                    continue
                try:
                    objects.extend(self.objects(resources, namespace))
                    successful += 1
                except RuntimeError as exc:
                    failed = True
                    failures.append(f'{namespace} {label}: {exc}')
            if successful:
                jobs = normalize(objects, self.context, namespace, project, self.models)
                if failed and namespace in self.namespace_jobs:
                    # Never turn a partially inaccessible owner graph into duplicate rows.
                    jobs = self.namespace_jobs[namespace]
                    age = max(0, time.monotonic() - self.namespace_fetched[namespace])
                    self.warnings.append(f'{namespace}: retaining previous snapshot ({age:.0f}s old) after partial query failure')
                elif not failed:
                    self.namespace_jobs[namespace] = jobs
                    self.namespace_fetched[namespace] = time.monotonic()
                result.extend(jobs)
            elif namespace in self.namespace_jobs:
                result.extend(self.namespace_jobs[namespace])
                age = max(0, time.monotonic() - self.namespace_fetched[namespace])
                self.warnings.append(f'{namespace}: retaining previous snapshot ({age:.0f}s old) after query failure')
        if failures and not result and not self.namespace_jobs:
            raise RuntimeError('; '.join(failures))
        self.warnings.extend(failures)
        return [j for j in result if (not me or j.user == self.user) and
                (j.state in self.states if self.states else self.include_finished or j.state not in TERMINAL)]
