"""Offline Run:AI fixtures: never contact a cluster or execute inside Pods."""
import ast
from copy import deepcopy
import io
import math
import subprocess
import unittest
from unittest.mock import Mock, patch

from rich.console import Console
import hpc_dashboard as ui
import hpc_dashboard_runai as r


NS = 'runai-test'
OWNER = 'me@example.org'


def obj(kind, name, uid, **values):
    result = dict(kind=kind, name=name, uid=uid, namespace=NS, annotations={},
                  labels={}, spec_annotations={}, spec_labels={},
                  template_annotations={}, template_labels={}, owners=[], owned=[],
                  containers=[], template_containers=[], init_containers=[],
                  conditions=[], container_statuses=[])
    result.update(values)
    return result


def container(cpu='500m', memory='2Gi', gpu=None):
    requests = {'cpu': cpu, 'memory': memory}
    if gpu is not None:
        requests['nvidia.com/gpu'] = str(gpu)
    return dict(name='main', resources=dict(requests=requests, limits={}), gpu_limit=[])


def fixtures(state='Running'):
    work = obj('TrainingWorkload', 'train', 'w1', phase=state, user=OWNER,
               gpu='0.5', cpu='0.5', memory='2Gi',
               owned=[dict(uid='j1', kind='RunaiJob', name='train')],
               spec_annotations={'purpose': 'test [bold]literal[/bold]'},
               spec_labels={'project': 'my-research'}, start='2026-01-01T00:00:00Z')
    job = obj('RunaiJob', 'train', 'j1', parallelism='1',
              annotations={'runai-calculated-status': state, 'runai-total-requested-gpus': '0.5'},
              template_annotations={'gpu-fraction': '0.5', 'user': OWNER},
              template_containers=[container()])
    pod = obj('Pod', 'train-0-0', 'p1', phase='Running', node='node-a',
              owners=[dict(uid='j1', kind='RunaiJob', name='train')],
              annotations={'gpu-fraction': '0.5'}, containers=[container()],
              start='2026-01-01T00:00:02Z')
    return [work, job, pod]


class QuantityTests(unittest.TestCase):
    def test_cpu_memory_quantities(self):
        for value, expected in [('500m', .5), ('.25', .25), ('2', 2), ('1e-3', .001), ('250000u', .25), ('100000000n', .1)]:
            self.assertAlmostEqual(r.quantity(value, cpu=True), expected)
        for value, expected in [('1Gi', 1024), ('2Mi', 2), ('1G', 1e9 / 1024**2), ('1024', 1/1024), ('0', 0), ('1e6', 1e6/1024**2)]:
            self.assertAlmostEqual(r.quantity(value), expected)
        self.assertIsNone(r.quantity('1e999999'))
        for value in ('bad', '-1', None, 'NaN', 'inf'):
            self.assertIsNone(r.quantity(value))
            self.assertIsNone(r.numeric(value))

    def test_effective_init_request(self):
        p = obj('Pod', 'p', 'p', containers=[container()], init_containers=[container(cpu='2')])
        self.assertEqual(r.effective_resource(p, 'cpu'), 2)
        p['containers'].append(container(cpu='250m'))
        self.assertEqual(r.effective_resource(p, 'cpu'), 2)
        p['containers'][0]['resources']['requests'].pop('cpu')
        self.assertIsNone(r.effective_resource(p, 'cpu'))

    def test_restartable_init_sidecars_overhead_and_pod_level_requests(self):
        p = obj('Pod', 'p', 'p', containers=[container(cpu='1')],
                init_containers=[dict(container(cpu='1'), restart_policy='Always'), container(cpu='3')],
                overhead={'cpu': '100m'})
        self.assertAlmostEqual(r.effective_resource(p, 'cpu'), 4.1)
        p['pod_requests'] = {'cpu': '8'}
        self.assertAlmostEqual(r.effective_resource(p, 'cpu'), 8.1)

    def test_zero_unset_gpu_and_memory_based_fraction(self):
        p = obj('Pod', 'p', 'p', containers=[container()])
        self.assertEqual(r.gpu_request(p, p['containers']), 0)
        p['annotations']['gpu-memory'] = '4096'
        self.assertIsNone(r.gpu_request(p, p['containers']))
        p['annotations']['gpu-fraction'] = '0.25'
        p['annotations']['gpu-fraction-num-devices'] = '2'
        self.assertEqual(r.gpu_request(p, p['containers']), .5)
        p['annotations'] = {}
        p['containers'][0]['resources']['limits']['nvidia.com/mig-1g.5gb'] = '1'
        self.assertIsNone(r.gpu_request(p, p['containers']))


class NormalizeTests(unittest.TestCase):
    def test_deduplicate_three_layers_and_fractional_cpu(self):
        jobs = r.normalize(fixtures(), 'cluster', NS, 'test', {'node-a': 'A100'}, now=1767225610)
        self.assertEqual(len(jobs), 1)
        j = jobs[0]
        self.assertEqual((j.uid, j.user, j.state), ('w1', OWNER, 'RUNNING'))
        self.assertEqual(j.identity, ('runai', 'cluster', NS, 'w1'))
        self.assertEqual((j.cpus, j.memory), (.5, 2048))
        self.assertIn('requested 0.5 · allocated 0.5 · A100', j.gpu)
        self.assertEqual(j.pod_summary, '1 running / 0 pending Pods')
        self.assertEqual(j.comment, 'test [bold]literal[/bold]')
        self.assertEqual(j.tag, 'my-research')
        self.assertEqual(j.elapsed, 10)
        self.assertFalse(j.steps_queried)

    def test_legacy_job_without_modern_workload(self):
        jobs = r.normalize(fixtures()[1:], 'c', NS, 'test')
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].uid, 'j1')
        self.assertEqual(jobs[0].user, OWNER)

    def test_distributed_retry_does_not_inflate_current_resources(self):
        data = fixtures()
        data[0]['kind'] = 'DistributedWorkload'
        data[0]['owned'].append(dict(uid='j2', kind='RunaiJob', name='worker'))
        worker = deepcopy(data[1]); worker.update(uid='j2', name='worker')
        p2 = deepcopy(data[2]); p2.update(uid='p2', name='worker-0-0', node='node-b')
        p2['owners'] = [dict(uid='j2')]
        old = deepcopy(data[2]); old.update(uid='p-old', phase='Failed', node='old-node')
        old['containers'] = [container(cpu='99', memory='99Gi')]
        jobs = r.normalize([*data, worker, p2, old], 'c', NS, 'test')
        self.assertEqual(len(jobs), 1)
        self.assertEqual((jobs[0].cpus, jobs[0].memory), (1, 4096))
        self.assertEqual(jobs[0].nodes, 'node-a, node-b')
        self.assertIn('requested 1 · allocated 1', jobs[0].gpu)
        self.assertIn('2 running / 0 pending Pods · 1 failed attempts', jobs[0].pod_summary)

    def test_top_owner_label_links_pod_without_intermediate_controller(self):
        data = fixtures()
        data[2]['owners'] = [dict(uid='missing-controller')]
        data[2]['labels'] = {'run.ai/top-owner-uid': 'w1'}
        jobs = r.normalize([data[0], data[2]], 'c', NS, 'test')
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].uid, 'w1')

    def test_recreated_name_does_not_attach_old_pod_by_name(self):
        data = fixtures()
        data[2]['owners'] = [dict(uid='old-job')]
        data[2]['labels'] = {'run.ai/top-owner-uid': 'old-workload', 'workloadKind': 'TrainingWorkload', 'workloadName': 'train'}
        for has_top_label in (True, False):
            if not has_top_label:
                data[2]['labels'].pop('run.ai/top-owner-uid')
            jobs = r.normalize([data[0], data[2]], 'c', NS, 'test')
            self.assertEqual(len(jobs), 2)
            self.assertNotEqual(jobs[0].identity, jobs[1].identity)

    def test_terminal_completed_duration_and_failed_attempts(self):
        data = fixtures('Completed')
        data[0]['end'] = '2026-01-01T00:00:08Z'
        data[2]['phase'] = 'Succeeded'
        j = r.normalize(data, 'c', NS, 'test', now=1767225999)[0]
        self.assertEqual(j.state, 'SUCCEEDED')
        self.assertEqual(j.elapsed, 8)
        self.assertIn('allocated 0', j.gpu)
        self.assertEqual(r.state_of(obj('TrainingWorkload', 'x', 'x', phase='Failed'))[0], 'FAILED')

    def test_pending_waiting_reason_and_unknown_limit(self):
        data = fixtures('Pending')
        data[2].update(phase='Pending', node=None, conditions=[dict(type='PodScheduled', status='False', message='Insufficient nvidia.com/gpu')])
        j = r.normalize(data, 'c', NS, 'test')[0]
        self.assertEqual((j.elapsed, j.limit, j.nodes), (0, None, ''))
        self.assertEqual(j.reason, 'Insufficient nvidia.com/gpu')
        self.assertIn('allocated 0', j.gpu)

    def test_crashloop_reason_and_burst_memory_limit(self):
        data = fixtures()
        data[2]['container_statuses'] = [dict(waiting='CrashLoopBackOff')]
        data[2]['containers'][0]['gpu_limit'] = ['0.8']
        j = r.normalize(data, 'c', NS, 'test')[0]
        self.assertEqual(j.reason, 'CrashLoopBackOff')
        self.assertIn('GPU memory limit 0.8', j.gpu)

    def test_completed_workload_after_pod_and_controller_cleanup(self):
        w = fixtures('Completed')[0]
        w['end'] = '2026-01-01T00:00:10Z'
        j = r.normalize([w], 'c', NS, 'test')[0]
        self.assertEqual((j.cpus, j.memory), (.5, 2048))
        self.assertEqual(j.elapsed, 10)

    def test_unknown_timing_and_resources_not_zero(self):
        j = r.normalize([obj('TrainingWorkload', 'x', 'x', phase='Completed')], 'c', NS, 'test')[0]
        self.assertIsNone(j.elapsed)
        self.assertIsNone(j.cpus)
        self.assertIn('CPU request unknown', j.resource_details)
        self.assertEqual(r.state_of(obj('RunaiJob', 'x', 'x', suspend='true'))[0], 'SUSPENDED')

    def test_owner_cycle_is_bounded(self):
        data = fixtures()
        data[0]['owners'] = [dict(uid='j1')]
        self.assertTrue(r.normalize(data, 'c', NS, 'test'))

    def test_render_sanitizes_metadata_and_has_no_slurm_steps(self):
        data = fixtures()
        data[0]['spec_annotations']['purpose'] = 'literal [red]hello[/red]\x1b\nnext'
        j = r.normalize(data, 'c', NS, 'test')[0]
        for width in (40, 100):
            lines = list(ui.render([j], OWNER, width=width).renderables)
            self.assertTrue(all(line.cell_len <= width for line in lines))
            text = ''.join(line.plain.strip() for line in lines)
            self.assertIn('[red]hello[/red]', text)
            self.assertNotIn('\x1b', text)
            self.assertNotIn('running steps', text)
            self.assertIn('Purpose', text)
            self.assertIn('my-research', text)


class QueueTests(unittest.TestCase):
    def queue(self, **kwargs):
        queue = r.RunaiQueue(projects=['test'], user=OWNER, **kwargs)
        queue.initialized = True
        queue.context = 'cluster'
        queue.resources = {'trainingworkloads', 'runaijobs'}
        queue.models_at = math.inf
        return queue

    def queries(self, queue, data):
        def objects(types, ns):
            if types == ['pods']:
                return [o for o in deepcopy(data) if o['kind'] == 'Pod']
            if types == ['runaijobs']:
                return [o for o in deepcopy(data) if o['kind'] == 'RunaiJob']
            return [o for o in deepcopy(data) if o['kind'] not in ('Pod', 'RunaiJob')]
        queue.objects = Mock(side_effect=objects)

    def test_default_excludes_finished_and_state_filter_can_include_terminal(self):
        data = fixtures('Completed')
        for flags, expected in [({}, 0), ({'include_finished': True}, 1), ({'states': ['succeeded']}, 1), ({'include_finished': True, 'states': ['running']}, 0)]:
            q = self.queue(**flags); self.queries(q, data)
            self.assertEqual(len(q.fetch()), expected)
        for terminal in ('Failed', 'Cancelled', 'Stopped', 'Deleted'):
            q = self.queue(); self.queries(q, fixtures(terminal))
            self.assertEqual(q.fetch(), [])

    def test_me_uses_sso_identity_not_unix_login(self):
        q = self.queue(); self.queries(q, fixtures())
        self.assertEqual(len(q.fetch('local-unix-name')), 1)
        q.user = 'another@example.org'
        self.assertEqual(q.fetch('local-unix-name'), [])

    def test_multiple_namespaces_keep_distinct_identity(self):
        q = self.queue(); q.projects = ['a', 'b']; self.queries(q, fixtures())
        jobs = q.fetch()
        self.assertEqual(len(jobs), 2)
        self.assertNotEqual(jobs[0].identity, jobs[1].identity)
        self.assertNotEqual(jobs[0].id, jobs[1].id)

    def test_partial_failure_retains_previous_graph_and_warns(self):
        q = self.queue(); self.queries(q, fixtures())
        original = q.fetch()
        q.objects.side_effect = [RuntimeError('permission denied'), fixtures()[1:2], fixtures()[2:]]
        recovered = q.fetch()
        self.assertEqual([j.uid for j in recovered], [j.uid for j in original])
        self.assertTrue(any('retaining previous' in w for w in q.warnings))

    def test_total_failure_is_not_empty_queue(self):
        q = self.queue(); q.objects = Mock(side_effect=RuntimeError('offline'))
        with self.assertRaisesRegex(RuntimeError, 'offline'):
            q.fetch()

    def test_failed_namespace_does_not_hide_other_namespace(self):
        q = self.queue(); q.projects = ['a', 'b']
        self.queries(q, fixtures())
        good = q.objects.side_effect
        q.objects.side_effect = lambda types, ns: good(types, ns) if ns == 'runai-a' else (_ for _ in ()).throw(RuntimeError('forbidden'))
        self.assertEqual(len(q.fetch()), 1)
        self.assertTrue(q.warnings)

    @patch('hpc_dashboard_runai.shutil.which', return_value='/bin/runai')
    def test_project_discovery_only_accessible_projects_not_kube_project_catalog(self, which):
        q = self.queue(); q.projects = []
        q.run = Mock(return_value='PROJECT DEPARTMENT DESERVED GPUs MANAGED NAMESPACE\na (default) dept 3 runai-a\nb dept 3 runai-b\n')
        q.discover()
        self.assertEqual(q.scopes, [('a', 'runai-a'), ('b', 'runai-b')])
        q.discover(); self.assertEqual(q.run.call_count, 1)

    def test_project_rediscovery_rejects_changed_global_context(self):
        q = self.queue(); q.projects = []; q.scopes = [('test', NS)]
        q.discovered_at = -math.inf
        q.run = Mock(return_value='different-cluster\n')
        with self.assertRaisesRegex(RuntimeError, 'context changed'):
            q.discover()
        q.run.assert_called_once_with(['kubectl', 'config', 'current-context'])

    def test_namespace_validation_and_scope_dedup(self):
        q = self.queue(); q.namespaces = ['runai-test']
        q.discover(); self.assertEqual(q.scopes, [('test', NS)])
        q = self.queue(); q.projects = ['../../bad']
        with self.assertRaisesRegex(RuntimeError, 'Invalid'):
            q.discover()

    def test_context_pinned_and_identity_determined_once(self):
        q = r.RunaiQueue(projects=['test'])
        q.run = Mock(return_value='cluster\n')
        q.kubectl = Mock(side_effect=[OWNER, 'runaijobs.run.ai\ntrainingworkloads.run.ai\n'])
        q.initialize(); q.initialize()
        self.assertEqual(q.context, 'cluster')
        self.assertEqual(q.user, OWNER)
        self.assertEqual(q.run.call_count, 1)
        q.kubectl.assert_any_call(['auth', 'whoami', '-o', 'jsonpath={.status.userInfo.username}'])

    def test_initialization_retry_does_not_turn_default_context_explicit(self):
        q = r.RunaiQueue(user=OWNER)
        q.run = Mock(return_value='cluster\n')
        q.kubectl = Mock(side_effect=[RuntimeError('temporary error'), 'runaijobs.run.ai'])
        with self.assertRaisesRegex(RuntimeError, 'temporary'):
            q.initialize()
        q.initialize()
        self.assertTrue(q.initialized)
        self.assertFalse(q.explicit_context)

    def test_explicit_context_requires_scopes(self):
        q = r.RunaiQueue(context='other', user=OWNER)
        with self.assertRaisesRegex(RuntimeError, '--context requires'):
            q.initialize()

    @patch('hpc_dashboard_runai.subprocess.run')
    def test_only_read_commands_and_bounded_errors_no_secret_stderr(self, run):
        q = self.queue(timeout=2)
        run.return_value = subprocess.CompletedProcess([], 1, '', 'Forbidden: sensitive-stderr-not-for-display')
        with self.assertRaisesRegex(RuntimeError, 'permission denied') as error:
            q.kubectl(['get', 'pods', '-n', NS])
        self.assertNotIn('sensitive', str(error.exception))
        self.assertEqual(run.call_args.args[0][:4], ['kubectl', '--request-timeout=2s', '--context', 'cluster'])
        self.assertEqual(run.call_args.kwargs['timeout'], 2)
        run.side_effect = subprocess.TimeoutExpired('kubectl', 2)
        with self.assertRaisesRegex(RuntimeError, 'within 2 seconds'):
            q.kubectl(['get', 'pods'])
        run.side_effect = FileNotFoundError()
        with self.assertRaisesRegex(RuntimeError, 'not installed'):
            q.kubectl(['get', 'pods'])

    def test_projection_parser_never_eval_and_malformed_response(self):
        q = self.queue()
        q.kubectl = Mock(return_value='[{"name":"line\\x1b\\v\\a", "uid":"u",},]')
        parsed = q.objects(['pods'], NS)
        self.assertEqual(parsed[0]['name'], 'line\x1b\v\a')
        self.assertIn('go-template=', q.kubectl.call_args.args[0][-1])
        for value in ('{}', '__import__("os").system("false")', '[1]', 'broken'):
            q.kubectl.return_value = value
            with self.assertRaisesRegex(RuntimeError, 'Invalid projected'):
                q.objects(['pods'], NS)

    def test_projection_allowlists_no_arbitrary_secrets_commands_or_env(self):
        self.assertNotIn('runai-cli-command', r.PROJECTION)
        self.assertNotIn('last-applied-configuration', r.PROJECTION)
        self.assertNotIn('.spec.volumes', r.PROJECTION)
        self.assertNotIn('.command', r.PROJECTION)
        self.assertIn('eq .name "RUNAI_GPU_MEMORY_LIMIT"', r.PROJECTION)
        self.assertNotIn('.envFrom', r.PROJECTION)
        self.assertIn('gpu-fraction', r.PROJECTION)

    @patch('hpc_dashboard.time.monotonic', return_value=100)
    def test_stale_namespace_snapshot_has_age_and_does_not_advance_elapsed(self, clock):
        q = self.queue(); self.queries(q, fixtures())
        cache = ui.QueueCache(q)
        original = cache.fetch()[0]
        clock.return_value = 115
        q.objects.side_effect = RuntimeError('offline')
        retained = cache.fetch()[0]
        self.assertEqual(retained.elapsed, original.elapsed)
        self.assertIn('15s old', cache.error)
        clock.return_value = 117
        self.assertEqual(cache.fetch()[0].elapsed, original.elapsed)

    @patch('hpc_dashboard.time.monotonic', return_value=100)
    def test_queue_cache_no_steps_queries_and_preserves_warnings(self, clock):
        q = self.queue(); self.queries(q, fixtures())
        q.fetch_steps = Mock(side_effect=AssertionError('must not call Slurm steps'))
        cache = ui.QueueCache(q)
        jobs = cache.fetch()
        self.assertFalse(jobs[0].steps_queried)
        q.fetch_steps.assert_not_called()
        clock.return_value = 105
        self.assertEqual(cache.fetch()[0].elapsed, jobs[0].elapsed + 5)


class CLITests(unittest.TestCase):
    @patch('hpc_dashboard.shutil.which')
    def test_auto_detection_and_ambiguity(self, which):
        for installed, expected in [({'squeue'}, 'slurm'), ({'kubectl'}, 'runai')]:
            which.side_effect = lambda cmd: '/bin/' + cmd if cmd in installed else None
            self.assertEqual(ui.choose_backend('auto'), expected)
        which.side_effect = lambda cmd: '/bin/' + cmd
        with self.assertRaisesRegex(RuntimeError, 'Both'):
            ui.choose_backend('auto')
        self.assertEqual(ui.choose_backend('runai'), 'runai')
        which.side_effect = lambda cmd: None
        with self.assertRaisesRegex(RuntimeError, 'No scheduler'):
            ui.choose_backend('auto')

    @patch('hpc_dashboard_runai.RunaiQueue')
    @patch('hpc_dashboard_telemetry.Telemetry')
    def test_runai_cli_never_launches_slurm_telemetry(self, telemetry, queue):
        q = queue.return_value
        q.fetch.return_value = r.normalize(fixtures(), 'cluster', NS, 'test')
        q.supports_steps = False; q.warnings = []; q.user = OWNER
        q.scopes = [('test', NS)]; q.context = 'cluster'
        for flags in ([], ['--usage'], ['--no-usage']):
            with patch.object(ui.sys, 'argv', ['hpcd', '--backend', 'runai', '--no-watch', '--no-color', '--project', 'test', '--me', *flags]), patch('sys.stdout', new=io.StringIO()) as output:
                self.assertEqual(ui.main(), 0)
                self.assertIn('MY JOBS', output.getvalue())
                self.assertIn(OWNER, output.getvalue())
            telemetry.assert_not_called()


if __name__ == '__main__':
    unittest.main()
