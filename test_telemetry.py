"""Unit tests only: all Slurm commands and process launches are mocked."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import telemetry as t


def job(**values):
    defaults = dict(id='123', user='me', state='RUNNING', nodes='node[01-02]',
                    cpus=8, memory=8192, gpu='2 × GPU')
    defaults.update(values)
    return SimpleNamespace(**defaults)


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = patch.object(t, 'cache_root', return_value=self.root)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.addCleanup(self.temp.cleanup)

    def directory(self):
        path = self.root / '123'
        path.mkdir(exist_ok=True)
        return path

    @patch.object(t.subprocess, 'Popen')
    @patch.object(t, 'run', return_value='')
    def test_only_own_running_jobs(self, run, popen):
        result = t.Telemetry('me').fetch([job(user='other'), job(state='PENDING'),
                                         job(id='../bad')])
        self.assertEqual({}, result)
        run.assert_not_called()
        popen.assert_not_called()

    @patch.object(t.subprocess, 'Popen')
    @patch.object(t, 'live_step', return_value=False)
    def test_detached_launch_and_budget(self, live, popen):
        path = self.directory()
        telemetry = t.Telemetry('me')
        for _ in range(4):
            telemetry._launch(job(), path)
            meta = t.read_json(path / 'meta.json')
            meta['heartbeat'] = 0
            t.atomic_json(path / 'meta.json', meta)
        self.assertEqual(2, popen.call_count)
        args, kwargs = popen.call_args
        self.assertEqual(['supervisor', '123', '2', '8', '8192'], args[0][-5:])
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(t.subprocess.DEVNULL, kwargs['stdin'])
        self.assertEqual(2, t.read_json(path / 'meta.json')['starts'])

    @patch.object(t.subprocess, 'Popen')
    def test_stale_live_step_and_scheduler_failure_never_launch(self, popen):
        path = self.directory()
        for state in (True, None):
            with patch.object(t, 'live_step', return_value=state):
                t.Telemetry('me')._launch(job(), path)
        popen.assert_not_called()

    @patch.object(t.subprocess, 'Popen')
    @patch.object(t, 'live_step', return_value=False)
    def test_locks_prevent_concurrent_launch(self, live, popen):
        path = self.directory()
        for name in ('launch.lock', 'run.lock'):
            with t.lock(path / name) as acquired:
                self.assertTrue(acquired)
                t.Telemetry('me')._launch(job(), path)
        popen.assert_not_called()

    def test_atomic_private_files_and_corruption(self):
        path = self.root / 'test.json'
        t.atomic_json(path, {'hello': 1})
        self.assertEqual({'hello': 1}, t.read_json(path))
        self.assertEqual(0o600, path.stat().st_mode & 0o777)
        self.assertEqual([path], list(self.root.iterdir()))
        for content in ('{', '[]'):
            path.write_text(content)
            self.assertEqual({}, t.read_json(path))

    def test_hostlist_count(self):
        for text, expected in [('n1', 1), ('n[01-03,08]', 4),
                               ('n[01-03],m[1-2]', 5), ('r[1-2]n[1-3]', 6)]:
            self.assertEqual(expected, t.node_count(text), text)

    @patch.object(t.Telemetry, '_launch')
    @patch.object(t, 'run', return_value='123.batch|2|00:10|1G\n123.extern|1|00:10|8G')
    def test_accounting_throttled_and_sample_aggregation(self, run, launch):
        path = self.directory()
        for host in ('n1', 'n2'):
            t.atomic_json(path / f'sample-{host}.json',
                          {'time': 100, 'cpu_cores': 2, 'ram_mib': 1024,
                           'gpus': [{'uuid': host, 'util': 50, 'used': 1024, 'total': 2048}]})
        telemetry = t.Telemetry('me')
        with patch.object(t.time, 'time', return_value=100):
            status = telemetry.fetch([job()])['123']
        self.assertIn('CPU 4.0/8 cores (50%)', status)
        self.assertIn('cgroup RAM 2.0 GiB', status)
        self.assertIn('VRAM 2.0/4.0 GiB', status)
        with patch.object(t.time, 'time', return_value=101):
            telemetry.fetch([job()])
        self.assertEqual(1, run.call_count)
        self.assertNotIn('123.extern', telemetry.accounting['123'])

    @patch.object(t.Telemetry, '_launch')
    @patch.object(t, 'run', return_value='')
    def test_partial_and_stale_samples_not_reported_as_full_job(self, run, launch):
        path = self.directory()
        t.atomic_json(path / 'sample-n1.json',
                      {'time': 100, 'cpu_cores': 2, 'ram_mib': 1024, 'gpus': []})
        with patch.object(t.time, 'time', return_value=100):
            status = t.Telemetry('me').fetch([job()])['123']
        self.assertIn('CPU unavailable', status)
        self.assertNotIn('cgroup RAM', status)
        with patch.object(t.time, 'time', return_value=200):
            status = t.Telemetry('me').fetch([job(nodes='n1')])['123']
        self.assertIn('CPU unavailable', status)

    def test_gpu_exact_allocation_ids_and_uuid(self):
        output = '0, GPU-a, 20, 100, 1000\n1, GPU-b, 70, 200, 1000\n'
        with patch.dict(os.environ, {'SLURM_STEP_GPUS': '', 'SLURM_JOB_GPUS': '1', 'CUDA_VISIBLE_DEVICES': '0'}), \
                patch.object(t, 'nvidia_query', return_value=(output, {'status': 'ok'})), \
                patch.object(t, 'process_gpu_uuids', return_value=({'GPU-b'}, 'verified process contexts')):
            sample = t.gpu_sample()
        self.assertEqual(['GPU-b'], [g['uuid'] for g in sample])
        with patch.dict(os.environ, {'SLURM_STEP_GPUS': '', 'SLURM_JOB_GPUS': 'GPU-a'}), \
                patch.object(t, 'nvidia_query', return_value=(output, {'status': 'ok'})), \
                patch.object(t, 'process_gpu_uuids', return_value=({'GPU-b'}, 'verified process contexts')):
            self.assertEqual('GPU-a', t.gpu_sample()[0]['uuid'])

    def test_step_gpu_precedence_and_diagnostics(self):
        output = '0, GPU-a, 20, 100, 1000\n1, GPU-b, 70, 200, 1000\n'
        detail = {}
        with patch.dict(os.environ, {'SLURM_STEP_GPUS': '1', 'SLURM_JOB_GPUS': '0',
                                     'CUDA_VISIBLE_DEVICES': '0', 'SECRET': 'hidden'}), \
                patch.object(t, 'nvidia_query', return_value=(output, {'status': 'ok'})), \
                patch.object(t, 'process_gpu_uuids', return_value=({'GPU-b'}, 'verified process contexts')):
            self.assertEqual('GPU-b', t.gpu_sample(detail)[0]['uuid'])
        self.assertEqual('SLURM_STEP_GPUS', detail['source'])
        self.assertEqual(['1'], detail['allowed_ids'])
        self.assertNotIn('hidden', json.dumps(detail))
        self.assertEqual({'0', '1', '2', 'GPU-a'}, t.gpu_ids('0-2,GPU-a'))

    def test_gpu_failures_classified(self):
        with patch.dict(os.environ, {'SLURM_STEP_GPUS': 'GPU-a'}), \
                patch.object(t.subprocess, 'run', return_value=SimpleNamespace(returncode=9, stderr='secret')):
            detail = {}
            self.assertEqual([], t.gpu_sample(detail))
        self.assertEqual('nvidia-smi failed', detail['status'])
        self.assertEqual(9, detail['returncode'])
        self.assertNotIn('secret', json.dumps(detail))
        for error, expected in [(FileNotFoundError(), 'nvidia-smi missing'),
                                (t.subprocess.TimeoutExpired('nvidia-smi', 2), 'nvidia-smi timeout')]:
            with patch.object(t.subprocess, 'run', side_effect=error):
                self.assertEqual(expected, t.nvidia_query()[1]['status'])

    def test_reload_execs_same_collector_without_new_step(self):
        with patch.object(t, 'source_version', return_value=(2, 10)), \
                patch.object(t.os, 'execv') as execute:
            t.reload_collector((1, 10), '123')
            self.assertEqual(['collector', '123'], execute.call_args.args[1][-2:])
        with patch.object(t, 'source_version', return_value=(1, 10)), \
                patch.object(t.os, 'execv') as execute:
            t.reload_collector((1, 10), '123')
            execute.assert_not_called()

    def test_numeric_gpu_ids_require_verified_contexts(self):
        with patch.dict(os.environ, {'SLURM_STEP_GPUS': '0'}), \
                patch.object(t, 'process_gpu_uuids', return_value=(set(), 'no verified process contexts')), \
                patch.object(t, 'nvidia_query') as query:
            detail = {}
            self.assertEqual([], t.gpu_sample(detail))
            self.assertEqual('GPU identity unverified', detail['status'])
            query.assert_not_called()

    def test_process_gpu_uuid_requires_same_uid_and_exact_job(self):
        output = '101, GPU-a\n102, GPU-b\n103, GPU-c\n'
        paths = {'/proc/101/cgroup': '3:cpu:/slurm/uid_1/job_123/step_0/task_0',
                 '/proc/102/cgroup': '3:cpu:/slurm/uid_1/job_1234/step_0/task_0',
                 '/proc/103/cgroup': '3:cpu:/slurm/uid_1/job_123/step_0/task_0'}
        def stat(path):
            return SimpleNamespace(st_uid=os.getuid() + (str(path) == '/proc/103'))
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), \
                patch.object(t, 'nvidia_query', return_value=(output, {'status': 'ok'})), \
                patch.object(Path, 'stat', stat), \
                patch.object(Path, 'read_text', lambda p: paths[str(p)]):
            self.assertEqual({'GPU-a'}, t.process_gpu_uuids()[0])

    def test_malformed_metadata_and_samples(self):
        path = self.root / 'meta.json'
        path.write_text('{"heartbeat": "bad", "starts": -9}')
        self.assertEqual(0, t.read_meta(path)['heartbeat'])
        self.assertEqual(t.MAX_STARTS, t.read_meta(path)['starts'])
        for content in ('{', '{}', '[]'):
            path.write_text(content)
            self.assertEqual(t.MAX_STARTS, t.read_meta(path)['starts'])
        for content in ('{"time": "bad"}', '{"time": NaN}',
                        '{"time": 1, "cpu_cores": "bad"}', '{"time": 1, "gpus": [1]}'):
            path.write_text(content)
            self.assertEqual({}, t.read_sample(path))

    def test_lock_only_cleanup_has_no_scheduler_query(self):
        path = self.directory()
        (path / 'run.lock').touch()
        with patch.object(t, 'run') as run:
            t.cleanup(self.root, now=t.time.time() + 2 * t.TTL)
            run.assert_not_called()

    def test_cleanup_throttled(self):
        path = self.directory()
        t.atomic_json(path / 'meta.json', {'starts': 2})
        with patch.object(t.time, 'time', return_value=t.time.time() + 2 * t.TTL), \
                patch.object(t, 'job_ended', return_value=False) as ended:
            t.cleanup(self.root)
            t.cleanup(self.root)
            ended.assert_called_once()

    def test_scrub_inherited_slurm_environment(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'other', 'SRUN_CPUS_PER_TASK': '99',
                                     'PATH': '/bin'}):
            env = t.launch_env()
        self.assertNotIn('SLURM_JOB_ID', env)
        self.assertNotIn('SRUN_CPUS_PER_TASK', env)
        self.assertEqual('/bin', env['PATH'])

    def test_cgroup_hybrid_reads_job_not_service(self):
        files = {'/proc/self/cgroup': '12:memory:/slurm/uid_1/job_123/step_batch/task_0\n'
                 '3:cpu,cpuacct:/slurm/uid_1/job_123/step_batch/task_0\n'
                 '0::/system.slice/slurmd.service\n',
                 '/sys/fs/cgroup/memory/slurm/uid_1/job_123/memory.usage_in_bytes': str(2 * 1024**2),
                 '/sys/fs/cgroup/cpu,cpuacct/slurm/uid_1/job_123/cpuacct.usage': '3000000000'}
        with patch.object(Path, 'read_text', lambda p: files[str(p)]), \
                patch.object(Path, 'exists', lambda p: str(p) in files):
            self.assertEqual((3, 2), t.cgroup_sample('123'))
            self.assertEqual((None, None), t.cgroup_sample('other'))

    def test_cleanup_preserves_live_budget_and_lock_inodes(self):
        path = self.directory()
        t.atomic_json(path / 'meta.json', {'starts': 2})
        (path / 'run.lock').touch()
        inode = (path / 'run.lock').stat().st_ino
        with patch.object(t, 'run', return_value='RUNNING'):
            t.cleanup(self.root, now=t.time.time() + 2 * t.TTL)
        self.assertTrue((path / 'meta.json').exists())
        with patch.object(t, 'run', return_value=''):
            t.cleanup(self.root, now=t.time.time() + 2 * t.TTL)
        self.assertFalse((path / 'meta.json').exists())
        self.assertEqual(inode, (path / 'run.lock').stat().st_ino)

    def test_completion_expiry_ignores_late_cached_readers(self):
        path = self.directory()
        ended_at = t.time.time() - t.TTL - 1
        t.atomic_json(path / 'meta.json', {'starts': 2, 'ended_at': ended_at})
        (path / 'seen').touch()
        with patch.object(t, 'job_ended', return_value=True):
            t.cleanup(self.root, now=t.time.time())
        self.assertFalse((path / 'meta.json').exists())
        self.assertFalse((path / 'seen').exists())

    @patch.object(t.signal, 'signal')
    @patch.object(t, 'live_step', return_value=False)
    @patch.object(t, 'job_ended', return_value=False)
    @patch.object(t.time, 'sleep', side_effect=RuntimeError('injected failure'))
    @patch.object(t.subprocess, 'Popen')
    def test_running_supervisor_exception_terminates_only_its_launcher(self, popen, sleep, ended, live, signals):
        proc = Mock()
        proc.poll.return_value = None
        popen.return_value = proc
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            t.supervisor('123', 1, 1, 1024)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=10)
        self.assertEqual(0, t.read_meta(self.directory() / 'meta.json')['heartbeat'])

    def test_ended_job_error_fallback_fails_closed(self):
        with patch.object(t, 'run', side_effect=[None, None]):
            self.assertFalse(t.job_ended('123'))
        with patch.object(t, 'run', side_effect=[None, '123\n124\n']):
            self.assertFalse(t.job_ended('123'))
        with patch.object(t, 'run', side_effect=[None, '124\n']):
            self.assertTrue(t.job_ended('123'))

    @patch.object(t, 'cleanup')
    @patch.object(t.time, 'sleep')
    @patch.object(t, 'live_step', return_value=False)
    @patch.object(t, 'run', return_value='')
    @patch.object(t.subprocess, 'Popen')
    def test_confirmed_end_retains_then_cleans_without_slurm_step(self, popen, run, live, sleep, cleanup):
        proc = Mock()
        proc.poll.return_value = 0
        popen.return_value = proc
        t.supervisor('123', 1, 1, 1024)
        sleep.assert_called_once_with(t.TTL + 1)
        self.assertGreater(t.read_meta(self.directory() / 'meta.json')['ended_at'], 0)
        cleanup.assert_called_once_with(self.root)
        proc.terminate.assert_not_called()

    @patch.object(t.time, 'sleep')
    @patch.object(t, 'live_step', return_value=False)
    @patch.object(t, 'run', return_value='RUNNING')
    @patch.object(t.subprocess, 'Popen')
    def test_supervisor_one_bounded_overlapping_step(self, popen, run, live, sleep):
        proc = Mock()
        proc.poll.return_value = 0
        popen.return_value = proc
        t.supervisor('123', 2, 8, 8192)
        command = popen.call_args.args[0]
        for arg in ('--overlap', '--exact', '--immediate=10', '--nodes=2',
                    '--ntasks-per-node=1', '--cpus-per-task=1', '--job-name=sj-telemetry'):
            self.assertIn(arg, command)
        self.assertNotIn('--gres=none', command)
        proc.terminate.assert_not_called()
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
