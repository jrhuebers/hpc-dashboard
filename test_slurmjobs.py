"""Offline unit tests: no scheduler calls, sleeps, or new allocations."""
import io
import json
import math
import subprocess
import unittest
from unittest.mock import Mock, patch

from rich.console import Console
import slurmjobs as s


def wrapped(value, **flags):
    return dict(number=value, set=True, infinite=False, **flags)


def raw_job(**changes):
    raw = dict(job_id=42, name='example', user_name='me', job_state=['RUNNING'],
               partition='compute', nodes='node01', cpus=wrapped(8),
               node_count=wrapped(2), start_time=wrapped(900), end_time=wrapped(0),
               time_limit=wrapped(60), comment='a comment')
    raw.update(changes)
    return raw


def legacy(**changes):
    fields = dict(jid='42', name='example', user='me', state='RUNNING',
                  partition='compute', nodes='node[01-02]', cpu='8', mem='2Gc',
                  nnode='2', elapsed='1:02', limit='1-02:03:04', gres='gpu:a100:2',
                  comment='a comment', reason='None')
    fields.update(changes)
    return s.SEPARATOR.join(fields.values())


class JsonTests(unittest.TestCase):
    def test_wrapped_numbers(self):
        self.assertEqual(s.number(wrapped(8)), 8)
        self.assertEqual(s.number(dict(number=8, set=False), 7), 7)
        self.assertEqual(s.number(None, 7), 7)
        self.assertEqual(s.number(dict(infinite=True)), math.inf)
        job = s.from_json(raw_job(), 1000)
        self.assertEqual((job.cpus, job.elapsed, job.limit), (8, 100, 3600))

    def test_allocated_vs_requested(self):
        raw = raw_job(tres_alloc_str='cpu=4,node=1,mem=8G,gres/gpu=2,gres/gpu:a100=2',
                      tres_req_str='cpu=16,node=2,mem=64G,gres/gpu=4')
        job = s.from_json(raw, 1000)
        self.assertEqual((job.cpus, job.memory, job.gpu), (4, 8192, '2 × a100'))
        raw['job_state'] = ['PENDING']
        job = s.from_json(raw, 1000)
        self.assertEqual((job.cpus, job.memory, job.elapsed), (16, 65536, 0))
        self.assertEqual(job.gpu, '4 × GPU (type unspecified)')

    def test_requested_fallback(self):
        job = s.from_json(raw_job(tres_req_str='cpu=12,node=3,mem=1.5G'), 1000)
        self.assertEqual((job.cpus, job.memory), (12, 1536))

    def test_memory_per_cpu_and_node(self):
        for field, amount, expected in [('memory_per_cpu', 1024, 8192),
                                         ('memory_per_node', 2048, 4096)]:
            with self.subTest(field=field):
                self.assertEqual(s.from_json(raw_job(**{field: wrapped(amount)}), 1000).memory,
                                 expected)
        self.assertEqual(s.from_json(raw_job(memory_per_cpu=512, memory_per_node=8192), 1000).memory, 4096)

    def test_zero_and_unset_memory(self):
        for field in ('memory_per_cpu', 'memory_per_node'):
            with self.subTest(field=field):
                job = s.from_json(raw_job(**{field: wrapped(0)}), 1000)
                self.assertEqual(job.memory, 0)
                self.assertEqual(s.memory_label(job.memory), 'all node RAM')
                self.assertIsNone(s.from_json(raw_job(**{field: dict(number=5, set=False)}), 1000).memory)
        self.assertIsNone(s.from_json(raw_job(), 1000).memory)

    def test_gres_gpu(self):
        cases = [({}, '', 'no GPU'), ({'gres/gpu': '0'}, '', 'no GPU'),
                 ({'gres/gpu': '3'}, '', '3 × GPU (type unspecified)'),
                 ({'gres/gpu': '3', 'gres/gpu:a100': '2', 'gres/gpu:v100': '1'}, '', '2 × a100, 1 × v100'),
                 ({}, 'gpu:a100:2(S:0-1),gpu:1', '2 × a100, 1 × GPU (type unspecified)')]
        for resources, gres, expected in cases:
            with self.subTest(resources=resources, gres=gres):
                self.assertEqual(s.gpu_label(resources, gres), expected)
        self.assertEqual(s.from_json(raw_job(tres_per_node='gpu:h100:4'), 1000).gpu, '4 × h100')

    def test_arrays_and_heterogeneous_jobs(self):
        for values, expected in [({'array_job_id': wrapped(30), 'array_task_id': wrapped(7)}, '30_7'),
                                 ({'array_job_id': 30, 'array_task_string': '1-8%2'}, '30_[1-8%2]'),
                                 ({'array_job_id': 30, 'array_task_id': 4294967294}, '42'),
                                 ({'array_job_id': 30, 'array_task_id': dict(set=False)}, '42'),
                                 ({'het_job_id': wrapped(10), 'het_job_offset': wrapped(2)}, '10+2')]:
            with self.subTest(values=values):
                self.assertEqual(s.from_json(raw_job(**values), 1000).id, expected)

    def test_pending_future_start_and_finished_times(self):
        for changes, expected in [({'job_state': ['PENDING'], 'start_time': wrapped(800)}, 0),
                                  ({'start_time': wrapped(1100)}, 0),
                                  ({'start_time': wrapped(0)}, 0),
                                  ({'end_time': wrapped(950)}, 50)]:
            with self.subTest(changes=changes):
                self.assertEqual(s.from_json(raw_job(**changes), 1000).elapsed, expected)

    def test_suspension_resume_and_unlimited(self):
        values = dict(suspend_time=wrapped(960), pre_sus_time=wrapped(40),
                      time_limit=dict(infinite=True))
        job = s.from_json(raw_job(job_state=['SUSPENDED'], **values), 1000)
        self.assertEqual(job.elapsed, 40)
        self.assertEqual(s.duration(job.limit), 'unlimited')
        self.assertEqual(s.from_json(raw_job(**values), 1000).elapsed, 80)
        self.assertEqual(s.from_json(raw_job(end_time=wrapped(980), **values), 1000).elapsed, 60)
        self.assertIsNone(s.from_json(raw_job(time_limit=dict(set=False)), 1000).limit)

    def test_reason_and_cleaning(self):
        job = s.from_json(raw_job(name='[red]name[/red]\x1b\t', comment='line1\nline2',
                                 state_description='Resources', state_reason='Priority'), 1000)
        self.assertEqual(job.name, '[red]name[/red]  ')
        self.assertEqual(job.comment, 'line1\nline2')
        self.assertEqual(job.reason, 'Resources')


class LegacyTests(unittest.TestCase):
    def test_legacy_cpu_and_node_memory(self):
        for mem, expected in [('2Gc', 16384), ('2Gn', 4096), ('2048', 4096), ('0n', 0), ('N/A', None)]:
            with self.subTest(mem=mem):
                job = s.from_text(legacy(mem=mem))
                self.assertEqual(job.memory, expected)
                self.assertEqual((job.elapsed, job.limit, job.gpu), (62, 93784, '2 × a100 per node'))

    def test_comments_arrays_and_unlimited(self):
        job = s.from_text(legacy(jid='42_[1-9%2]', name='long | name', comment='[red]hello|world[/red]', limit='UNLIMITED'))
        self.assertEqual(job.id, '42_[1-9%2]')
        self.assertEqual(job.comment, '[red]hello|world[/red]')
        self.assertEqual(job.limit, math.inf)
        for comment in ('(null)', 'N/A'):
            self.assertEqual(s.from_text(legacy(comment=comment)).comment, '')

    def test_gres_prefix_unknown_and_multi_node(self):
        for gres, expected in [('gres/gpu:1', '1 × GPU (type unspecified)'),
                               ('gres/gpu:h100:2', '2 × h100'),
                               ('gpu:1', '1 × GPU (type unspecified)'),
                               ('N/A', 'GPU unknown'), ('gres/gpu', 'GPU unknown'),
                               ('(null)', 'no GPU')]:
            with self.subTest(gres=gres):
                self.assertEqual(s.from_text(legacy(gres=gres, nnode='1')).gpu, expected)
        self.assertEqual(s.from_text(legacy(gres='gres/gpu:1', nnode='3')).gpu,
                         '1 × GPU (type unspecified) per node')

    def test_malformed_line(self):
        with self.assertRaisesRegex(ValueError, 'Cannot parse legacy'):
            s.from_text('not a valid record')

    def test_duration_and_memory_units(self):
        for text, seconds in [('10', 600), ('2:03', 123), ('1:02:03', 3723), ('2-01:02:03', 176523), ('INFINITE', math.inf), ('N/A', None)]:
            self.assertEqual(s.parse_duration(text), seconds)
        for value, expected in [('1024K', 1), ('1.5G', 1536), ('1T', 1048576), ('bad', None)]:
            self.assertEqual(s.memory_mib(value), expected)
        self.assertEqual(s.duration(-1), '0:00')
        self.assertEqual(s.duration(93784), '1-02:03:04')


class QueueTests(unittest.TestCase):
    @patch('slurmjobs.subprocess.run')
    def test_timeout_and_missing_command(self, run):
        queue = s.Queue(timeout=2)
        run.side_effect = subprocess.TimeoutExpired('squeue', 2)
        with self.assertRaisesRegex(RuntimeError, 'within 2 seconds'):
            queue.fetch()
        run.assert_called_once_with(['squeue', '--json'], capture_output=True, text=True, timeout=2, check=False)
        run.side_effect = FileNotFoundError()
        with self.assertRaisesRegex(RuntimeError, 'not installed'):
            queue.fetch()

    @patch('slurmjobs.time.time', return_value=1000)
    @patch('slurmjobs.subprocess.run')
    def test_json_fetch(self, run, clock):
        run.return_value = subprocess.CompletedProcess([], 0, json.dumps({'jobs': [raw_job()]}), '')
        jobs = s.Queue().fetch('me')
        self.assertEqual(jobs[0].elapsed, 100)
        self.assertEqual(run.call_args.args[0], ['squeue', '--json', '--user', 'me'])

    @patch('slurmjobs.subprocess.run')
    def test_unsupported_json_fallback_is_remembered(self, run):
        for error in ('unrecognized option --json', 'unknown option json', 'invalid option json', 'json not supported'):
            with self.subTest(error=error):
                run.reset_mock()
                run.side_effect = [subprocess.CompletedProcess([], 1, '', error),
                                   subprocess.CompletedProcess([], 0, legacy() + '\n', ''),
                                   subprocess.CompletedProcess([], 0, '', '')]
                queue = s.Queue()
                self.assertEqual(queue.fetch('me')[0].id, '42')
                self.assertTrue(queue.legacy)
                self.assertEqual(queue.fetch('me'), [])
                self.assertEqual(run.call_count, 3)
                self.assertEqual(run.call_args_list[1].args[0],
                                 ['squeue', '--noheader', '--noconvert', '--format', s.SEPARATOR.join(s.FIELDS), '--user', 'me'])
                self.assertNotIn('--json', run.call_args.args[0])

    @patch('slurmjobs.subprocess.run')
    def test_real_failures_do_not_trigger_fallback(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, '', 'controller unavailable')
        queue = s.Queue()
        with self.assertRaisesRegex(RuntimeError, 'controller unavailable'):
            queue.fetch()
        self.assertFalse(queue.legacy)
        self.assertEqual(run.call_count, 1)

    @patch('slurmjobs.subprocess.run')
    def test_json_errors_and_invalid_payload(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, '{"errors":["failure"]}', '')
        with self.assertRaisesRegex(RuntimeError, 'Slurm returned errors'):
            s.Queue().fetch()
        run.return_value = subprocess.CompletedProcess([], 0, 'not json', '')
        with self.assertRaises(ValueError):
            s.Queue().fetch()
        run.return_value = subprocess.CompletedProcess([], 0, '{}', '')
        self.assertEqual(s.Queue().fetch(), [])


class CacheTests(unittest.TestCase):
    @patch('slurmjobs.time.monotonic')
    def test_refresh_throttles_queue_and_updates_elapsed(self, clock):
        clock.return_value = 100
        job = s.from_json(raw_job(), 1000)
        queue = Mock()
        queue.fetch.return_value = [job]
        cache = s.QueueCache(queue, 'me')
        self.assertEqual(cache.fetch()[0].elapsed, 100)
        clock.return_value = 101
        self.assertEqual(cache.fetch()[0].elapsed, 101)
        self.assertEqual(job.elapsed, 100)
        queue.fetch.assert_called_once_with('me')
        clock.return_value = 110
        cache.fetch()
        self.assertEqual(queue.fetch.call_count, 2)

    @patch('slurmjobs.time.monotonic')
    def test_failure_preserves_good_snapshot_and_backs_off(self, clock):
        clock.return_value = 100
        queue = Mock()
        queue.fetch.return_value = [s.from_json(raw_job(), 1000)]
        cache = s.QueueCache(queue)
        cache.fetch()
        clock.return_value = 110
        queue.fetch.side_effect = RuntimeError('offline')
        with self.assertRaises(RuntimeError):
            cache.fetch()
        clock.return_value = 111
        self.assertEqual(cache.fetch()[0].elapsed, 111)
        self.assertEqual(queue.fetch.call_count, 2)

    @patch('slurmjobs.time.monotonic', return_value=100)
    def test_pending_elapsed_not_extrapolated(self, clock):
        queue = Mock()
        queue.fetch.return_value = [s.from_json(raw_job(job_state=['PENDING']), 1000)]
        cache = s.QueueCache(queue)
        cache.fetch()
        clock.return_value = 105
        self.assertEqual(cache.fetch()[0].elapsed, 0)


class RenderingTests(unittest.TestCase):
    def text(self, jobs, width=80, compact=False):
        buffer = io.StringIO()
        console = Console(file=buffer, width=width, color_system=None, highlight=False)
        console.print(s.render(jobs, 'me', compact=compact, width=width))
        return buffer.getvalue()

    def test_comments_and_markup_are_literal(self):
        job = s.from_json(raw_job(name='[red]name[/red]', comment='[bold]comment[/bold]'), 1000)
        output = self.text([job])
        self.assertIn('[red]name[/red]', output)
        self.assertIn('[bold]comment[/bold]', output)
        title = list(s.job_lines(job))[0]
        self.assertEqual([span.style for span in title.spans], ['green'])

    def test_long_names_comments_wrap_without_loss(self):
        name = 'long-name-' * 15
        comment = '[red]' + 'x' * 150 + '[/red]\nsecond line'
        job = s.from_json(raw_job(name=name, comment=comment), 1000)
        lines = s.wrap_lines(list(s.job_lines(job)), 40)
        self.assertTrue(all(len(line.plain) <= 40 for line in lines))
        joined = ''.join(line.plain.strip() for line in lines)
        self.assertIn(name, joined)
        self.assertIn('[red]' + 'x' * 150 + '[/red]', joined)
        self.assertNotIn('…', joined)
        self.assertTrue(any(line.plain.startswith(' ' * 15 + 'x') for line in lines))

    def test_grouping_and_state_order(self):
        jobs = [s.from_json(raw_job(job_id=2, job_state=['PENDING']), 1000),
                s.from_json(raw_job(job_id=3), 1000),
                s.from_json(raw_job(job_id=4, user_name='zoe'), 1000),
                s.from_json(raw_job(job_id=5, user_name='amy'), 1000)]
        output = self.text(jobs)
        self.assertIn('MY JOBS · me · 2 jobs', output)
        self.assertIn('OTHER USERS · 2 jobs', output)
        self.assertLess(output.index('3 · example'), output.index('2 · example'))
        self.assertLess(output.index('amy'), output.index('zoe'))
        self.assertIn('  5 · example', output)

    def test_only_section_boundary_has_blank_line(self):
        own = s.from_json(raw_job(), 1000)
        others = [s.from_json(raw_job(job_id=i, user_name=user), 1000)
                  for i, user in [(2, 'alice'), (3, 'bob')]]
        for jobs in ([own], [], [own, *others], others):
            lines = self.text(jobs).splitlines()
            blanks = [i for i, line in enumerate(lines) if not line.strip()]
            if any(job.user != 'me' for job in jobs):
                boundary = next(i for i, line in enumerate(lines) if line.startswith('OTHER USERS'))
                self.assertEqual(blanks, [boundary - 1])
            else:
                self.assertEqual(blanks, [])

    def test_header_colors(self):
        from rich.style import Style
        jobs = [s.from_json(raw_job(), 1000),
                s.from_json(raw_job(job_id=2, user_name='alice'), 1000)]
        lines = list(s.render(jobs, 'me').renderables)
        headers = [line for line in lines if line.plain.startswith(('MY JOBS', 'OTHER USERS')) or line.plain == 'alice']
        self.assertEqual(len(headers), 3)
        for header in headers:
            style = Style.parse(header.style)
            self.assertEqual(style.color.get_truecolor(), (255, 255, 255))
            self.assertEqual(style.bgcolor.get_truecolor(), (0, 0, 0))

    def test_usage_literal_wrapping_and_own_jobs_only(self):
        own = s.from_json(raw_job(), 1000)
        other = s.from_json(raw_job(job_id=43, user_name='other'), 1000)
        usage = {'42': 'CPU 1/8 cores · RAM 1 GiB · GPU [red]not markup[/red]',
                 '43': 'must not appear'}
        group = s.render([own, other], 'me', width=40, usage=usage)
        lines = list(group.renderables)
        self.assertTrue(all(len(line.plain) <= 40 for line in lines))
        joined = ''.join(line.plain.strip() for line in lines)
        self.assertIn('[red]not markup[/red]', joined)
        self.assertNotIn('must not appear', joined)
        self.assertTrue(any(line.plain.startswith(' ' * 13) for line in lines))

    def test_watch_default_one_second(self):
        with patch('slurmjobs.sys.argv', ['sj', '--help']):
            with patch('sys.stdout', new=io.StringIO()) as output:
                with self.assertRaises(SystemExit) as exc:
                    s.main()
            self.assertEqual(exc.exception.code, 0)
            self.assertIn('default: 1;', output.getvalue())

    def test_pending_reason_compact_and_empty(self):
        job = s.from_json(raw_job(job_state=['PENDING'], state_reason='Priority', nodes='', comment=''), 1000)
        output = self.text([job], width=120, compact=True)
        self.assertIn('Waiting    Priority', output)
        self.assertIn('not allocated', output)
        self.assertIn('Time 0:00/1:00:00', output)
        self.assertNotIn('Comment', output)
        self.assertIn('No active jobs.', self.text([]))


if __name__ == '__main__':
    unittest.main()
