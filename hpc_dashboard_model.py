"""Scheduler-neutral display records. No scheduler calls or side effects."""
from dataclasses import dataclass


def clean(value):
    # Names and metadata are untrusted terminal input, not Rich markup.
    return ''.join(c if c.isprintable() or c == '\n' else ' ' for c in str(value or ''))


@dataclass
class Job:
    id: str
    name: str
    user: str
    state: str
    partition: str
    nodes: str
    cpus: float | None
    memory: float | None
    gpu: str
    elapsed: float | None
    limit: float | None
    comment: str = ''
    reason: str = ''
    step_count: int | None = None
    steps_queried: bool = False
    backend: str = 'slurm'
    context: str = ''
    namespace: str = ''
    uid: str = ''
    pod_summary: str = ''
    tag: str = ''
    resource_details: str = ''
    raw_state: str = ''

    @property
    def identity(self):
        return (self.backend, self.context, self.namespace, self.uid or self.id)
