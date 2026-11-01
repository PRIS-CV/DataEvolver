"""Host-registered external training, with durable intent before process launch.

The caller supplies an approved command and exact source hashes. A model cannot
select an executable. An uncertain launch is reconciled from its native receipt;
it is never resubmitted under the same job identity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import signal
import time

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.harness.state_store import CampaignLease, atomic_write_json
from dataevolver.harness.training_feedback import ExternalTrainingRunAdapter


class RegisteredTrainingRunner(ExternalTrainingRunAdapter):
    def __init__(self, *, store, work_root, command, source_hashes, environment,
                 max_wall_seconds):
        super().__init__(store=store, work_root=work_root)
        if (not command or not all(isinstance(v, str) and v for v in command)
                or not Path(command[0]).is_absolute() or not source_hashes
                or type(max_wall_seconds) is not int or max_wall_seconds < 1):
            raise ValueError('training requires a registered command, source inventory and timeout')
        self.command, self.sources = tuple(command), dict(source_hashes)
        self.environment, self.timeout = dict(environment), max_wall_seconds

    def _sources(self):
        if any(file_sha256(Path(path)) != digest for path, digest in self.sources.items()):
            raise ValueError('registered external training implementation changed')

    def run(self, job, *, external_job_id, result_store, result_path, runner_inputs=None):
        self._sources()
        # ArtifactRef validates normalized relative paths before any subprocess.
        expected = ArtifactRef(result_store, result_path, '0' * 64)
        destination = self.store.roots[result_store] / expected.path
        if not destination.resolve().is_relative_to(self.store.roots[result_store]):
            raise ValueError('training receipt escaped its registered result store')
        request = {'job': job.to_dict(), 'external_job_id': external_job_id,
            'stores': {k: str(v) for k, v in self.store.roots.items()},
            'result_store': result_store, 'result_path': result_path,
            'inputs': runner_inputs or {}}
        binding = payload_sha256({'request': request, 'command': self.command,
            'sources': self.sources, 'environment': self.environment, 'timeout': self.timeout})
        directory = self.root / (job.job_id + '.runner')
        with CampaignLease(directory / '.lease', ttl_seconds=self.timeout + 60):
            self.submit(job, external_job_id=external_job_id)
            path = directory / 'launch.json'
            old = json.loads(path.read_text()) if path.exists() else None
            if old is not None and old['binding'] != binding:
                raise ValueError('external training launch binding changed')
            if destination.exists():
                if old is None:
                    raise ValueError('native training receipt has no launch intent')
                receipt = ArtifactRef(result_store, result_path, file_sha256(destination))
                return self.collect(job, receipt)
            if old is not None:
                raise RuntimeError('recorded or uncertain external training launch cannot be resubmitted')
            request_path = directory / 'request.json'
            atomic_write_json(request_path, request)
            record = {'binding': binding, 'state': 'STARTED', 'started_at': time.time(),
                      'native_receipt': str(destination), 'command': list(self.command)}
            atomic_write_json(path, record)
            environment = {**os.environ, **self.environment}
            for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
                environment.pop(key, None)
            started = time.monotonic()
            try:
                with (directory / 'runner.log').open('a') as log:
                    process = subprocess.Popen([*self.command, '--request', str(request_path)],
                        env=environment, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    record['pid'] = process.pid
                    atomic_write_json(path, record)
                    try:
                        returncode = process.wait(timeout=self.timeout)
                    except BaseException:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                        raise
                record.update(state='EXITED', exit_code=returncode,
                              wall_seconds=time.monotonic() - started, finished_at=time.time())
                atomic_write_json(path, record)
                if returncode:
                    raise RuntimeError('external training failed; native log and exit retained')
                if not destination.is_file():
                    raise RuntimeError('external trainer exited without its exact native receipt')
                return self.collect(job, ArtifactRef(result_store, result_path, file_sha256(destination)))
            except BaseException as exc:
                if record['state'] == 'STARTED':
                    record.update(state='UNCERTAIN', error=type(exc).__name__,
                        wall_seconds=time.monotonic() - started, finished_at=time.time())
                    atomic_write_json(path, record)
                raise
