import test from 'node:test';
import assert from 'node:assert/strict';
import { cp, mkdtemp, rm, readFile, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import { root } from './build.mjs';

test('app builds and serves when copied without the Harness or repository root', async () => {
  const scratch = await mkdtemp(join(tmpdir(), 'dataevolver-standalone-page-'));
  try {
    const app = join(scratch, 'standalone');
    await cp(root, app, { recursive: true, filter: path => !['dist', 'node_modules', '.npm', '__pycache__'].some(name => path.split(/[\\/]/).includes(name)) });
    const build = spawnSync(process.execPath, [join(app, 'tooling/build.mjs')], {
      cwd: scratch, encoding: 'utf8', timeout: 60000,
      env: { ...process.env, TRACE_API_BASE_URL: '/' },
    });
    assert.equal(build.status, 0, build.stderr);
    await access(join(app, 'dist/index.html'));
    await access(join(app, 'dist/traces/index.html'));
    for (const name of ['dist', 'src', 'experiments', 'package.json']) await assert.rejects(access(join(scratch, name)));
    for (const name of ['backend', 'tooling', 'deploy', 'package.json']) await assert.rejects(access(join(app, 'dist', name)));
    const { createTraceServer } = await import(pathToFileURL(join(app, 'backend/server.mjs')));
    const server = await createTraceServer();
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    try {
      const base = `http://127.0.0.1:${server.address().port}`;
      assert.equal((await fetch(`${base}/`)).status, 200);
      assert.equal((await fetch(`${base}/traces/`)).status, 200);
      assert.equal((await (await fetch(`${base}/api/health`)).json()).cases, 2);
      for (const path of ['/src/dataevolver/__init__.py', '/experiments/', '/backend/server.mjs', '/package.json']) assert.equal((await fetch(base + path)).status, 404);
    } finally { await new Promise(resolve => server.close(resolve)); }
  } finally { await rm(scratch, { recursive: true, force: true }); }
});

test('Docker build context is limited to the application folder', async () => {
  const compose = await readFile(join(root, 'deploy/compose.yaml'), 'utf8');
  assert.match(compose, /context: \.\.\n/);
  assert.match(compose, /dockerfile: deploy\/Dockerfile/);
  const docker = await readFile(join(root, 'deploy/Dockerfile'), 'utf8');
  assert.doesNotMatch(docker, /COPY\s+\.\.\//);
  assert.match(docker, /COPY frontend \.\/frontend/);
  assert.match(docker, /backend\/server\.mjs/);
});
