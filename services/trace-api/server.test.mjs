import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, mkdir, symlink, writeFile, readFile, cp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { request } from 'node:http';
import { fileURLToPath } from 'node:url';
import { containedFile, createTraceServer, loadArchive } from './server.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
let server, base, scratch;
before(async () => {
  server = await createTraceServer({ staticRoot: join(root, 'web'), allowedOrigins: ['https://pris-cv.github.io'], dev: true });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${server.address().port}`;
  scratch = await mkdtemp(join(tmpdir(), 'dataevolver-traces-test-'));
});
after(async () => {
  await new Promise(resolve => server.close(resolve));
  await rm(scratch, { recursive: true, force: true });
});
test('health discloses only archive status', async () => {
  const r = await fetch(`${base}/api/health`), data = await r.json();
  assert.equal(r.status, 200); assert.equal(data.mode, 'curated-archive-read-only');
  assert.equal(data.cases, 2); assert.match(data.revision, /^[a-f0-9]{64}$/);
});
test('manifest, case and round agree exactly with exported evidence', async () => {
  const expected = JSON.parse(await readFile(join(root, 'web/traces/traces.json'), 'utf8'));
  assert.deepEqual(await (await fetch(`${base}/api/traces`)).json(), expected);
  for (const c of expected.cases) {
    assert.deepEqual(await (await fetch(`${base}/api/traces/${c.id}`)).json(), c);
    for (const r of c.rounds) assert.deepEqual(await (await fetch(`${base}/api/traces/${c.id}/rounds/${r.round}`)).json(), r);
  }
});
test('only registered image bytes are served', async () => {
  const r = await fetch(`${base}/api/assets/grounding-r0.png`);
  assert.equal(r.headers.get('content-type'), 'image/png');
  assert.deepEqual(Buffer.from(await r.arrayBuffer()), await readFile(join(root, 'web/traces/images/grounding-r0.png')));
  assert.equal((await fetch(`${base}/api/assets/private.png`)).status, 404);
});
test('API cannot start, stop or mutate experiments', async () => {
  for (const method of ['POST', 'PUT', 'PATCH', 'DELETE']) assert.equal((await fetch(`${base}/api/traces`, { method })).status, 405);
  for (const path of ['/api/runs', '/api/start', '/api/traces/unknown', '/api/traces/count/rounds/99', '/api/traces/count/rounds/-1']) assert.equal((await fetch(base + path)).status, 404);
});
test('CORS allows only explicit read-only origins, never credentials', async () => {
  let r = await fetch(`${base}/api/traces`, { headers: { Origin: 'https://pris-cv.github.io' } });
  assert.equal(r.headers.get('access-control-allow-origin'), 'https://pris-cv.github.io');
  assert.equal(r.headers.get('access-control-allow-credentials'), null);
  r = await fetch(`${base}/api/traces`, { headers: { Origin: 'https://unlisted.example' } });
  assert.equal(r.headers.get('access-control-allow-origin'), null);
  r = await fetch(`${base}/api/traces`, { method: 'OPTIONS', headers: { Origin: 'https://pris-cv.github.io', 'Access-Control-Request-Method': 'GET' } });
  assert.equal(r.status, 204);
  assert.equal((await fetch(`${base}/api/traces`, { method: 'OPTIONS', headers: { Origin: 'https://unlisted.example' } })).status, 403);
  assert.equal((await fetch(`${base}/api/traces`, { method: 'OPTIONS', headers: { Origin: 'https://pris-cv.github.io', 'Access-Control-Request-Method': 'POST' } })).status, 405);
});
test('HEAD has headers but no body; static page and config are usable', async () => {
  const head = await fetch(`${base}/api/traces`, { method: 'HEAD' });
  assert.equal(await head.text(), ''); assert.ok(Number(head.headers.get('content-length')) > 0);
  assert.match(await (await fetch(`${base}/traces/config.js`)).text(), /apiBase: "\/"/);
  assert.equal((await fetch(`${base}/traces/`)).status, 200);
  assert.equal((await fetch(`${base}/traces?lang=zh`, { redirect: 'manual' })).headers.get('location'), '/traces/?lang=zh');
  assert.equal((await fetch(`${base}/traces/build_showcase.py`)).status, 404);
});
test('raw traversal and malformed URLs are rejected before file lookup', async () => {
  for (const path of ['/../package.json', '/%2e%2e/package.json', '/%2eenv', '/api/assets/%2e%2e/private.png', '/bad%00path', '/bad%ZZpath']) {
    const status = await new Promise((resolve, reject) => {
      const req = request(base, { path }, r => { r.resume(); resolve(r.statusCode); }); req.on('error', reject); req.end();
    });
    assert.equal(status, 400, path);
  }
});
test('symlink cannot escape the public root', async () => {
  await mkdir(join(scratch, 'public'));
  await writeFile(join(scratch, 'secret.json'), 'secret');
  await symlink(join(scratch, 'secret.json'), join(scratch, 'public/leak.json'));
  await assert.rejects(containedFile(join(scratch, 'public'), 'leak.json'));
});
test('changed original bytes fail archive admission', async () => {
  const bad = join(scratch, 'bad-archive');
  await cp(join(root, 'web/traces'), bad, { recursive: true });
  await writeFile(join(bad, 'images/grounding-r0.png'), 'modified');
  await assert.rejects(loadArchive(bad), /checksum/);
});
test('static preview does not pretend to have a backend', async () => {
  const preview = await createTraceServer({ staticRoot: join(root, 'web'), apiEnabled: false });
  await new Promise(resolve => preview.listen(0, '127.0.0.1', resolve));
  try { assert.equal((await fetch(`http://127.0.0.1:${preview.address().port}/api/health`)).status, 404); }
  finally { await new Promise(resolve => preview.close(resolve)); }
});
