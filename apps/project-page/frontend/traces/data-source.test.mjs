import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { loadTraces, normalizeApiBase, validateManifest } from './data-source.js';
const snapshot = JSON.parse(await readFile(new URL('./traces.json', import.meta.url), 'utf8'));

test('bundled archive needs no network and is unchanged', async () => {
  const result = await loadTraces({ snapshot, fetchImpl() { throw Error('Unexpected request'); } });
  assert.equal(result.mode, 'archive');
  assert.equal(result.data, snapshot);
});
test('same-origin API maps images while preserving bundled evidence', async () => {
  const result = await loadTraces({ snapshot, apiBase: '/', fetchImpl: async (url, options) => {
    assert.equal(url, '/api/traces');
    assert.equal(options.credentials, 'omit');
    return { ok: true, json: async () => structuredClone(snapshot) };
  }});
  assert.equal(result.mode, 'connected');
  assert.equal(result.data.cases[0].rounds[0].image, '/api/assets/grounding-r0.png');
  assert.equal(snapshot.cases[0].rounds[0].image, 'images/grounding-r0.png');
});
test('HTTPS API uses the configured origin, without query-string overrides', async () => {
  const result = await loadTraces({ snapshot, apiBase: 'https://traces.example.org', fetchImpl: async url => {
    assert.equal(url, 'https://traces.example.org/api/traces');
    return { ok: true, json: async () => structuredClone(snapshot) };
  }});
  assert.equal(result.data.cases[1].rounds[2].image, 'https://traces.example.org/api/assets/count-r2.png');
});
for (const [name, fetchImpl] of [
  ['network failure', async () => { throw Error('offline'); }],
  ['HTTP error', async () => ({ ok: false })],
  ['invalid JSON schema', async () => ({ ok: true, json: async () => ({ cases: [] }) })],
  ['JSON parse error', async () => ({ ok: true, json: async () => { throw Error('parse'); } })],
]) test(`${name} explicitly falls back to local archive`, async () => {
  const result = await loadTraces({ snapshot, apiBase: '/', fetchImpl });
  assert.equal(result.mode, 'fallback');
  assert.equal(result.data, snapshot);
});
test('request timeout is bounded and falls back', async () => {
  const result = await loadTraces({ snapshot, apiBase: '/', timeoutMs: 10, fetchImpl: (_, { signal }) => new Promise((_, reject) => signal.addEventListener('abort', () => reject(Error('aborted')))) });
  assert.equal(result.mode, 'fallback');
});
test('API configuration rejects mixed content, credentials and arbitrary paths', () => {
  for (const url of ['http://public.example.org', 'https://u:p@example.org', 'https://example.org/private', 'javascript:alert(1)', 'https://example.org/?secret=x']) assert.throws(() => normalizeApiBase(url));
  assert.equal(normalizeApiBase('http://127.0.0.1:8787'), 'http://127.0.0.1:8787');
});
test('public contract rejects private fields and untrusted asset references', () => {
  for (const mutate of [x => x.cases[0].rounds[0].image = 'https://other.example/image.png', x => x.cases[0].id = '<script>', x => x.secret = '/aaaidata/private', x => x.cases[0].date = '<img>']) {
    const invalid = structuredClone(snapshot); mutate(invalid);
    assert.throws(() => validateManifest(invalid));
  }
});
