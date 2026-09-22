import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, access } from 'node:fs/promises';
import { resolve } from 'node:path';
import { build, root, publicAsset } from './build.mjs';

test('build contains the homepage, subpage, six original frames and runtime config', async () => {
  const destination = await build({ apiBase: '/' });
  for (const file of ['index.html', 'index_zh.html', 'traces/index.html', 'traces/trace.js', 'traces/data-source.js', 'traces/traces.json', 'traces/evidence/grounding.json', 'traces/evidence/count.json']) await access(resolve(destination, file));
  for (const [id, n] of [['grounding', 2], ['count', 4]]) for (let i = 0; i < n; i++) {
    const image = `traces/images/${id}-r${i}.png`;
    assert.deepEqual(await readFile(resolve(destination, image)), await readFile(resolve(root, 'web', image)));
  }
  assert.match(await readFile(resolve(destination, 'traces/config.js'), 'utf8'), /"apiBase":"\/"/);
  await assert.rejects(access(resolve(destination, 'traces/build_showcase.py')));
  await assert.rejects(access(resolve(destination, 'traces/data-source.test.mjs')));
  await build({ apiBase: '' }); // Leave the default Pages-ready static build.
});
test('build rejects invalid API config before touching output', async () => {
  await assert.rejects(build({ apiBase: 'http://untrusted.example' }));
});
test('publish filter excludes source, hidden files and tests', () => {
  for (const file of ['.env', 'build.py', 'server.mjs', 'README.md', 'example.test.js']) assert.equal(publicAsset(file), false);
  for (const file of ['index.html', 'data.json', 'trace.css', 'image.png']) assert.equal(publicAsset(file), true);
});
