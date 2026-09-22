import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import data from './model-cases-data.js';
import { renderModelCases } from './model-cases.js';

test('model comparison has six original frames and matching public JSON', async () => {
  const json = JSON.parse(await readFile(new URL('./model-cases.json', import.meta.url)));
  assert.deepEqual(data, json);
  assert.equal(data.images.length, 6);
  for (const frame of data.images) {
    assert.match(frame.image, /^images\/models-medium-[a-z-]+\.(png|jpg)$/);
    const bytes = await readFile(new URL(frame.image, import.meta.url));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), frame.sha256);
    assert.equal(frame.transform, 'byte-identical copy');
  }
});
test('comparison keeps generator boundaries, blocked outcome and provenance limits', () => {
  assert.equal(data.findings.experiment_status, 'blocked');
  assert.equal(data.findings.vlm_used, false);
  assert.equal(data.kind, 'historical_configuration_comparison_not_continuous_agent_trace');
  assert.equal(data.images.find(i => i.id === 'flux-after').evidence_level, 'archived_report_image_hash');
  for (const forbidden of ['/aaaidata/', '/data/', '/home/', '/Users/', 'raw_response', '<think>', 'api_key']) assert.ok(!JSON.stringify(data).includes(forbidden));
  for (const zh of [true, false]) {
    const html = renderModelCases(zh);
    assert.equal((html.match(/loading="lazy"/g) || []).length, 6);
    assert.match(html, /model-cases.json/);
    assert.match(html, zh ? /不能把它们串成同一模型/ : /different generators/);
  }
});
