import test from 'node:test';
import assert from 'node:assert/strict';
import { buildArguments } from '../commands.mjs';

const config = { campaignRoot: 'campaigns with spaces', maxActions: 2 };

test('rejects unsupported commands and fields before any dispatch', () => {
  for (const operation of ['evolve', 'toString', '__proto__']) {
    assert.throws(() => buildArguments({ operation }, config), /Unsupported/);
  }
  assert.throws(() => buildArguments({ operation: 'status', campaign: 'task', command: 'anything' }, config), /not valid/);
  assert.throws(() => buildArguments({ operation: 'step', campaign: 'task', goal: 'implicit-create.json' }, config), /not valid/);
});

test('keeps creation identity and shell-like paths as literal argv values', () => {
  assert.throws(() => buildArguments({ operation: 'create', goal: 'goal.json' }, config), /idempotencyKey/);
  const goal = 'prepared/$(not-a-command) "goal".json';
  assert.deepEqual(buildArguments({ operation: 'create', goal, idempotencyKey: 'stable-id' }, config), [
    'create', `--goal=${goal}`, '--idempotency-key=stable-id', '--campaign-root=campaigns with spaces',
  ]);
});

test('bounds step actions by the deployment limit', () => {
  for (const maxActions of [0, -1, 1.5, 3, NaN]) {
    assert.throws(() => buildArguments({ operation: 'step', campaign: 'task', maxActions }, config), /maxActions/);
  }
  assert.deepEqual(buildArguments({ operation: 'step', campaign: 'task', maxActions: 1 }, config), [
    'step', '--campaign=task', '--campaign-root=campaigns with spaces', '--max-actions=1',
  ]);
});

test('requires explicit acknowledgement for continuous resume', () => {
  assert.throws(() => buildArguments({ operation: 'resume', campaign: 'task' }, config), /resumeExecution/);
  assert.deepEqual(buildArguments({ operation: 'resume', campaign: 'task', resumeExecution: true }, config), [
    'resume', '--campaign=task', '--campaign-root=campaigns with spaces',
  ]);
});

test('requires a complete native worker compilation request', () => {
  assert.throws(() => buildArguments({ operation: 'worker-compile', profile: 'worker.json', campaign: 'task' }, config), /output/);
  assert.deepEqual(buildArguments({ operation: 'worker-compile', profile: 'worker.json', output: 'prepared', campaign: 'task' }, config), [
    'worker-compile', '--profile=worker.json', '--output=prepared', '--campaign=task', '--campaign-root=campaigns with spaces',
  ]);
});
