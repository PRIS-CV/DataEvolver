/** Explicit bindings to the public DataEvolver CLI; no process or model work. */

/** @type {Record<string, {argv: string[], required: string[], optional?: string[], campaignRoot?: boolean}>} */
const routes = {
  catalog: { argv: ['operator-sdk', 'catalog'], required: [] },
  'compile-plan': { argv: ['compile-plan'], required: ['plan', 'profile', 'evaluator', 'output', 'campaign'] },
  'worker-compile': { argv: ['worker-compile'], required: ['profile', 'output', 'campaign'], campaignRoot: true },
  create: { argv: ['create'], required: ['goal', 'idempotencyKey'], optional: ['dryRun'], campaignRoot: true },
  step: { argv: ['step'], required: ['campaign'], optional: ['maxActions', 'dryRun'], campaignRoot: true },
  status: { argv: ['status'], required: ['campaign'], campaignRoot: true },
  explain: { argv: ['explain'], required: ['campaign'], optional: ['limit'], campaignRoot: true },
  replay: { argv: ['replay'], required: ['campaign'], campaignRoot: true },
  pause: { argv: ['pause'], required: ['campaign'], campaignRoot: true },
  resume: { argv: ['resume'], required: ['campaign'], optional: ['resumeExecution'], campaignRoot: true },
  operators: { argv: ['operators'], required: ['campaign'], campaignRoot: true },
};

/** @type {Record<string, string>} */
const flags = {
  plan: '--plan', profile: '--profile', evaluator: '--evaluator', output: '--output',
  campaign: '--campaign', goal: '--goal', idempotencyKey: '--idempotency-key',
};

export const operations = Object.freeze(Object.keys(routes));

/** @param {unknown} value @param {string} label @returns {string} */
export function nonemptyText(value, label) {
  if (typeof value !== 'string' || !value.trim() || value.includes('\0')) {
    throw new Error(`${label} must be a non-empty string without NUL bytes`);
  }
  return value;
}

/** @param {unknown} value @param {string} label @param {number} [maximum] @returns {number} */
export function positiveInteger(value, label, maximum = Number.MAX_SAFE_INTEGER) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1 || value > maximum) {
    throw new Error(`${label} must be an integer between 1 and ${maximum}`);
  }
  return value;
}

/**
 * Bind one tool request without interpreting paths, shell text or task state.
 * @param {{operation: string, [key: string]: unknown}} args
 * @param {{campaignRoot: string, maxActions: number}} config
 * @returns {string[]}
 */
export function buildArguments(args, config) {
  if (!Object.hasOwn(routes, args.operation)) throw new Error('Unsupported DataEvolver operation');
  const route = routes[args.operation];
  const allowed = new Set(['operation', ...route.required, ...(route.optional ?? [])]);
  for (const key of Object.keys(args)) {
    if (!allowed.has(key)) throw new Error(`${key} is not valid for ${args.operation}`);
  }
  const argv = [...route.argv];
  for (const field of route.required) {
    argv.push(`${flags[field]}=${nonemptyText(args[field], field)}`);
  }
  if (route.campaignRoot) argv.push(`--campaign-root=${config.campaignRoot}`);
  if (args.operation === 'step') {
    const count = positiveInteger(args.maxActions ?? config.maxActions, 'maxActions', config.maxActions);
    argv.push(`--max-actions=${count}`);
  }
  if (args.limit !== undefined) argv.push(`--limit=${positiveInteger(args.limit, 'limit')}`);
  if (args.dryRun !== undefined) {
    if (typeof args.dryRun !== 'boolean') throw new Error('dryRun must be a boolean');
    if (args.dryRun) argv.push('--dry-run');
  }
  if (args.operation === 'resume' && args.resumeExecution !== true) {
    throw new Error('resume calls Controller.run() immediately. Set resumeExecution=true only for authorized continuous execution.');
  }
  return argv;
}
