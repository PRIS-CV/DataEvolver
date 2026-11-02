import { posix, win32 } from 'node:path';
import Schema from '@deepseek-ai/schemastery';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { buildArguments, nonemptyText, operations, positiveInteger } from './commands.mjs';

/** @typedef {import('@deepseek-ai/cordis').Context & {
 * subprocess: import('@deepseek-ai/dsh-subprocess').SubprocessRuntime,
 * sandbox: import('@deepseek-ai/dsh-sandbox').SandboxProvider,
 * sandboxPolicy: import('@deepseek-ai/dsh-sandbox-policy').SandboxPolicyService
 * }} PluginContext */
/** @typedef {{command: string[], workingDirectory: string, campaignRoot: string,
 * maxActions: number, timeoutMs: number, graceMs: number, maxOutputBytes: number,
 * forwardEnv: string[]}} PluginConfig */

export const name = 'dataevolver';
export const inject = ['tools', 'subprocess', 'sandbox', 'sandboxPolicy'];

/** @type {Schema<PluginConfig>} */
export const Config = Schema.object({
  command: Schema.array(Schema.string()).default(['dataevolver']),
  workingDirectory: Schema.string().default(''),
  campaignRoot: Schema.string().default('.dataevolver/campaigns'),
  maxActions: Schema.number().default(1),
  timeoutMs: Schema.number().default(300000),
  graceMs: Schema.number().default(5000),
  maxOutputBytes: Schema.number().default(262144),
  forwardEnv: Schema.array(Schema.string()).default([]),
});

const streamSchema = /** @type {const} */ ({
  type: 'object', additionalProperties: false,
  properties: {
    text: { type: 'string', required: true },
    truncated: { type: 'boolean', required: true },
    spillPath: { type: 'string' },
  },
});

/** @param {PluginContext} ctx @param {PluginConfig} config */
export function apply(ctx, config) {
  if (!config.command.length) throw new Error('command must contain an executable');
  config.command.forEach((part) => nonemptyText(part, 'command argument'));
  nonemptyText(config.campaignRoot, 'campaignRoot');
  for (const key of /** @type {const} */ (['maxActions', 'timeoutMs', 'graceMs', 'maxOutputBytes'])) {
    positiveInteger(config[key], key, 2147483647);
  }
  if (config.workingDirectory) {
    nonemptyText(config.workingDirectory, 'workingDirectory');
    if (!posix.isAbsolute(config.workingDirectory) && !win32.isAbsolute(config.workingDirectory)) {
      throw new Error('workingDirectory must be absolute in the subprocess execution world');
    }
  }
  for (const key of config.forwardEnv) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) throw new Error('forwardEnv must contain environment variable names');
  }

  const lifetime = new AbortController();
  /** @type {Set<import('@deepseek-ai/dsh-subprocess').SubprocessHandle>} */
  const active = new Set();
  ctx.effect(() => async () => {
    lifetime.abort(new Error('DataEvolver plugin unloaded; inspect campaign state before continuing'));
    await Promise.allSettled([...active].flatMap((handle) => [handle.done, handle.waitForExit()]));
  });

  ctx.tools.register(defineTool({
    name: 'dataevolver_harness',
    description: 'Compile, execute and inspect DataEvolver image and 3D data synthesis campaigns. '
      + 'catalog lists registered operators; compile-plan binds a data graph; worker-compile binds a native worker profile. '
      + 'create requires a goal and idempotencyKey. step advances bounded Controller actions, not image counts or seconds. '
      + 'resume immediately calls Controller.run(); use it only for authorized continuous execution and set resumeExecution=true. '
      + 'Paths belong to the configured subprocess execution world. Inspect status/explain after interruption; never blindly retry execution.',
    parameters: {
      operation: { type: 'string', enum: operations, required: true },
      plan: { type: 'string', description: 'Plan JSON path for compile-plan.' },
      profile: { type: 'string', description: 'Binding profile for compile-plan, or native worker profile for worker-compile.' },
      evaluator: { type: 'string', description: 'Evaluator JSON path for compile-plan.' },
      output: { type: 'string', description: 'New prepared output directory for either compiler.' },
      campaign: { type: 'string', description: 'Campaign ID; required except for catalog and create.' },
      goal: { type: 'string', description: 'Compiled goal.json path; required for create.' },
      idempotencyKey: { type: 'string', description: 'Stable creation identity; required for create.' },
      maxActions: { type: 'integer', description: 'step only: positive count, bounded by the deployment maxActions.' },
      limit: { type: 'integer', description: 'explain only: positive event count.' },
      dryRun: { type: 'boolean', description: 'create/step only: pass the existing CLI dry-run flag.' },
      resumeExecution: { type: 'boolean', description: 'resume only: true acknowledges immediate continuous execution; it does not grant user authorization.' },
    },
    output: {
      schema: {
        type: 'object', additionalProperties: false,
        properties: {
          operation: { type: 'string', required: true },
          cwd: { type: 'string', required: true },
          exitCode: { oneOf: [{ type: 'integer' }, { type: 'null' }], required: true },
          signal: { oneOf: [{ type: 'string' }, { type: 'null' }], required: true },
          timedOut: { type: 'boolean', required: true },
          stdout: { ...streamSchema, required: true },
          stderr: { ...streamSchema, required: true },
          receipt: { type: 'json', required: true },
        },
      },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
    },
    async execute(args, exec) {
      const commandArgs = buildArguments(args, config);
      const session = exec.agent?.session;
      const policy = ctx.sandboxPolicy.resolve({ session });
      const cwd = config.workingDirectory || session?.header.cwd || policy.workspaceRoot;
      const deadline = AbortSignal.timeout(config.timeoutMs);
      const signal = AbortSignal.any([exec.signal, lifetime.signal, deadline]);
      signal.throwIfAborted();
      /** @type {Record<string, string>} */
      const env = {};
      for (const key of config.forwardEnv) {
        const value = process.env[key];
        if (value === undefined) throw new Error(`Configured environment variable is unset: ${key}`);
        env[key] = value;
      }
      const executable = await ctx.subprocess.resolveExecutable(config.command[0], env, signal);
      let argv = [executable, ...config.command.slice(1), ...commandArgs];
      if (policy.mode !== 'danger-full-access') {
        const confined = await ctx.sandbox.confine(argv, { ...policy, mode: policy.mode }, signal);
        argv = confined.argv;
      }
      signal.throwIfAborted();
      const handle = ctx.subprocess.spawn({
        argv, cwd, env, graceMs: config.graceMs, signal,
        stdio: {
          stdin: 'ignore',
          stdout: { maxBytes: config.maxOutputBytes },
          stderr: { maxBytes: config.maxOutputBytes },
        },
      });
      active.add(handle);
      try {
        const outcome = await handle.done;
        if (signal.aborted) await handle.waitForExit();
        exec.signal.throwIfAborted();
        lifetime.signal.throwIfAborted();
        /** @param {import('@deepseek-ai/dsh-subprocess').SubprocessOutputReader | undefined} reader */
        const output = (reader) => {
          if (!reader) throw new Error('Subprocess provider did not return its configured output reader');
          const stream = reader.readFrom(0);
          return { text: stream.text, truncated: stream.lossy,
            ...(stream.spillPath !== undefined ? { spillPath: stream.spillPath } : {}) };
        };
        const stdout = output(handle.collected.stdout);
        const stderr = output(handle.collected.stderr);
        let receipt = null;
        if (!stdout.truncated) {
          try { receipt = JSON.parse(stdout.text); } catch { /* Raw stdout remains available for non-JSON output. */ }
        }
        return { operation: args.operation, cwd, ...outcome, timedOut: deadline.aborted, stdout, stderr, receipt };
      } finally {
        active.delete(handle);
      }
    },
  }));
}
