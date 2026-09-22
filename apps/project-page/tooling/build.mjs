import { cp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { realpathSync } from 'node:fs';
import { resolve, extname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { createHash } from 'node:crypto';
import { normalizeApiBase, validateManifest } from '../frontend/traces/data-source.js';

export const root = fileURLToPath(new URL('../', import.meta.url));
const extensions = new Set(['.html', '.css', '.js', '.json', '.png', '.jpg', '.jpeg', '.webp', '.avif', '.svg', '.gif', '.ico', '.mp4', '.webm', '.woff', '.woff2', '.pdf', '.txt']);

export function publicAsset(path, directory = false) {
  if (basename(path).startsWith('.')) return false;
  return directory || (extensions.has(extname(path).toLowerCase()) && !path.endsWith('.test.js'));
}

export async function build({ apiBase = process.env.TRACE_API_BASE_URL || '' } = {}) {
  normalizeApiBase(apiBase); // Validate before replacing an earlier build.
  const manifest = validateManifest(JSON.parse(await readFile(resolve(root, 'frontend/traces/traces.json'), 'utf8')));
  const script = await readFile(resolve(root, 'frontend/traces/traces-data.js'), 'utf8');
  const embedded = JSON.parse(script.replace(/^window\.TRACE_SHOWCASE = /, '').trim().replace(/;$/, ''));
  if (!isDeepStrictEqual(manifest, embedded)) throw new Error('Embedded trace data differs from JSON archive');
  for (const c of manifest.cases) {
    const evidence = JSON.parse(await readFile(resolve(root, `frontend/traces/evidence/${c.id}.json`), 'utf8'));
    if (!isDeepStrictEqual(c, evidence)) throw new Error('Evidence download differs from the displayed archive');
    for (const r of c.rounds) {
      const bytes = await readFile(resolve(root, 'frontend/traces', r.image));
      const expected = c.provenance.find(p => p.file === r.image.slice(7) && p.transform === 'byte-identical copy');
      if (!expected || createHash('sha256').update(bytes).digest('hex') !== expected.sha256) throw new Error('Original image checksum mismatch');
    }
  }
  const comparison = JSON.parse(await readFile(resolve(root, 'frontend/traces/model-cases.json'), 'utf8'));
  const moduleText = await readFile(resolve(root, 'frontend/traces/model-cases-data.js'), 'utf8');
  const comparisonModule = JSON.parse(moduleText.replace(/^export default /, '').trim().replace(/;$/, ''));
  if (!isDeepStrictEqual(comparison, comparisonModule)) throw new Error('Model comparison differs from public JSON');
  if (comparison.schema_version !== 'dataevolver.public_model_comparison.v1' || comparison.images?.length !== 6) throw new Error('Invalid model comparison');
  for (const frame of comparison.images) {
    if (!/^images\/models-medium-[a-z-]+\.(png|jpg)$/.test(frame.image)) throw new Error('Invalid comparison image path');
    const bytes = await readFile(resolve(root, 'frontend/traces', frame.image));
    if (createHash('sha256').update(bytes).digest('hex') !== frame.sha256) throw new Error('Model comparison image checksum mismatch');
  }
  const source = resolve(root, 'frontend'), destination = resolve(root, 'dist');
  // Only the fixed, generated output directory is replaced; source data is untouched.
  await rm(destination, { recursive: true, force: true });
  async function copyDirectory(from, to) {
    await mkdir(to, { recursive: true });
    for (const entry of await readdir(from, { withFileTypes: true })) {
      if (entry.isSymbolicLink()) throw new Error(`Symlinks are not published: ${entry.name}`);
      if (!publicAsset(entry.name, entry.isDirectory())) continue;
      if (entry.isDirectory()) await copyDirectory(resolve(from, entry.name), resolve(to, entry.name));
      else if (entry.isFile()) await cp(resolve(from, entry.name), resolve(to, entry.name));
    }
  }
  await copyDirectory(source, destination);
  await writeFile(resolve(destination, 'traces/config.js'), `window.TRACE_CONFIG = ${JSON.stringify({ apiBase }).replaceAll('<', '\\u003c')};\n`);
  await writeFile(resolve(destination, '.nojekyll'), '');
  return destination;
}

if (process.argv[1] && realpathSync(resolve(process.argv[1])) === realpathSync(fileURLToPath(import.meta.url))) {
  console.log(`Project Page built: ${await build()}`);
}
