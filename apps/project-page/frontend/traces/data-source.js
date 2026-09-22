/** Public archive contract shared by the browser and the read-only API. */
export function validateManifest(data) {
  if (data?.schema_version !== 'dataevolver.public_trace.v1' || !Array.isArray(data.cases)) throw new Error('Unsupported trace archive');
  const expected = { grounding: 2, count: 4 };
  if (data.cases.length !== 2 || new Set(data.cases.map(c => c.id)).size !== 2) throw new Error('Unexpected case set');
  for (const c of data.cases) {
    if (!Object.hasOwn(expected, c.id) || c.rounds?.length !== expected[c.id] || !/^\d{4}-\d{2}-\d{2}$/.test(c.date)) throw new Error('Invalid case');
    if (typeof c.source_run !== 'string' || !Array.isArray(c.provenance)) throw new Error('Missing provenance');
    for (const [i, r] of c.rounds.entries()) {
      if (r.round !== i || r.image !== `images/${c.id}-r${i}.png`) throw new Error('Invalid round or image path');
      if (c.id === 'grounding') {
        if (!Number.isFinite(r.hybrid_score) || !Array.isArray(r.camera?.location)) throw new Error('Missing grounding review');
      } else {
        if (!['failed_constraints', 'passed_constraints', 'uncertain_constraints'].every(k => Array.isArray(r[k]) && r[k].every(x => typeof x === 'string'))) throw new Error('Missing constraint review');
        if (typeof r.prompt !== 'string' || typeof r.rewrite_reason !== 'string') throw new Error('Missing prompt');
      }
    }
    if (c.id === 'grounding') {
      const evidence = c.execution_evidence?.b?.[0];
      if (!evidence || !['bottom_z_before', 'bottom_z_after', 'ground_z'].every(k => Number.isFinite(evidence[k])) || !Array.isArray(c.parameter_diff)) throw new Error('Missing execution evidence');
    }
  }
  // This contract is deliberately scoped to the curated gallery, not arbitrary
  // private Harness events. New case kinds need a reviewed renderer and export.
  const text = JSON.stringify(data);
  for (const forbidden of ['/aaaidata/', '/home/', '/Users/', 'raw_response', '<think>', 'api_key']) {
    if (text.includes(forbidden)) throw new Error('Non-public fields in archive');
  }
  return data;
}

export function normalizeApiBase(value) {
  if (!value) return null;
  if (value === '/') return '';
  const url = new URL(value);
  const local = ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname);
  if ((url.protocol !== 'https:' && !(url.protocol === 'http:' && local)) || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('API base must be an HTTPS origin, localhost origin, or /');
  }
  return url.origin;
}

export async function loadTraces({ snapshot, apiBase = '', fetchImpl = globalThis.fetch, timeoutMs = 5000 }) {
  validateManifest(snapshot);
  if (!apiBase) return { data: snapshot, mode: 'archive', base: null };
  const abort = new AbortController();
  const timeout = setTimeout(() => abort.abort(), timeoutMs);
  try {
    const base = normalizeApiBase(apiBase);
    const response = await fetchImpl(`${base}/api/traces`, { signal: abort.signal, credentials: 'omit', cache: 'no-store' });
    if (!response.ok) throw new Error('Trace service unavailable');
    const data = validateManifest(await response.json());
    for (const c of data.cases) for (const r of c.rounds) r.image = `${base}/api/assets/${r.image.slice(7)}`;
    return { data, mode: 'connected', base };
  } catch (_) {
    return { data: snapshot, mode: 'fallback', base: null };
  } finally {
    clearTimeout(timeout);
  }
}
