import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { readFile, realpath, stat } from 'node:fs/promises';
import { resolve, relative, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateManifest } from '../../web/traces/data-source.js';

const repository = fileURLToPath(new URL('../../', import.meta.url));
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml', '.webp': 'image/webp', '.avif': 'image/avif', '.ico': 'image/x-icon', '.gif': 'image/gif', '.woff': 'font/woff', '.woff2': 'font/woff2', '.mp4': 'video/mp4', '.webm': 'video/webm', '.pdf': 'application/pdf', '.txt': 'text/plain; charset=utf-8' };
const digest = buffer => createHash('sha256').update(buffer).digest('hex');

export async function containedFile(root, name) {
  const resolvedRoot = await realpath(root);
  const target = await realpath(resolve(resolvedRoot, name));
  const path = relative(resolvedRoot, target);
  if (!path || path === '..' || path.startsWith(`..${sep}`) || resolve(resolvedRoot, path) !== target) throw new Error('Path outside public root');
  if (!(await stat(target)).isFile()) throw new Error('Not a public file');
  return target;
}

export async function loadArchive(dataRoot) {
  const raw = await readFile(await containedFile(dataRoot, 'traces.json'));
  const manifest = validateManifest(JSON.parse(raw));
  const images = new Map();
  for (const c of manifest.cases) {
    for (const r of c.rounds) {
      const buffer = await readFile(await containedFile(dataRoot, r.image));
      const record = c.provenance.find(p => p.file === r.image.slice(7) && p.transform === 'byte-identical copy');
      if (!record || digest(buffer) !== record.sha256) throw new Error('Archive image checksum mismatch');
      images.set(r.image.slice(7), buffer);
    }
  }
  return { manifest, images, revision: digest(raw) };
}

export async function createTraceServer({
  dataRoot = resolve(repository, 'web/traces'),
  staticRoot = resolve(repository, 'dist'),
  allowedOrigins = [], apiEnabled = true, dev = false,
} = {}) {
  const archive = apiEnabled ? await loadArchive(dataRoot) : null;
  const origins = new Set(allowedOrigins);
  if (origins.has('*')) throw new Error('Use explicit allowed origins, not wildcard CORS');
  for (const origin of origins) if (new URL(origin).origin !== origin) throw new Error('Invalid CORS origin');

  const server = createServer(async (req, res) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    res.setHeader('X-Frame-Options', 'SAMEORIGIN');
    const send = (status, body, type = 'application/json; charset=utf-8', cache = 'no-store') => {
      const bytes = Buffer.isBuffer(body) ? body : Buffer.from(typeof body === 'string' ? body : JSON.stringify(body));
      res.writeHead(status, { 'Content-Type': type, 'Content-Length': bytes.length, 'Cache-Control': cache });
      res.end(req.method === 'HEAD' ? undefined : bytes);
    };
    let pathname;
    try {
      const rawPath = decodeURIComponent((req.url || '/').split('?')[0]);
      if (rawPath.includes('\\') || rawPath.includes('\0') || rawPath.split('/').some(s => s.startsWith('.'))) return send(400, { error: 'Invalid path' });
      pathname = new URL(req.url, 'http://localhost').pathname;
    } catch { return send(400, { error: 'Invalid URL' }); }
    const api = pathname.startsWith('/api/');
    const origin = req.headers.origin;
    if (api) {
      res.setHeader('Vary', 'Origin');
      if (origin && origins.has(origin)) res.setHeader('Access-Control-Allow-Origin', origin);
      // Non-listed origins may read via same-origin navigation, but never receive
      // cross-origin access. No credentials, write endpoints, or private records.
    }
    if (req.method === 'OPTIONS') {
      if (!api || !apiEnabled) return send(404, { error: 'Not found' });
      if (!origin || !origins.has(origin)) return send(403, { error: 'Origin not allowed' });
      if (req.headers['access-control-request-method'] && !['GET', 'HEAD'].includes(req.headers['access-control-request-method'])) return send(405, { error: 'Read-only API' });
      res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
      res.setHeader('Access-Control-Max-Age', '600');
      return send(204, '');
    }
    if (!['GET', 'HEAD'].includes(req.method)) {
      res.setHeader('Allow', 'GET, HEAD, OPTIONS');
      return send(405, { error: 'Read-only service' });
    }
    try {
      if (api) {
        if (!apiEnabled) return send(404, { error: 'API not available in static preview' });
        if (pathname === '/api/health') return send(200, { status: 'ok', mode: 'curated-archive-read-only', cases: archive.manifest.cases.length, revision: archive.revision });
        if (pathname === '/api/traces') return send(200, archive.manifest);
        const match = pathname.match(/^\/api\/traces\/([a-z0-9-]+)(?:\/rounds\/(0|[1-9]\d*))?$/);
        if (match) {
          const c = archive.manifest.cases.find(x => x.id === match[1]);
          const value = match[2] === undefined ? c : c?.rounds.find(r => r.round === Number(match[2]));
          return value ? send(200, value) : send(404, { error: 'Trace or round not found' });
        }
        const image = pathname.match(/^\/api\/assets\/([a-z0-9-]+\.png)$/);
        if (image && archive.images.has(image[1])) return send(200, archive.images.get(image[1]), 'image/png', 'public, max-age=300');
        return send(404, { error: 'Not found' });
      }
      if (!staticRoot) return send(404, { error: 'API-only deployment' });
      if (pathname === '/traces') {
        res.writeHead(308, { Location: '/traces/' + new URL(req.url, 'http://localhost').search });
        return res.end();
      }
      if (dev && pathname === '/traces/config.js') return send(200, 'window.TRACE_CONFIG = { apiBase: "/" };', mime['.js']);
      const name = pathname.endsWith('/') ? `${pathname.slice(1)}index.html` : pathname.slice(1);
      if (!Object.hasOwn(mime, extname(name))) return send(404, { error: 'Not found' });
      const file = await containedFile(staticRoot, name);
      return send(200, await readFile(file), mime[extname(file)]);
    } catch (error) {
      if (['ENOENT', 'ENOTDIR', 'EACCES'].includes(error.code) || error.message === 'Path outside public root' || error.message === 'Not a public file') return send(404, { error: 'Not found' });
      console.error('Trace request failed:', error.message);
      return send(500, { error: 'Unable to read public archive' });
    }
  });
  server.requestTimeout = 15000;
  server.headersTimeout = 10000;
  server.keepAliveTimeout = 5000;
  return server;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const dev = process.argv.includes('--dev');
  const onlyApi = process.argv.includes('--api-only');
  const staticOnly = process.argv.includes('--static-only');
  const host = process.env.HOST || '127.0.0.1';
  const port = Number(process.env.PORT || (dev ? 4173 : 8787));
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid PORT');
  const server = await createTraceServer({
    dataRoot: process.env.TRACE_DATA_DIR || resolve(repository, 'web/traces'),
    staticRoot: onlyApi ? null : resolve(repository, dev ? 'web' : 'dist'),
    allowedOrigins: (process.env.TRACE_ALLOWED_ORIGINS || '').split(',').map(x => x.trim()).filter(Boolean),
    apiEnabled: !staticOnly, dev,
  });
  server.listen(port, host, () => console.log(`DataEvolver ${staticOnly ? 'static preview' : 'read-only trace service'}: http://${host}:${port}${onlyApi ? '/api/health' : '/traces/'}`));
  for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => {
    server.close(() => process.exit(0));
    setTimeout(() => { server.closeAllConnections(); process.exit(0); }, 5000).unref();
  });
}
