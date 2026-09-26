// Minimal static file server for development and the browser tests (WebGPU needs a secure
// context, and localhost counts as one).
//   node tools/serve.mjs [--port 8080] [--mount /models/smollm=/path/to/snapshot]
import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.bin': 'application/octet-stream',
  '.safetensors': 'application/octet-stream',
};

export function serve({ root, port = 0, mounts = {} }) {
  const server = createServer((req, res) => {
    let path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    let base = root;
    for (const [prefix, dir] of Object.entries(mounts)) {
      if (path === prefix || path.startsWith(prefix + '/')) {
        base = dir;
        path = path.slice(prefix.length) || '/';
        break;
      }
    }
    if (path.endsWith('/')) path += 'index.html';
    const file = join(base, normalize(path).replace(/^([/\\])+/, ''));
    if (!file.startsWith(resolve(base))) {
      res.writeHead(403).end();
      return;
    }
    let st;
    try {
      st = statSync(file);
    } catch {
      res.writeHead(404).end('not found');
      return;
    }
    res.writeHead(200, {
      'content-type': TYPES[extname(file)] || 'application/octet-stream',
      'content-length': st.size,
      'cache-control': 'no-store',
    });
    if (req.method === 'HEAD') res.end();
    else createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(port, '127.0.0.1', () => ok(server)));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  let port = 8080;
  const mounts = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--port') port = Number(args[++i]);
    else if (args[i] === '--mount') {
      const [prefix, dir] = args[++i].split('=');
      mounts[prefix] = resolve(dir);
    }
  }
  const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
  const server = await serve({ root, port, mounts });
  console.log(`serving ${root} at http://localhost:${server.address().port}/`);
}
