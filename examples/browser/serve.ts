/**
 * Minimal Bun dev server with on-the-fly TS bundling for the demo entry.
 *
 * Hits to `/main.ts` are bundled via Bun's transpiler; everything else is
 * served as a static file. Keep dependencies zero (no Vite).
 */

const ROOT = new URL('.', import.meta.url).pathname;
const SRC_ROOT = new URL('../../src/', import.meta.url).pathname;

const TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.ts': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8'
} as Record<string, string>;

const bundleEntry = async (entry: string): Promise<Response> => {
    const result = await Bun.build({
        entrypoints: [entry],
        target: 'browser',
        format: 'esm',
        sourcemap: 'inline',
    });
    if (!result.success) {
        const logs = result.logs.map(l => String(l)).join('\n');
        return new Response('// build failed\n' + logs, {
            status: 500,
            headers: { 'Content-Type': 'text/javascript' }
        });
    }
    const out = result.outputs[0];
    const text = await out.text();
    return new Response(text, {
        headers: { 'Content-Type': 'text/javascript; charset=utf-8' }
    });
};

const port = Number(process.env.PORT ?? 5173);
Bun.serve({
    port,
    async fetch(req) {
        const url = new URL(req.url);
        let path = url.pathname;
        if (path === '/') path = '/index.html';

        // Bundle TS entry points on demand.
        if (path.endsWith('.ts')) {
            const fsPath = ROOT + path.replace(/^\//, '');
            const file = Bun.file(fsPath);
            if (await file.exists()) return bundleEntry(fsPath);
        }

        // Static fallback.
        const fsPath = ROOT + path.replace(/^\//, '');
        const file = Bun.file(fsPath);
        if (await file.exists()) {
            const ext = path.slice(path.lastIndexOf('.'));
            const ct = TYPES[ext] ?? 'application/octet-stream';
            return new Response(file, { headers: { 'Content-Type': ct } });
        }

        return new Response('not found', { status: 404 });
    }
});
console.log(`droplet demo on http://localhost:${port}`);
