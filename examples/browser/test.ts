/**
 * Intensive test suite for droplet's renderer + voxelizer.
 *
 * Runs in the browser against a real WebGPU device. Each test reports
 * PASS/FAIL with a one-line diagnostic. Tests are ordered cheap-to-expensive
 * so failures surface fast.
 */

import {
    createDevice,
    Renderer,
    voxelize,
    mat4LookAt,
    type SplatData,
    type Vec3,
} from '../../src/index.ts';

const results = document.getElementById('results')!;

const print = (msg: string, cls = 'info') => {
    const span = document.createElement('span');
    span.className = cls;
    span.textContent = msg + '\n';
    results.appendChild(span);
};

interface TestCase {
    name: string;
    slow?: boolean;
    fn: (ctx: TestCtx) => Promise<void>;
}

interface TestCtx {
    device: GPUDevice;
    renderer: Renderer;
    expect(cond: boolean, msg: string): void;
    expectClose(a: number, b: number, eps: number, msg: string): void;
}

const tests: TestCase[] = [];

const test = (name: string, fn: TestCase['fn'], slow = false) => tests.push({ name, fn, slow });

// ---------- helpers ----------
const oneSplat = (
    pos: Vec3, color: [number, number, number] = [1, 0, 0],
    logScale = -2.5, opLogit = 3
): SplatData => ({
    count: 1,
    positions: new Float32Array(pos),
    rotations: new Float32Array([1, 0, 0, 0]),
    logScales: new Float32Array([logScale, logScale, logScale]),
    opacityLogits: new Float32Array([opLogit]),
    colorsDC: new Float32Array([color[0] - 0.5, color[1] - 0.5, color[2] - 0.5]),
    shBands: 0,
});

const manySplats = (n: number, layout: 'sphere' | 'random' | 'line' = 'sphere'): SplatData => {
    const positions = new Float32Array(n * 3);
    const rotations = new Float32Array(n * 4);
    const logScales = new Float32Array(n * 3);
    const opacityLogits = new Float32Array(n);
    const colorsDC = new Float32Array(n * 3);
    let seed = 1;
    const rand = () => { seed = (seed * 1664525 + 1013904223) | 0; return ((seed >>> 0) % 10000) / 10000; };
    for (let i = 0; i < n; i++) {
        if (layout === 'sphere') {
            const t = i / n;
            const phi = Math.acos(1 - 2 * t);
            const theta = Math.PI * (1 + Math.sqrt(5)) * i;
            positions[i * 3 + 0] = Math.cos(theta) * Math.sin(phi);
            positions[i * 3 + 1] = Math.sin(theta) * Math.sin(phi);
            positions[i * 3 + 2] = Math.cos(phi);
        } else if (layout === 'random') {
            positions[i * 3 + 0] = (rand() - 0.5) * 4;
            positions[i * 3 + 1] = (rand() - 0.5) * 4;
            positions[i * 3 + 2] = (rand() - 0.5) * 4;
        } else { // line along z
            positions[i * 3 + 0] = 0;
            positions[i * 3 + 1] = 0;
            positions[i * 3 + 2] = (i / n) * 2 - 1;
        }
        rotations[i * 4 + 0] = 1;
        logScales[i * 3 + 0] = -3.2;
        logScales[i * 3 + 1] = -3.2;
        logScales[i * 3 + 2] = -3.2;
        opacityLogits[i] = 2.5;
        colorsDC[i * 3 + 0] = (rand() - 0.5);
        colorsDC[i * 3 + 1] = (rand() - 0.5);
        colorsDC[i * 3 + 2] = (rand() - 0.5);
    }
    return { count: n, positions, rotations, logScales, opacityLogits, colorsDC, shBands: 0 };
};

const renderFront = async (
    renderer: Renderer, data: SplatData, w = 128, h = 128
): Promise<Uint8Array> => {
    const view = mat4LookAt([0, 0, 4], [0, 0, 0], [0, 1, 0]);
    return renderer.render(data, {
        width: w, height: h, viewMatrix: view,
        fovY: Math.PI / 3, near: 0.01, far: 1000
    }, { r: 0, g: 0, b: 0, a: 1 });
};

const nonBlackCount = (rgba: Uint8Array): number => {
    let n = 0;
    for (let i = 0; i < rgba.length; i += 4) {
        if (rgba[i] > 5 || rgba[i + 1] > 5 || rgba[i + 2] > 5) n++;
    }
    return n;
};

// ---------- tests ----------
test('empty scene → all black', async ({ renderer, expect }) => {
    const empty: SplatData = {
        count: 0,
        positions: new Float32Array(0), rotations: new Float32Array(0),
        logScales: new Float32Array(0), opacityLogits: new Float32Array(0),
        colorsDC: new Float32Array(0), shBands: 0
    };
    const rgba = await renderFront(renderer, empty);
    const nb = nonBlackCount(rgba);
    expect(nb === 0, `non-black pixels = ${nb}, expected 0`);
});

test('one splat at origin renders symmetrically', async ({ renderer, expect }) => {
    const data = oneSplat([0, 0, 0]);
    const rgba = await renderFront(renderer, data, 128, 128);
    // sample center vs ±10 in x and y — should be near-symmetric
    const at = (x: number, y: number) => {
        const i = (y * 128 + x) * 4;
        return rgba[i];
    };
    const c = at(64, 64);
    const l = at(54, 64), r = at(74, 64);
    const u = at(64, 54), d = at(64, 74);
    expect(c > 100, `center R=${c}, expected > 100`);
    expect(Math.abs(l - r) < 10, `left=${l} right=${r} should be ~equal`);
    expect(Math.abs(u - d) < 10, `up=${u} down=${d} should be ~equal`);
});

test('three splats: red/green/blue at distinct positions', async ({ renderer, expect }) => {
    const data: SplatData = {
        count: 3,
        positions: new Float32Array([0, 0, 1, 0.5, 0, 0.8, -0.5, 0.3, 0.8]),
        rotations: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
        logScales: new Float32Array([-2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5]),
        opacityLogits: new Float32Array([3, 3, 3]),
        colorsDC: new Float32Array([0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5]),
        shBands: 0,
    };
    const rgba = await renderFront(renderer, data, 256, 256);
    // Each splat should render in its own region. Sample dominant colors.
    let rDom = 0, gDom = 0, bDom = 0;
    for (let i = 0; i < rgba.length; i += 4) {
        const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
        if (r > 50 && r > g && r > b) rDom++;
        if (g > 50 && g > r && g > b) gDom++;
        if (b > 50 && b > r && b > g) bDom++;
    }
    expect(rDom > 50 && gDom > 50 && bDom > 50,
        `r-dom=${rDom}, g-dom=${gDom}, b-dom=${bDom} (each should be > 50)`);
});

test('depth sort: front splat occludes back splat at same screen pos', async ({ renderer, expect }) => {
    // Red in front of green at the same x,y.
    const data: SplatData = {
        count: 2,
        positions: new Float32Array([0, 0, 1, 0, 0, -1]),
        rotations: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0]),
        logScales: new Float32Array([-2.5, -2.5, -2.5, -2.5, -2.5, -2.5]),
        opacityLogits: new Float32Array([4, 4]),
        colorsDC: new Float32Array([0.5, -0.5, -0.5, -0.5, 0.5, -0.5]),
        shBands: 0,
    };
    const rgba = await renderFront(renderer, data, 128, 128);
    const i = (64 * 128 + 64) * 4;
    const r = rgba[i], g = rgba[i + 1];
    expect(r > 200 && g < 50, `center R=${r} G=${g} — red should win, green hidden`);
});

test('5000 splats sphere renders dense disc', async ({ renderer, expect }) => {
    const data = manySplats(5000, 'sphere');
    const rgba = await renderFront(renderer, data, 256, 256);
    const nb = nonBlackCount(rgba);
    // disc area ≈ π·r² where r ≈ focal/distance = 221/3 = ~73. area ≈ 17000.
    // Allow factor of 2 either way.
    expect(nb > 10000 && nb < 30000,
        `non-black pixels = ${nb} (expected ~10000-30000 for full sphere disc)`);
});

test('chunk boundary: rendering 200K splats matches 100K twice', async ({ renderer, expect }) => {
    // Build 200K random splats in a tight cluster so EVERY pixel sees many.
    // Compare: render with default chunkCap (single chunk) vs forced multi-chunk.
    // Skip if not enough memory.
    const n = 200_000;
    const data = manySplats(n, 'random');
    const single = await renderFront(renderer, data, 64, 64);

    // Re-render forcing 4 chunks (50K cap).
    const multi = new Renderer((renderer as any).device, { chunkCap: 50_000 });
    const multiRgba = await renderFront(multi, data, 64, 64);
    multi.destroy();

    // Same scene, same camera → identical compositing.
    let maxDiff = 0;
    for (let i = 0; i < single.length; i++) {
        const d = Math.abs(single[i] - multiRgba[i]);
        if (d > maxDiff) maxDiff = d;
    }
    expect(maxDiff <= 2,
        `single vs 4-chunk max pixel diff = ${maxDiff} (>2 means chunk accumulation broken)`);
});

test('large pair count: 200K dense splats don\'t crash on dispatch limits', async ({ renderer, expect }) => {
    // 200K splats with LARGE log-scale → wide footprint → many pairs per splat.
    const n = 200_000;
    const data: SplatData = {
        count: n,
        positions: new Float32Array(n * 3),
        rotations: new Float32Array(n * 4),
        logScales: new Float32Array(n * 3),
        opacityLogits: new Float32Array(n),
        colorsDC: new Float32Array(n * 3),
        shBands: 0,
    };
    for (let i = 0; i < n; i++) {
        // tight cluster, large splats → 100k+ pairs per dispatch
        data.positions[i * 3 + 0] = (Math.random() - 0.5) * 0.1;
        data.positions[i * 3 + 1] = (Math.random() - 0.5) * 0.1;
        data.positions[i * 3 + 2] = (Math.random() - 0.5) * 0.1;
        data.rotations[i * 4 + 0] = 1;
        data.logScales[i * 3 + 0] = -2.5;
        data.logScales[i * 3 + 1] = -2.5;
        data.logScales[i * 3 + 2] = -2.5;
        data.opacityLogits[i] = 0;
        data.colorsDC[i * 3 + 0] = 0.5;
    }
    // Just need this to not throw and produce *some* output.
    const rgba = await renderFront(renderer, data, 256, 256);
    const nb = nonBlackCount(rgba);
    expect(nb > 100, `expected dense rendering, got ${nb} non-black pixels (may indicate crash)`);
});

test('1M splats render without dispatch overflow', async ({ renderer, expect }) => {
    const n = 1_000_000;
    const data = manySplats(n, 'random');
    const rgba = await renderFront(renderer, data, 256, 256);
    const nb = nonBlackCount(rgba);
    expect(nb > 1000, `1M random splats produced ${nb} non-black pixels (crash likely if 0)`);
}, true);

test('voxelize: 5000 sphere produces > 0 blocks', async ({ device, expect }) => {
    const data = manySplats(5000, 'sphere');
    const result = await voxelize(device, data, {
        voxelResolution: 0.05, opacityCutoff: 0.1
    });
    expect(result.buffer.count > 0,
        `voxelize produced ${result.buffer.count} blocks (expected > 0)`);
});

test('voxelize: empty scene returns 0 blocks gracefully', async ({ device, expect }) => {
    const empty: SplatData = {
        count: 0,
        positions: new Float32Array(0), rotations: new Float32Array(0),
        logScales: new Float32Array(0), opacityLogits: new Float32Array(0),
        colorsDC: new Float32Array(0), shBands: 0
    };
    const result = await voxelize(device, empty, {
        voxelResolution: 0.05, opacityCutoff: 0.1
    });
    expect(result.buffer.count === 0,
        `empty scene gave ${result.buffer.count} blocks (expected 0)`);
});

// ---------- runner ----------
const run = async (includeSlow: boolean) => {
    results.innerHTML = '';
    print(`initializing…`);
    const device = await createDevice();
    const errors: string[] = [];
    device.addEventListener('uncapturederror', (ev: any) =>
        errors.push(ev.error?.message ?? String(ev.error)));
    const renderer = new Renderer(device);

    const ctx: TestCtx = {
        device, renderer,
        expect(cond, msg) { if (!cond) throw new Error(msg); },
        expectClose(a, b, eps, msg) {
            if (Math.abs(a - b) > eps) throw new Error(`${msg}: |${a} - ${b}| > ${eps}`);
        }
    };

    let pass = 0, fail = 0, skip = 0;
    for (const t of tests) {
        if (t.slow && !includeSlow) { skip++; print(`SKIP  ${t.name}`, 'info'); continue; }
        const t0 = performance.now();
        try {
            await t.fn(ctx);
            const dt = performance.now() - t0;
            pass++;
            print(`PASS  ${t.name}  (${dt.toFixed(0)} ms)`, 'pass');
        } catch (e) {
            fail++;
            const dt = performance.now() - t0;
            print(`FAIL  ${t.name}  (${dt.toFixed(0)} ms): ${(e as Error).message}`, 'fail');
        }
        // Surface GPU errors that fired during the test, even if test "passed".
        if (errors.length > 0) {
            for (const e of errors) print(`      gpu-error: ${e}`, 'fail');
            errors.length = 0;
        }
    }
    renderer.destroy();
    print(`\n${pass} pass · ${fail} fail · ${skip} skip`,
        fail === 0 ? 'pass' : 'fail');
};

document.getElementById('run')!.addEventListener('click', () => run(false));
document.getElementById('run-slow')!.addEventListener('click', () => run(true));
// Auto-run on load
run(false);
