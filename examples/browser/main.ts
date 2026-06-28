/**
 * Browser demo: fetch a real .ply Gaussian-splat scene and render it with
 * an orbit camera. Synthetic-sphere fallback is built-in if scene.ply is
 * absent or fails to fetch.
 *
 * Controls:
 *   - drag                orbit camera around scene center
 *   - scroll wheel        dolly in/out
 */

import {
    createDevice,
    Renderer,
    loadPlySplats,
    splatBounds,
    mat4LookAt,
    type SplatData,
    type Vec3,
} from '../../src/index.ts';

const stats = document.getElementById('stats')!;
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

const log = (msg: string, cls = '') => {
    const div = document.createElement('div');
    if (cls) div.className = cls;
    div.textContent = msg;
    stats.appendChild(div);
};

const buildSphereShell = (n: number, radius: number): SplatData => {
    const positions = new Float32Array(n * 3);
    const rotations = new Float32Array(n * 4);
    const logScales = new Float32Array(n * 3);
    const opacityLogits = new Float32Array(n);
    const colorsDC = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        const t = i / n;
        const phi = Math.acos(1 - 2 * t);
        const theta = Math.PI * (1 + Math.sqrt(5)) * i;
        positions[i * 3 + 0] = radius * Math.cos(theta) * Math.sin(phi);
        positions[i * 3 + 1] = radius * Math.sin(theta) * Math.sin(phi);
        positions[i * 3 + 2] = radius * Math.cos(phi);
        rotations[i * 4 + 0] = 1;
        logScales[i * 3 + 0] = -3.2;
        logScales[i * 3 + 1] = -3.2;
        logScales[i * 3 + 2] = -3.2;
        opacityLogits[i] = 2.5;
        colorsDC[i * 3 + 0] = Math.cos(theta) * 0.5;
        colorsDC[i * 3 + 1] = Math.sin(theta) * 0.5;
        colorsDC[i * 3 + 2] = Math.cos(phi) * 0.5;
    }
    return { count: n, positions, rotations, logScales, opacityLogits, colorsDC,
             shBands: 0 };
};

/**
 * Try to load scene.ply from the dev server. Returns null on any failure so
 * the demo falls back to the synthetic shell.
 */
const tryLoadPly = async (url: string): Promise<SplatData | null> => {
    try {
        const t0 = performance.now();
        const r = await fetch(url);
        if (!r.ok) return null;
        const buf = await r.arrayBuffer();
        log(`fetched ${url} (${(buf.byteLength / 1024 / 1024).toFixed(1)} MB) in `
            + `${(performance.now() - t0).toFixed(0)} ms`);
        const t1 = performance.now();
        const data = loadPlySplats(buf);
        log(`parsed ${data.count.toLocaleString()} splats in `
            + `${(performance.now() - t1).toFixed(0)} ms (SH bands: ${data.shBands})`);
        return data;
    } catch (e) {
        log(`PLY load failed: ${(e as Error).message}`, 'err');
        return null;
    }
};

/**
 * Orbit camera around a target point. Pitch is clamped so up never flips.
 */
class OrbitCamera {
    yaw = 0;
    pitch = 0;
    dist: number;
    target: Vec3;

    constructor(target: Vec3, dist: number) {
        this.target = target;
        this.dist = dist;
    }

    eye(): Vec3 {
        const cp = Math.cos(this.pitch);
        return [
            this.target[0] + this.dist * cp * Math.sin(this.yaw),
            this.target[1] + this.dist * Math.sin(this.pitch),
            this.target[2] + this.dist * cp * Math.cos(this.yaw)
        ];
    }

    view() { return mat4LookAt(this.eye(), this.target, [0, 1, 0]); }
}

const main = async () => {
    try {
        log(`initializing WebGPU…`);
        const device = await createDevice();
        device.addEventListener('uncapturederror', (ev: any) => {
            log(`GPU error: ${ev.error?.message ?? ev.error}`, 'err');
        });
        log(`device acquired`);

        // Real scene first; synthetic if missing.
        let data = await tryLoadPly('/scene.ply');
        if (!data) {
            log(`scene.ply not found — using synthetic 5000-splat sphere`);
            data = buildSphereShell(5000, 1.0);
        }

        // Center camera on scene bounds; distance = 1.5 × half-diagonal.
        const b = splatBounds(data);
        const center: Vec3 = [
            (b.min[0] + b.max[0]) * 0.5,
            (b.min[1] + b.max[1]) * 0.5,
            (b.min[2] + b.max[2]) * 0.5,
        ];
        const half = Math.hypot(
            b.max[0] - b.min[0],
            b.max[1] - b.min[1],
            b.max[2] - b.min[2]
        ) * 0.5;
        const cam = new OrbitCamera(center, Math.max(0.5, half * 1.5));
        log(`scene bounds: center=(${center.map(v => v.toFixed(2)).join(', ')}) `
            + `radius≈${half.toFixed(2)}`);

        const renderer = new Renderer(device);
        const fpsRow = document.createElement('div');
        stats.appendChild(fpsRow);

        let pending = false;
        const draw = async () => {
            if (pending) return;
            pending = true;
            try {
                const t0 = performance.now();
                const rgba = await renderer.render(
                    data!,
                    {
                        width: canvas.width,
                        height: canvas.height,
                        viewMatrix: cam.view(),
                        fovY: Math.PI / 3,
                        near: 0.01, far: 1000,
                    },
                    { r: 0.04, g: 0.05, b: 0.07, a: 1.0 }
                );
                const ms = performance.now() - t0;
                const out = new Uint8ClampedArray(rgba);
                const img = new ImageData(out, canvas.width, canvas.height);
                ctx.putImageData(img, 0, 0);
                fpsRow.textContent =
                    `render: ${ms.toFixed(1)} ms  |  yaw=${(cam.yaw * 180 / Math.PI).toFixed(0)}° `
                    + `pitch=${(cam.pitch * 180 / Math.PI).toFixed(0)}° dist=${cam.dist.toFixed(2)}`;
            } finally {
                pending = false;
            }
        };

        await draw();
        log(`drag to orbit · wheel to dolly`);

        let dragging = false;
        let lastX = 0, lastY = 0;
        canvas.addEventListener('pointerdown', (e) => {
            dragging = true;
            lastX = e.clientX; lastY = e.clientY;
            canvas.setPointerCapture(e.pointerId);
        });
        canvas.addEventListener('pointerup', (e) => {
            dragging = false;
            canvas.releasePointerCapture(e.pointerId);
        });
        canvas.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX; lastY = e.clientY;
            cam.yaw -= dx * 0.005;
            cam.pitch += dy * 0.005;
            const pmax = Math.PI / 2 - 0.01;
            if (cam.pitch > pmax) cam.pitch = pmax;
            if (cam.pitch < -pmax) cam.pitch = -pmax;
            draw();
        });
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = Math.exp(e.deltaY * 0.001);
            cam.dist = Math.min(half * 10, Math.max(0.1, cam.dist * factor));
            draw();
        }, { passive: false });
    } catch (e) {
        log(`error: ${(e as Error).message}`, 'err');
        console.error(e);
    }
};

main();
