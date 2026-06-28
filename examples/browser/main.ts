/**
 * Browser demo: build a synthetic spherical-shell scene of 5,000 colored
 * splats, render it with mouse-orbit camera, voxelize once at the start.
 *
 * Controls:
 *   - drag                orbit camera around the origin
 *   - scroll wheel        dolly in/out
 */

import {
    createDevice,
    Renderer,
    voxelize,
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
        rotations[i * 4 + 0] = 1; rotations[i * 4 + 1] = 0;
        rotations[i * 4 + 2] = 0; rotations[i * 4 + 3] = 0;
        logScales[i * 3 + 0] = -3.2;
        logScales[i * 3 + 1] = -3.2;
        logScales[i * 3 + 2] = -3.2;
        opacityLogits[i] = 2.5;
        colorsDC[i * 3 + 0] = Math.cos(theta) * 0.5;
        colorsDC[i * 3 + 1] = Math.sin(theta) * 0.5;
        colorsDC[i * 3 + 2] = Math.cos(phi) * 0.5;
    }
    return {
        count: n,
        positions, rotations, logScales, opacityLogits, colorsDC,
        shBands: 0
    };
};

/**
 * Orbit camera state. The camera sits on a sphere of radius `dist` around
 * the origin, parameterised by yaw (around world-up) and pitch (elevation).
 *
 * Pitch is clamped to (-89°, 89°) so the up vector never flips.
 */
class OrbitCamera {
    yaw = 0;
    pitch = 0;
    dist = 4;

    eye(): Vec3 {
        const cp = Math.cos(this.pitch);
        return [
            this.dist * cp * Math.sin(this.yaw),
            this.dist * Math.sin(this.pitch),
            this.dist * cp * Math.cos(this.yaw)
        ];
    }

    view() {
        return mat4LookAt(this.eye(), [0, 0, 0], [0, 1, 0]);
    }
}

const main = async () => {
    try {
        log(`initializing WebGPU…`);
        const device = await createDevice();
        device.addEventListener('uncapturederror', (ev: any) => {
            log(`GPU error: ${ev.error?.message ?? ev.error}`, 'err');
        });
        log(`device acquired`);

        const data = buildSphereShell(5000, 1.0);
        log(`built ${data.count} synthetic splats`);

        const renderer = new Renderer(device);
        const cam = new OrbitCamera();

        // --- Voxelize once up front ---
        const t1 = performance.now();
        const result = await voxelize(device, data, {
            voxelResolution: 0.05,
            opacityCutoff: 0.1
        });
        const voxMs = performance.now() - t1;
        log(`voxelize: ${voxMs.toFixed(1)} ms — `
            + `${result.numBlocksX}×${result.numBlocksY}×${result.numBlocksZ} blocks, `
            + `${result.buffer.count} non-empty, `
            + `${result.buffer.popcount()} solid voxels`);

        // --- Render loop, on-demand ---
        const fpsRow = document.createElement('div');
        stats.appendChild(fpsRow);

        let pending = false;
        const draw = async () => {
            if (pending) return;
            pending = true;
            try {
                const t0 = performance.now();
                const rgba = await renderer.render(
                    data,
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
                // The renderer's Uint8Array shares an ArrayBuffer with no
                // particular ownership; copy into a fresh ArrayBuffer so the
                // ImageData constructor (which rejects SharedArrayBuffer)
                // accepts it.
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

        // --- Mouse input ---
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
            // 0.005 rad/px feels right at 640x480
            cam.yaw -= dx * 0.005;
            cam.pitch += dy * 0.005;
            // clamp pitch so up-vector never flips
            const pmax = Math.PI / 2 - 0.01;
            if (cam.pitch > pmax) cam.pitch = pmax;
            if (cam.pitch < -pmax) cam.pitch = -pmax;
            draw();
        });
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = Math.exp(e.deltaY * 0.001);
            cam.dist = Math.min(50, Math.max(1.5, cam.dist * factor));
            draw();
        }, { passive: false });
    } catch (e) {
        log(`error: ${(e as Error).message}`, 'err');
        console.error(e);
    }
};

main();
