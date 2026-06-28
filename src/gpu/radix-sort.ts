/**
 * GPU radix sort (u32 key + u32 value).
 *
 * Independent implementation. 4-bit radix per pass, 8 passes for 32-bit keys.
 *
 * Layout:
 *  - Per workgroup, each thread bins its element into one of 16 buckets.
 *  - Histogram built per workgroup, summed across workgroups, prefix-summed.
 *  - Each thread writes its element to its final slot.
 *
 * This is the simpler "global histogram, scatter" form rather than tile-shuffle;
 * fast enough for our pair counts (typical 1080p scene: < 8M pairs).
 */

import { SHADER_PRELUDE } from '../render/shaders/constants.ts';
import { emptyBuffer } from './device.ts';

export const RADIX_BUCKETS = 16;
export const RADIX_BITS = 4;
export const RADIX_WG = 256;

const histogramShader = () => /* wgsl */`
${SHADER_PRELUDE}

struct Uniforms {
    n: u32,
    shift: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> globalHist: array<atomic<u32>>;

var<workgroup> local: array<atomic<u32>, ${RADIX_BUCKETS}>;

@compute @workgroup_size(${RADIX_WG})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    if (lid.x < ${RADIX_BUCKETS}u) { atomicStore(&local[lid.x], 0u); }
    workgroupBarrier();

    if (gid.x < u.n) {
        let k = (keys[gid.x] >> u.shift) & 0xfu;
        atomicAdd(&local[k], 1u);
    }
    workgroupBarrier();

    if (lid.x < ${RADIX_BUCKETS}u) {
        atomicAdd(&globalHist[lid.x], atomicLoad(&local[lid.x]));
    }
}
`;

const scatterShader = () => /* wgsl */`
${SHADER_PRELUDE}

struct Uniforms {
    n: u32,
    shift: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> inKeys: array<u32>;
@group(0) @binding(2) var<storage, read> inVals: array<u32>;
@group(0) @binding(3) var<storage, read_write> outKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> outVals: array<u32>;
@group(0) @binding(5) var<storage, read_write> offsets: array<atomic<u32>>;

@compute @workgroup_size(${RADIX_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.n) { return; }
    let k = inKeys[gid.x];
    let v = inVals[gid.x];
    let bucket = (k >> u.shift) & 0xfu;
    let dst = atomicAdd(&offsets[bucket], 1u);
    outKeys[dst] = k;
    outVals[dst] = v;
}
`;

/**
 * Radix sort keys (u32) and parallel values (u32). Operates entirely on the
 * GPU. The input buffers are swapped (ping-pong) — the caller should use the
 * returned buffer pair as the new "current" set.
 *
 * Note: this performs 8 passes (4 bits each) which is plenty for tile-id
 * sorting. For larger key spaces extend the loop.
 */
export class GpuRadixSorter {
    private device: GPUDevice;
    private histPipeline: GPUComputePipeline;
    private scatterPipeline: GPUComputePipeline;
    private uniformBuf: GPUBuffer;
    private histBuf: GPUBuffer;
    private offsetsBuf: GPUBuffer;
    private maxN: number = 0;
    private pongKeys: GPUBuffer | null = null;
    private pongVals: GPUBuffer | null = null;

    constructor(device: GPUDevice) {
        this.device = device;
        this.histPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: histogramShader() }), entryPoint: 'main' }
        });
        this.scatterPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: scatterShader() }), entryPoint: 'main' }
        });
        this.uniformBuf = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.histBuf = device.createBuffer({
            size: RADIX_BUCKETS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.offsetsBuf = device.createBuffer({
            size: RADIX_BUCKETS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
    }

    private ensurePong(n: number) {
        if (n > this.maxN) {
            this.pongKeys?.destroy();
            this.pongVals?.destroy();
            this.pongKeys = emptyBuffer(this.device, n * 4,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'radix-pong-keys');
            this.pongVals = emptyBuffer(this.device, n * 4,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'radix-pong-vals');
            this.maxN = n;
        }
    }

    /**
     * Sort `n` elements. Returns the buffers (key, value) holding the sorted
     * result; this may be the input buffers or the internal pong buffers.
     */
    async sort(
        keys: GPUBuffer, vals: GPUBuffer, n: number, keyBits: number = 32
    ): Promise<{ keys: GPUBuffer; vals: GPUBuffer }> {
        this.ensurePong(n);
        let inKeys = keys, inVals = vals;
        let outKeys = this.pongKeys!, outVals = this.pongVals!;
        const passes = Math.ceil(keyBits / RADIX_BITS);

        for (let p = 0; p < passes; p++) {
            const shift = p * RADIX_BITS;
            // Reset histogram + write uniform.
            this.device.queue.writeBuffer(this.uniformBuf, 0, new Uint32Array([n, shift]));
            this.device.queue.writeBuffer(this.histBuf, 0, new Uint32Array(RADIX_BUCKETS));

            const histBg = this.device.createBindGroup({
                layout: this.histPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuf } },
                    { binding: 1, resource: { buffer: inKeys } },
                    { binding: 2, resource: { buffer: this.histBuf } }
                ]
            });
            const enc1 = this.device.createCommandEncoder();
            const p1 = enc1.beginComputePass();
            p1.setPipeline(this.histPipeline);
            p1.setBindGroup(0, histBg);
            p1.dispatchWorkgroups(Math.ceil(n / RADIX_WG));
            p1.end();
            this.device.queue.submit([enc1.finish()]);

            // CPU readback histogram → exclusive scan → write offsets.
            const stage = this.device.createBuffer({
                size: RADIX_BUCKETS * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const enc2 = this.device.createCommandEncoder();
            enc2.copyBufferToBuffer(this.histBuf, 0, stage, 0, RADIX_BUCKETS * 4);
            this.device.queue.submit([enc2.finish()]);
            await stage.mapAsync(GPUMapMode.READ);
            const h = new Uint32Array(stage.getMappedRange().slice(0));
            stage.unmap(); stage.destroy();
            const offsets = new Uint32Array(RADIX_BUCKETS);
            let sum = 0;
            for (let b = 0; b < RADIX_BUCKETS; b++) {
                offsets[b] = sum; sum += h[b];
            }
            this.device.queue.writeBuffer(this.offsetsBuf, 0, offsets);

            const scatBg = this.device.createBindGroup({
                layout: this.scatterPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuf } },
                    { binding: 1, resource: { buffer: inKeys } },
                    { binding: 2, resource: { buffer: inVals } },
                    { binding: 3, resource: { buffer: outKeys } },
                    { binding: 4, resource: { buffer: outVals } },
                    { binding: 5, resource: { buffer: this.offsetsBuf } }
                ]
            });
            const enc3 = this.device.createCommandEncoder();
            const p3 = enc3.beginComputePass();
            p3.setPipeline(this.scatterPipeline);
            p3.setBindGroup(0, scatBg);
            p3.dispatchWorkgroups(Math.ceil(n / RADIX_WG));
            p3.end();
            this.device.queue.submit([enc3.finish()]);

            // Swap.
            const tk = inKeys; inKeys = outKeys; outKeys = tk;
            const tv = inVals; inVals = outVals; outVals = tv;
        }

        return { keys: inKeys, vals: inVals };
    }
}
