/**
 * Renderer orchestrator. The public entry point that drives the full
 * cull → sort → project → bin → rasterize → finalize pipeline.
 *
 * Memory model: all per-render scratch buffers are owned by `Renderer`, sized
 * once on first use and grown on demand. Repeated calls to `render()` reuse
 * them.
 */

import type { SplatData, RenderCamera, BackgroundRGBA } from '../types.ts';
import { splatInputStride, numSHCoeffsPerChannel } from '../types.ts';
import { buildCameraBasis } from './camera.ts';
import {
    preprocess, packChunkInput, createPreprocessScratch,
    type PreprocessScratch
} from './preprocess.ts';
import { TILE_SIZE } from './shaders/constants.ts';
import { projectShader } from './shaders/project.ts';
import {
    prefixScanBlocksShader, prefixAddBlockOffsetsShader, PREFIX_BLOCK
} from './shaders/prefix-sum.ts';
import { emitPairsShader } from './shaders/emit-pairs.ts';
import {
    initTileOffsetsShader, findBoundariesShader, finalizeTileOffsetsShader
} from './shaders/find-boundaries.ts';
import { rasterizeShader } from './shaders/rasterize.ts';
import { finalizeShader } from './shaders/finalize.ts';
import { emptyBuffer, uploadBuffer, readbackBuffer } from '../gpu/device.ts';
import { GpuRadixSorter } from '../gpu/radix-sort.ts';

/** Maximum splats per project+rasterize chunk. */
const CHUNK_CAP = 200_000;
/**
 * Maximum tile coverage per splat (clamp to prevent runaway pair growth).
 * Picked to comfortably cover a viewport-sized splat at 1080p: 1920/16 ×
 * 1080/16 = 120 × 68 ≈ 8160 tiles. The previous value of 1024 silently
 * truncated large close-up splats to ~32×32 tiles, leaving the rest of
 * the splat's footprint unrendered — visible as feathery directional
 * edges along the partial-emit boundary.
 */
const MAX_COVERAGE_PER_SPLAT = 16_384;

export interface RendererOptions {
    /** Cap on splats per project+rasterize dispatch. Default 200k. */
    chunkCap?: number;
    /** Cap on tile coverage per splat. Default 1024. */
    maxCoveragePerSplat?: number;
}

export class Renderer {
    private device: GPUDevice;
    private opts: Required<RendererOptions>;
    private cpuScratch: PreprocessScratch;
    private sorter: GpuRadixSorter;

    // Pipelines (lazy, parameterized by SH bands + projection).
    private pipelineCache = new Map<string, GpuPipelines>();

    // Owned reusable buffers.
    private splatInputBuf: GPUBuffer | null = null;
    private splatInputCapacity = 0;
    private projectedBuf: GPUBuffer | null = null;
    private projectedCapacity = 0;
    private tileCountsBuf: GPUBuffer | null = null;
    private tileCountsCapacity = 0;
    private blockSumsBuf: GPUBuffer | null = null;
    private pairTilesBuf: GPUBuffer | null = null;
    private pairSplatsBuf: GPUBuffer | null = null;
    private pairCapacity = 0;
    private tileOffsetsBuf: GPUBuffer | null = null;
    private tileOffsetsCapacity = 0;
    private accumBuf: GPUBuffer | null = null;
    private outRgbaBuf: GPUBuffer | null = null;
    private accumPixels = 0;

    constructor(device: GPUDevice, opts: RendererOptions = {}) {
        this.device = device;
        this.opts = {
            chunkCap: opts.chunkCap ?? CHUNK_CAP,
            maxCoveragePerSplat: opts.maxCoveragePerSplat ?? MAX_COVERAGE_PER_SPLAT
        };
        this.cpuScratch = createPreprocessScratch();
        this.sorter = new GpuRadixSorter(device);
    }

    private getPipelines(bands: 0 | 1 | 2 | 3, proj: 'pinhole' | 'equirect'): GpuPipelines {
        const key = `${bands}-${proj}`;
        let p = this.pipelineCache.get(key);
        if (!p) {
            p = buildPipelines(this.device, bands, proj);
            this.pipelineCache.set(key, p);
        }
        return p;
    }

    private ensureSplatInput(stride: number, chunkCap: number) {
        const need = stride * chunkCap * 4;
        if (need > this.splatInputCapacity) {
            this.splatInputBuf?.destroy();
            this.splatInputBuf = emptyBuffer(this.device, need,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'splat-input');
            this.splatInputCapacity = need;
        }
    }

    private ensureProjected(chunkCap: number) {
        // 3 vec4<f32> per splat
        const need = chunkCap * 3 * 16;
        if (need > this.projectedCapacity) {
            this.projectedBuf?.destroy();
            this.projectedBuf = emptyBuffer(this.device, need,
                GPUBufferUsage.STORAGE, 'projected');
            this.projectedCapacity = need;
        }
    }

    private ensureTileCounts(chunkCap: number) {
        const need = Math.max(chunkCap, PREFIX_BLOCK) * 4;
        if (need > this.tileCountsCapacity) {
            this.tileCountsBuf?.destroy();
            this.tileCountsBuf = emptyBuffer(this.device, need,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'tile-counts');
            this.tileCountsCapacity = need;
            const blocks = Math.ceil(chunkCap / PREFIX_BLOCK);
            this.blockSumsBuf?.destroy();
            this.blockSumsBuf = emptyBuffer(this.device, Math.max(64, blocks * 4),
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'block-sums');
        }
    }

    private ensurePairs(numPairs: number) {
        const need = Math.max(numPairs, 1) * 4;
        if (need > this.pairCapacity) {
            this.pairTilesBuf?.destroy();
            this.pairSplatsBuf?.destroy();
            this.pairTilesBuf = emptyBuffer(this.device, need,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'pair-tiles');
            this.pairSplatsBuf = emptyBuffer(this.device, need,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'pair-splats');
            this.pairCapacity = need;
        }
    }

    private ensureTileOffsets(numTiles: number) {
        const need = (numTiles + 1) * 4;
        if (need > this.tileOffsetsCapacity) {
            this.tileOffsetsBuf?.destroy();
            this.tileOffsetsBuf = emptyBuffer(this.device, need,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'tile-offsets');
            this.tileOffsetsCapacity = need;
        }
    }

    private ensureImage(pixels: number) {
        if (pixels > this.accumPixels) {
            this.accumBuf?.destroy();
            this.outRgbaBuf?.destroy();
            this.accumBuf = emptyBuffer(this.device, pixels * 16,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'accum');
            this.outRgbaBuf = emptyBuffer(this.device, pixels * 4,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'out-rgba');
            this.accumPixels = pixels;
            this.accumInit = null;  // regrow init scratch
        }
    }

    /** Lazily-cached "(0,0,0,1) per pixel" Float32Array used to clear accum. */
    private accumInit: Float32Array | null = null;
    private getAccumInit(pixels: number): Float32Array {
        if (!this.accumInit || this.accumInit.length < pixels * 4) {
            const arr = new Float32Array(pixels * 4);
            for (let p = 0; p < pixels; p++) arr[p * 4 + 3] = 1;  // T = 1
            this.accumInit = arr;
        }
        return this.accumInit;
    }

    /**
     * Render one frame. Returns the RGBA8 pixel data, length = width*height*4.
     */
    async render(
        data: SplatData,
        camera: RenderCamera,
        background: BackgroundRGBA = { r: 0, g: 0, b: 0, a: 1 }
    ): Promise<Uint8Array> {
        const basis = buildCameraBasis(camera);
        const bands = data.shBands;
        const projection = basis.projection;
        const pipelines = this.getPipelines(bands, projection);
        const stride = splatInputStride(bands);

        const tilesX = Math.ceil(camera.width / TILE_SIZE);
        const tilesY = Math.ceil(camera.height / TILE_SIZE);
        const numTiles = tilesX * tilesY;
        const pixels = camera.width * camera.height;

        // ---- 1. CPU cull + sort ----
        const pre = preprocess(data, basis, this.cpuScratch);
        const numVisible = pre.numVisible;

        this.ensureImage(pixels);
        this.ensureTileOffsets(numTiles);
        // Init the accumulator to (R=0, G=0, B=0, T=1) per pixel. The
        // rasterize shader RESUMES from this state across chunks; starting
        // with T=1 (full transmittance) is required or chunk 0's results
        // collapse to zero alpha.
        const initArr = this.getAccumInit(pixels);
        this.device.queue.writeBuffer(this.accumBuf!, 0, initArr.buffer, 0, pixels * 16);

        if (numVisible === 0) {
            return this.composite(camera, background, pipelines, numTiles);
        }

        // ---- 2. Chunked project + binning + rasterize ----
        const chunkCap = Math.min(this.opts.chunkCap, numVisible);
        const stagingChunk = new Float32Array(stride * chunkCap);
        this.ensureSplatInput(stride, chunkCap);
        this.ensureProjected(chunkCap);
        this.ensureTileCounts(chunkCap);

        for (let chunkStart = 0; chunkStart < numVisible; chunkStart += chunkCap) {
            const chunkCount = Math.min(chunkCap, numVisible - chunkStart);
            packChunkInput(data, pre.sortedIndices, chunkStart, chunkCount, stagingChunk);
            this.device.queue.writeBuffer(
                this.splatInputBuf!, 0,
                stagingChunk.buffer, stagingChunk.byteOffset,
                chunkCount * stride * 4
            );

            // Reset per-chunk transient buffers.
            this.device.queue.writeBuffer(this.tileCountsBuf!, 0, new Uint32Array(chunkCap));

            // ---- project ----
            const projUniform = this.makeProjectUniforms(
                camera, basis, chunkCount, tilesX, tilesY, stride, bands
            );
            const projBg = this.device.createBindGroup({
                layout: pipelines.project.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: projUniform } },
                    { binding: 1, resource: { buffer: this.splatInputBuf! } },
                    { binding: 2, resource: { buffer: this.projectedBuf! } },
                    { binding: 3, resource: { buffer: this.tileCountsBuf! } }
                ]
            });
            const enc = this.device.createCommandEncoder();
            {
                const p = enc.beginComputePass();
                p.setPipeline(pipelines.project);
                p.setBindGroup(0, projBg);
                p.dispatchWorkgroups(Math.ceil(chunkCount / 64));
                p.end();
            }
            this.device.queue.submit([enc.finish()]);

            // ---- prefix-sum tile counts ----
            const numBlocks = Math.ceil(chunkCount / PREFIX_BLOCK);
            const scanU = this.makeScanUniform(chunkCount);
            const scanBg = this.device.createBindGroup({
                layout: pipelines.prefixScan.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.tileCountsBuf! } },
                    { binding: 1, resource: { buffer: this.blockSumsBuf! } },
                    { binding: 2, resource: { buffer: scanU } }
                ]
            });
            const enc2 = this.device.createCommandEncoder();
            {
                const p = enc2.beginComputePass();
                p.setPipeline(pipelines.prefixScan);
                p.setBindGroup(0, scanBg);
                p.dispatchWorkgroups(numBlocks);
                p.end();
            }
            this.device.queue.submit([enc2.finish()]);

            // CPU exclusive-scan over block sums → write back.
            const blockSums = await readbackBuffer(this.device, this.blockSumsBuf!, numBlocks * 4);
            const blockSumsU32 = new Uint32Array(blockSums.buffer, 0, numBlocks);
            const blockPrefix = new Uint32Array(numBlocks);
            let sum = 0;
            for (let b = 0; b < numBlocks; b++) {
                blockPrefix[b] = sum; sum += blockSumsU32[b];
            }
            const numPairs = sum;
            this.device.queue.writeBuffer(this.blockSumsBuf!, 0, blockPrefix);

            const addBg = this.device.createBindGroup({
                layout: pipelines.prefixAdd.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.tileCountsBuf! } },
                    { binding: 1, resource: { buffer: this.blockSumsBuf! } },
                    { binding: 2, resource: { buffer: scanU } }
                ]
            });
            const enc3 = this.device.createCommandEncoder();
            {
                const p = enc3.beginComputePass();
                p.setPipeline(pipelines.prefixAdd);
                p.setBindGroup(0, addBg);
                p.dispatchWorkgroups(numBlocks);
                p.end();
            }
            this.device.queue.submit([enc3.finish()]);

            if (numPairs === 0) continue;
            this.ensurePairs(numPairs);

            // ---- emit pairs ----
            const emitU = this.makeEmitUniform(tilesX, tilesY, chunkCount);
            const emitBg = this.device.createBindGroup({
                layout: pipelines.emitPairs.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: emitU } },
                    { binding: 1, resource: { buffer: this.projectedBuf! } },
                    { binding: 2, resource: { buffer: this.tileCountsBuf! } },
                    { binding: 3, resource: { buffer: this.pairTilesBuf! } },
                    { binding: 4, resource: { buffer: this.pairSplatsBuf! } }
                ]
            });
            const enc4 = this.device.createCommandEncoder();
            {
                const p = enc4.beginComputePass();
                p.setPipeline(pipelines.emitPairs);
                p.setBindGroup(0, emitBg);
                p.dispatchWorkgroups(Math.ceil(chunkCount / 64));
                p.end();
            }
            this.device.queue.submit([enc4.finish()]);

            // ---- sort pairs by tile id ----
            // CPU stable sort: read back, sort, write back. The GPU radix
            // sort that previously lived here was unstable across passes —
            // tile slices got scrambled, leaving most pixels with empty
            // slices. CPU radix sort on a few million u32 pairs is fast
            // enough (~1-2ms) and correct. A stable GPU sort is a future
            // optimization.
            const sortedTiles = this.pairTilesBuf!;
            const sortedSplats = this.pairSplatsBuf!;
            await this.cpuSortPairs(sortedTiles, sortedSplats, numPairs);

            // ---- find tile boundaries ----
            // Init tileOffsets to sentinel.
            const initU = this.makeInitUniform(numTiles);
            const initBg = this.device.createBindGroup({
                layout: pipelines.initOffsets.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.tileOffsetsBuf! } },
                    { binding: 1, resource: { buffer: initU } }
                ]
            });
            const enc5 = this.device.createCommandEncoder();
            {
                const p = enc5.beginComputePass();
                p.setPipeline(pipelines.initOffsets);
                p.setBindGroup(0, initBg);
                p.dispatchWorkgroups(Math.ceil((numTiles + 1) / 64));
                p.end();
            }
            this.device.queue.submit([enc5.finish()]);

            // find-boundaries uses 2D dispatch to stay under WebGPU's
            // 65535 per-dimension workgroup cap. A linear thread index is
            // reconstructed in the shader as gid.y * stride + gid.x.
            const fbDispatch = dispatch2D(numPairs, 64);
            const findU = this.makeFindUniform(numPairs, numTiles, fbDispatch.stride);
            const findBg = this.device.createBindGroup({
                layout: pipelines.findBoundaries.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: findU } },
                    { binding: 1, resource: { buffer: sortedTiles } },
                    { binding: 2, resource: { buffer: this.tileOffsetsBuf! } }
                ]
            });
            const enc6 = this.device.createCommandEncoder();
            {
                const p = enc6.beginComputePass();
                p.setPipeline(pipelines.findBoundaries);
                p.setBindGroup(0, findBg);
                p.dispatchWorkgroups(fbDispatch.x, fbDispatch.y);
                p.end();
            }
            this.device.queue.submit([enc6.finish()]);

            const finalBg = this.device.createBindGroup({
                layout: pipelines.finalizeOffsets.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: findU } },
                    { binding: 1, resource: { buffer: this.tileOffsetsBuf! } }
                ]
            });
            const enc7 = this.device.createCommandEncoder();
            {
                const p = enc7.beginComputePass();
                p.setPipeline(pipelines.finalizeOffsets);
                p.setBindGroup(0, finalBg);
                p.dispatchWorkgroups(1);
                p.end();
            }
            this.device.queue.submit([enc7.finish()]);

            // ---- rasterize ----
            const rastU = this.makeRastUniform(tilesX, tilesY, camera.width, camera.height);
            const rastBg = this.device.createBindGroup({
                layout: pipelines.rasterize.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: rastU } },
                    { binding: 1, resource: { buffer: this.projectedBuf! } },
                    { binding: 2, resource: { buffer: this.tileOffsetsBuf! } },
                    { binding: 3, resource: { buffer: sortedSplats } },
                    { binding: 4, resource: { buffer: this.accumBuf! } }
                ]
            });
            const enc8 = this.device.createCommandEncoder();
            {
                const p = enc8.beginComputePass();
                p.setPipeline(pipelines.rasterize);
                p.setBindGroup(0, rastBg);
                p.dispatchWorkgroups(tilesX, tilesY);
                p.end();
            }
            this.device.queue.submit([enc8.finish()]);
        }

        return this.composite(camera, background, pipelines, numTiles);
    }

    private async composite(
        camera: RenderCamera,
        bg: BackgroundRGBA,
        pipelines: GpuPipelines,
        _numTiles: number
    ): Promise<Uint8Array> {
        const pixels = camera.width * camera.height;
        const u = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const ub = new ArrayBuffer(32);
        const dv = new DataView(ub);
        dv.setUint32(0, camera.width, true);
        dv.setUint32(4, camera.height, true);
        dv.setFloat32(8, bg.r, true);
        dv.setFloat32(12, bg.g, true);
        dv.setFloat32(16, bg.b, true);
        dv.setFloat32(20, bg.a, true);
        this.device.queue.writeBuffer(u, 0, ub);

        const bgrp = this.device.createBindGroup({
            layout: pipelines.finalize.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: u } },
                { binding: 1, resource: { buffer: this.accumBuf! } },
                { binding: 2, resource: { buffer: this.outRgbaBuf! } }
            ]
        });
        const enc = this.device.createCommandEncoder();
        {
            const p = enc.beginComputePass();
            p.setPipeline(pipelines.finalize);
            p.setBindGroup(0, bgrp);
            p.dispatchWorkgroups(
                Math.ceil(camera.width / 8),
                Math.ceil(camera.height / 8)
            );
            p.end();
        }
        this.device.queue.submit([enc.finish()]);

        const out = await readbackBuffer(this.device, this.outRgbaBuf!, pixels * 4);
        u.destroy();
        return out;
    }

    private makeProjectUniforms(
        cam: RenderCamera,
        basis: ReturnType<typeof buildCameraBasis>,
        numSplats: number,
        tilesX: number, tilesY: number,
        stride: number, bands: 0 | 1 | 2 | 3
    ): GPUBuffer {
        // mat4 (64B) + 11 u32/f32 (44B) + pad → 128B
        const size = 128;
        const buf = this.device.createBuffer({
            size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const ab = new ArrayBuffer(size);
        const f = new Float32Array(ab);
        const u = new Uint32Array(ab);
        f.set(basis.viewMatrix, 0);
        f[16] = cam.width;
        f[17] = cam.height;
        f[18] = basis.focal;
        f[19] = basis.cx;
        f[20] = basis.cy;
        f[21] = cam.near;
        f[22] = cam.far;
        u[23] = numSplats;
        u[24] = tilesX;
        u[25] = tilesY;
        u[26] = this.opts.maxCoveragePerSplat;
        u[27] = bands;
        u[28] = stride;
        this.device.queue.writeBuffer(buf, 0, ab);
        return buf;
    }

    private makeScanUniform(n: number): GPUBuffer {
        const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(buf, 0, new Uint32Array([n, 0, 0, 0]));
        return buf;
    }

    private makeEmitUniform(tilesX: number, tilesY: number, n: number): GPUBuffer {
        const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(buf, 0, new Uint32Array([tilesX, tilesY, n, this.opts.maxCoveragePerSplat]));
        return buf;
    }

    private makeInitUniform(n: number): GPUBuffer {
        const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(buf, 0, new Uint32Array([n, 0, 0, 0]));
        return buf;
    }

    private makeFindUniform(numPairs: number, numTiles: number, stride: number): GPUBuffer {
        const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(buf, 0, new Uint32Array([numPairs, numTiles, stride, 0]));
        return buf;
    }

    private makeRastUniform(tilesX: number, tilesY: number, w: number, h: number): GPUBuffer {
        const buf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(buf, 0, new Uint32Array([tilesX, tilesY, w, h, 0, 0, 0, 0]));
        return buf;
    }

    /**
     * Stable CPU sort of (tileId, splatIdx) pairs by tileId. Used in place
     * of an unstable GPU radix sort. Reads N u32 pairs back to CPU,
     * sorts via counting sort (since tile-id range is small and bounded),
     * writes the sorted result back to GPU.
     */
    private async cpuSortPairs(
        tilesBuf: GPUBuffer, splatsBuf: GPUBuffer, n: number
    ): Promise<void> {
        if (n <= 1) return;
        const tRaw = await readbackBuffer(this.device, tilesBuf, n * 4);
        const sRaw = await readbackBuffer(this.device, splatsBuf, n * 4);
        const tIn = new Uint32Array(tRaw.buffer, 0, n);
        const sIn = new Uint32Array(sRaw.buffer, 0, n);

        // Find max tile id and use counting sort. Tile IDs are small bounded
        // integers (= numTiles which is ≤ ~100k at 4K), so counting sort wins.
        let maxT = 0;
        for (let i = 0; i < n; i++) {
            if (tIn[i] > maxT) maxT = tIn[i];
        }
        const buckets = maxT + 1;
        const counts = new Uint32Array(buckets);
        for (let i = 0; i < n; i++) counts[tIn[i]]++;
        let sum = 0;
        for (let b = 0; b < buckets; b++) {
            const c = counts[b];
            counts[b] = sum;
            sum += c;
        }
        const tOut = new Uint32Array(n);
        const sOut = new Uint32Array(n);
        for (let i = 0; i < n; i++) {
            const t = tIn[i];
            const dst = counts[t]++;
            tOut[dst] = t;
            sOut[dst] = sIn[i];
        }
        this.device.queue.writeBuffer(tilesBuf, 0, tOut.buffer, 0, n * 4);
        this.device.queue.writeBuffer(splatsBuf, 0, sOut.buffer, 0, n * 4);
    }

    /** Release all GPU resources. */
    destroy(): void {
        this.splatInputBuf?.destroy();
        this.projectedBuf?.destroy();
        this.tileCountsBuf?.destroy();
        this.blockSumsBuf?.destroy();
        this.pairTilesBuf?.destroy();
        this.pairSplatsBuf?.destroy();
        this.tileOffsetsBuf?.destroy();
        this.accumBuf?.destroy();
        this.outRgbaBuf?.destroy();
    }
}

/**
 * Compute a 2D workgroup dispatch that linearly covers `n` items at
 * `wgSize` items per workgroup, splitting across X and Y when the
 * single-dim workgroup count would exceed WebGPU's 65535 limit.
 *
 * The shader reconstructs the linear index as `gid.y * stride + gid.x`,
 * where `stride = x * wgSize`. Threads with linear index >= n must
 * early-return.
 */
const dispatch2D = (
    n: number, wgSize: number
): { x: number; y: number; stride: number } => {
    const totalWg = Math.max(1, Math.ceil(n / wgSize));
    if (totalWg <= 65535) return { x: totalWg, y: 1, stride: totalWg * wgSize };
    // Pick X = 32768 so stride is a round number and Y stays small even
    // at the largest realistic pair counts (1 G pairs → Y ~ 500).
    const x = 32768;
    const y = Math.ceil(totalWg / x);
    return { x, y, stride: x * wgSize };
};

interface GpuPipelines {
    project: GPUComputePipeline;
    prefixScan: GPUComputePipeline;
    prefixAdd: GPUComputePipeline;
    emitPairs: GPUComputePipeline;
    initOffsets: GPUComputePipeline;
    findBoundaries: GPUComputePipeline;
    finalizeOffsets: GPUComputePipeline;
    rasterize: GPUComputePipeline;
    finalize: GPUComputePipeline;
}

const buildPipelines = (
    device: GPUDevice, bands: 0 | 1 | 2 | 3, proj: 'pinhole' | 'equirect'
): GpuPipelines => {
    const mk = (code: string) => device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code }), entryPoint: 'main' }
    });
    return {
        project: mk(projectShader(bands, proj)),
        prefixScan: mk(prefixScanBlocksShader()),
        prefixAdd: mk(prefixAddBlockOffsetsShader()),
        emitPairs: mk(emitPairsShader()),
        initOffsets: mk(initTileOffsetsShader()),
        findBoundaries: mk(findBoundariesShader()),
        finalizeOffsets: mk(finalizeTileOffsetsShader()),
        rasterize: mk(rasterizeShader()),
        finalize: mk(finalizeShader())
    };
};
