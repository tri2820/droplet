/**
 * Voxelization orchestrator. Public API: `voxelize()`.
 *
 * Pipeline:
 *   1. Compute per-Gaussian world AABB (3-sigma extent).
 *   2. Build BVH for broad-phase queries.
 *   3. Walk the grid in 16×16×16-block batches; per batch query BVH for
 *      overlapping splats.
 *   4. Submit batches to GPU in mega-flushes (capped to avoid TDR timeouts).
 *   5. Decode result masks into a sparse `BlockMaskBuffer`.
 *
 * Output is the sparse occupancy bitmap that downstream collision stages
 * (fill, carve, mesh) operate on.
 */

import type { SplatData, Bounds } from '../types.ts';
import { computeAabbExtents } from '../spatial/aabb.ts';
import { GaussianBVH } from '../spatial/bvh.ts';
import { BlockMaskBuffer } from './block-mask-buffer.ts';
import { voxelizeShader } from './shaders/voxelize.ts';
import { emptyBuffer, uploadBuffer, readbackBuffer } from '../gpu/device.ts';

const MAX_BLOCKS_PER_BATCH = 4096;
const MEGA_MAX_BATCHES = 256;
const MEGA_MAX_INDICES = 2 * 1024 * 1024;
/** Floats per Gaussian record in the GPU buffer (matches WGSL struct). */
const GAUSSIAN_STRIDE = 16;

export interface VoxelizeOptions {
    /** Voxel side length in world units. Default 0.05. */
    voxelResolution?: number;
    /** Opacity threshold for "solid". Default 0.1. */
    opacityCutoff?: number;
    /** Override grid bounds. Defaults to data's bounding box (block-aligned). */
    bounds?: Bounds;
}

export interface VoxelizeResult {
    buffer: BlockMaskBuffer;
    bounds: Bounds;
    voxelResolution: number;
    numBlocksX: number;
    numBlocksY: number;
    numBlocksZ: number;
}

interface PendingBatch {
    indexOffset: number;
    indexCount: number;
    blockMinX: number;
    blockMinY: number;
    blockMinZ: number;
    numBlocksX: number;
    numBlocksY: number;
    numBlocksZ: number;
    bx: number; by: number; bz: number;
}

export const voxelize = async (
    device: GPUDevice,
    data: SplatData,
    opts: VoxelizeOptions = {}
): Promise<VoxelizeResult> => {
    const voxelRes = opts.voxelResolution ?? 0.05;
    const opacityCutoff = opts.opacityCutoff ?? 0.1;
    const blockSize = 4 * voxelRes;

    // 1. AABB + BVH.
    const extents = computeAabbExtents(data);
    const bvh = new GaussianBVH(data.positions, extents);

    // 2. Grid bounds (block-aligned).
    const bounds = opts.bounds ?? computeBlockAlignedBounds(data, extents, blockSize);
    const numBlocksX = Math.round((bounds.max[0] - bounds.min[0]) / blockSize);
    const numBlocksY = Math.round((bounds.max[1] - bounds.min[1]) / blockSize);
    const numBlocksZ = Math.round((bounds.max[2] - bounds.min[2]) / blockSize);
    const bStride = numBlocksX * numBlocksY;

    // 3. Upload Gaussians to GPU once.
    const gaussianBuf = uploadGaussians(device, data, extents);
    const uniformBuf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const ub = new ArrayBuffer(16);
    const ubF = new Float32Array(ub);
    const ubU = new Uint32Array(ub);
    ubF[0] = opacityCutoff;
    ubF[1] = voxelRes;
    ubU[2] = MAX_BLOCKS_PER_BATCH;
    device.queue.writeBuffer(uniformBuf, 0, ub);

    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ code: voxelizeShader() }),
            entryPoint: 'main'
        }
    });

    // 4. Per-batch buffers, sized once.
    const indexBuf = emptyBuffer(device, MEGA_MAX_INDICES * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'voxel-indices');
    const batchInfoBuf = emptyBuffer(device, MEGA_MAX_BATCHES * 32,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'voxel-batch-info');
    const resultsBuf = emptyBuffer(device, MEGA_MAX_BATCHES * MAX_BLOCKS_PER_BATCH * 2 * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'voxel-results');

    const buffer = new BlockMaskBuffer();
    const indexScratch = new Uint32Array(MEGA_MAX_INDICES);
    let indexOffset = 0;
    const pending: PendingBatch[] = [];

    const flush = async () => {
        if (pending.length === 0) return;

        // Upload index list + batch infos.
        device.queue.writeBuffer(indexBuf, 0,
            indexScratch.buffer, 0, indexOffset * 4);

        const infoArr = new ArrayBuffer(pending.length * 32);
        const infoU = new Uint32Array(infoArr);
        const infoF = new Float32Array(infoArr);
        for (let i = 0; i < pending.length; i++) {
            const b = pending[i];
            const o = i * 8;
            infoU[o + 0] = b.indexOffset;
            infoU[o + 1] = b.indexCount;
            infoU[o + 2] = b.numBlocksX;
            infoU[o + 3] = b.numBlocksY;
            infoU[o + 4] = b.numBlocksZ;
            infoF[o + 5] = b.blockMinX;
            infoF[o + 6] = b.blockMinY;
            infoF[o + 7] = b.blockMinZ;
        }
        device.queue.writeBuffer(batchInfoBuf, 0, infoArr);

        // Zero-fill the results region we're about to dispatch into.
        const resBytes = pending.length * MAX_BLOCKS_PER_BATCH * 2 * 4;
        device.queue.writeBuffer(resultsBuf, 0, new Uint8Array(resBytes));

        const bg = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuf } },
                { binding: 1, resource: { buffer: gaussianBuf } },
                { binding: 2, resource: { buffer: indexBuf } },
                { binding: 3, resource: { buffer: resultsBuf } },
                { binding: 4, resource: { buffer: batchInfoBuf } }
            ]
        });
        const enc = device.createCommandEncoder();
        const p = enc.beginComputePass();
        p.setPipeline(pipeline);
        p.setBindGroup(0, bg);
        p.dispatchWorkgroups(MAX_BLOCKS_PER_BATCH, 1, pending.length);
        p.end();
        device.queue.submit([enc.finish()]);

        const raw = await readbackBuffer(device, resultsBuf, resBytes);
        const masks = new Uint32Array(raw.buffer, 0, resBytes / 4);
        for (let b = 0; b < pending.length; b++) {
            const batch = pending[b];
            const batchResultOffset = b * MAX_BLOCKS_PER_BATCH * 2;
            const totalBlocks = batch.numBlocksX * batch.numBlocksY * batch.numBlocksZ;
            for (let bi = 0; bi < totalBlocks; bi++) {
                const lo = masks[batchResultOffset + bi * 2];
                const hi = masks[batchResultOffset + bi * 2 + 1];
                if (lo === 0 && hi === 0) continue;
                const lx = bi % batch.numBlocksX;
                const ly = ((bi / batch.numBlocksX) | 0) % batch.numBlocksY;
                const lz = (bi / (batch.numBlocksX * batch.numBlocksY)) | 0;
                const ax = batch.bx + lx;
                const ay = batch.by + ly;
                const az = batch.bz + lz;
                const flat = ax + ay * numBlocksX + az * bStride;
                buffer.addBlock(flat, lo, hi);
            }
        }

        pending.length = 0;
        indexOffset = 0;
    };

    const batchSize = 16; // 16x16x16-block tiles
    for (let bz = 0; bz < numBlocksZ; bz += batchSize) {
        for (let by = 0; by < numBlocksY; by += batchSize) {
            for (let bx = 0; bx < numBlocksX; bx += batchSize) {
                const cx = Math.min(batchSize, numBlocksX - bx);
                const cy = Math.min(batchSize, numBlocksY - by);
                const cz = Math.min(batchSize, numBlocksZ - bz);

                const minX = bounds.min[0] + bx * blockSize;
                const minY = bounds.min[1] + by * blockSize;
                const minZ = bounds.min[2] + bz * blockSize;
                const maxX = minX + cx * blockSize;
                const maxY = minY + cy * blockSize;
                const maxZ = minZ + cz * blockSize;

                let count = bvh.queryOverlappingInto(
                    minX, minY, minZ, maxX, maxY, maxZ,
                    indexScratch, indexOffset
                );
                if (count === -1) {
                    // Overflow: flush + retry.
                    await flush();
                    count = bvh.queryOverlappingInto(
                        minX, minY, minZ, maxX, maxY, maxZ,
                        indexScratch, 0
                    );
                }
                if (count === 0) continue;

                pending.push({
                    indexOffset,
                    indexCount: count,
                    blockMinX: minX,
                    blockMinY: minY,
                    blockMinZ: minZ,
                    numBlocksX: cx,
                    numBlocksY: cy,
                    numBlocksZ: cz,
                    bx, by, bz
                });
                indexOffset += count;

                if (pending.length >= MEGA_MAX_BATCHES || indexOffset >= MEGA_MAX_INDICES) {
                    await flush();
                }
            }
        }
        await flush();
    }
    await flush();

    gaussianBuf.destroy();
    uniformBuf.destroy();
    indexBuf.destroy();
    batchInfoBuf.destroy();
    resultsBuf.destroy();

    return {
        buffer,
        bounds,
        voxelResolution: voxelRes,
        numBlocksX, numBlocksY, numBlocksZ
    };
};

const uploadGaussians = (
    device: GPUDevice,
    data: SplatData,
    extents: Float32Array
): GPUBuffer => {
    const n = data.count;
    const arr = new Float32Array(n * GAUSSIAN_STRIDE);
    for (let i = 0; i < n; i++) {
        const o = i * GAUSSIAN_STRIDE;
        arr[o + 0] = data.positions[i * 3 + 0];
        arr[o + 1] = data.positions[i * 3 + 1];
        arr[o + 2] = data.positions[i * 3 + 2];
        arr[o + 3] = data.opacityLogits[i];
        arr[o + 4] = data.rotations[i * 4 + 0];
        arr[o + 5] = data.rotations[i * 4 + 1];
        arr[o + 6] = data.rotations[i * 4 + 2];
        arr[o + 7] = data.rotations[i * 4 + 3];
        arr[o + 8] = data.logScales[i * 3 + 0];
        arr[o + 9] = data.logScales[i * 3 + 1];
        arr[o + 10] = data.logScales[i * 3 + 2];
        arr[o + 11] = extents[i * 3 + 0];
        arr[o + 12] = extents[i * 3 + 1];
        arr[o + 13] = extents[i * 3 + 2];
    }
    return uploadBuffer(device, arr,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'gaussians');
};

const computeBlockAlignedBounds = (
    data: SplatData,
    extents: Float32Array,
    blockSize: number
): Bounds => {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < data.count; i++) {
        const px = data.positions[i * 3 + 0];
        const py = data.positions[i * 3 + 1];
        const pz = data.positions[i * 3 + 2];
        const ex = extents[i * 3 + 0];
        const ey = extents[i * 3 + 1];
        const ez = extents[i * 3 + 2];
        if (px - ex < minX) minX = px - ex;
        if (py - ey < minY) minY = py - ey;
        if (pz - ez < minZ) minZ = pz - ez;
        if (px + ex > maxX) maxX = px + ex;
        if (py + ey > maxY) maxY = py + ey;
        if (pz + ez > maxZ) maxZ = pz + ez;
    }
    // Round outward to block boundaries.
    const align = (v: number, dir: number) =>
        dir < 0 ? Math.floor(v / blockSize) * blockSize : Math.ceil(v / blockSize) * blockSize;
    return {
        min: [align(minX, -1), align(minY, -1), align(minZ, -1)],
        max: [align(maxX, 1), align(maxY, 1), align(maxZ, 1)]
    };
};
