/**
 * Exclusive prefix sum (scan). Used to convert per-splat tile-coverage counts
 * into emit offsets so each splat can write its (tile, splatIdx) pairs into
 * disjoint slots without atomics.
 *
 * Implementation: simple two-level Blelloch scan in two dispatches:
 *  1. `scanBlocksShader` — per-workgroup local exclusive scan, plus per-block
 *     sums into a second buffer.
 *  2. `addBlockOffsetsShader` — after the host scans the per-block sums (CPU
 *     fallback for small block counts), this pass adds each block's prefix to
 *     its elements.
 *
 * For our use case the block count is small (max 2M elements / 256 per block
 * = ~8k blocks) so a CPU exclusive-scan over block sums is fine and avoids
 * a third pass.
 */

import { SHADER_PRELUDE } from './constants.ts';

export const PREFIX_BLOCK = 256;

export const prefixScanBlocksShader = () => /* wgsl */`
${SHADER_PRELUDE}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(2) var<uniform> uN: u32;

const BLOCK: u32 = ${PREFIX_BLOCK}u;
var<workgroup> tmp: array<u32, ${PREFIX_BLOCK * 2}>;

@compute @workgroup_size(${PREFIX_BLOCK})
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let blockOffset = wgId.x * BLOCK;
    let i = blockOffset + tid;
    var v: u32 = 0u;
    if (i < uN) { v = data[i]; }
    tmp[tid] = v;
    workgroupBarrier();

    // Hillis-Steele inclusive scan in shared memory.
    var step: u32 = 1u;
    while (step < BLOCK) {
        var t: u32 = 0u;
        if (tid >= step) { t = tmp[tid - step]; }
        workgroupBarrier();
        tmp[tid] = tmp[tid] + t;
        workgroupBarrier();
        step = step * 2u;
    }

    // Convert inclusive -> exclusive by subtracting own value.
    let inc = tmp[tid];
    let exc = inc - v;
    if (i < uN) { data[i] = exc; }

    // Last thread writes block sum.
    if (tid == BLOCK - 1u) {
        blockSums[wgId.x] = inc;
    }
}
`;

export const prefixAddBlockOffsetsShader = () => /* wgsl */`
${SHADER_PRELUDE}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> blockOffsets: array<u32>;
@group(0) @binding(2) var<uniform> uN: u32;

const BLOCK: u32 = ${PREFIX_BLOCK}u;

@compute @workgroup_size(${PREFIX_BLOCK})
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let i = wgId.x * BLOCK + lid.x;
    if (i >= uN) { return; }
    data[i] = data[i] + blockOffsets[wgId.x];
}
`;
