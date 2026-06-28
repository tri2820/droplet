/**
 * CPU-side preprocessing for the renderer:
 *   1. Frustum cull and depth-sort visible splats (front-to-back).
 *   2. Pack visible splats into a tight Float32Array for upload.
 *
 * The packed layout matches the project shader's expected stride
 * (`splatInputStride(bands)` in types.ts):
 *
 *   [0..2]   pos.xyz
 *   [3..6]   rot.w, rot.x, rot.y, rot.z
 *   [7..9]   log_scale.xyz
 *   [10]     opacity (logit)
 *   [11..13] DC color
 *   [14..]   SH coefs, channel-major: R[0..M-1], G[0..M-1], B[0..M-1]
 */

import type { SplatData } from '../types.ts';
import { numSHCoeffsPerChannel, splatInputStride } from '../types.ts';
import {
    radixSortIndicesByFloat,
    createRadixScratch,
    type RadixSortScratch
} from '../spatial/radix-sort.ts';
import { cullToVisible, type CameraBasis } from './camera.ts';

export interface PreprocessOutput {
    /** Number of visible splats (≤ data.count). */
    numVisible: number;
    /** Visible splat indices, sorted front-to-back. Aliased into scratch. */
    sortedIndices: Uint32Array;
    /** Camera-space depth for each visible splat (parallel to sortedIndices). */
    depths: Float32Array;
}

export interface PreprocessScratch {
    indices: Uint32Array;
    depths: Float32Array;
    radix: RadixSortScratch;
    capacity: number;
}

export const createPreprocessScratch = (): PreprocessScratch => ({
    indices: new Uint32Array(0),
    depths: new Float32Array(0),
    radix: createRadixScratch(),
    capacity: 0
});

const ensureCap = (s: PreprocessScratch, n: number) => {
    if (s.capacity < n) {
        s.indices = new Uint32Array(n);
        s.depths = new Float32Array(n);
        s.capacity = n;
    }
};

/**
 * Cull + sort. Returns aliased views into scratch — do not mutate.
 */
export const preprocess = (
    data: SplatData,
    basis: CameraBasis,
    scratch: PreprocessScratch
): PreprocessOutput => {
    ensureCap(scratch, data.count);
    const numVisible = cullToVisible(
        data.positions, data.count, basis, scratch.indices, scratch.depths
    );
    radixSortIndicesByFloat(scratch.indices, scratch.depths, scratch.radix, numVisible);
    // Re-gather depths in the post-sort order so the parallel view is correct.
    // (radix-sort only permutes indices; depths[] still holds the visible-order keys.)
    // We rebuild depths from positions on demand if needed downstream.
    return {
        numVisible,
        sortedIndices: scratch.indices,
        depths: scratch.depths
    };
};

/**
 * Pack a range of visible splats (chunk) into a tight Float32Array for upload
 * to the project shader.
 *
 * @param data        - Source splat data.
 * @param sortedIdx   - Front-to-back sorted indices into `data`.
 * @param chunkStart  - First index in the visible range.
 * @param chunkCount  - Number of splats in the chunk.
 * @param out         - Float32Array of length >= chunkCount * stride.
 */
export const packChunkInput = (
    data: SplatData,
    sortedIdx: Uint32Array,
    chunkStart: number,
    chunkCount: number,
    out: Float32Array
): void => {
    const bands = data.shBands;
    const m = numSHCoeffsPerChannel(bands);
    const stride = splatInputStride(bands);

    const pos = data.positions;
    const rot = data.rotations;
    const ls = data.logScales;
    const op = data.opacityLogits;
    const dc = data.colorsDC;
    const sh = data.colorsSH;

    for (let i = 0; i < chunkCount; i++) {
        const g = sortedIdx[chunkStart + i];
        const o = i * stride;
        out[o + 0] = pos[g * 3 + 0];
        out[o + 1] = pos[g * 3 + 1];
        out[o + 2] = pos[g * 3 + 2];
        out[o + 3] = rot[g * 4 + 0];
        out[o + 4] = rot[g * 4 + 1];
        out[o + 5] = rot[g * 4 + 2];
        out[o + 6] = rot[g * 4 + 3];
        out[o + 7] = ls[g * 3 + 0];
        out[o + 8] = ls[g * 3 + 1];
        out[o + 9] = ls[g * 3 + 2];
        out[o + 10] = op[g];
        out[o + 11] = dc[g * 3 + 0];
        out[o + 12] = dc[g * 3 + 1];
        out[o + 13] = dc[g * 3 + 2];
        if (bands > 0 && sh) {
            // Channel-major SH: src layout per Gaussian is also channel-major.
            // We assume the source `colorsSH` is laid out as
            //   sh[g * (3 * m) + 0..m-1]      = R coeffs
            //   sh[g * (3 * m) + m..2m-1]     = G coeffs
            //   sh[g * (3 * m) + 2m..3m-1]    = B coeffs
            const srcBase = g * 3 * m;
            const dstBase = o + 14;
            for (let k = 0; k < 3 * m; k++) {
                out[dstBase + k] = sh[srcBase + k];
            }
        }
    }
};
