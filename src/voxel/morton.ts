/**
 * Morton (Z-order) encoding for 4x4x4 blocks.
 *
 * Within a 4³ block, voxels are addressed by a Morton index in [0, 64). Each
 * coord uses 2 bits, interleaved as z2 y2 x2 z1 y1 x1.
 *
 * The encode/decode functions are used both CPU-side (block iteration) and
 * shader-side (the WGSL prelude redefines them). See `gpu-voxelization.ts`.
 */

export const mortonEncode3 = (x: number, y: number, z: number): number => {
    return ((x & 1) << 0) | ((y & 1) << 1) | ((z & 1) << 2)
         | ((x & 2) << 2) | ((y & 2) << 3) | ((z & 2) << 4);
};

export const mortonDecode3 = (m: number): readonly [number, number, number] => {
    return [
        (m & 1) | ((m >> 2) & 2),
        ((m >> 1) & 1) | ((m >> 3) & 2),
        ((m >> 2) & 1) | ((m >> 4) & 2)
    ];
};
