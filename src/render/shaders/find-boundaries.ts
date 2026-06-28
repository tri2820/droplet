/**
 * Find per-tile slice boundaries in the sorted pair list.
 *
 * After tile-id radix sort, contiguous runs of identical tile IDs in the
 * `pairTiles` array are one tile's splat slice. We need:
 *
 *   tileOffsets[t]   = index of first pair with tileId == t
 *   tileOffsets[T]   = numPairs   (sentinel for last tile's end)
 *
 * Each thread looks at index i and i-1; if they differ, write boundaries for
 * all tile IDs in between. Initial atomicMin to `numPairs + 1` ensures empty
 * tiles get a valid sentinel.
 */

import { SHADER_PRELUDE } from './constants.ts';

export const initTileOffsetsShader = () => /* wgsl */`
${SHADER_PRELUDE}

@group(0) @binding(0) var<storage, read_write> tileOffsets: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> uN: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i > uN) { return; }
    atomicStore(&tileOffsets[i], 0xffffffffu);
}
`;

export const findBoundariesShader = () => /* wgsl */`
${SHADER_PRELUDE}

struct Uniforms {
    numPairs: u32,
    numTiles: u32,
    // For 2D dispatch: stride = dispatchX * 64 (workgroup_size.x).
    // Linear pair index = gid.y * stride + gid.x. Required because a
    // single-dim dispatch caps at 65535 workgroups in X, but real
    // scenes can produce > 4 M pairs per chunk → > 65 K workgroups.
    stride: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> pairTiles: array<u32>;
@group(0) @binding(2) var<storage, read_write> tileOffsets: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * u.stride + gid.x;
    if (i >= u.numPairs) { return; }
    let cur = pairTiles[i];
    if (i == 0u) {
        // First pair: all tiles up to cur start at 0; tiles >= cur start at 0.
        // Use atomicMin so we don't clobber if later threads write smaller values
        // (they can't, but the API requires atomic).
        atomicMin(&tileOffsets[cur], 0u);
    } else {
        let prev = pairTiles[i - 1u];
        if (cur != prev) {
            // Every tile in (prev, cur] starts at i.
            for (var t = prev + 1u; t <= cur; t = t + 1u) {
                atomicMin(&tileOffsets[t], i);
            }
        }
    }
}
`;

export const finalizeTileOffsetsShader = () => /* wgsl */`
${SHADER_PRELUDE}

struct Uniforms {
    numPairs: u32,
    numTiles: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> tileOffsets: array<u32>;

// Fix-up: any tile that wasn't touched still has 0xffffffff. Sweep right-to-left
// to fill them with the next valid offset. Single-thread pass; numTiles is small
// (8160 at 1080p with TILE_SIZE=16).
@compute @workgroup_size(1)
fn main() {
    tileOffsets[u.numTiles] = u.numPairs;
    var fill: u32 = u.numPairs;
    var i: u32 = u.numTiles;
    loop {
        if (i == 0u) { break; }
        i = i - 1u;
        if (tileOffsets[i] == 0xffffffffu) {
            tileOffsets[i] = fill;
        } else {
            fill = tileOffsets[i];
        }
    }
}
`;
