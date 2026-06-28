/**
 * Emit (tileId, splatIdx) pairs.
 *
 * For each splat with non-zero coverage, walk its tile AABB and write a pair
 * for each tile it touches. Pairs are written into the slot reserved by the
 * prefix-sum result (`emitOffsets`). No atomics; offsets are disjoint.
 */

import { SHADER_PRELUDE } from './constants.ts';

export const emitPairsShader = () => /* wgsl */`
${SHADER_PRELUDE}

struct Uniforms {
    tilesX: u32,
    tilesY: u32,
    numSplats: u32,
    maxCoverage: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> projected: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> emitOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> pairTiles: array<u32>;
@group(0) @binding(4) var<storage, read_write> pairSplats: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.numSplats) { return; }

    let v0 = projected[idx * 3u + 0u];
    let screenX = v0.x;
    let screenY = v0.y;
    let radius = v0.z;
    if (radius <= 0.0) { return; }

    let minTileX = u32(max(0.0, floor((screenX - radius) / f32(TILE_SIZE))));
    let minTileY = u32(max(0.0, floor((screenY - radius) / f32(TILE_SIZE))));
    let maxTileX = u32(min(f32(u.tilesX) - 1.0, floor((screenX + radius) / f32(TILE_SIZE))));
    let maxTileY = u32(min(f32(u.tilesY) - 1.0, floor((screenY + radius) / f32(TILE_SIZE))));
    if (maxTileX < minTileX || maxTileY < minTileY) { return; }

    var offset = emitOffsets[idx];
    var written: u32 = 0u;
    for (var ty = minTileY; ty <= maxTileY; ty = ty + 1u) {
        for (var tx = minTileX; tx <= maxTileX; tx = tx + 1u) {
            if (written >= u.maxCoverage) { return; }
            let tileId = ty * u.tilesX + tx;
            pairTiles[offset] = tileId;
            pairSplats[offset] = idx;
            offset = offset + 1u;
            written = written + 1u;
        }
    }
}
`;
