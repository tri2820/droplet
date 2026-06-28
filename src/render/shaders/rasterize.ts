/**
 * Tile-binned rasterizer. One workgroup per tile, TILE_SIZE×TILE_SIZE threads
 * (= one thread per pixel).
 *
 * Each pixel walks its tile's depth-sorted splat slice and accumulates color
 * via front-to-back alpha-over compositing:
 *
 *   color += T · α_i · color_i
 *   T     *= (1 - α_i)
 *
 * Once T < MIN_TRANSMITTANCE the pixel is saturated and exits.
 *
 * Output: an RGBA storage buffer (`outRGBA`) holding accumulated color +
 *         residual transmittance. The host composites the background.
 */

import { SHADER_PRELUDE } from './constants.ts';

export const rasterizeShader = () => /* wgsl */`
${SHADER_PRELUDE}

struct Uniforms {
    tilesX: u32,
    tilesY: u32,
    imageWidth: u32,
    imageHeight: u32,
    _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> projected: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(3) var<storage, read> pairSplats: array<u32>;
@group(0) @binding(4) var<storage, read_write> outRGBA: array<vec4<f32>>;

@compute @workgroup_size(${16}, ${16}, 1)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    if (wgId.x >= u.tilesX || wgId.y >= u.tilesY) { return; }
    let tileIdx = wgId.y * u.tilesX + wgId.x;
    let sliceStart = tileOffsets[tileIdx];
    let sliceEnd = tileOffsets[tileIdx + 1u];

    let pixelX = wgId.x * TILE_SIZE + lid.x;
    let pixelY = wgId.y * TILE_SIZE + lid.y;
    if (pixelX >= u.imageWidth || pixelY >= u.imageHeight) { return; }
    let pixelIdx = pixelY * u.imageWidth + pixelX;

    let px = f32(pixelX) + 0.5;
    let py = f32(pixelY) + 0.5;

    var color = vec3<f32>(0.0);
    var T: f32 = 1.0;

    for (var i = sliceStart; i < sliceEnd; i = i + 1u) {
        if (T < MIN_TRANSMITTANCE) { break; }
        let splatIdx = pairSplats[i];
        let v0 = projected[splatIdx * 3u + 0u];
        let dx = px - v0.x;
        let dy = py - v0.y;
        let r = v0.z;
        if (r <= 0.0 || abs(dx) > r || abs(dy) > r) { continue; }
        let v1 = projected[splatIdx * 3u + 1u];
        let power = -0.5 * (v1.x * dx * dx + 2.0 * v1.y * dx * dy + v1.z * dy * dy);
        if (power > 0.0) { continue; }
        let alpha = min(OPACITY_CAP, v1.w * max(0.0, exp(power) - GAUSSIAN_FLOOR));
        if (alpha < MIN_ALPHA) { continue; }
        let weight = T * alpha;
        let v2 = projected[splatIdx * 3u + 2u];
        color = color + weight * v2.rgb;
        T = T * (1.0 - alpha);
    }

    outRGBA[pixelIdx] = vec4<f32>(color, T);
}
`;
