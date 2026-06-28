/**
 * Final composite — pulls the accumulated (color, T) image and composites
 * over the background, producing 8-bit RGBA for canvas display.
 *
 * Pixel output: rgb = color + T * background; alpha = 1 - T (or always 1 if
 * we're rendering to canvas).
 */

import { SHADER_PRELUDE } from './constants.ts';

export const finalizeShader = () => /* wgsl */`
${SHADER_PRELUDE}

struct Uniforms {
    width: u32,
    height: u32,
    bgR: f32,
    bgG: f32,
    bgB: f32,
    bgA: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> inRGBA: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> outRGBA: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let i = gid.y * u.width + gid.x;
    let v = inRGBA[i];
    let bg = vec3<f32>(u.bgR, u.bgG, u.bgB);
    let rgb = clamp(v.rgb + v.a * bg, vec3<f32>(0.0), vec3<f32>(1.0));
    let aOut = clamp(1.0 - v.a + v.a * u.bgA, 0.0, 1.0);
    let r = u32(rgb.r * 255.0 + 0.5);
    let g = u32(rgb.g * 255.0 + 0.5);
    let b = u32(rgb.b * 255.0 + 0.5);
    let a = u32(aOut * 255.0 + 0.5);
    outRGBA[i] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
`;
