/**
 * Project compute shader.
 *
 * For each visible splat:
 *  1. World->camera transform.
 *  2. Build 3D covariance Σ = R · diag(s²) · Rᵀ from quaternion + log-scales.
 *  3. Apply EWA Jacobian for perspective projection → 2D covariance Σ'.
 *  4. Add `AA_DILATION_COV` to diagonal (anti-aliasing for sub-pixel splats).
 *  5. Invert Σ' to get conic (the quadratic form for the 2D Gaussian).
 *  6. Evaluate SH band 0..bands against view direction → RGB.
 *  7. Write projected record: (screenX, screenY, radius), conic + alpha, RGB.
 *  8. Compute tile-coverage AABB and write per-splat tile count (clamped).
 *
 * Output layout (3 vec4<f32> per splat, packed contiguously):
 *   slot 0: vec4(screenX, screenY, radius, depth)
 *   slot 1: vec4(conicA, conicB, conicC, alpha)
 *   slot 2: vec4(R, G, B, _pad)
 */

import { SHADER_PRELUDE } from './constants.ts';

export const projectShader = (bands: 0 | 1 | 2 | 3, projection: 'pinhole' | 'equirect') => /* wgsl */`
${SHADER_PRELUDE}

struct Uniforms {
    viewMatrix: mat4x4<f32>,
    width: f32,
    height: f32,
    focal: f32,
    cx: f32,
    cy: f32,
    near: f32,
    far: f32,
    numSplats: u32,
    tilesX: u32,
    tilesY: u32,
    maxCoverage: u32,
    shBands: u32,
    inputStride: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> splatInput: array<f32>;
@group(0) @binding(2) var<storage, read_write> projected: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> tileCounts: array<u32>;

// Rotate vector v by quaternion (w, x, y, z). Rodrigues form.
fn quatRot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.yzw, v);
    return v + q.x * t + cross(q.yzw, t);
}

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

${bands === 0 ? '' : evalSHDecl(bands)}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.numSplats) { return; }

    let base = idx * u.inputStride;
    let pos = vec3<f32>(splatInput[base + 0u], splatInput[base + 1u], splatInput[base + 2u]);
    let rot = vec4<f32>(
        splatInput[base + 3u], splatInput[base + 4u],
        splatInput[base + 5u], splatInput[base + 6u]
    );
    let logScale = vec3<f32>(
        splatInput[base + 7u], splatInput[base + 8u], splatInput[base + 9u]
    );
    let opacityLogit = splatInput[base + 10u];
    let dc = vec3<f32>(
        splatInput[base + 11u], splatInput[base + 12u], splatInput[base + 13u]
    );
    let scale = exp(logScale);

    // World -> camera
    let posCam4 = u.viewMatrix * vec4<f32>(pos, 1.0);
    let posCam = posCam4.xyz;
    let depth = -posCam.z;

    ${projection === 'pinhole' ? pinholeProjectBody() : equirectProjectBody()}

    // 3D covariance Σ = R · diag(s²) · Rᵀ. We build via the columns of R·diag(s).
    let s = scale;
    // Rotation matrix columns by rotating basis vectors.
    let rx = quatRot(rot, vec3<f32>(1.0, 0.0, 0.0)) * s.x;
    let ry = quatRot(rot, vec3<f32>(0.0, 1.0, 0.0)) * s.y;
    let rz = quatRot(rot, vec3<f32>(0.0, 0.0, 1.0)) * s.z;
    // Σ = M Mᵀ where M = [rx ry rz]
    let cov3_00 = rx.x * rx.x + ry.x * ry.x + rz.x * rz.x;
    let cov3_01 = rx.x * rx.y + ry.x * ry.y + rz.x * rz.y;
    let cov3_02 = rx.x * rx.z + ry.x * ry.z + rz.x * rz.z;
    let cov3_11 = rx.y * rx.y + ry.y * ry.y + rz.y * rz.y;
    let cov3_12 = rx.y * rx.z + ry.y * ry.z + rz.y * rz.z;
    let cov3_22 = rx.z * rx.z + ry.z * ry.z + rz.z * rz.z;

    // Transform Σ into camera frame: Σ_cam = V · Σ · Vᵀ, where V is the upper-left 3x3
    // of view matrix. Since the rotation we used for the Gaussian basis is already in
    // world space, we now need V·Σ·Vᵀ. The upper-left 3x3 of u.viewMatrix:
    let v00 = u.viewMatrix[0].x; let v01 = u.viewMatrix[1].x; let v02 = u.viewMatrix[2].x;
    let v10 = u.viewMatrix[0].y; let v11 = u.viewMatrix[1].y; let v12 = u.viewMatrix[2].y;
    let v20 = u.viewMatrix[0].z; let v21 = u.viewMatrix[1].z; let v22 = u.viewMatrix[2].z;

    // V·Σ
    let a00 = v00 * cov3_00 + v01 * cov3_01 + v02 * cov3_02;
    let a01 = v00 * cov3_01 + v01 * cov3_11 + v02 * cov3_12;
    let a02 = v00 * cov3_02 + v01 * cov3_12 + v02 * cov3_22;
    let a10 = v10 * cov3_00 + v11 * cov3_01 + v12 * cov3_02;
    let a11 = v10 * cov3_01 + v11 * cov3_11 + v12 * cov3_12;
    let a12 = v10 * cov3_02 + v11 * cov3_12 + v12 * cov3_22;
    let a20 = v20 * cov3_00 + v21 * cov3_01 + v22 * cov3_02;
    let a21 = v20 * cov3_01 + v21 * cov3_11 + v22 * cov3_12;
    let a22 = v20 * cov3_02 + v21 * cov3_12 + v22 * cov3_22;
    // (V·Σ)·Vᵀ
    let cc00 = a00 * v00 + a01 * v01 + a02 * v02;
    let cc01 = a00 * v10 + a01 * v11 + a02 * v12;
    let cc02 = a00 * v20 + a01 * v21 + a02 * v22;
    let cc11 = a10 * v10 + a11 * v11 + a12 * v12;
    let cc12 = a10 * v20 + a11 * v21 + a12 * v22;
    let cc22 = a20 * v20 + a21 * v21 + a22 * v22;

    // EWA Jacobian J for pinhole projection:
    //   J = [[f/z, 0, -f*x/z²],
    //        [0, f/z, -f*y/z²]]
    let z = posCam.z;
    let invZ = 1.0 / z;
    let j00 = u.focal * invZ;
    let j02 = -u.focal * posCam.x * invZ * invZ;
    let j11 = u.focal * invZ;
    let j12 = -u.focal * posCam.y * invZ * invZ;

    // Σ_2D = J · Σ_cam · Jᵀ  (only top-left 2x2 of result needed)
    let b00 = j00 * cc00 + j02 * cc02;
    let b01 = j00 * cc01 + j02 * cc12;
    let b02 = j00 * cc02 + j02 * cc22;
    let b10 = j11 * cc01 + j12 * cc02;
    let b11 = j11 * cc11 + j12 * cc12;
    let b12 = j11 * cc12 + j12 * cc22;

    // cov2D = b * J^T. J^T has rows [j00,0], [0,j11], [j02,j12], so:
    //   cov2D[0,0] = b00*j00 + b02*j02
    //   cov2D[0,1] = b01*j11 + b02*j12
    //   cov2D[1,1] = b11*j11 + b12*j12
    var cov2_00 = b00 * j00 + b02 * j02;
    var cov2_01 = b01 * j11 + b02 * j12;
    var cov2_11 = b11 * j11 + b12 * j12;

    // Apply anti-aliasing dilation: adds a 0.3px Gaussian to the splat.
    cov2_00 = cov2_00 + AA_DILATION_COV;
    cov2_11 = cov2_11 + AA_DILATION_COV;

    // Invert 2x2 covariance to get conic.
    let det = cov2_00 * cov2_11 - cov2_01 * cov2_01;
    if (det <= 1.0e-6) {
        tileCounts[idx] = 0u;
        projected[idx * 3u + 0u] = vec4<f32>(0.0, 0.0, -1.0, 0.0);
        return;
    }
    let invDet = 1.0 / det;
    let conicA = cov2_11 * invDet;
    let conicB = -cov2_01 * invDet;
    let conicC = cov2_00 * invDet;

    // Screen-space radius (3σ AABB radius) from eigenvalue bound.
    let mid = 0.5 * (cov2_00 + cov2_11);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let radiusRaw = SIGMA_CUTOFF * sqrt(lambda1);

    // Outlier-splat fade. A handful of trained splats project to
    // screen-spanning footprints (close-by mega-splats or training
    // pathologies); without a fade they tint the whole frame and
    // saturate the rasterizer's tile coverage. Linearly drop alpha
    // from 1 → 0 as the un-clamped radius grows past fadeStart,
    // fully discarded past fadeEnd. Thresholds are fractions of
    // image height so the SAME world-space splats fade at every
    // render resolution. Matches splat-transform's RADIUS_FADE_*_FRAC.
    let fadeStart = RADIUS_FADE_START_FRAC * u.height;
    let fadeEnd = RADIUS_FADE_END_FRAC * u.height;
    let radiusFade = clamp((fadeEnd - radiusRaw) / (fadeEnd - fadeStart), 0.0, 1.0);
    if (radiusFade <= 0.0) {
        tileCounts[idx] = 0u;
        projected[idx * 3u + 0u] = vec4<f32>(0.0, 0.0, -1.0, 0.0);
        return;
    }
    let radius = ceil(min(radiusRaw, fadeEnd));

    // View-dependent color via SH evaluation.
    var color = dc;
    ${bands === 0 ? '' : `
    let viewDir = normalize(posCam);
    color = color + evalSH(idx, viewDir);
    `}
    // Clamp the per-splat color to [0, 1]. The SH-C0 + 0.5 convention
    // assumes the model was trained to land in this range, but real
    // scenes commonly contain a small fraction of splats with
    // out-of-range bright values (training pathologies near highlights
    // / lights). Without a clamp these saturate to white halos when
    // many overlap. We clamp per-splat rather than tone-map post-
    // composite so each splat's contribution stays physically bounded.
    color = clamp(color + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));

    let alpha = sigmoid(opacityLogit) * radiusFade;

    projected[idx * 3u + 0u] = vec4<f32>(screenX, screenY, radius, depth);
    projected[idx * 3u + 1u] = vec4<f32>(conicA, conicB, conicC, alpha);
    projected[idx * 3u + 2u] = vec4<f32>(color, 0.0);

    // Tile coverage AABB.
    let minTileX = u32(max(0.0, floor((screenX - radius) / f32(TILE_SIZE))));
    let minTileY = u32(max(0.0, floor((screenY - radius) / f32(TILE_SIZE))));
    let maxTileX = u32(min(f32(u.tilesX) - 1.0, floor((screenX + radius) / f32(TILE_SIZE))));
    let maxTileY = u32(min(f32(u.tilesY) - 1.0, floor((screenY + radius) / f32(TILE_SIZE))));
    var count: u32 = 0u;
    if (maxTileX >= minTileX && maxTileY >= minTileY) {
        count = (maxTileX - minTileX + 1u) * (maxTileY - minTileY + 1u);
        if (count > u.maxCoverage) { count = u.maxCoverage; }
    }
    tileCounts[idx] = count;
}
`;

const pinholeProjectBody = () => /* wgsl */`
    if (depth <= u.near || depth >= u.far) {
        tileCounts[idx] = 0u;
        projected[idx * 3u + 0u] = vec4<f32>(0.0, 0.0, -1.0, 0.0);
        return;
    }
    // Pinhole projection. World +X projects to screen +X (right). World +Y
    // is "up", but canvas Y grows downward, so screenY subtracts the y term.
    let projX = posCam.x * u.focal / -posCam.z;
    let projY = -posCam.y * u.focal / -posCam.z;
    let screenX = projX + u.cx;
    let screenY = projY + u.cy;
`;

const equirectProjectBody = () => /* wgsl */`
    let r = length(posCam);
    if (r <= u.near) {
        tileCounts[idx] = 0u;
        projected[idx * 3u + 0u] = vec4<f32>(0.0, 0.0, -1.0, 0.0);
        return;
    }
    let lon = atan2(posCam.x, -posCam.z);
    let lat = asin(posCam.y / r);
    let screenX = (lon / (3.14159265 * 2.0) + 0.5) * u.width;
    let screenY = (0.5 - lat / 3.14159265) * u.height;
`;

const evalSHDecl = (bands: 1 | 2 | 3) => {
    const M = bands === 1 ? 3 : bands === 2 ? 8 : 15;
    return /* wgsl */`
// SH evaluation. Coeffs are channel-major in splatInput starting at offset 14.
// For each splat we have ${M} coeffs per channel.
fn evalSH(idx: u32, dir: vec3<f32>) -> vec3<f32> {
    let base = idx * u.inputStride + 14u;
    let x = dir.x; let y = dir.y; let z = dir.z;
    let M: u32 = ${M}u;
    var r: f32 = 0.0; var g: f32 = 0.0; var b: f32 = 0.0;
    // Band 1: l=1
    let SH1: f32 = 0.4886025119029199;
    r = r + (-SH1 * y) * splatInput[base + 0u];
    r = r + ( SH1 * z) * splatInput[base + 1u];
    r = r + (-SH1 * x) * splatInput[base + 2u];
    g = g + (-SH1 * y) * splatInput[base + M + 0u];
    g = g + ( SH1 * z) * splatInput[base + M + 1u];
    g = g + (-SH1 * x) * splatInput[base + M + 2u];
    b = b + (-SH1 * y) * splatInput[base + 2u*M + 0u];
    b = b + ( SH1 * z) * splatInput[base + 2u*M + 1u];
    b = b + (-SH1 * x) * splatInput[base + 2u*M + 2u];
    ${bands >= 2 ? `
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let yz = y * z; let xz = x * z;
    let SH2_0 = 1.0925484305920792;
    let SH2_1 = -1.0925484305920792;
    let SH2_2 = 0.31539156525252005;
    let SH2_3 = -1.0925484305920792;
    let SH2_4 = 0.5462742152960396;
    let band2 = array<f32, 5>(
        SH2_0 * xy,
        SH2_1 * yz,
        SH2_2 * (2.0 * zz - xx - yy),
        SH2_3 * xz,
        SH2_4 * (xx - yy)
    );
    for (var k: u32 = 0u; k < 5u; k = k + 1u) {
        r = r + band2[k] * splatInput[base + 3u + k];
        g = g + band2[k] * splatInput[base + M + 3u + k];
        b = b + band2[k] * splatInput[base + 2u*M + 3u + k];
    }
    ` : ''}
    ${bands >= 3 ? `
    let SH3_0 = -0.5900435899266435;
    let SH3_1 =  2.890611442640554;
    let SH3_2 = -0.4570457994644658;
    let SH3_3 =  0.3731763325901154;
    let SH3_4 = -0.4570457994644658;
    let SH3_5 =  1.445305721320277;
    let SH3_6 = -0.5900435899266435;
    let band3 = array<f32, 7>(
        SH3_0 * y * (3.0 * xx - yy),
        SH3_1 * xy * z,
        SH3_2 * y * (4.0 * zz - xx - yy),
        SH3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        SH3_4 * x * (4.0 * zz - xx - yy),
        SH3_5 * z * (xx - yy),
        SH3_6 * x * (xx - 3.0 * yy)
    );
    for (var k: u32 = 0u; k < 7u; k = k + 1u) {
        r = r + band3[k] * splatInput[base + 8u + k];
        g = g + band3[k] * splatInput[base + M + 8u + k];
        b = b + band3[k] * splatInput[base + 2u*M + 8u + k];
    }
    ` : ''}
    return vec3<f32>(r, g, b);
}
`;
};
