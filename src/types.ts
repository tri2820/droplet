/**
 * Shared types for droplet.
 *
 * The core data structure is `SplatData`: a struct-of-arrays Gaussian buffer
 * with explicit float channels. We keep position/rotation/scale/opacity/SH
 * separate so consumers can stream parts of the data without re-packing.
 */

import type { Vec3, Mat4 } from './math/index.ts';

/**
 * Gaussian splat data, SoA. All arrays are the same length N.
 *
 * - `positions`: float32 length 3N — xyz, world space
 * - `rotations`: float32 length 4N — quaternion [w, x, y, z], unit length
 * - `logScales`: float32 length 3N — ln(scale_xyz), so exp(logScale) = sigma
 * - `opacityLogits`: float32 length N — pre-sigmoid opacity
 * - `colorsDC`: float32 length 3N — SH band-0 coefficients (or plain RGB / sqrt(C0))
 * - `colorsSH`: optional float32 — channel-major view-dependent SH terms.
 *   Layout per Gaussian: [R[0..N-1], G[0..N-1], B[0..N-1]] where
 *   N = numCoeffsPerChannel(bands). See `numSHCoeffsPerChannel`.
 */
export interface SplatData {
    count: number;
    positions: Float32Array;
    rotations: Float32Array;
    logScales: Float32Array;
    opacityLogits: Float32Array;
    colorsDC: Float32Array;
    colorsSH?: Float32Array;
    shBands: 0 | 1 | 2 | 3;
}

/** Number of SH coefficients per color channel for a given band count. */
export const numSHCoeffsPerChannel = (bands: 0 | 1 | 2 | 3): number => {
    return bands === 0 ? 0 : bands === 1 ? 3 : bands === 2 ? 8 : 15;
};

/**
 * Total floats per Gaussian when packed for the GPU project shader. Layout:
 *
 *   [0..2]   pos.xyz
 *   [3..6]   rot.w, rot.x, rot.y, rot.z
 *   [7..9]   log_scale.xyz
 *   [10]     opacity (logit)
 *   [11..13] f_dc.rgb
 *   [14..]   SH channel-major
 */
export const splatInputStride = (bands: 0 | 1 | 2 | 3): number => {
    return 14 + 3 * numSHCoeffsPerChannel(bands);
};

/** Axis-aligned bounding box. */
export interface Bounds {
    min: Vec3;
    max: Vec3;
}

/** Background color, RGBA in [0, 1]. */
export interface BackgroundRGBA {
    r: number;
    g: number;
    b: number;
    a: number;
}

/** Pinhole or equirectangular projection. Pinhole is the common case. */
export type Projection = 'pinhole' | 'equirect';

/**
 * Camera parameters for one render. `viewMatrix` is world->camera (col-major
 * Float32Array(16)). `fovY` is vertical field of view in radians (pinhole only).
 */
export interface RenderCamera {
    width: number;
    height: number;
    viewMatrix: Mat4;
    fovY: number;
    near: number;
    far: number;
    projection?: Projection;
}
