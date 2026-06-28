/**
 * Per-Gaussian world-space AABB computation.
 *
 * Each splat is an ellipsoid (Σ from quat * diag(scale²) * quat^T). The
 * world-space AABB of the 3-sigma ellipsoid is what voxelization and BVH
 * queries use to broad-phase reject.
 *
 * Closed form: for a unit-axis ellipsoid with semi-axes s_x, s_y, s_z rotated
 * by rotation matrix R, the AABB half-extent along world axis i is
 *   ext_i = sqrt(sum_j (R_ij * s_j)^2) * 3
 * (3-sigma is the truncation radius used by the rasterizer too).
 */

import type { SplatData } from '../types.ts';

const SIGMA_CUTOFF = 3.0;

/**
 * Compute per-Gaussian world-space 3sigma AABB extents. Returns a Float32Array
 * of length 3N (x, y, z half-extents per Gaussian, in world units).
 */
export const computeAabbExtents = (data: SplatData): Float32Array => {
    const n = data.count;
    const out = new Float32Array(n * 3);
    const rot = data.rotations;
    const ls = data.logScales;

    for (let i = 0; i < n; i++) {
        const qw = rot[i * 4 + 0];
        const qx = rot[i * 4 + 1];
        const qy = rot[i * 4 + 2];
        const qz = rot[i * 4 + 3];
        const sx = Math.exp(ls[i * 3 + 0]);
        const sy = Math.exp(ls[i * 3 + 1]);
        const sz = Math.exp(ls[i * 3 + 2]);

        // Build rotation matrix R from quaternion. Standard formula.
        const xx = qx * qx, yy = qy * qy, zz = qz * qz;
        const xy = qx * qy, xz = qx * qz, yz = qy * qz;
        const wx = qw * qx, wy = qw * qy, wz = qw * qz;

        const r00 = 1 - 2 * (yy + zz);
        const r01 = 2 * (xy - wz);
        const r02 = 2 * (xz + wy);
        const r10 = 2 * (xy + wz);
        const r11 = 1 - 2 * (xx + zz);
        const r12 = 2 * (yz - wx);
        const r20 = 2 * (xz - wy);
        const r21 = 2 * (yz + wx);
        const r22 = 1 - 2 * (xx + yy);

        // ext_i = sqrt( (R_i0 * s_x)^2 + (R_i1 * s_y)^2 + (R_i2 * s_z)^2 ) * sigma
        const ex = Math.sqrt((r00 * sx) ** 2 + (r01 * sy) ** 2 + (r02 * sz) ** 2) * SIGMA_CUTOFF;
        const ey = Math.sqrt((r10 * sx) ** 2 + (r11 * sy) ** 2 + (r12 * sz) ** 2) * SIGMA_CUTOFF;
        const ez = Math.sqrt((r20 * sx) ** 2 + (r21 * sy) ** 2 + (r22 * sz) ** 2) * SIGMA_CUTOFF;

        out[i * 3 + 0] = ex;
        out[i * 3 + 1] = ey;
        out[i * 3 + 2] = ez;
    }
    return out;
};
