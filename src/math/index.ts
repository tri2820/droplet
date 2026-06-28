/**
 * Minimal vec/quat/mat helpers. Independent — no gl-matrix or PlayCanvas.
 *
 * All matrices are stored column-major as Float32Array(16) to match WebGPU's
 * mat4x4<f32> upload convention. Vectors are plain `[x, y, z]` tuples.
 */

export type Vec3 = readonly [number, number, number];
export type Quat = readonly [number, number, number, number]; // [w, x, y, z]
export type Mat4 = Float32Array; // length 16, column-major

export const vec3 = (x: number, y: number, z: number): Vec3 => [x, y, z];

export const vec3Sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const vec3Add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const vec3Dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const vec3Cross = (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
];
export const vec3Len = (v: Vec3): number => Math.hypot(v[0], v[1], v[2]);
export const vec3Norm = (v: Vec3): Vec3 => {
    const l = vec3Len(v);
    return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
};
export const vec3Scale = (v: Vec3, s: number): Vec3 => [v[0] * s, v[1] * s, v[2] * s];

export const mat4 = (): Mat4 => new Float32Array(16);

export const mat4Identity = (): Mat4 => {
    const m = new Float32Array(16);
    m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
    return m;
};

/**
 * Build a right-handed look-at view matrix (camera looking down -z).
 * Output is the world->camera transform.
 */
export const mat4LookAt = (eye: Vec3, target: Vec3, up: Vec3): Mat4 => {
    const f = vec3Norm(vec3Sub(target, eye));      // forward
    const s = vec3Norm(vec3Cross(f, up));          // right
    const u = vec3Cross(s, f);                     // true up

    const m = new Float32Array(16);
    // column-major
    m[0] = s[0]; m[1] = u[0]; m[2] = -f[0]; m[3] = 0;
    m[4] = s[1]; m[5] = u[1]; m[6] = -f[1]; m[7] = 0;
    m[8] = s[2]; m[9] = u[2]; m[10] = -f[2]; m[11] = 0;
    m[12] = -vec3Dot(s, eye);
    m[13] = -vec3Dot(u, eye);
    m[14] = vec3Dot(f, eye);
    m[15] = 1;
    return m;
};

/** Multiply a 4x4 matrix by a vec3 (treated as point: w=1). Returns Vec3 + w. */
export const mat4TransformPoint = (m: Mat4, p: Vec3): [number, number, number, number] => [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15]
];

/**
 * Rotate a vec3 by a quaternion (w, x, y, z) using Rodrigues' formula:
 *   v' = v + 2w * (q.xyz x v) + 2 * q.xyz x (q.xyz x v)
 */
export const quatRotate = (q: Quat, v: Vec3): Vec3 => {
    const qx = q[1], qy = q[2], qz = q[3], qw = q[0];
    const tx = 2 * (qy * v[2] - qz * v[1]);
    const ty = 2 * (qz * v[0] - qx * v[2]);
    const tz = 2 * (qx * v[1] - qy * v[0]);
    return [
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx)
    ];
};

export const quatNorm = (q: Quat): Quat => {
    const l = Math.hypot(q[0], q[1], q[2], q[3]);
    return l > 0 ? [q[0] / l, q[1] / l, q[2] / l, q[3] / l] : [1, 0, 0, 0];
};
