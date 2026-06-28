/**
 * Camera basis derivation used by both CPU cull/sort and GPU shaders.
 *
 * For pinhole projection, the GPU receives:
 *  - view matrix (world -> camera)
 *  - focal length in pixels (= imageHeight / (2 * tan(fovY/2)))
 *  - principal point (px, py) at image center
 *  - near plane (cull threshold)
 *
 * For equirect, there's no focal length; the GPU spreads the sphere across
 * the image and culls by radius around camera. (Pinhole is the default and
 * the only one most users will need.)
 */

import type { RenderCamera, Projection } from '../types.ts';
import type { Mat4 } from '../math/index.ts';

export interface CameraBasis {
    projection: Projection;
    viewMatrix: Mat4;
    width: number;
    height: number;
    near: number;
    far: number;
    /** Pinhole only. Focal length in pixels. */
    focal: number;
    /** Principal point. */
    cx: number;
    cy: number;
}

export const buildCameraBasis = (cam: RenderCamera): CameraBasis => {
    const projection = cam.projection ?? 'pinhole';
    const focal = cam.height / (2 * Math.tan(cam.fovY / 2));
    return {
        projection,
        viewMatrix: cam.viewMatrix,
        width: cam.width,
        height: cam.height,
        near: cam.near,
        far: cam.far,
        focal,
        cx: cam.width / 2,
        cy: cam.height / 2
    };
};

/**
 * CPU near-plane cull. Returns the count of visible splats and writes their
 * indices into `outIndices` and their camera-space z (depth) into `outDepths`.
 *
 * The depth is camera-space z (negative-z forward in right-handed view); we
 * use `-z` so smaller values mean closer, which is what the sort wants for
 * front-to-back ordering.
 */
export const cullToVisible = (
    positions: Float32Array,
    n: number,
    basis: CameraBasis,
    outIndices: Uint32Array,
    outDepths: Float32Array
): number => {
    const m = basis.viewMatrix;
    let count = 0;
    if (basis.projection === 'pinhole') {
        // Camera-space z must be > near (we use the negated form: -z_cam > near).
        for (let i = 0; i < n; i++) {
            const x = positions[i * 3 + 0];
            const y = positions[i * 3 + 1];
            const z = positions[i * 3 + 2];
            // World -> camera (we only need the z component).
            const cz = m[2] * x + m[6] * y + m[10] * z + m[14];
            const depth = -cz; // forward distance
            if (depth > basis.near && depth < basis.far) {
                outIndices[count] = i;
                outDepths[count] = depth;
                count++;
            }
        }
    } else {
        // Equirect: cull by sphere of radius near around camera.
        const nearSq = basis.near * basis.near;
        for (let i = 0; i < n; i++) {
            const x = positions[i * 3 + 0];
            const y = positions[i * 3 + 1];
            const z = positions[i * 3 + 2];
            const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
            const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
            const cz = m[2] * x + m[6] * y + m[10] * z + m[14];
            const r2 = cx * cx + cy * cy + cz * cz;
            if (r2 > nearSq) {
                outIndices[count] = i;
                outDepths[count] = Math.sqrt(r2);
                count++;
            }
        }
    }
    return count;
};
