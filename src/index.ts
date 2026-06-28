/**
 * droplet — WebGPU 3D Gaussian Splatting renderer and voxelizer.
 *
 * Independent reimplementation of the rendering and voxel-collision pipelines
 * shipped in PlayCanvas SuperSplat (splat-transform). No PlayCanvas or
 * ML-framework dependencies — direct WebGPU only.
 *
 * Public surface:
 *  - `Renderer`         — tile-binned compute rasterizer for forward rendering
 *  - `voxelize`         — sparse voxel-occupancy generator for collision
 *  - `createDevice`     — WebGPU adapter/device convenience
 *  - `SplatData`        — input data layout (struct-of-arrays float32)
 *  - `RenderCamera`     — view/projection params
 */

export type {
    SplatData,
    RenderCamera,
    Projection,
    BackgroundRGBA,
    Bounds
} from './types.ts';
export { numSHCoeffsPerChannel, splatInputStride } from './types.ts';

export { Renderer, type RendererOptions } from './render/pipeline.ts';
export { buildCameraBasis, type CameraBasis } from './render/camera.ts';

export { voxelize, type VoxelizeOptions, type VoxelizeResult } from './voxel/voxelize.ts';
export { BlockMaskBuffer, type VoxelBlock } from './voxel/block-mask-buffer.ts';

export { createDevice, type DeviceOptions } from './gpu/device.ts';

export {
    mat4LookAt, mat4Identity, vec3,
    type Vec3, type Mat4, type Quat
} from './math/index.ts';
