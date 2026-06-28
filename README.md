# droplet

WebGPU 3D Gaussian Splatting renderer and voxelizer.

Independent reimplementation of the rendering and voxel-collision pipelines from PlayCanvas SuperSplat (`splat-transform`). No PlayCanvas, no Burn, no ML framework — direct WebGPU only. TypeScript end to end.

## Status

- Forward renderer: cull → CPU global depth sort → GPU project → tile binning (prefix-sum, emit pairs, GPU radix sort, find boundaries) → tile-binned rasterizer → composite.
- Voxelizer: AABB extents → BVH broad-phase → cooperative-tiled GPU voxelization → sparse `BlockMaskBuffer`.

## Requirements

- Chrome 134+, Edge, Android Chrome, or any browser with WebGPU enabled.
- Bun 1.3+ (for development).

## Install

```sh
bun install
```

## Use

```ts
import {
    createDevice,
    Renderer,
    voxelize,
    mat4LookAt,
    type SplatData,
} from 'droplet';

const device = await createDevice();

const data: SplatData = {
    count: N,
    positions: posF32,      // 3N
    rotations: rotF32,      // 4N, quaternion [w,x,y,z]
    logScales: scaleF32,    // 3N, ln(sigma_xyz)
    opacityLogits: opF32,   // N
    colorsDC: dcF32,        // 3N, SH band-0 or RGB / sqrt(C0)
    shBands: 0,             // 0..3
};

// ---- Render ----
const renderer = new Renderer(device);
const rgba = await renderer.render(data, {
    width: 1280, height: 720,
    viewMatrix: mat4LookAt([0, 0, 4], [0, 0, 0], [0, 1, 0]),
    fovY: Math.PI / 3,
    near: 0.01, far: 1000,
});
// rgba is Uint8Array of length width*height*4 (RGBA8)

// ---- Voxelize for collision ----
const result = await voxelize(device, data, {
    voxelResolution: 0.05,
    opacityCutoff: 0.1,
});
console.log(`${result.buffer.count} non-empty blocks`);
```

## Layout

```
src/
  index.ts                    public exports
  types.ts                    SplatData, RenderCamera, layout helpers
  math/                       Vec3, Quat, Mat4 (no gl-matrix dep)
  gpu/
    device.ts                 WebGPU adapter + buffer helpers
    radix-sort.ts             GPU radix sort (u32 key + u32 value)
  spatial/
    aabb.ts                   Per-Gaussian 3-sigma world AABB
    bvh.ts                    Broad-phase BVH over Gaussian AABBs
    radix-sort.ts             CPU radix sort by float depth
  render/
    pipeline.ts               Renderer orchestrator
    camera.ts                 CameraBasis + CPU cull
    preprocess.ts             cull + sort + chunk packing
    shaders/                  WGSL strings
      constants.ts            TILE_SIZE etc + shared prelude
      project.ts              project shader (pinhole / equirect)
      prefix-sum.ts           two-level scan
      emit-pairs.ts           per-splat tile pair emit
      find-boundaries.ts      slice boundary atomic-min
      rasterize.ts            tile-binned compute rasterizer
      finalize.ts             composite + 8-bit pack
  voxel/
    voxelize.ts               voxelizer orchestrator
    block-mask-buffer.ts      sparse occupancy storage
    morton.ts                 Morton encode/decode
    shaders/
      voxelize.ts             cooperative-tiled voxelize WGSL
```

## Algorithm notes

**Renderer.** Tile-binned 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023), adapted from SuperSplat's `splat-transform`. Each pixel walks only its tile's depth-sorted splat slice and accumulates color via front-to-back alpha-over with transmittance early-out at `T < 1e-4`.

**Voxelizer.** For each voxel, sum density contribution σ from every overlapping Gaussian (closest-point-in-cube + Mahalanobis distance²), convert via Beer-Lambert `α = 1 - exp(-σ)`, threshold to get a bit. 64 voxels per block, 2 × u32 bitmask per block, sparse storage. One workgroup per block; 64 threads cooperatively load 64 Gaussians at a time into shared memory, then every thread evaluates against the cached tile — 64× memory-traffic reduction over naive per-voxel reads.

## What's intentionally not here

- Backward pass / training. droplet renders, it doesn't train. For training use [Brush](https://github.com/ArthurBrussee/brush).
- LOD streaming, SOG compression, PLY loading. Bring your own loader.
- Carve / fill / mesh collision stages — only the voxelization is here. Those stages are grid arithmetic on the produced `BlockMaskBuffer` and can be layered on without touching the GPU code.

## License

MIT.
