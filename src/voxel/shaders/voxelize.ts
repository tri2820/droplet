/**
 * Voxelization compute shader.
 *
 * One workgroup per 4×4×4 block. 64 threads, each thread = one voxel inside
 * the block. The thread's local invocation index is decoded via Morton into
 * (x, y, z) within the block (0..3 each).
 *
 * Cooperative tiled loading: all 64 threads cooperate to load a tile of 64
 * Gaussians into workgroup-sharedG memory, then every thread evaluates against
 * the same tile before loading the next. This is the 64× memory-traffic
 * reduction trick from SuperSplat's gpu-voxelization.
 *
 * Output: 2 × u32 per block (block linear index → occupancy mask), packed
 * into `results[batchIdx * maxBlocksPerBatch * 2 + blockId * 2 + (0|1)]`.
 */

export const voxelizeShader = () => /* wgsl */`
struct Uniforms {
    opacityCutoff: f32,
    voxelResolution: f32,
    maxBlocksPerBatch: u32,
};

struct Gaussian {
    posX: f32, posY: f32, posZ: f32,
    opacityLogit: f32,
    rotW: f32, rotX: f32, rotY: f32, rotZ: f32,
    scaleX: f32, scaleY: f32, scaleZ: f32,
    extentX: f32, extentY: f32, extentZ: f32,
    _pad0: f32, _pad1: f32,
};

struct BatchInfo {
    indexOffset: u32,
    indexCount: u32,
    numBlocksX: u32,
    numBlocksY: u32,
    numBlocksZ: u32,
    blockMinX: f32,
    blockMinY: f32,
    blockMinZ: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<u32>;
@group(0) @binding(4) var<storage, read> batchInfos: array<BatchInfo>;

const TILE_SIZE: u32 = 64u;
var<workgroup> sharedG: array<Gaussian, 64>;
var<workgroup> blockMasks: array<atomic<u32>, 2>;

fn mortonToXYZ(m: u32) -> vec3<u32> {
    return vec3<u32>(
        (m & 1u) | ((m >> 2u) & 2u),
        ((m >> 1u) & 1u) | ((m >> 3u) & 2u),
        ((m >> 2u) & 1u) | ((m >> 4u) & 2u)
    );
}

fn evalGaussian(voxelCenter: vec3<f32>, voxelHalf: f32, g: Gaussian) -> f32 {
    let gc = vec3<f32>(g.posX, g.posY, g.posZ);
    let diff = voxelCenter - gc;
    let ext = vec3<f32>(g.extentX, g.extentY, g.extentZ);
    if (any(abs(diff) > (ext + voxelHalf))) { return 0.0; }

    // Closest point in voxel cube to Gaussian center.
    let closest = clamp(gc, voxelCenter - voxelHalf, voxelCenter + voxelHalf);
    let cd = closest - gc;

    // Inverse-rotate by quaternion (negate xyz). Rodrigues.
    let qxyz = vec3<f32>(-g.rotX, -g.rotY, -g.rotZ);
    let t = 2.0 * cross(qxyz, cd);
    let localDiff = cd + g.rotW * t + cross(qxyz, t);

    // Mahalanobis distance squared.
    let invScale = vec3<f32>(exp(-g.scaleX), exp(-g.scaleY), exp(-g.scaleZ));
    let scaled = localDiff * invScale;
    let d2 = dot(scaled, scaled);

    let opacity = 1.0 / (1.0 + exp(-g.opacityLogit));
    return opacity * exp(-0.5 * d2);
}

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) voxelIdx: u32,
    @builtin(workgroup_id) wgId: vec3<u32>
) {
    let batchIdx = wgId.z;
    let flatBlockId = wgId.x;
    let info = batchInfos[batchIdx];

    let totalBlocks = info.numBlocksX * info.numBlocksY * info.numBlocksZ;
    if (flatBlockId >= totalBlocks) { return; }

    let blockX = flatBlockId % info.numBlocksX;
    let blockY = (flatBlockId / info.numBlocksX) % info.numBlocksY;
    let blockZ = flatBlockId / (info.numBlocksX * info.numBlocksY);

    let localPos = mortonToXYZ(voxelIdx);
    let blockMin = vec3<f32>(info.blockMinX, info.blockMinY, info.blockMinZ);
    let blockOffset = vec3<f32>(f32(blockX), f32(blockY), f32(blockZ)) * 4.0 * u.voxelResolution;
    let voxelCenter = blockMin + blockOffset + (vec3<f32>(localPos) + 0.5) * u.voxelResolution;
    let voxelHalf = u.voxelResolution * 0.5;

    if (voxelIdx < 2u) { atomicStore(&blockMasks[voxelIdx], 0u); }
    workgroupBarrier();

    var totalSigma = 0.0;
    let nIdx = info.indexCount;
    let nTiles = (nIdx + TILE_SIZE - 1u) / TILE_SIZE;

    for (var tile = 0u; tile < nTiles; tile = tile + 1u) {
        let loadIdx = tile * TILE_SIZE + voxelIdx;
        if (loadIdx < nIdx) {
            let gIdx = indices[info.indexOffset + loadIdx];
            sharedG[voxelIdx] = gaussians[gIdx];
        }
        workgroupBarrier();

        if (totalSigma < 7.0) {
            let thisTile = min(TILE_SIZE, nIdx - tile * TILE_SIZE);
            for (var c = 0u; c < thisTile; c = c + 1u) {
                totalSigma = totalSigma + evalGaussian(voxelCenter, voxelHalf, sharedG[c]);
                if (totalSigma >= 7.0) { break; }
            }
        }
        workgroupBarrier();
    }

    // Beer-Lambert: opacity = 1 - exp(-sigma).
    let opacity = 1.0 - exp(-totalSigma);
    let isSolid = opacity >= u.opacityCutoff;

    let linearIdx = localPos.z * 16u + localPos.y * 4u + localPos.x;
    if (isSolid) {
        atomicOr(&blockMasks[linearIdx >> 5u], 1u << (linearIdx & 31u));
    }
    workgroupBarrier();

    if (voxelIdx < 2u) {
        let base = batchIdx * u.maxBlocksPerBatch * 2u + flatBlockId * 2u;
        results[base + voxelIdx] = atomicLoad(&blockMasks[voxelIdx]);
    }
}
`;
