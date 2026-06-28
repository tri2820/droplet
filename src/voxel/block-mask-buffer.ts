/**
 * Sparse storage for the voxel grid.
 *
 * The grid is divided into 4×4×4-voxel blocks. Each block holds a 64-bit
 * occupancy mask (2 × u32). Only non-empty blocks are stored.
 *
 * Lookup is by integer block index (block coordinates flattened as
 *   `bx + by * numBlocksX + bz * numBlocksX * numBlocksY`).
 *
 * The buffer is append-only during voxelization. Downstream stages (fill,
 * carve) walk it block-by-block.
 */

export interface VoxelBlock {
    /** Flat block index. */
    blockIdx: number;
    /** Low 32 voxel bits (linear index 0..31). */
    maskLo: number;
    /** High 32 voxel bits (linear index 32..63). */
    maskHi: number;
}

export class BlockMaskBuffer {
    /** Map blockIdx -> entry index in `blocks`. */
    private byIdx: Map<number, number> = new Map();
    /** Dense list of non-empty blocks. */
    blocks: VoxelBlock[] = [];

    /** Insert or OR-merge a block. */
    addBlock(blockIdx: number, maskLo: number, maskHi: number): void {
        if (maskLo === 0 && maskHi === 0) return;
        const ex = this.byIdx.get(blockIdx);
        if (ex === undefined) {
            this.byIdx.set(blockIdx, this.blocks.length);
            this.blocks.push({ blockIdx, maskLo, maskHi });
        } else {
            const b = this.blocks[ex];
            b.maskLo = (b.maskLo | maskLo) >>> 0;
            b.maskHi = (b.maskHi | maskHi) >>> 0;
        }
    }

    get(blockIdx: number): VoxelBlock | undefined {
        const ex = this.byIdx.get(blockIdx);
        return ex === undefined ? undefined : this.blocks[ex];
    }

    /** Total bit count across all blocks. */
    popcount(): number {
        let s = 0;
        for (const b of this.blocks) {
            s += popcount32(b.maskLo) + popcount32(b.maskHi);
        }
        return s;
    }

    get count(): number { return this.blocks.length; }
}

const popcount32 = (v: number): number => {
    v = v - ((v >>> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
    return ((((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24);
};
