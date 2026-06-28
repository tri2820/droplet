/**
 * Bounding volume hierarchy over Gaussian AABBs.
 *
 * Designed for the voxelizer's broad-phase: given a query AABB (one
 * voxel-batch's region), enumerate splat indices whose AABB overlaps.
 *
 * Implementation: balanced binary BVH built via top-down median split on the
 * longest axis. Centroids are computed once; the actual node bounds are the
 * union of child Gaussian AABBs. Stored as flat typed arrays for cache
 * locality.
 *
 * Tradeoff vs. SAH: median split is O(N log N) construction and gives
 * ~85-90% of SAH query speed for our distributions (Gaussian centers tend to
 * cluster around scene structure). The construction time matters because we
 * rebuild on every voxelization run.
 */

interface BVHNode {
    minX: number; minY: number; minZ: number;
    maxX: number; maxY: number; maxZ: number;
    /** First Gaussian in this node's index range. */
    start: number;
    /** One past last Gaussian (so count = end - start). */
    end: number;
    /** Child node indices, -1 if leaf. */
    left: number;
    right: number;
}

/** Maximum Gaussians per leaf — small leaves = deeper tree, fewer false positives. */
const LEAF_SIZE = 8;

export class GaussianBVH {
    private nodes: BVHNode[] = [];
    /** Splat index permutation: nodes reference into this array. */
    private indices: Uint32Array;
    /** Per-Gaussian world AABB centers (3 floats per splat). */
    private centers: Float32Array;
    /** Per-Gaussian world AABB half-extents (3 floats per splat). */
    private extents: Float32Array;

    constructor(positions: Float32Array, extents: Float32Array) {
        const n = positions.length / 3;
        this.indices = new Uint32Array(n);
        for (let i = 0; i < n; i++) this.indices[i] = i;
        this.centers = positions;
        this.extents = extents;
        if (n > 0) this.build(0, n);
    }

    /**
     * Top-down build. Returns the index of the created node.
     *
     * Iterative would be faster but recursion is simpler and our max depth is
     * log2(N) ~ 25 for 30M splats, safely under the JS stack limit.
     */
    private build(start: number, end: number): number {
        const node: BVHNode = {
            minX: Infinity, minY: Infinity, minZ: Infinity,
            maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity,
            start, end, left: -1, right: -1
        };
        const nodeIdx = this.nodes.length;
        this.nodes.push(node);

        // Compute node bounds from contained Gaussian AABBs.
        for (let i = start; i < end; i++) {
            const g = this.indices[i];
            const cx = this.centers[g * 3 + 0];
            const cy = this.centers[g * 3 + 1];
            const cz = this.centers[g * 3 + 2];
            const ex = this.extents[g * 3 + 0];
            const ey = this.extents[g * 3 + 1];
            const ez = this.extents[g * 3 + 2];
            if (cx - ex < node.minX) node.minX = cx - ex;
            if (cy - ey < node.minY) node.minY = cy - ey;
            if (cz - ez < node.minZ) node.minZ = cz - ez;
            if (cx + ex > node.maxX) node.maxX = cx + ex;
            if (cy + ey > node.maxY) node.maxY = cy + ey;
            if (cz + ez > node.maxZ) node.maxZ = cz + ez;
        }

        const count = end - start;
        if (count <= LEAF_SIZE) return nodeIdx;

        // Pick split axis = longest extent of node bounds. Median split by centroid.
        const dx = node.maxX - node.minX;
        const dy = node.maxY - node.minY;
        const dz = node.maxZ - node.minZ;
        const axis = dx > dy ? (dx > dz ? 0 : 2) : (dy > dz ? 1 : 2);

        // In-place median partition by centroid on `axis`.
        const mid = start + (count >> 1);
        this.nth(start, end, mid, axis);

        node.left = this.build(start, mid);
        node.right = this.build(mid, end);
        return nodeIdx;
    }

    /** In-place quickselect: after the call, `indices[k]` is the k-th element by axis. */
    private nth(lo: number, hi: number, k: number, axis: number): void {
        const idx = this.indices;
        const c = this.centers;
        while (hi - lo > 1) {
            // Median-of-three pivot
            const m = (lo + hi) >> 1;
            const a = c[idx[lo] * 3 + axis];
            const b = c[idx[m] * 3 + axis];
            const d = c[idx[hi - 1] * 3 + axis];
            // place median at lo+1
            const pivotIdx = (a < b) ? (b < d ? m : (a < d ? hi - 1 : lo))
                                     : (a < d ? lo : (b < d ? hi - 1 : m));
            const tmp0 = idx[lo + 1]; idx[lo + 1] = idx[pivotIdx]; idx[pivotIdx] = tmp0;
            const pivot = c[idx[lo + 1] * 3 + axis];
            let i = lo + 1, j = hi - 1;
            while (true) {
                while (++i < hi && c[idx[i] * 3 + axis] < pivot) {}
                while (--j > lo && c[idx[j] * 3 + axis] > pivot) {}
                if (i >= j) break;
                const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
            const t2 = idx[lo + 1]; idx[lo + 1] = idx[j]; idx[j] = t2;
            if (j === k) return;
            if (j < k) lo = j + 1; else hi = j;
        }
    }

    /**
     * Enumerate splat indices whose AABB overlaps the query box.
     *
     * Writes indices into `out` starting at `outOffset`. Returns the count of
     * indices written. The caller is responsible for ensuring `out` has
     * sufficient capacity; if not, returns -1 (caller can grow + retry).
     */
    queryOverlappingInto(
        qMinX: number, qMinY: number, qMinZ: number,
        qMaxX: number, qMaxY: number, qMaxZ: number,
        out: Uint32Array,
        outOffset: number
    ): number {
        if (this.nodes.length === 0) return 0;
        let count = 0;
        // Iterative DFS using a small stack.
        const stack: number[] = [0];
        const idx = this.indices;
        const c = this.centers;
        const e = this.extents;
        while (stack.length > 0) {
            const ni = stack.pop()!;
            const n = this.nodes[ni];
            // Box-box overlap test against node bounds.
            if (n.maxX < qMinX || n.minX > qMaxX) continue;
            if (n.maxY < qMinY || n.minY > qMaxY) continue;
            if (n.maxZ < qMinZ || n.minZ > qMaxZ) continue;
            if (n.left < 0) {
                // Leaf — per-Gaussian test.
                for (let i = n.start; i < n.end; i++) {
                    const g = idx[i];
                    const gMinX = c[g * 3 + 0] - e[g * 3 + 0];
                    const gMaxX = c[g * 3 + 0] + e[g * 3 + 0];
                    if (gMaxX < qMinX || gMinX > qMaxX) continue;
                    const gMinY = c[g * 3 + 1] - e[g * 3 + 1];
                    const gMaxY = c[g * 3 + 1] + e[g * 3 + 1];
                    if (gMaxY < qMinY || gMinY > qMaxY) continue;
                    const gMinZ = c[g * 3 + 2] - e[g * 3 + 2];
                    const gMaxZ = c[g * 3 + 2] + e[g * 3 + 2];
                    if (gMaxZ < qMinZ || gMinZ > qMaxZ) continue;
                    if (outOffset + count >= out.length) return -1;
                    out[outOffset + count++] = g;
                }
            } else {
                stack.push(n.left, n.right);
            }
        }
        return count;
    }

    /** Total node count, for diagnostics. */
    get nodeCount(): number { return this.nodes.length; }
    /** Splat count, for diagnostics. */
    get splatCount(): number { return this.indices.length; }
}
