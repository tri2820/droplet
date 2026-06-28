/**
 * Float32 radix sort by depth. Used for the CPU global depth sort in the
 * renderer pipeline.
 *
 * Sorts indices (Uint32Array) by an associated Float32Array of keys. Output
 * places indices of smallest depths first (front-to-back, which is what the
 * rasterizer expects so it can early-out on saturated transmittance).
 *
 * Key trick: floats are bit-mapped to unsigned ints so radix sort works on
 * them directly:
 *   positive: flip sign bit
 *   negative: flip all bits
 */

const PASSES = 4;
const BUCKET_BITS = 8;
const BUCKETS = 1 << BUCKET_BITS;

/** Convert IEEE-754 float to a sortable u32. */
const floatToSortable = (f: number, view: DataView): number => {
    view.setFloat32(0, f, true);
    const u = view.getUint32(0, true);
    return (u & 0x80000000) ? ~u : (u ^ 0x80000000);
};

export interface RadixSortScratch {
    keys: Uint32Array;       // sortable u32 of depths
    indicesTmp: Uint32Array; // pong buffer for indices
    keysTmp: Uint32Array;    // pong buffer for keys
    histogram: Uint32Array;  // [PASSES * BUCKETS]
    capacity: number;
}

export const createRadixScratch = (): RadixSortScratch => ({
    keys: new Uint32Array(0),
    indicesTmp: new Uint32Array(0),
    keysTmp: new Uint32Array(0),
    histogram: new Uint32Array(PASSES * BUCKETS),
    capacity: 0
});

const ensureCapacity = (s: RadixSortScratch, n: number) => {
    if (s.capacity < n) {
        s.keys = new Uint32Array(n);
        s.indicesTmp = new Uint32Array(n);
        s.keysTmp = new Uint32Array(n);
        s.capacity = n;
    }
};

/**
 * Sort `indices` in-place by ascending `depths[i]` (front-to-back).
 *
 * `indices` and `depths` are PARALLEL arrays of length `n`: position i in
 * both refers to the same logical splat. They are permuted together. (The
 * radix sort only writes the index permutation; depth values are read once
 * up front, converted to sortable u32, and the parallel u32 keys are
 * shuffled alongside the indices each pass.)
 *
 * @param indices - Uint32Array of splat indices to sort (in-place).
 * @param depths  - Float32Array of depths, parallel to `indices`.
 * @param scratch - Reusable scratch (grows on demand).
 * @param n       - Number of elements to sort.
 */
export const radixSortIndicesByFloat = (
    indices: Uint32Array,
    depths: Float32Array,
    scratch: RadixSortScratch,
    n: number
): void => {
    if (n <= 1) return;
    ensureCapacity(scratch, n);

    const keys = scratch.keys;
    const buf = new ArrayBuffer(4);
    const view = new DataView(buf);
    for (let i = 0; i < n; i++) {
        keys[i] = floatToSortable(depths[i], view);
    }

    // Build histograms for all passes in one walk.
    const hist = scratch.histogram;
    hist.fill(0, 0, PASSES * BUCKETS);
    for (let i = 0; i < n; i++) {
        const k = keys[i];
        hist[(k & 0xff)]++;
        hist[BUCKETS + ((k >>> 8) & 0xff)]++;
        hist[2 * BUCKETS + ((k >>> 16) & 0xff)]++;
        hist[3 * BUCKETS + ((k >>> 24) & 0xff)]++;
    }
    // Prefix sums per pass.
    for (let p = 0; p < PASSES; p++) {
        let sum = 0;
        for (let b = 0; b < BUCKETS; b++) {
            const c = hist[p * BUCKETS + b];
            hist[p * BUCKETS + b] = sum;
            sum += c;
        }
    }

    let inIdx = indices, inKey = keys;
    let outIdx = scratch.indicesTmp, outKey = scratch.keysTmp;
    for (let p = 0; p < PASSES; p++) {
        const shift = p * BUCKET_BITS;
        const base = p * BUCKETS;
        for (let i = 0; i < n; i++) {
            const k = inKey[i];
            const b = (k >>> shift) & 0xff;
            const dst = hist[base + b]++;
            outIdx[dst] = inIdx[i];
            outKey[dst] = k;
        }
        const ti = inIdx; inIdx = outIdx; outIdx = ti;
        const tk = inKey; inKey = outKey; outKey = tk;
    }
    if (inIdx !== indices) indices.set(inIdx.subarray(0, n));
};
