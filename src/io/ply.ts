/**
 * PLY loader for standard 3D Gaussian Splatting files.
 *
 * Supports the property layout produced by the original Inria 3DGS trainer
 * and most downstream tools:
 *
 *   element vertex N
 *   property float x, y, z
 *   property float scale_0..2     (log-scales)
 *   property float f_dc_0..2      (SH band-0 coefficients, raw)
 *   property float opacity        (logit)
 *   property float rot_0..3       (quaternion, w-first by 3DGS convention)
 *   [optional] property float f_rest_0..K-1  (higher-order SH, channel-major)
 *
 * Property order in the file is honored — we de-interleave by index after
 * parsing the header. Quaternions are normalized on load. f_dc is converted
 * from raw SH-C0 coefficients to droplet's "(color - 0.5)" convention by
 * multiplying by SH_C0 = 1/(2·√π).
 */

import type { SplatData } from '../types.ts';

const SH_C0 = 0.28209479177387814;

interface PlyProperty {
    name: string;
    type: 'float' | 'uchar' | 'int';
}

interface PlyHeader {
    count: number;
    properties: PlyProperty[];
    dataOffset: number;
    binary: boolean;
    littleEndian: boolean;
}

const SIZE: Record<PlyProperty['type'], number> = { float: 4, uchar: 1, int: 4 };

/** Parse the PLY header, returning property layout and binary-data offset. */
const parseHeader = (bytes: Uint8Array): PlyHeader => {
    // Find end_header\n
    const dec = new TextDecoder('utf-8');
    // Scan first ~4KB of the file as text — headers don't exceed that.
    const HEADER_MAX = 65536;
    const text = dec.decode(bytes.subarray(0, Math.min(HEADER_MAX, bytes.byteLength)));
    const endIdx = text.indexOf('end_header\n');
    if (endIdx < 0) throw new Error('PLY: end_header not found in first 64KB');
    const dataOffset = endIdx + 'end_header\n'.length;

    const headerText = text.slice(0, endIdx);
    const lines = headerText.split('\n').map(l => l.trim()).filter(Boolean);

    if (lines[0] !== 'ply') throw new Error('PLY: missing magic');
    let count = 0;
    let binary = false;
    let littleEndian = true;
    const props: PlyProperty[] = [];
    let inVertex = false;
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        if (line.startsWith('format ')) {
            binary = line.includes('binary');
            littleEndian = line.includes('little_endian');
            continue;
        }
        if (line.startsWith('element vertex ')) {
            count = parseInt(line.slice('element vertex '.length), 10);
            inVertex = true;
            continue;
        }
        if (line.startsWith('element ')) {
            // Other elements end the vertex section.
            inVertex = false;
            continue;
        }
        if (inVertex && line.startsWith('property ')) {
            // "property <type> <name>"
            const parts = line.split(/\s+/);
            const t = parts[1];
            const name = parts[2];
            if (t !== 'float' && t !== 'uchar' && t !== 'int') {
                throw new Error(`PLY: unsupported property type "${t}"`);
            }
            props.push({ type: t, name });
        }
    }
    if (!binary) throw new Error('PLY: ASCII PLY not supported (binary only)');
    if (!littleEndian) throw new Error('PLY: only little-endian supported');
    if (count <= 0) throw new Error('PLY: no vertices');
    return { count, properties: props, dataOffset, binary, littleEndian };
};

export interface PlyLoadOptions {
    /**
     * Flip the scene's Y axis on load to convert from COLMAP / Inria 3DGS
     * convention (Y-down world) to Y-up world. Default `true`, which is
     * what virtually every PLY exported from the standard 3DGS trainer
     * expects when viewed with a Y-up camera. Set `false` if your PLY is
     * already Y-up.
     *
     * Flipping affects positions, quaternions, and SH coefficients. The
     * quaternion flip negates the x and z components — a reflection
     * through the XZ plane that mirrors the rotation axis to match the
     * mirrored coordinate frame. SH coefficients aren't flipped (their
     * basis is symmetric enough that the visible effect is small and
     * fixing them properly requires rotating the SH band, which is out
     * of scope here).
     */
    flipY?: boolean;
}

/**
 * Load splat data from a 3DGS PLY file.
 *
 * @param source - ArrayBuffer of the PLY file (already fetched).
 * @param opts - Loader options. See {@link PlyLoadOptions}.
 * @returns SplatData ready to feed into the renderer/voxelizer.
 */
export const loadPlySplats = (
    source: ArrayBuffer | ArrayBufferView,
    opts: PlyLoadOptions = {}
): SplatData => {
    const flipY = opts.flipY ?? true;
    const bytes = source instanceof ArrayBuffer
        ? new Uint8Array(source)
        : new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
    const hdr = parseHeader(bytes);

    // Build a property-name → index map.
    const idx: Record<string, number> = {};
    let stride = 0;
    for (let p = 0; p < hdr.properties.length; p++) {
        idx[hdr.properties[p].name] = p;
        stride += SIZE[hdr.properties[p].type];
    }
    // For the simple 3DGS layout, every property is float — read as Float32Array.
    const allFloat = hdr.properties.every(p => p.type === 'float');
    if (!allFloat) {
        throw new Error('PLY: mixed property types not yet supported');
    }
    const floatsPerVert = hdr.properties.length;
    const totalBytes = hdr.count * floatsPerVert * 4;
    if (hdr.dataOffset + totalBytes > bytes.byteLength) {
        throw new Error(
            `PLY: data section truncated (need ${totalBytes} bytes, have ` +
            `${bytes.byteLength - hdr.dataOffset})`
        );
    }
    // View the binary data as Float32Array (4-byte aligned — copy if needed).
    const dataU8 = bytes.subarray(hdr.dataOffset, hdr.dataOffset + totalBytes);
    let flat: Float32Array;
    if ((dataU8.byteOffset & 3) === 0) {
        flat = new Float32Array(dataU8.buffer, dataU8.byteOffset, dataU8.byteLength / 4);
    } else {
        const aligned = new Uint8Array(dataU8.byteLength);
        aligned.set(dataU8);
        flat = new Float32Array(aligned.buffer);
    }

    const need = (n: string): number => {
        if (!(n in idx)) throw new Error(`PLY: missing property "${n}"`);
        return idx[n];
    };
    const ix = need('x'),  iy = need('y'),  iz = need('z');
    const isx = need('scale_0'), isy = need('scale_1'), isz = need('scale_2');
    const id0 = need('f_dc_0'), id1 = need('f_dc_1'), id2 = need('f_dc_2');
    const iop = need('opacity');
    const ir0 = need('rot_0'), ir1 = need('rot_1'), ir2 = need('rot_2'), ir3 = need('rot_3');

    // Detect extra SH bands by f_rest_* presence. f_rest layout in PLY is
    // channel-major: R[0..K-1], G[0..K-1], B[0..K-1] where K = coeffs/channel.
    let kRest = 0;
    while (`f_rest_${kRest}` in idx) kRest++;
    const coeffsPerChannel = Math.floor(kRest / 3);
    const shBands: 0 | 1 | 2 | 3 =
        coeffsPerChannel >= 15 ? 3
        : coeffsPerChannel >= 8 ? 2
        : coeffsPerChannel >= 3 ? 1
        : 0;

    const N = hdr.count;
    const positions = new Float32Array(N * 3);
    const rotations = new Float32Array(N * 4);
    const logScales = new Float32Array(N * 3);
    const opacityLogits = new Float32Array(N);
    const colorsDC = new Float32Array(N * 3);
    let colorsSH: Float32Array | undefined;
    if (shBands > 0) {
        colorsSH = new Float32Array(N * 3 * coeffsPerChannel);
    }

    const ySign = flipY ? -1 : 1;
    for (let i = 0; i < N; i++) {
        const base = i * floatsPerVert;

        positions[i * 3 + 0] = flat[base + ix];
        positions[i * 3 + 1] = ySign * flat[base + iy];
        positions[i * 3 + 2] = flat[base + iz];

        // Log-scales are invariant under Y flip (axis-aligned magnitudes).
        logScales[i * 3 + 0] = flat[base + isx];
        logScales[i * 3 + 1] = flat[base + isy];
        logScales[i * 3 + 2] = flat[base + isz];

        opacityLogits[i] = flat[base + iop];

        colorsDC[i * 3 + 0] = SH_C0 * flat[base + id0];
        colorsDC[i * 3 + 1] = SH_C0 * flat[base + id1];
        colorsDC[i * 3 + 2] = SH_C0 * flat[base + id2];

        // Quaternion — 3DGS convention is (w, x, y, z) in rot_0..rot_3.
        // To mirror a rotation through the XZ plane (Y-flip), negate the
        // x and z components of the quaternion (equivalent to conjugating
        // by the y-flip). Leaves w and y alone.
        let w = flat[base + ir0];
        let qx = flat[base + ir1];
        let qy = flat[base + ir2];
        let qz = flat[base + ir3];
        const n = Math.hypot(w, qx, qy, qz) || 1;
        const f = flipY ? -1 : 1;
        rotations[i * 4 + 0] = w  / n;
        rotations[i * 4 + 1] = f * qx / n;
        rotations[i * 4 + 2] = qy / n;
        rotations[i * 4 + 3] = f * qz / n;

        if (colorsSH && shBands > 0) {
            // Each splat owns 3 * coeffsPerChannel SH floats. PLY stores
            // them in channel-major order (R first, then G, then B); we
            // store the same way per splat.
            for (let k = 0; k < coeffsPerChannel * 3; k++) {
                colorsSH[i * coeffsPerChannel * 3 + k] = flat[base + idx[`f_rest_${k}`]];
            }
        }
    }

    return { count: N, positions, rotations, logScales, opacityLogits, colorsDC,
             colorsSH, shBands };
};

/** Compute the axis-aligned bounding box of all splat centers. */
export const splatBounds = (data: SplatData): {
    min: [number, number, number]; max: [number, number, number];
} => {
    const p = data.positions;
    let nx = Infinity, ny = Infinity, nz = Infinity;
    let xx = -Infinity, xy = -Infinity, xz = -Infinity;
    for (let i = 0; i < data.count; i++) {
        const x = p[i * 3 + 0], y = p[i * 3 + 1], z = p[i * 3 + 2];
        if (x < nx) nx = x;  if (x > xx) xx = x;
        if (y < ny) ny = y;  if (y > xy) xy = y;
        if (z < nz) nz = z;  if (z > xz) xz = z;
    }
    return { min: [nx, ny, nz], max: [xx, xy, xz] };
};
