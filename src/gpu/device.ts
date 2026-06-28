/**
 * Thin WebGPU device wrapper. Provides:
 *  - adapter/device acquisition
 *  - buffer/binding-group helpers
 *  - a minimal "compute pass" builder for one-shot dispatches
 *
 * No PlayCanvas dependency. The only external assumption is that the host
 * has `navigator.gpu` (Chrome 134+, Edge, Android Chrome).
 */

export interface DeviceOptions {
    /** Optional pre-acquired GPUDevice (e.g. when sharing with a render context). */
    device?: GPUDevice;
    /** Power preference; defaults to 'high-performance'. */
    powerPreference?: GPUPowerPreference;
    /** Required features. The init will throw if unavailable. */
    requiredFeatures?: GPUFeatureName[];
    /** Required limits override. */
    requiredLimits?: Record<string, number>;
}

/**
 * Acquire a WebGPU device.
 *
 * @throws if WebGPU is unavailable or the adapter request fails.
 */
export const createDevice = async (opts: DeviceOptions = {}): Promise<GPUDevice> => {
    if (opts.device) return opts.device;
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
        throw new Error('WebGPU not available. Requires Chrome 134+, Edge, or Android Chrome.');
    }
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: opts.powerPreference ?? 'high-performance'
    });
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice({
        requiredFeatures: opts.requiredFeatures,
        requiredLimits: opts.requiredLimits
    });
    return device;
};

/** Create a GPU buffer from a TypedArray, with explicit usage. */
export const uploadBuffer = (
    device: GPUDevice,
    data: ArrayBufferView,
    usage: GPUBufferUsageFlags,
    label?: string
): GPUBuffer => {
    const buf = device.createBuffer({
        label,
        size: Math.max(16, data.byteLength + ((4 - (data.byteLength & 3)) & 3)),
        usage,
        mappedAtCreation: true
    });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buf.unmap();
    return buf;
};

/** Create an empty GPU buffer. Size is rounded up to multiple of 4. */
export const emptyBuffer = (
    device: GPUDevice,
    sizeBytes: number,
    usage: GPUBufferUsageFlags,
    label?: string
): GPUBuffer => {
    return device.createBuffer({
        label,
        size: Math.max(16, sizeBytes + ((4 - (sizeBytes & 3)) & 3)),
        usage
    });
};

/** Readback an entire storage buffer to a Uint8Array (one-shot helper). */
export const readbackBuffer = async (
    device: GPUDevice,
    src: GPUBuffer,
    size: number
): Promise<Uint8Array> => {
    const staging = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, staging, 0, size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
};
