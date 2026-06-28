/** Statistics on the loaded splat data. */
import { loadPlySplats } from '../../src/index.ts';

export const inspect = async () => {
    const r = await fetch('/scene.ply');
    const buf = await r.arrayBuffer();
    const data = loadPlySplats(buf);
    const n = data.count;

    // f_dc stats — pre-scaled by SH_C0 (~0.28209) in loader.
    let dcMin = Infinity, dcMax = -Infinity, dcSum = 0, dcAbsSum = 0;
    for (let i = 0; i < data.colorsDC.length; i++) {
        const v = data.colorsDC[i];
        if (v < dcMin) dcMin = v;
        if (v > dcMax) dcMax = v;
        dcSum += v;
        dcAbsSum += Math.abs(v);
    }

    // Implied color = dc + 0.5 (no SH bands).
    let cMin = Infinity, cMax = -Infinity, overOneCount = 0;
    for (let i = 0; i < data.colorsDC.length; i++) {
        const c = Math.max(0, data.colorsDC[i] + 0.5);
        if (c < cMin) cMin = c;
        if (c > cMax) cMax = c;
        if (c > 1) overOneCount++;
    }

    // Opacity stats.
    let opMin = Infinity, opMax = -Infinity, alphaSum = 0;
    let alphaOver99 = 0;
    for (let i = 0; i < data.opacityLogits.length; i++) {
        const o = data.opacityLogits[i];
        if (o < opMin) opMin = o;
        if (o > opMax) opMax = o;
        const a = 1 / (1 + Math.exp(-o));
        alphaSum += a;
        if (a > 0.99) alphaOver99++;
    }

    // Log-scale stats (radius in world units = exp(logScale)).
    let lsMin = Infinity, lsMax = -Infinity;
    let sigmaP99 = 0;
    const sigmas: number[] = [];
    for (let i = 0; i < data.logScales.length; i++) {
        const s = data.logScales[i];
        if (s < lsMin) lsMin = s;
        if (s > lsMax) lsMax = s;
        sigmas.push(Math.exp(s));
    }
    sigmas.sort((a, b) => a - b);
    sigmaP99 = sigmas[Math.floor(sigmas.length * 0.99)];
    const sigmaP50 = sigmas[Math.floor(sigmas.length * 0.5)];
    const sigmaMax = sigmas[sigmas.length - 1];

    // Sample 5 random splats fully.
    const samples = [];
    for (let k = 0; k < 5; k++) {
        const i = Math.floor((k + 0.5) / 5 * n);
        samples.push({
            i,
            pos: [data.positions[i*3], data.positions[i*3+1], data.positions[i*3+2]].map(v => v.toFixed(2)),
            rot: [data.rotations[i*4], data.rotations[i*4+1], data.rotations[i*4+2], data.rotations[i*4+3]].map(v => v.toFixed(3)),
            logScale: [data.logScales[i*3], data.logScales[i*3+1], data.logScales[i*3+2]].map(v => v.toFixed(2)),
            sigma: [Math.exp(data.logScales[i*3]), Math.exp(data.logScales[i*3+1]), Math.exp(data.logScales[i*3+2])].map(v => v.toFixed(3)),
            opacityLogit: data.opacityLogits[i].toFixed(2),
            alpha: (1/(1+Math.exp(-data.opacityLogits[i]))).toFixed(3),
            colorsDC: [data.colorsDC[i*3], data.colorsDC[i*3+1], data.colorsDC[i*3+2]].map(v => v.toFixed(3)),
            implColor: [data.colorsDC[i*3]+0.5, data.colorsDC[i*3+1]+0.5, data.colorsDC[i*3+2]+0.5].map(v => v.toFixed(3)),
        });
    }

    return {
        count: n,
        dc: { min: dcMin.toFixed(3), max: dcMax.toFixed(3),
              mean: (dcSum / data.colorsDC.length).toFixed(3),
              absMean: (dcAbsSum / data.colorsDC.length).toFixed(3) },
        impliedColor: { min: cMin.toFixed(3), max: cMax.toFixed(3),
                        pctOverOne: (overOneCount * 100 / data.colorsDC.length).toFixed(1) + '%' },
        opacity: { logitMin: opMin.toFixed(2), logitMax: opMax.toFixed(2),
                   meanAlpha: (alphaSum / n).toFixed(3),
                   pctAlphaOver99: (alphaOver99 * 100 / n).toFixed(1) + '%' },
        logScale: { min: lsMin.toFixed(2), max: lsMax.toFixed(2),
                    sigmaP50: sigmaP50.toFixed(3), sigmaP99: sigmaP99.toFixed(3),
                    sigmaMax: sigmaMax.toFixed(3) },
        samples
    };
};
