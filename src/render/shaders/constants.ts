/**
 * Shared constants for renderer shaders.
 *
 * `TILE_SIZE`        — pixel side length of one tile (workgroup size).
 * `SIGMA_CUTOFF`     — Gaussian truncation in standard deviations.
 * `AA_DILATION_COV`  — adds 0.3px to the diagonal of 2D covariance to
 *                      prevent sub-pixel splats from disappearing under
 *                      perspective.
 * `OPACITY_CAP`      — clamps a single splat's contribution to avoid
 *                      numerical issues with α near 1.
 * `MIN_TRANSMITTANCE`— pixel early-out when remaining T drops below this.
 * `MIN_ALPHA`        — skip splat if its α at this pixel is too small to
 *                      matter (perf opt).
 * `GAUSSIAN_FLOOR`   — subtracted from exp(power) so alpha → 0 exactly at
 *                      the 3-sigma truncation radius. Matches SuperSplat
 *                      (and the PlayCanvas engine) — eliminates faint ring
 *                      artifacts.
 */
export const TILE_SIZE = 16;
export const SIGMA_CUTOFF = 3.0;
export const AA_DILATION_COV = 0.3;
export const OPACITY_CAP = 0.999;
export const MIN_TRANSMITTANCE = 1.0e-4;
export const MIN_ALPHA = 1.0 / 255.0;
/** exp(-0.5 * SIGMA_CUTOFF^2) = exp(-4.5) ≈ 0.0111. */
export const GAUSSIAN_FLOOR = Math.exp(-0.5 * SIGMA_CUTOFF * SIGMA_CUTOFF);

/**
 * Outlier-splat fade thresholds, expressed as fractions of image height.
 *
 * A splat whose un-clamped 3σ screen-radius exceeds `START × height`
 * starts fading; at `END × height` its alpha reaches 0 and the splat is
 * discarded. Defends against pathological mega-splats that would
 * otherwise tint the whole frame. Image-height-relative so the same
 * world-space splats fade at any render resolution. Matches splat-transform.
 */
export const RADIUS_FADE_START_FRAC = 1024 / 1080;
export const RADIUS_FADE_END_FRAC = 2048 / 1080;

/** WGSL prelude with the constants above, prepended to every shader. */
export const SHADER_PRELUDE = /* wgsl */`
const TILE_SIZE: u32 = ${TILE_SIZE}u;
const SIGMA_CUTOFF: f32 = ${SIGMA_CUTOFF};
const AA_DILATION_COV: f32 = ${AA_DILATION_COV};
const OPACITY_CAP: f32 = ${OPACITY_CAP};
const MIN_TRANSMITTANCE: f32 = ${MIN_TRANSMITTANCE};
const MIN_ALPHA: f32 = ${MIN_ALPHA};
const GAUSSIAN_FLOOR: f32 = ${GAUSSIAN_FLOOR.toExponential(10)};
const RADIUS_FADE_START_FRAC: f32 = ${RADIUS_FADE_START_FRAC};
const RADIUS_FADE_END_FRAC: f32 = ${RADIUS_FADE_END_FRAC};
`;
