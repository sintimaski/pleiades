// Haversine angular distance (arcsec) for one (ra, dec) pair. All coordinates in degrees.
// coords is interleaved: [ra_a, dec_a, ra_b, dec_b] per pair (4 f32 per index).
// Formula: a = sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δλ/2); θ = 2 asin(min(√a, 1)); arcsec = θ * RAD_TO_ARCSEC.

const DEG_TO_RAD: f32 = 0.017453292519943295;  // π/180
const RAD_TO_ARCSEC: f32 = 206264.80624709636; // 3600 * 180/π

@group(0) @binding(0) var<storage, read> coords: array<f32>;  // 4 floats per pair: ra_a, dec_a, ra_b, dec_b
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&out)) {
        return;
    }
    let base = i * 4u;
    let lon1 = coords[base + 0u] * DEG_TO_RAD;
    let lat1 = coords[base + 1u] * DEG_TO_RAD;
    let lon2 = coords[base + 2u] * DEG_TO_RAD;
    let lat2 = coords[base + 3u] * DEG_TO_RAD;
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let sin_dlat = sin(dlat * 0.5);
    let sin_dlon = sin(dlon * 0.5);
    let a = sin_dlat * sin_dlat + cos(lat1) * cos(lat2) * sin_dlon * sin_dlon;
    let sqrt_a = sqrt(min(a, 1.0));
    let c = 2.0 * asin(sqrt_a);
    out[i] = c * RAD_TO_ARCSEC;
}
