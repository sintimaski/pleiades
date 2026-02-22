// Haversine + radius filter + compact output. Only writes (a_ix, b_ix, sep) for pairs with sep <= radius.
// Pairs: 24 bytes each (ra_a, dec_a, ra_b, dec_b: f32; a_ix, b_ix: u32).
// Push constant: radius_arcsec (f32).

const DEG_TO_RAD: f32 = 0.017453292519943295;
const RAD_TO_ARCSEC: f32 = 206264.80624709636;

struct Pair {
    ra_a: f32,
    dec_a: f32,
    ra_b: f32,
    dec_b: f32,
    a_ix: u32,
    b_ix: u32,
}

struct Match {
    a_ix: u32,
    b_ix: u32,
    sep: f32,
}

struct Params {
    radius_arcsec: f32,
}

@group(0) @binding(0) var<storage, read> pairs: array<Pair>;
@group(0) @binding(1) var<storage, read_write> count: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> matches: array<Match>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&pairs)) {
        return;
    }
    let p = pairs[i];
    let lon1 = p.ra_a * DEG_TO_RAD;
    let lat1 = p.dec_a * DEG_TO_RAD;
    let lon2 = p.ra_b * DEG_TO_RAD;
    let lat2 = p.dec_b * DEG_TO_RAD;
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let sin_dlat = sin(dlat * 0.5);
    let sin_dlon = sin(dlon * 0.5);
    let a = sin_dlat * sin_dlat + cos(lat1) * cos(lat2) * sin_dlon * sin_dlon;
    let sqrt_a = sqrt(min(a, 1.0));
    let c = 2.0 * asin(sqrt_a);
    let sep = c * RAD_TO_ARCSEC;
    if (sep <= params.radius_arcsec) {
        let idx = atomicAdd(&count, 1u);
        matches[idx].a_ix = p.a_ix;
        matches[idx].b_ix = p.b_ix;
        matches[idx].sep = sep;
    }
}
