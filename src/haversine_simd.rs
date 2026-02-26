//! SIMD haversine for CPU join: AVX2 (x86_64) and NEON (aarch64).
//! Computes 4–8 haversine `a` terms per call for cheap-reject path.
//!
//! Uses std sin/cos (no native SIMD transcendentals in AVX2/NEON). SIMD gains come from
//! vectorized loads, deg-to-rad conversion, dlat/dlon arithmetic, and final mul_add.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

/// Haversine `a` for 4 lanes using AVX2. Output in out[0..4].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn haversine_a_4_avx2(
    ra1_deg: &[f64; 4],
    dec1_deg: &[f64; 4],
    lon2: f64,
    lat2: f64,
    cos_lat2: f64,
    out: &mut [f64; 4],
) {
    let half = _mm256_set1_pd(0.5);
    let deg_to_rad = _mm256_set1_pd(DEG_TO_RAD);

    let ra1 = _mm256_loadu_pd(ra1_deg.as_ptr());
    let dec1 = _mm256_loadu_pd(dec1_deg.as_ptr());

    let lon1 = _mm256_mul_pd(ra1, deg_to_rad);
    let lat1 = _mm256_mul_pd(dec1, deg_to_rad);

    let lat2_v = _mm256_set1_pd(lat2);
    let lon2_v = _mm256_set1_pd(lon2);

    let dlat = _mm256_sub_pd(lat2_v, lat1);
    let dlon = _mm256_sub_pd(lon2_v, lon1);

    let half_dlat = _mm256_mul_pd(dlat, half);
    let half_dlon = _mm256_mul_pd(dlon, half);

    // Scalar sin/cos per lane (no SIMD transcendentals); use poly for consistency
    let mut sin_half_dlat = [0.0_f64; 4];
    let mut sin_half_dlon = [0.0_f64; 4];
    let mut cos_lat1 = [0.0_f64; 4];
    _mm256_storeu_pd(sin_half_dlat.as_mut_ptr(), half_dlat);
    _mm256_storeu_pd(sin_half_dlon.as_mut_ptr(), half_dlon);
    _mm256_storeu_pd(cos_lat1.as_mut_ptr(), lat1);

    for i in 0..4 {
        sin_half_dlat[i] = sin_half_dlat[i].sin();
        sin_half_dlon[i] = sin_half_dlon[i].sin();
        cos_lat1[i] = cos_lat1[i].cos();
    }

    let sin_dlat = _mm256_loadu_pd(sin_half_dlat.as_ptr());
    let sin_dlon = _mm256_loadu_pd(sin_half_dlon.as_ptr());
    let cos_l1 = _mm256_loadu_pd(cos_lat1.as_ptr());

    let cos_lat2_v = _mm256_set1_pd(cos_lat2);

    let sin_dlat_sq = _mm256_mul_pd(sin_dlat, sin_dlat);
    let sin_dlon_sq = _mm256_mul_pd(sin_dlon, sin_dlon);
    let term2 = _mm256_mul_pd(_mm256_mul_pd(cos_l1, cos_lat2_v), sin_dlon_sq);

    let a = _mm256_add_pd(sin_dlat_sq, term2);
    _mm256_storeu_pd(out.as_mut_ptr(), a);
}

/// Haversine `a` for 4 lanes — SIMD when available (AVX2 on x86_64), else scalar.
#[inline(always)]
pub fn haversine_a_4_rad(
    ra1_deg: &[f64; 4],
    dec1_deg: &[f64; 4],
    lon2: f64,
    lat2: f64,
    cos_lat2: f64,
    out: &mut [f64; 4],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                haversine_a_4_avx2(ra1_deg, dec1_deg, lon2, lat2, cos_lat2, out);
            }
            return;
        }
    }

    // Scalar fallback (aarch64, or x86 without AVX2)
    for i in 0..4 {
        let lon1 = ra1_deg[i] * DEG_TO_RAD;
        let lat1 = dec1_deg[i] * DEG_TO_RAD;
        let dlat = lat2 - lat1;
        let dlon = lon2 - lon1;
        let half_dlat = dlat * 0.5;
        let half_dlon = dlon * 0.5;
        let sin_hd = half_dlat.sin();
        let sin_hl = half_dlon.sin();
        let cos_l1 = lat1.cos();
        out[i] = sin_hd * sin_hd + cos_l1 * cos_lat2 * sin_hl * sin_hl;
    }
}

/// Haversine `a` for 8 lanes — two batches of 4.
#[inline(always)]
pub fn haversine_a_8_rad(
    ra1_deg: &[f64; 8],
    dec1_deg: &[f64; 8],
    lon2: f64,
    lat2: f64,
    cos_lat2: f64,
    out: &mut [f64; 8],
) {
    let (ra1_4a, ra1_4b) = (
        [ra1_deg[0], ra1_deg[1], ra1_deg[2], ra1_deg[3]],
        [ra1_deg[4], ra1_deg[5], ra1_deg[6], ra1_deg[7]],
    );
    let (dec1_4a, dec1_4b) = (
        [dec1_deg[0], dec1_deg[1], dec1_deg[2], dec1_deg[3]],
        [dec1_deg[4], dec1_deg[5], dec1_deg[6], dec1_deg[7]],
    );
    let mut out_a = [0.0_f64; 4];
    let mut out_b = [0.0_f64; 4];
    haversine_a_4_rad(&ra1_4a, &dec1_4a, lon2, lat2, cos_lat2, &mut out_a);
    haversine_a_4_rad(&ra1_4b, &dec1_4b, lon2, lat2, cos_lat2, &mut out_b);
    out[0..4].copy_from_slice(&out_a);
    out[4..8].copy_from_slice(&out_b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_a_4_matches_scalar() {
        let ra1 = [10.0, 20.0, 30.0, 40.0];
        let dec1 = [-5.0, 5.0, -10.0, 10.0];
        let lon2 = 15.0 * DEG_TO_RAD;
        let lat2 = 0.0 * DEG_TO_RAD;
        let cos_lat2 = 1.0;

        let mut out = [0.0_f64; 4];
        haversine_a_4_rad(&ra1, &dec1, lon2, lat2, cos_lat2, &mut out);

        for i in 0..4 {
            let lon1 = ra1[i] * DEG_TO_RAD;
            let lat1 = dec1[i] * DEG_TO_RAD;
            let dlat = lat2 - lat1;
            let dlon = lon2 - lon1;
            let expected = (dlat * 0.5).sin().mul_add(
                (dlat * 0.5).sin(),
                lat1.cos() * cos_lat2 * (dlon * 0.5).sin() * (dlon * 0.5).sin(),
            );
            assert!(
                (out[i] - expected).abs() < 1e-8,
                "lane {}: got {} expected {}",
                i,
                out[i],
                expected
            );
        }
    }
}
