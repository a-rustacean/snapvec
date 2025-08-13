#[cfg(f16_use_scalar)]
fn l2sq_distance_f16_scalar(a: &[f16], b: &[f16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = (a[i] as f32) - (b[i] as f32);
        sum += diff * diff;
    }
    sum
}

#[cfg(f16_use_avx2)]
#[target_feature(enable = "avx2,f16c")]
unsafe fn l2sq_distance_f16_avx2(a: &[f16], b: &[f16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= len {
        let a_ptr = a.as_ptr().add(i) as *const __m128i;
        let b_ptr = b.as_ptr().add(i) as *const __m128i;

        let a_vec1 = _mm_loadu_si128(a_ptr);
        let a_vec2 = _mm_loadu_si128(a_ptr.add(1));
        let b_vec1 = _mm_loadu_si128(b_ptr);
        let b_vec2 = _mm_loadu_si128(b_ptr.add(1));

        let a_float1 = _mm256_cvtph_ps(a_vec1);
        let a_float2 = _mm256_cvtph_ps(a_vec2);
        let b_float1 = _mm256_cvtph_ps(b_vec1);
        let b_float2 = _mm256_cvtph_ps(b_vec2);

        let diff1 = _mm256_sub_ps(a_float1, b_float1);
        let diff2 = _mm256_sub_ps(a_float2, b_float2);
        let sq1 = _mm256_mul_ps(diff1, diff1);
        let sq2 = _mm256_mul_ps(diff2, diff2);
        sum = _mm256_add_ps(sum, sq1);
        sum = _mm256_add_ps(sum, sq2);

        i += 16;
    }

    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let mut sum_sse = _mm_add_ps(sum_high, sum_low);
    sum_sse = _mm_hadd_ps(sum_sse, sum_sse);
    sum_sse = _mm_hadd_ps(sum_sse, sum_sse);
    let mut total = _mm_cvtss_f32(sum_sse);

    while i < len {
        let diff = a[i] as f32 - b[i] as f32;
        total += diff * diff;
        i += 1;
    }

    total
}

#[cfg(f16_use_sse2)]
#[target_feature(enable = "sse2,f16c")]
unsafe fn l2sq_distance_f16_sse2(a: &[f16], b: &[f16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 8 <= len {
        let a_ptr = a.as_ptr().add(i) as *const __m128i;
        let b_ptr = b.as_ptr().add(i) as *const __m128i;

        let a_vec = _mm_loadu_si128(a_ptr);
        let b_vec = _mm_loadu_si128(b_ptr);

        let a_low = _mm_cvtph_ps(a_vec);
        let b_low = _mm_cvtph_ps(b_vec);
        let diff_low = _mm_sub_ps(a_low, b_low);
        let sq_low = _mm_mul_ps(diff_low, diff_low);
        sum = _mm_add_ps(sum, sq_low);

        let a_high = _mm_srli_si128(a_vec, 8);
        let b_high = _mm_srli_si128(b_vec, 8);
        let a_high_f32 = _mm_cvtph_ps(a_high);
        let b_high_f32 = _mm_cvtph_ps(b_high);
        let diff_high = _mm_sub_ps(a_high_f32, b_high_f32);
        let sq_high = _mm_mul_ps(diff_high, diff_high);
        sum = _mm_add_ps(sum, sq_high);

        i += 8;
    }

    let sum2 = _mm_movehl_ps(sum, sum);
    let sum = _mm_add_ps(sum, sum2);
    let sum2 = _mm_shuffle_ps(sum, sum, 0x55);
    let mut total = _mm_cvtss_f32(_mm_add_ps(sum, sum2));

    while i < len {
        let diff = a[i] as f32 - b[i] as f32;
        total += diff * diff;
        i += 1;
    }

    total
}

#[cfg(f16_use_neon)]
#[target_feature(enable = "neon")]
unsafe fn l2sq_distance_f16_neon(a: &[f16], b: &[f16]) -> f32 {
    use core::arch::aarch64::*;

    let len = a.len();
    let mut total = 0.0f32;
    let mut i = 0;
    let mut sum = vdupq_n_f32(0.0);

    while i + 8 <= len {
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);

        let a_vec = vld1q_f16(a_ptr);
        let b_vec = vld1q_f16(b_ptr);

        let a_low = vget_low_f16(a_vec);
        let b_low = vget_low_f16(b_vec);
        let a_low_f32 = vcvt_f32_f16(a_low);
        let b_low_f32 = vcvt_f32_f16(b_low);

        let a_high = vget_high_f16(a_vec);
        let b_high = vget_high_f16(b_vec);
        let a_high_f32 = vcvt_f32_f16(a_high);
        let b_high_f32 = vcvt_f32_f16(b_high);

        let diff_low = vsubq_f32(a_low_f32, b_low_f32);
        let diff_high = vsubq_f32(a_high_f32, b_high_f32);

        let sq_low = vmulq_f32(diff_low, diff_low);
        let sq_high = vmulq_f32(diff_high, diff_high);

        sum = vaddq_f32(sum, sq_low);
        sum = vaddq_f32(sum, sq_high);

        i += 8;
    }

    total += vaddvq_f32(sum);

    while i < len {
        let diff = a[i] as f32 - b[i] as f32;
        total += diff * diff;
        i += 1;
    }

    total
}

pub fn l2sq_distance_f16(a: &[f16], b: &[f16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(f16_use_avx2)]
    {
        unsafe { l2sq_distance_f16_avx2(a, b) }
    }

    #[cfg(f16_use_sse2)]
    {
        unsafe { l2sq_distance_f16_sse2(a, b) }
    }

    #[cfg(f16_use_neon)]
    {
        unsafe { l2sq_distance_f16_neon(a, b) }
    }

    #[cfg(f16_use_scalar)]
    {
        l2sq_distance_f16_scalar(a, b)
    }
}
