#[cfg(use_scalar)]
pub fn l2sq_distance_u8_scalar(a: &[u8], b: &[u8]) -> f32 {
    let len = a.len();
    let mut total_sum: u64 = 0;
    for i in 0..len {
        let diff = a[i] as i32 - b[i] as i32;
        total_sum += (diff * diff) as u64;
    }
    total_sum as f32
}

#[cfg(use_avx2)]
#[target_feature(enable = "avx2")]
pub unsafe fn l2sq_distance_u8_avx2(a: &[u8], b: &[u8]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut total_sum: u64 = 0;
    let mut i = 0;

    if len >= 32 {
        for chunk in (0..(len - 31)).step_by(32) {
            let a_vec = _mm256_loadu_si256(a_ptr.add(chunk) as *const _);
            let b_vec = _mm256_loadu_si256(b_ptr.add(chunk) as *const _);

            // Convert to 16-bit signed integers to handle negative differences
            let a_low128 = _mm256_extracti128_si256(a_vec, 0);
            let a_high128 = _mm256_extracti128_si256(a_vec, 1);
            let b_low128 = _mm256_extracti128_si256(b_vec, 0);
            let b_high128 = _mm256_extracti128_si256(b_vec, 1);

            let a_low = _mm256_cvtepu8_epi16(a_low128);
            let a_high = _mm256_cvtepu8_epi16(a_high128);
            let b_low = _mm256_cvtepu8_epi16(b_low128);
            let b_high = _mm256_cvtepu8_epi16(b_high128);

            // Compute differences
            let diff_low = _mm256_sub_epi16(a_low, b_low);
            let diff_high = _mm256_sub_epi16(a_high, b_high);

            // Square the differences using multiply and accumulate
            let sq_low = _mm256_madd_epi16(diff_low, diff_low);
            let sq_high = _mm256_madd_epi16(diff_high, diff_high);

            // Sum and accumulate
            let mut sum_vector = _mm256_setzero_si256();
            sum_vector = _mm256_add_epi32(sum_vector, sq_low);
            sum_vector = _mm256_add_epi32(sum_vector, sq_high);

            let low = _mm256_castsi256_si128(sum_vector);
            let high = _mm256_extracti128_si256(sum_vector, 1);
            let sum128 = _mm_add_epi32(low, high);
            let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
            let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
            let sum_chunk = _mm_cvtsi128_si32(sum32) as u32;

            total_sum += sum_chunk as u64;
        }
        i = (len / 32) * 32;
    }

    for j in i..len {
        let diff = a[j] as i32 - b[j] as i32;
        total_sum += (diff * diff) as u64;
    }

    total_sum as f32
}

#[cfg(use_sse2)]
#[target_feature(enable = "sse2")]
pub unsafe fn l2sq_distance_u8_sse2(a: &[u8], b: &[u8]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut total_sum: u64 = 0;
    let mut i = 0;

    if len >= 16 {
        for chunk in (0..(len - 15)).step_by(16) {
            let a_vec = _mm_loadu_si128(a_ptr.add(chunk) as *const _);
            let b_vec = _mm_loadu_si128(b_ptr.add(chunk) as *const _);

            // Convert to 16-bit signed integers to handle negative differences
            let a_low = _mm_cvtepu8_epi16(a_vec);
            let a_high = _mm_cvtepu8_epi16(_mm_srli_si128(a_vec, 8));
            let b_low = _mm_cvtepu8_epi16(b_vec);
            let b_high = _mm_cvtepu8_epi16(_mm_srli_si128(b_vec, 8));

            // Compute differences
            let diff_low = _mm_sub_epi16(a_low, b_low);
            let diff_high = _mm_sub_epi16(a_high, b_high);

            // Square the differences using multiply and accumulate
            let sq_low = _mm_madd_epi16(diff_low, diff_low);
            let sq_high = _mm_madd_epi16(diff_high, diff_high);

            // Sum and accumulate
            let sum1 = _mm_add_epi32(sq_low, sq_high);
            let sum_shift = _mm_srli_si128(sum1, 8);
            let sum2 = _mm_add_epi32(sum1, sum_shift);
            let sum_shift = _mm_srli_si128(sum2, 4);
            let sum3 = _mm_add_epi32(sum2, sum_shift);
            let sum_chunk = _mm_cvtsi128_si32(sum3) as u32;

            total_sum += sum_chunk as u64;
        }
        i = (len / 16) * 16;
    }

    for j in i..len {
        let diff = a[j] as i32 - b[j] as i32;
        total_sum += (diff * diff) as u64;
    }

    total_sum as f32
}

#[cfg(use_neon)]
#[target_feature(enable = "neon")]
pub unsafe fn l2sq_distance_u8_neon(a: &[u8], b: &[u8]) -> f32 {
    use core::arch::aarch64::*;

    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut total_sum: u64 = 0;
    let mut i = 0;

    if len >= 16 {
        for chunk in (0..(len - 15)).step_by(16) {
            let a_vec = vld1q_u8(a_ptr.add(chunk));
            let b_vec = vld1q_u8(b_ptr.add(chunk));

            let a_low = vget_low_u8(a_vec);
            let a_high = vget_high_u8(a_vec);
            let b_low = vget_low_u8(b_vec);
            let b_high = vget_high_u8(b_vec);

            // Convert to 16-bit signed integers
            let a_low_s16 = vreinterpretq_s16_u16(vmovl_u8(a_low));
            let a_high_s16 = vreinterpretq_s16_u16(vmovl_u8(a_high));
            let b_low_s16 = vreinterpretq_s16_u16(vmovl_u8(b_low));
            let b_high_s16 = vreinterpretq_s16_u16(vmovl_u8(b_high));

            // Compute differences
            let diff_low = vsubq_s16(a_low_s16, b_low_s16);
            let diff_high = vsubq_s16(a_high_s16, b_high_s16);

            // Square the differences
            let sq_low = vmulq_s16(diff_low, diff_low);
            let sq_high = vmulq_s16(diff_high, diff_high);

            // Convert to 32-bit and sum
            let sum_low = vpaddlq_s16(sq_low);
            let sum_high = vpaddlq_s16(sq_high);

            let sum = vaddq_s32(sum_low, sum_high);
            let sum_chunk = vaddvq_s32(sum) as u32;

            total_sum += sum_chunk as u64;
        }
        i = (len / 16) * 16;
    }

    for j in i..len {
        let diff = a[j] as i32 - b[j] as i32;
        total_sum += (diff * diff) as u64;
    }

    total_sum as f32
}

pub fn l2sq_distance_u8(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(use_avx2)]
    {
        unsafe { l2sq_distance_u8_avx2(a, b) }
    }

    #[cfg(use_sse2)]
    {
        unsafe { l2sq_distance_u8_sse2(a, b) }
    }

    #[cfg(use_neon)]
    {
        unsafe { l2sq_distance_u8_neon(a, b) }
    }

    #[cfg(use_scalar)]
    {
        l2sq_distance_u8_scalar(a, b)
    }
}
