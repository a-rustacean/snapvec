#[cfg(use_scalar)]
pub fn dot_product_u8_scalar(a: &[u8], b: &[u8]) -> f32 {
    let len = a.len();
    let mut total_sum: u64 = 0;
    for i in 0..len {
        total_sum += a[i] as u64 * b[i] as u64;
    }
    total_sum as f32
}

#[cfg(use_avx2)]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_u8_avx2(a: &[u8], b: &[u8]) -> f32 {
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

            let a_low128 = _mm256_extracti128_si256(a_vec, 0);
            let a_high128 = _mm256_extracti128_si256(a_vec, 1);
            let b_low128 = _mm256_extracti128_si256(b_vec, 0);
            let b_high128 = _mm256_extracti128_si256(b_vec, 1);

            let a_low = _mm256_cvtepu8_epi16(a_low128);
            let a_high = _mm256_cvtepu8_epi16(a_high128);
            let b_low = _mm256_cvtepu8_epi16(b_low128);
            let b_high = _mm256_cvtepu8_epi16(b_high128);

            let prod_low = _mm256_mullo_epi16(a_low, b_low);
            let prod_high = _mm256_mullo_epi16(a_high, b_high);

            let prod_low_low128 = _mm256_extracti128_si256(prod_low, 0);
            let prod_low_high128 = _mm256_extracti128_si256(prod_low, 1);
            let prod_high_low128 = _mm256_extracti128_si256(prod_high, 0);
            let prod_high_high128 = _mm256_extracti128_si256(prod_high, 1);

            let prod_low_low = _mm256_cvtepu16_epi32(prod_low_low128);
            let prod_low_high = _mm256_cvtepu16_epi32(prod_low_high128);
            let prod_high_low = _mm256_cvtepu16_epi32(prod_high_low128);
            let prod_high_high = _mm256_cvtepu16_epi32(prod_high_high128);

            let mut sum_vector = _mm256_setzero_si256();
            sum_vector = _mm256_add_epi32(sum_vector, prod_low_low);
            sum_vector = _mm256_add_epi32(sum_vector, prod_low_high);
            sum_vector = _mm256_add_epi32(sum_vector, prod_high_low);
            sum_vector = _mm256_add_epi32(sum_vector, prod_high_high);

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
        total_sum += a[j] as u64 * b[j] as u64;
    }

    total_sum as f32
}

#[cfg(use_sse2)]
#[target_feature(enable = "sse2")]
pub unsafe fn dot_product_u8_sse2(a: &[u8], b: &[u8]) -> f32 {
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

            let a_low = _mm_cvtepu8_epi16(a_vec);
            let a_high = _mm_cvtepu8_epi16(_mm_srli_si128(a_vec, 8));
            let b_low = _mm_cvtepu8_epi16(b_vec);
            let b_high = _mm_cvtepu8_epi16(_mm_srli_si128(b_vec, 8));

            let prod_low = _mm_mullo_epi16(a_low, b_low);
            let prod_high = _mm_mullo_epi16(a_high, b_high);

            let prod_low_low = _mm_cvtepu16_epi32(prod_low);
            let prod_low_high = _mm_cvtepu16_epi32(_mm_srli_si128(prod_low, 8));
            let prod_high_low = _mm_cvtepu16_epi32(prod_high);
            let prod_high_high = _mm_cvtepu16_epi32(_mm_srli_si128(prod_high, 8));

            let sum1 = _mm_add_epi32(prod_low_low, prod_low_high);
            let sum2 = _mm_add_epi32(prod_high_low, prod_high_high);
            let sum = _mm_add_epi32(sum1, sum2);

            let sum_shift = _mm_srli_si128(sum, 8);
            let sum2 = _mm_add_epi32(sum, sum_shift);
            let sum_shift = _mm_srli_si128(sum2, 4);
            let sum3 = _mm_add_epi32(sum2, sum_shift);
            let sum_chunk = _mm_cvtsi128_si32(sum3) as u32;

            total_sum += sum_chunk as u64;
        }
        i = (len / 16) * 16;
    }

    for j in i..len {
        total_sum += a[j] as u64 * b[j] as u64;
    }

    total_sum as f32
}

#[cfg(use_neon)]
#[target_feature(enable = "neon")]
pub unsafe fn dot_product_u8_neon(a: &[u8], b: &[u8]) -> f32 {
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

            let a_low = vmovl_u8(a_low);
            let a_high = vmovl_u8(a_high);
            let b_low = vmovl_u8(b_low);
            let b_high = vmovl_u8(b_high);

            let prod_low = vmulq_u16(a_low, b_low);
            let prod_high = vmulq_u16(a_high, b_high);

            let sum_low = vpaddlq_u16(prod_low);
            let sum_high = vpaddlq_u16(prod_high);

            let sum = vaddq_u32(sum_low, sum_high);

            let sum_chunk = vaddvq_u32(sum) as u32;

            total_sum += sum_chunk as u64;
        }
        i = (len / 16) * 16;
    }

    for j in i..len {
        total_sum += a[j] as u64 * b[j] as u64;
    }

    total_sum as f32
}

pub fn dot_product_u8(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(use_avx2)]
    {
        unsafe { dot_product_u8_avx2(a, b) }
    }

    #[cfg(use_sse2)]
    {
        unsafe { dot_product_u8_sse2(a, b) }
    }

    #[cfg(use_neon)]
    {
        unsafe { dot_product_u8_neon(a, b) }
    }

    #[cfg(use_scalar)]
    {
        dot_product_u8_scalar(a, b)
    }
}
