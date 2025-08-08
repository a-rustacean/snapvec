#[cfg(use_scalar)]
fn dot_product_i8_scalar(a: &[i8], b: &[i8]) -> f32 {
    let mut sum = 0;
    for i in 0..a.len() {
        sum += a[i] as i32 * b[i] as i32;
    }
    sum as f32
}

#[cfg(use_avx2)]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_i8_avx2(a: &[i8], b: &[i8]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let len = a.len();
    let mut i = 0;
    let mut total = 0;

    if len >= 32 {
        let mut acc = _mm256_setzero_si256();
        let end = len & !31;

        while i < end {
            let a_ptr = a.as_ptr().add(i);
            let b_ptr = b.as_ptr().add(i);

            let a_vec = _mm256_loadu_si256(a_ptr as *const __m256i);
            let b_vec = _mm256_loadu_si256(b_ptr as *const __m256i);

            let a_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_vec));
            let a_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec, 1));
            let b_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_vec));
            let b_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 1));

            let prod_low = _mm256_madd_epi16(a_low, b_low);
            let prod_high = _mm256_madd_epi16(a_high, b_high);

            let sum_chunk = _mm256_add_epi32(prod_low, prod_high);
            acc = _mm256_add_epi32(acc, sum_chunk);
            i += 32;
        }

        let acc_high = _mm256_extracti128_si256(acc, 1);
        let acc_low = _mm256_castsi256_si128(acc);
        let acc_low = _mm_add_epi32(acc_low, acc_high);
        let acc_low = _mm_hadd_epi32(acc_low, acc_low);
        let acc_low = _mm_hadd_epi32(acc_low, acc_low);
        total = _mm_cvtsi128_si32(acc_low) as i32;
    }

    while i < len {
        total += a[i] as i32 * b[i] as i32;
        i += 1;
    }

    total as f32
}

#[cfg(use_sse2)]
#[target_feature(enable = "sse2")]
unsafe fn dot_product_i8_sse2(a: &[i8], b: &[i8]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let len = a.len();
    let mut i = 0;
    let mut total = 0;

    if len >= 16 {
        let mut acc = _mm_setzero_si128();
        let end = len & !15;

        while i < end {
            let a_ptr = a.as_ptr().add(i);
            let b_ptr = b.as_ptr().add(i);

            let a_vec = _mm_loadu_si128(a_ptr as *const __m128i);
            let b_vec = _mm_loadu_si128(b_ptr as *const __m128i);

            let a_low = _mm_srai_epi16(_mm_unpacklo_epi8(a_vec, a_vec), 8);
            let a_high = _mm_srai_epi16(_mm_unpackhi_epi8(a_vec, a_vec), 8);
            let b_low = _mm_srai_epi16(_mm_unpacklo_epi8(b_vec, b_vec), 8);
            let b_high = _mm_srai_epi16(_mm_unpackhi_epi8(b_vec, b_vec), 8);

            let prod_low = _mm_madd_epi16(a_low, b_low);
            let prod_high = _mm_madd_epi16(a_high, b_high);

            let sum_chunk = _mm_add_epi32(prod_low, prod_high);
            acc = _mm_add_epi32(acc, sum_chunk);
            i += 16;
        }

        total = {
            let ptr = &acc as *const __m128i as *const i32;
            *ptr.offset(0) + *ptr.offset(1) + *ptr.offset(2) + *ptr.offset(3)
        };
    }

    while i < len {
        total += a[i] as i32 * b[i] as i32;
        i += 1;
    }

    total as f32
}

#[cfg(use_neon)]
#[target_feature(enable = "neon")]
unsafe fn dot_product_i8_neon(a: &[i8], b: &[i8]) -> f32 {
    use core::arch::aarch64::*;

    let len = a.len();
    let mut i = 0;
    let mut total = 0;

    if len >= 16 {
        let mut acc = vdupq_n_s32(0);
        let end = len & !15;

        while i < end {
            let a_ptr = a.as_ptr().add(i);
            let b_ptr = b.as_ptr().add(i);

            let a_vec = vld1q_s8(a_ptr);
            let b_vec = vld1q_s8(b_ptr);

            let a_low = vmovl_s8(vget_low_s8(a_vec));
            let a_high = vmovl_s8(vget_high_s8(a_vec));
            let b_low = vmovl_s8(vget_low_s8(b_vec));
            let b_high = vmovl_s8(vget_high_s8(b_vec));

            let prod_low = vmulq_s16(a_low, b_low);
            let prod_high = vmulq_s16(a_high, b_high);

            let sum_low = vpaddlq_s16(prod_low);
            let sum_high = vpaddlq_s16(prod_high);
            let sum_chunk = vaddq_s32(sum_low, sum_high);

            acc = vaddq_s32(acc, sum_chunk);
            i += 16;
        }

        total = vaddvq_s32(acc) as i32;
    }

    while i < len {
        total += a[i] as i32 * b[i] as i32;
        i += 1;
    }

    total as f32
}

pub fn dot_product_i8(a: &[i8], b: &[i8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(use_avx2)]
    {
        unsafe { dot_product_i8_avx2(a, b) }
    }

    #[cfg(use_sse2)]
    {
        unsafe { dot_product_i8_sse2(a, b) }
    }

    #[cfg(use_neon)]
    {
        unsafe { dot_product_i8_neon(a, b) }
    }

    #[cfg(use_scalar)]
    {
        dot_product_i8_scalar(a, b)
    }
}
