#[cfg(use_scalar)]
fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(use_avx2)]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let len = a.len();
    let mut i = 0;
    let mut acc = _mm256_setzero_ps();

    // Process 8 elements at a time
    while i + 8 <= len {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(a_vec, b_vec));
        i += 8;
    }

    // Reduce 256-bit vector to 128-bit vector
    let acc_high = _mm256_extractf128_ps(acc, 0x1);
    let acc_low = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(acc_high, acc_low);

    // Horizontal add 128-bit vector
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let sum_scalar = _mm_cvtss_f32(_mm_add_ss(sums, _mm_movehl_ps(sums, sums)));

    // Process remaining elements
    let mut total = sum_scalar;
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

#[cfg(use_sse2)]
#[target_feature(enable = "sse")]
unsafe fn dot_product_f32_sse2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let len = a.len();
    let mut i = 0;
    let mut acc = _mm_setzero_ps();

    // Process 4 elements at a time
    while i + 4 <= len {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        acc = _mm_add_ps(acc, _mm_mul_ps(a_vec, b_vec));
        i += 4;
    }

    // Horizontal add
    let shuf = _mm_movehl_ps(acc, acc);
    let sums = _mm_add_ps(acc, shuf);
    let sum_scalar = _mm_cvtss_f32(_mm_add_ss(sums, _mm_shuffle_ps(sums, sums, 0x55)));

    // Process remaining elements
    let mut total = sum_scalar;
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

#[cfg(use_neon)]
#[target_feature(enable = "neon")]
unsafe fn dot_product_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let len = a.len();
    let mut i = 0;
    let mut acc = vdupq_n_f32(0.0);

    // Process 4 elements at a time
    while i + 4 <= len {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        acc = vaddq_f32(acc, vmulq_f32(a_vec, b_vec));
        i += 4;
    }

    // Horizontal vector sum
    let sum_scalar = vaddvq_f32(acc);

    // Process remaining elements
    let mut total = sum_scalar;
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(use_avx2)]
    {
        unsafe { dot_product_f32_avx2(a, b) }
    }

    #[cfg(use_sse2)]
    {
        unsafe { dot_product_f32_sse2(a, b) }
    }

    #[cfg(use_neon)]
    {
        unsafe { dot_product_f32_neon(a, b) }
    }

    #[cfg(use_scalar)]
    {
        dot_product_f32_scalar(a, b)
    }
}
