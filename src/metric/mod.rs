#![allow(unsafe_op_in_unsafe_fn)]

mod dot_product;
mod l2sq_distance;

use core::{cmp::Ordering, f32};

use crate::{
    metric::dot_product::{dot_product_f16, dot_product_f32, dot_product_i8, dot_product_u8},
    metric::l2sq_distance::{
        l2sq_distance_f16, l2sq_distance_f32, l2sq_distance_i8, l2sq_distance_u8,
    },
    storage::{QuantVec, Quantization, RawVec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum DistanceMetricKind {
    #[default]
    Cos,
    L2sq,
    Dot,
}

type DistanceComputeFn = fn(&QuantVec, &QuantVec) -> f32;

pub struct DistanceMetric {
    kind: DistanceMetricKind,
    compute_fn: DistanceComputeFn,
}

fn i8_cos(a: &QuantVec, b: &QuantVec) -> f32 {
    let dot_product = dot_product_i8(a.as_i8_slice(), b.as_i8_slice());
    cosine_similarity_from_dot_product(dot_product, a.mag, b.mag)
}

fn u8_cos(a: &QuantVec, b: &QuantVec) -> f32 {
    let dot_product = dot_product_u8(a.as_u8_slice(), b.as_u8_slice());
    cosine_similarity_from_dot_product(dot_product, a.mag, b.mag)
}

fn f16_cos(a: &QuantVec, b: &QuantVec) -> f32 {
    let dot_product = dot_product_f16(a.as_f16_slice(), b.as_f16_slice());
    cosine_similarity_from_dot_product(dot_product, a.mag, b.mag)
}

fn f32_cos(a: &QuantVec, b: &QuantVec) -> f32 {
    let dot_product = dot_product_f32(a.as_f32_slice(), b.as_f32_slice());
    cosine_similarity_from_dot_product(dot_product, a.mag, b.mag)
}

fn i8_dot(a: &QuantVec, b: &QuantVec) -> f32 {
    dot_product_i8(a.as_i8_slice(), b.as_i8_slice())
}

fn u8_dot(a: &QuantVec, b: &QuantVec) -> f32 {
    dot_product_u8(a.as_u8_slice(), b.as_u8_slice())
}

fn f16_dot(a: &QuantVec, b: &QuantVec) -> f32 {
    dot_product_f16(a.as_f16_slice(), b.as_f16_slice())
}

fn f32_dot(a: &QuantVec, b: &QuantVec) -> f32 {
    dot_product_f32(a.as_f32_slice(), b.as_f32_slice())
}

fn i8_l2sq(a: &QuantVec, b: &QuantVec) -> f32 {
    l2sq_distance_i8(a.as_i8_slice(), b.as_i8_slice())
}

fn u8_l2sq(a: &QuantVec, b: &QuantVec) -> f32 {
    l2sq_distance_u8(a.as_u8_slice(), b.as_u8_slice())
}

fn f16_l2sq(a: &QuantVec, b: &QuantVec) -> f32 {
    l2sq_distance_f16(a.as_f16_slice(), b.as_f16_slice())
}

fn f32_l2sq(a: &QuantVec, b: &QuantVec) -> f32 {
    l2sq_distance_f32(a.as_f32_slice(), b.as_f32_slice())
}

impl DistanceMetric {
    pub fn new(kind: DistanceMetricKind, quantization: Quantization) -> Self {
        use DistanceMetricKind::*;
        use Quantization::*;

        let compute_fn = match (quantization, kind) {
            (I8, Cos) => i8_cos,
            (U8, Cos) => u8_cos,
            (F16, Cos) => f16_cos,
            (F32, Cos) => f32_cos,
            (I8, Dot) => i8_dot,
            (U8, Dot) => u8_dot,
            (F16, Dot) => f16_dot,
            (F32, Dot) => f32_dot,
            (I8, L2sq) => i8_l2sq,
            (U8, L2sq) => u8_l2sq,
            (F16, L2sq) => f16_l2sq,
            (F32, L2sq) => f32_l2sq,
        };

        Self { kind, compute_fn }
    }

    #[inline(always)]
    pub fn calculate(&self, a: &QuantVec, b: &QuantVec) -> f32 {
        (self.compute_fn)(a, b)
    }

    pub fn calculate_raw(&self, a: &RawVec, mag_a: f32, b: &RawVec, mag_b: f32) -> f32 {
        use DistanceMetricKind::*;
        match self.kind {
            Cos => {
                let dot_product = dot_product_f32(&a.vec, &b.vec);
                cosine_similarity_from_dot_product(dot_product, mag_a, mag_b)
            }
            Dot => dot_product_f32(&a.vec, &b.vec),
            L2sq => l2sq_distance_f32(&a.vec, &b.vec),
        }
    }

    pub fn cmp_score(&self, a: f32, b: f32) -> Ordering {
        use DistanceMetricKind::*;
        match self.kind {
            Cos => a.total_cmp(&b),
            L2sq => b.total_cmp(&a),
            Dot => a.total_cmp(&b),
        }
    }

    pub fn max_value(&self) -> f32 {
        use DistanceMetricKind::*;
        match self.kind {
            Cos => 2.0,
            L2sq => 0.0,
            Dot => f32::INFINITY,
        }
    }
}

// Magnitude for i8 vectors
pub fn magnitude_i8(v: &[i8]) -> f32 {
    dot_product_i8(v, v).sqrt()
}

// Magnitude for u8 vectors
pub fn magnitude_u8(v: &[u8]) -> f32 {
    dot_product_u8(v, v).sqrt()
}

// Magnitude for f16 vectors
pub fn magnitude_f16(v: &[f16]) -> f32 {
    dot_product_f16(v, v).sqrt()
}

// Magnitude for f32 vectors
pub fn magnitude_f32(v: &[f32]) -> f32 {
    dot_product_f32(v, v).sqrt()
}

pub fn cosine_similarity_from_dot_product(dot_product: f32, mag_a: f32, mag_b: f32) -> f32 {
    let denominator = mag_a * mag_b;

    if denominator == 0.0 {
        0.0
    } else {
        dot_product / denominator
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    // Tests for dot_product_u8
    #[test]
    fn test_dot_product_u8() {
        assert_eq!(dot_product_u8(&[1, 2, 3], &[4, 5, 6]), 32.0);
        assert_eq!(dot_product_u8(&[0, 0, 0], &[1, 2, 3]), 0.0);
        assert_eq!(dot_product_u8(&[], &[]), 0.0);
        assert_eq!(dot_product_u8(&[255], &[255]), 65025.0);
    }

    #[test]
    fn test_dot_product_u8_extended() {
        // Test single element
        assert_eq!(dot_product_u8(&[5], &[10]), 50.0);

        // Test larger vectors
        let a = (0..100).map(|x| (x % 256) as u8).collect::<Vec<_>>();
        let b = (0..100).map(|x| ((x + 1) % 256) as u8).collect::<Vec<_>>();
        let expected: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as u32 * *y as u32) as f32)
            .sum();
        assert_eq!(dot_product_u8(&a, &b), expected);

        // Test with max values
        assert_eq!(dot_product_u8(&[255, 255], &[255, 255]), 130050.0);

        // Test orthogonal-like vectors (alternating pattern)
        let a = vec![1, 0, 1, 0, 1, 0];
        let b = vec![0, 1, 0, 1, 0, 1];
        assert_eq!(dot_product_u8(&a, &b), 0.0);

        // Test power of 2 sizes to check vectorized implementations
        let a16 = vec![1u8; 16];
        let b16 = vec![2u8; 16];
        assert_eq!(dot_product_u8(&a16, &b16), 32.0);

        let a32 = vec![3u8; 32];
        let b32 = vec![4u8; 32];
        assert_eq!(dot_product_u8(&a32, &b32), 384.0);

        let a64 = vec![5u8; 64];
        let b64 = vec![6u8; 64];
        assert_eq!(dot_product_u8(&a64, &b64), 1920.0);
    }

    // Tests for dot_product_i8
    #[test]
    fn test_dot_product_i8() {
        assert_eq!(dot_product_i8(&[1, 2, 3], &[4, 5, 6]), 32.0);
        assert_eq!(dot_product_i8(&[-1, -2, 3], &[4, 5, -6]), -32.0);
        assert_eq!(dot_product_i8(&[0, 0, 0], &[1, 2, 3]), 0.0);
        assert_eq!(dot_product_i8(&[], &[]), 0.0);
    }

    #[test]
    fn test_dot_product_i8_extended() {
        // Test single element
        assert_eq!(dot_product_i8(&[5], &[-10]), -50.0);

        // Test extreme values
        assert_eq!(dot_product_i8(&[127], &[127]), 16129.0);
        assert_eq!(dot_product_i8(&[-128], &[-128]), 16384.0);
        assert_eq!(dot_product_i8(&[127], &[-128]), -16256.0);

        // Test mixed positive and negative
        assert_eq!(dot_product_i8(&[1, -2, 3, -4], &[-5, 6, -7, 8]), -70.0);

        // Test larger vectors with pattern
        let a = (0..50).map(|x| (x % 256) as i8).collect::<Vec<_>>();
        let b = (0..50).map(|x| ((x + 1) % 256) as i8).collect::<Vec<_>>();
        let expected: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as i32 * *y as i32) as f32)
            .sum();
        assert_eq!(dot_product_i8(&a, &b), expected);

        // Test power of 2 sizes for vectorized implementations
        let a16 = vec![1i8; 16];
        let b16 = vec![-2i8; 16];
        assert_eq!(dot_product_i8(&a16, &b16), -32.0);

        let a32 = vec![3i8; 32];
        let b32 = vec![-4i8; 32];
        assert_eq!(dot_product_i8(&a32, &b32), -384.0);

        // Test alternating signs
        let a = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let b = vec![-1, 1, -1, 1, -1, 1, -1, 1];
        assert_eq!(dot_product_i8(&a, &b), -8.0);
    }

    // Tests for dot_product_f16
    #[test]
    fn test_dot_product_f16() {
        let tol = 0.001;

        let a = [1.5, 2.5];
        let b = [3.0, 4.0];
        assert!((dot_product_f16(&a, &b) - 14.5).abs() < tol);

        let a = [-1.5, 2.5];
        let b = [3.0, -4.0];
        assert!((dot_product_f16(&a, &b) - (-14.5)).abs() < tol);

        assert_eq!(dot_product_f16(&[], &[]), 0.0);
    }

    #[test]
    fn test_dot_product_f16_extended() {
        let tol = 0.01; // f16 has lower precision

        // Test single element
        let result = dot_product_f16(&[2.5], &[4.0]);
        assert!((result - 10.0).abs() < tol);

        // Test with small values
        let a = [0.1, 0.2, 0.3];
        let b = [0.4, 0.5, 0.6];
        let expected = 0.1 * 0.4 + 0.2 * 0.5 + 0.3 * 0.6;
        assert!((dot_product_f16(&a, &b) - expected).abs() < tol);

        // Test with negative values
        let a = [-1.0, 2.0, -3.0];
        let b = [4.0, -5.0, 6.0];
        let expected = -4.0 + 2.0 * (-5.0) + (-3.0) * 6.0;
        assert!((dot_product_f16(&a, &b) - expected).abs() < tol);

        // Test zeros
        assert_eq!(dot_product_f16(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]), 0.0);
        assert_eq!(dot_product_f16(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]), 0.0);

        // Test orthogonal vectors
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!((dot_product_f16(&a, &b) - 0.0).abs() < tol);

        // Test power of 2 sizes for vectorized implementations
        let a8 = vec![1.5f16; 8];
        let b8 = vec![2.0f16; 8];
        assert!((dot_product_f16(&a8, &b8) - 24.0).abs() < tol);

        let a16 = vec![0.5f16; 16];
        let b16 = vec![3.0f16; 16];
        assert!((dot_product_f16(&a16, &b16) - 24.0).abs() < tol);
    }

    // Tests for dot_product_f32
    #[test]
    fn test_dot_product_f32() {
        assert_eq!(dot_product_f32(&[1.5, 2.5], &[3.0, 4.0]), 14.5);
        assert_eq!(dot_product_f32(&[-1.5, 2.5], &[3.0, -4.0]), -14.5);
        assert_eq!(dot_product_f32(&[0.0, 0.0], &[1.5, 2.5]), 0.0);
        assert_eq!(dot_product_f32(&[], &[]), 0.0);
    }

    #[test]
    fn test_dot_product_f32_extended() {
        let tol = 1e-6;

        // Test single element
        assert_eq!(dot_product_f32(&[2.5], &[4.0]), 10.0);

        // Test high precision
        let a = [0.123456789, 0.987654321];
        let b = [0.111111111, 0.222222222];
        let expected = 0.123456789 * 0.111111111 + 0.987654321 * 0.222222222;
        assert!((dot_product_f32(&a, &b) - expected).abs() < tol);

        // Test very small numbers
        let a = [1e-10, 2e-10];
        let b = [3e-10, 4e-10];
        let expected = 1e-10 * 3e-10 + 2e-10 * 4e-10;
        assert!((dot_product_f32(&a, &b) - expected).abs() < 1e-19);

        // Test large numbers
        let a = [1e6, 2e6];
        let b = [3e6, 4e6];
        let expected = 1e6 * 3e6 + 2e6 * 4e6;
        assert_eq!(dot_product_f32(&a, &b), expected);

        // Test mixed signs and magnitudes
        let a = [-1000.0, 0.001, 42.0];
        let b = [0.001, -1000.0, -1.0];
        let expected = -1000.0 * 0.001 + 0.001 * (-1000.0) - 42.0;
        assert!((dot_product_f32(&a, &b) - expected).abs() < tol);

        // Test special values
        assert!(dot_product_f32(&[f32::INFINITY], &[1.0]).is_infinite());
        assert!(dot_product_f32(&[f32::NAN], &[1.0]).is_nan());

        // Test power of 2 sizes for vectorized implementations
        let a4 = vec![1.25f32; 4];
        let b4 = vec![2.0f32; 4];
        assert_eq!(dot_product_f32(&a4, &b4), 10.0);

        let a8 = vec![1.5f32; 8];
        let b8 = vec![2.0f32; 8];
        assert_eq!(dot_product_f32(&a8, &b8), 24.0);

        let a16 = vec![0.5f32; 16];
        let b16 = vec![3.0f32; 16];
        assert_eq!(dot_product_f32(&a16, &b16), 24.0);

        // Test non-power-of-2 sizes to check remainder handling
        let a5 = vec![2.0f32; 5];
        let b5 = vec![3.0f32; 5];
        assert_eq!(dot_product_f32(&a5, &b5), 30.0);

        let a13 = vec![1.0f32; 13];
        let b13 = vec![4.0f32; 13];
        assert_eq!(dot_product_f32(&a13, &b13), 52.0);
    }

    // Tests for magnitude functions
    #[test]
    fn test_magnitude_u8() {
        assert_eq!(magnitude_u8(&[3, 4]), 5.0);
        assert_eq!(magnitude_u8(&[]), 0.0);
        assert_eq!(magnitude_u8(&[0, 0, 0]), 0.0);
        assert_eq!(magnitude_u8(&[1]), 1.0);

        // Test with larger values
        let v = vec![5u8; 16];
        let expected = (5.0_f32 * 5.0_f32 * 16.0_f32).sqrt();
        assert_eq!(magnitude_u8(&v), expected);
    }

    #[test]
    fn test_magnitude_i8() {
        assert_eq!(magnitude_i8(&[3, 4]), 5.0);
        assert_eq!(magnitude_i8(&[-3, 4]), 5.0);
        assert_eq!(magnitude_i8(&[]), 0.0);
        assert_eq!(magnitude_i8(&[0, 0, 0]), 0.0);
        assert_eq!(magnitude_i8(&[-1]), 1.0);

        // Test with extreme values
        assert_eq!(magnitude_i8(&[127]), 127.0);
        assert_eq!(magnitude_i8(&[-128]), 128.0);

        // Test with mixed signs
        let v = vec![1, -1, 1, -1];
        let expected = (4.0_f32).sqrt();
        assert_eq!(magnitude_i8(&v), expected);
    }

    #[test]
    fn test_magnitude_f16() {
        let tol = 0.01;

        let result = magnitude_f16(&[3.0, 4.0]);
        assert!((result - 5.0).abs() < tol);

        assert_eq!(magnitude_f16(&[]), 0.0);
        assert_eq!(magnitude_f16(&[0.0, 0.0, 0.0]), 0.0);

        let result = magnitude_f16(&[1.0]);
        assert!((result - 1.0).abs() < tol);

        // Test with negative values
        let result = magnitude_f16(&[-3.0, 4.0]);
        assert!((result - 5.0).abs() < tol);
    }

    #[test]
    fn test_magnitude_f32() {
        let tol = 1e-6;

        assert_eq!(magnitude_f32(&[3.0, 4.0]), 5.0);
        assert_eq!(magnitude_f32(&[]), 0.0);
        assert_eq!(magnitude_f32(&[0.0, 0.0, 0.0]), 0.0);
        assert_eq!(magnitude_f32(&[1.0]), 1.0);

        // Test with negative values
        assert_eq!(magnitude_f32(&[-3.0, 4.0]), 5.0);

        // Test with high precision
        let v = [0.1, 0.2, 0.3];
        let expected = (0.01 + 0.04 + 0.09_f32).sqrt();
        assert!((magnitude_f32(&v) - expected).abs() < tol);
    }

    // Tests for dot product properties and cross-validation
    #[test]
    fn test_dot_product_properties() {
        // Test commutativity: a·b = b·a
        let a_u8 = vec![1, 2, 3, 4];
        let b_u8 = vec![5, 6, 7, 8];
        assert_eq!(dot_product_u8(&a_u8, &b_u8), dot_product_u8(&b_u8, &a_u8));

        let a_i8 = vec![1, -2, 3, -4];
        let b_i8 = vec![-5, 6, -7, 8];
        assert_eq!(dot_product_i8(&a_i8, &b_i8), dot_product_i8(&b_i8, &a_i8));

        let a_f32 = vec![1.5, -2.5, 3.5, -4.5];
        let b_f32 = vec![-5.5, 6.5, -7.5, 8.5];
        assert_eq!(
            dot_product_f32(&a_f32, &b_f32),
            dot_product_f32(&b_f32, &a_f32)
        );

        // Test distributivity: a·(b+c) = a·b + a·c (approximated for integer types)
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = vec![7.0, 8.0, 9.0];
        let bc_sum: Vec<f32> = b.iter().zip(c.iter()).map(|(x, y)| x + y).collect();
        let left = dot_product_f32(&a, &bc_sum);
        let right = dot_product_f32(&a, &b) + dot_product_f32(&a, &c);
        assert!((left - right).abs() < 1e-6);

        // Test scalar multiplication: (ka)·b = k(a·b)
        let k = 2.5;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let ka: Vec<f32> = a.iter().map(|x| k * x).collect();
        let left = dot_product_f32(&ka, &b);
        let right = k * dot_product_f32(&a, &b);
        assert!((left - right).abs() < 1e-6);
    }

    // Test consistency between different vector sizes (to test vectorized vs scalar paths)
    #[test]
    fn test_vectorized_consistency() {
        // Test that results are consistent across different vector sizes
        // This helps verify that vectorized implementations produce the same results as scalar

        // For u8
        let base_u8 = [1, 2, 3, 4];
        let pattern_u8 = [5, 6, 7, 8];

        // Test various sizes to hit different code paths
        for size in [1, 2, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
            let a: Vec<u8> = base_u8.iter().cycle().take(size).cloned().collect();
            let b: Vec<u8> = pattern_u8.iter().cycle().take(size).cloned().collect();

            let result = dot_product_u8(&a, &b);
            let expected: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (*x as u32 * *y as u32) as f32)
                .sum();
            assert_eq!(result, expected, "u8 size {} failed", size);
        }

        // For i8
        let base_i8 = [1, -2, 3, -4];
        let pattern_i8 = [-5, 6, -7, 8];

        for size in [1, 2, 4, 8, 15, 16, 17, 31, 32, 33] {
            let a: Vec<i8> = base_i8.iter().cycle().take(size).cloned().collect();
            let b: Vec<i8> = pattern_i8.iter().cycle().take(size).cloned().collect();

            let result = dot_product_i8(&a, &b);
            let expected: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (*x as i32 * *y as i32) as f32)
                .sum();
            assert_eq!(result, expected, "i8 size {} failed", size);
        }

        // For f32
        let base_f32 = [1.5, -2.5, 3.5, -4.5];
        let pattern_f32 = [-5.5, 6.5, -7.5, 8.5];

        for size in [1, 2, 4, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<f32> = base_f32.iter().cycle().take(size).cloned().collect();
            let b: Vec<f32> = pattern_f32.iter().cycle().take(size).cloned().collect();

            let result = dot_product_f32(&a, &b);
            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            assert!(
                (result - expected).abs() < 1e-6,
                "f32 size {} failed: {} vs {}",
                size,
                result,
                expected
            );
        }
    }

    // Tests for cosine_similarity_from_dot_product
    #[test]
    fn test_cosine_similarity_from_dot_product() {
        assert_eq!(cosine_similarity_from_dot_product(10.0, 5.0, 2.0), 1.0);
        assert_eq!(cosine_similarity_from_dot_product(10.0, 0.0, 5.0), 0.0);
        assert_eq!(cosine_similarity_from_dot_product(10.0, 5.0, 0.0), 0.0);
        assert_eq!(cosine_similarity_from_dot_product(10.0, 0.0, 0.0), 0.0);
        assert_eq!(cosine_similarity_from_dot_product(10.0, -5.0, -2.0), 1.0);
        assert_eq!(cosine_similarity_from_dot_product(0.0, 5.0, 2.0), 0.0);
    }

    #[test]
    fn test_cosine_similarity_extended() {
        let tol = 1e-6;

        // Test perfect similarity (parallel vectors)
        assert_eq!(cosine_similarity_from_dot_product(25.0, 5.0, 5.0), 1.0);

        // Test perfect dissimilarity (anti-parallel vectors)
        assert_eq!(cosine_similarity_from_dot_product(-25.0, 5.0, 5.0), -1.0);

        // Test orthogonal vectors
        assert_eq!(cosine_similarity_from_dot_product(0.0, 5.0, 3.0), 0.0);

        // Test general case
        let dot_prod = 14.0;
        let mag_a = 5.0;
        let mag_b = 3.0;
        let expected = dot_prod / (mag_a * mag_b);
        let result = cosine_similarity_from_dot_product(dot_prod, mag_a, mag_b);
        assert!((result - expected).abs() < tol);

        // Test with negative magnitudes (should still work due to multiplication)
        assert_eq!(cosine_similarity_from_dot_product(-10.0, -5.0, 2.0), 1.0);
        assert_eq!(cosine_similarity_from_dot_product(-10.0, 5.0, -2.0), 1.0);
    }

    // Debug assertion tests (length mismatches)
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_mismatched_lengths_u8() {
        dot_product_u8(&[1, 2], &[1]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_mismatched_lengths_i8() {
        dot_product_i8(&[1, 2], &[1]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_mismatched_lengths_f16() {
        dot_product_f16(&[1.0], &[]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_mismatched_lengths_f32() {
        dot_product_f32(&[1.0, 2.0], &[1.0]);
    }
}
