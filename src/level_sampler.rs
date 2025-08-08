pub fn sample(max_level: u8, index: u32) -> (u32, u8) {
    let level = if index == 0 {
        max_level
    } else {
        let tz = index.trailing_zeros();
        if tz > max_level as u32 {
            max_level
        } else {
            tz as u8
        }
    };
    let sum = if index == 0 {
        0
    } else {
        let m = index - 1;
        let total_tz = if m == 0 { 0 } else { m - m.count_ones() };
        let base_power = (max_level as u32) + 1;
        let mut deduct = 0;
        if base_power < 32 {
            for shift in base_power..32 {
                let a = m >> shift;
                let b = if shift < 31 { m >> (shift + 1) } else { 0 };
                let count = a - b;
                let k = shift - base_power + 1;
                deduct += k * count;
            }
        }
        (max_level as u32) + total_tz - deduct
    };
    (sum, level)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cases() {
        assert_eq!(sample(10, 0), (0, 10)); // index=0: sum=0, level=10
        assert_eq!(sample(10, 1), (10, 0)); // index=1: sum=10 (from 0), level=0
        assert_eq!(sample(10, 2), (10, 1)); // index=2: sum=10+0, level=1
        assert_eq!(sample(10, 3), (11, 0)); // index=3: sum=10+0+1, level=0
        assert_eq!(sample(10, 4), (11, 2)); // index=4: sum=10+0+1+0, level=2
    }

    #[test]
    fn test_small_max_level() {
        assert_eq!(sample(2, 0), (0, 2));
        assert_eq!(sample(2, 1), (2, 0));
        assert_eq!(sample(2, 2), (2, 1));
        assert_eq!(sample(2, 3), (3, 0));
        assert_eq!(sample(2, 4), (3, 2));
    }

    #[test]
    fn test_max_level_zero() {
        assert_eq!(sample(0, 0), (0, 0));
        assert_eq!(sample(0, 1), (0, 0));
        assert_eq!(sample(0, 2), (0, 0));
    }
}
