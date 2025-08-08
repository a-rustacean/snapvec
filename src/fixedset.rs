use alloc::boxed::Box;

pub const fn next_pow2_u16(mut x: u16) -> usize {
    if x == 0 {
        return 1;
    }

    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    (x + 1) as usize
}

pub struct FixedSet {
    buckets: Box<[u64]>,
}

impl FixedSet {
    #[inline]
    pub fn new(len: u16) -> Self {
        Self {
            buckets: unsafe { Box::new_zeroed_slice(next_pow2_u16(len)).assume_init() },
        }
    }

    #[inline]
    pub fn insert(&mut self, value: u32) {
        let mask = (self.buckets.len() - 1) as u32;
        let bucket = (value >> 6) & mask;
        let bit_pos = value & 0x3f;
        self.buckets[bucket as usize] |= 1u64 << bit_pos;
    }

    #[inline]
    pub fn is_member(&self, value: u32) -> bool {
        let mask = (self.buckets.len() - 1) as u32;
        let bucket = (value >> 6) & mask;
        let bit_pos = value & 0x3f;
        (self.buckets[bucket as usize] & (1u64 << bit_pos)) != 0
    }
}
