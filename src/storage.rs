use core::{
    ops::Range,
    ptr::{self, Pointee},
    slice,
};

use crate::{
    arena::DynAlloc,
    metric::{magnitude_f16, magnitude_f32, magnitude_i8, magnitude_u8},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Quantization {
    #[default]
    I8,
    U8,
    F16,
    F32,
}

impl Quantization {
    #[inline]
    pub(crate) fn size(&self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[repr(C, align(4))]
#[derive(Debug)]
pub struct QuantVec {
    pub(crate) mag: f32,
    vec: [u8],
}

#[repr(C, align(4))]
pub struct RawVec {
    pub(crate) vec: [f32],
}

impl DynAlloc for QuantVec {
    type Metadata = (Quantization, u16);
    type Args = (*const f32, Range<f32>);

    const ALIGN: usize = 4;

    #[inline]
    fn size((quantization, len): Self::Metadata) -> usize {
        let multiplier = quantization.size();
        4 + len as usize * multiplier
    }

    #[inline]
    fn ptr_metadata((quantization, len): Self::Metadata) -> <Self as Pointee>::Metadata {
        let multiplier = quantization.size();
        len as usize * multiplier
    }

    unsafe fn new_at(
        ptr: *mut u8,
        (quantization, len): Self::Metadata,
        (raw_vec_ptr, range): Self::Args,
    ) {
        let raw_vec_ref: &[f32] = unsafe { &*ptr::from_raw_parts(raw_vec_ptr, len as usize) };
        let vec_ptr = unsafe { ptr.add(4) };

        let mag = match quantization {
            Quantization::I8 => {
                let vec_ptr = vec_ptr as *mut i8;
                for (i, dim) in raw_vec_ref.iter().enumerate() {
                    unsafe {
                        vec_ptr.add(i).write(
                            ((((dim - range.start) / (range.end - range.start)) * 255.0) - 128.0)
                                .clamp(-128.0, 127.0) as i8,
                        );
                    }
                }
                let vec_slice = unsafe { slice::from_raw_parts(vec_ptr, raw_vec_ref.len()) };
                magnitude_i8(vec_slice)
            }
            Quantization::U8 => {
                for (i, dim) in raw_vec_ref.iter().enumerate() {
                    unsafe {
                        vec_ptr.add(i).write(
                            (((dim - range.start) / (range.end - range.start)) * 255.0)
                                .clamp(0.0, 255.0) as u8,
                        );
                    }
                }
                let vec_slice = unsafe { slice::from_raw_parts(vec_ptr, raw_vec_ref.len()) };
                magnitude_u8(vec_slice)
            }
            Quantization::F16 => {
                let vec_ptr = vec_ptr as *mut f16;
                for (i, dim) in raw_vec_ref.iter().enumerate() {
                    unsafe {
                        vec_ptr.add(i).write(*dim as f16);
                    }
                }
                let vec_slice = unsafe { slice::from_raw_parts(vec_ptr, raw_vec_ref.len()) };
                magnitude_f16(vec_slice)
            }
            Quantization::F32 => {
                let vec_ptr = vec_ptr as *mut f32;
                unsafe {
                    ptr::copy_nonoverlapping(raw_vec_ptr, vec_ptr, len as usize);
                }
                let vec_slice = unsafe { slice::from_raw_parts(vec_ptr, raw_vec_ref.len()) };
                magnitude_f32(vec_slice)
            }
        };

        unsafe {
            (ptr as *mut f32).write(mag);
        }
    }
}

impl DynAlloc for RawVec {
    type Metadata = u16;
    type Args = *const f32;

    const ALIGN: usize = 4;

    #[inline]
    fn size(len: Self::Metadata) -> usize {
        4 * len as usize
    }

    #[inline]
    fn ptr_metadata(len: Self::Metadata) -> <Self as Pointee>::Metadata {
        len as usize
    }

    unsafe fn new_at(ptr: *mut u8, metadata: Self::Metadata, args: Self::Args) {
        unsafe {
            ptr::copy_nonoverlapping(args, ptr as *mut f32, metadata as usize);
        }
    }
}

impl QuantVec {
    pub fn as_i8_slice(&self) -> &[i8] {
        unsafe { slice::from_raw_parts(self.vec.as_ptr() as *mut i8, self.vec.len()) }
    }

    pub fn as_u8_slice(&self) -> &[u8] {
        &self.vec
    }

    pub fn as_f16_slice(&self) -> &[f16] {
        unsafe { slice::from_raw_parts(self.vec.as_ptr() as *mut f16, self.vec.len() / 2) }
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        unsafe { &*ptr::from_raw_parts(&self.vec as *const [u8] as *const f32, self.vec.len() / 4) }
    }
}
