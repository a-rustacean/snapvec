#![no_std]
#![feature(ptr_metadata, f16, new_zeroed_alloc, maybe_uninit_fill, allocator_api)]
#![cfg_attr(target_os = "macos", feature(stdarch_neon_f16))]

use core::num::NonZeroU32;

extern crate alloc;

mod arena;
mod backend;
mod fixedset;
mod graph;
mod handle;
mod level_sampler;
mod metric;
mod node;
mod storage;

#[derive(Debug, Clone, Copy)]
pub struct NodeId(pub NonZeroU32);

pub use backend::*;
pub use graph::{Graph, Options};
pub use metric::DistanceMetricKind;
pub use storage::Quantization;
