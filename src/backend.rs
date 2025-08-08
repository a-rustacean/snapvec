use core::{alloc::Layout, ptr::NonNull};

use alloc::alloc::{Allocator, Global, handle_alloc_error};

use crate::Options;

pub trait ArenaBackend {
    /// # Safety
    ///
    /// The implementation must ensure:
    /// - Returned memory is aligned to `layout.align()`
    /// - The pointer points to valid memory of at least `layout.size()` bytes
    /// - The memory remains valid until either:
    ///   - Arena reset/clear (if supported)
    ///   - Arena destruction
    ///   - Explicit deallocation via [`ArenaBackend::dealloc`] (if supported)
    ///
    /// Implementations must document:
    /// - The lifetime/validity guarantees of allocated memory
    /// - Required conditions before memory may be reused
    unsafe fn alloc(&mut self, layout: Layout) -> NonNull<u8>;

    /// # Safety
    ///
    /// The implementation must define and enforce:
    /// - Valid combinations of `ptr` and `layout`
    /// - Required ownership/borrowing conditions
    /// - Synchronization requirements
    ///
    /// Typical conditions include:
    /// - `ptr` was allocated by this allocator
    /// - `layout` matches the allocation layout
    /// - No live references to the memory exist
    /// - Deallocation occurs at proper arena lifecycle phase
    ///
    /// Implementations may choose to:
    /// - Support individual deallocation
    /// - Only allow bulk deallocation (arena reset)
    /// - Make this a no-op
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout);

    fn flush(&mut self);
}

pub struct ArenaBackendMemory<A: Allocator = Global> {
    alloc: A,
}

impl<A: Allocator> ArenaBackend for ArenaBackendMemory<A> {
    /// Allocates memory using the underlying allocator
    ///
    /// # Safety
    /// Satisfies ArenaBackend's safety requirements by:
    /// 1. Returning properly aligned memory (guaranteed by `Allocator`)
    /// 2. Providing memory valid for `layout.size()` bytes
    /// 3. Guaranteeing validity until either:
    ///    - Explicit [`ArenaBackendMemory::dealloc`] call OR
    ///    - Arena destruction
    /// 4. Preserving allocator's defined memory lifetime semantics
    unsafe fn alloc(&mut self, layout: Layout) -> NonNull<u8> {
        let ptr = match self.alloc.allocate(layout) {
            Ok(ptr) => ptr,
            Err(_) => handle_alloc_error(layout),
        };
        ptr.cast()
    }

    /// Deallocates memory using the underlying allocator
    ///
    /// # Safety
    /// Requires:
    /// 1. `ptr` was allocated by this allocator instance
    /// 2. `layout` matches the allocation's original layout
    /// 3. No live references exist to the memory region
    /// 4. Not called more than once for same pointer
    ///
    /// Satisfies ArenaBackend's safety contract by:
    /// - Releasing memory to allocator according to standard safety rules
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        unsafe {
            self.alloc.deallocate(ptr, layout);
        }
    }

    /// No-op for standard allocators
    ///
    /// Arena reset requires full destruction/reconstruction since
    /// individual deallocations are handled immediately
    fn flush(&mut self) {}
}

impl Default for ArenaBackendMemory {
    fn default() -> Self {
        Self { alloc: Global }
    }
}

pub trait GraphOptionsBackend {
    fn save_options(&mut self, options: &Options);
    fn load_options(&self) -> Options;

    fn save_id_counter(&mut self, counter: u32);
    fn load_id_counter(&self) -> u32;

    fn flush(&mut self);
}

#[derive(Default)]
pub struct GraphOptionsBackendMemory {
    options: Options,
    id_counter: u32,
}

impl GraphOptionsBackend for GraphOptionsBackendMemory {
    fn save_options(&mut self, options: &Options) {
        self.options = options.clone();
    }

    fn load_options(&self) -> Options {
        self.options.clone()
    }

    fn save_id_counter(&mut self, counter: u32) {
        self.id_counter = counter;
    }

    fn load_id_counter(&self) -> u32 {
        self.id_counter
    }

    fn flush(&mut self) {}
}

pub struct GraphBackend<A: ArenaBackend, G: GraphOptionsBackend> {
    pub raw_vec: A,
    pub quant_vec: A,
    pub nodes0: A,
    pub nodes1: A,
    pub options: G,
}

impl Default for GraphBackend<ArenaBackendMemory, GraphOptionsBackendMemory> {
    fn default() -> Self {
        Self {
            raw_vec: ArenaBackendMemory::default(),
            quant_vec: ArenaBackendMemory::default(),
            nodes0: ArenaBackendMemory::default(),
            nodes1: ArenaBackendMemory::default(),
            options: GraphOptionsBackendMemory::default(),
        }
    }
}
