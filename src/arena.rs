use core::{
    alloc::Layout,
    marker::PhantomData,
    mem,
    ops::{Index, IndexMut},
    ptr::{self, NonNull, Pointee},
};

use alloc::vec::Vec;

use crate::{backend::ArenaBackend, handle::Handle};

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm64ec",
    target_arch = "powerpc64",
))]
const CACHE_LINE_SIZE: usize = 128;
// arm, mips, mips64, sparc, and hexagon have 32-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_arm.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mips.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mipsle.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mips64x.go#L9
// - https://github.com/torvalds/linux/blob/3516bd729358a2a9b090c1905bd2a3fa926e24c6/arch/sparc/include/asm/cache.h#L17
// - https://github.com/torvalds/linux/blob/3516bd729358a2a9b090c1905bd2a3fa926e24c6/arch/hexagon/include/asm/cache.h#L12
#[cfg(any(
    target_arch = "arm",
    target_arch = "mips",
    target_arch = "mips32r6",
    target_arch = "mips64",
    target_arch = "mips64r6",
    target_arch = "sparc",
    target_arch = "hexagon",
))]
const CACHE_LINE_SIZE: usize = 32;
// m68k has 16-byte cache line size.
//
// Sources:
// - https://github.com/torvalds/linux/blob/3516bd729358a2a9b090c1905bd2a3fa926e24c6/arch/m68k/include/asm/cache.h#L9
#[cfg(target_arch = "m68k")]
const CACHE_LINE_SIZE: usize = 16;
// s390x has 256-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_s390x.go#L7
// - https://github.com/torvalds/linux/blob/3516bd729358a2a9b090c1905bd2a3fa926e24c6/arch/s390/include/asm/cache.h#L13
#[cfg(target_arch = "s390x")]
const CACHE_LINE_SIZE: usize = 256;
// x86, wasm, riscv, and sparc64 have 64-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/dda2991c2ea0c5914714469c4defc2562a907230/src/internal/cpu/cpu_x86.go#L9
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_wasm.go#L7
// - https://github.com/torvalds/linux/blob/3516bd729358a2a9b090c1905bd2a3fa926e24c6/arch/riscv/include/asm/cache.h#L10
// - https://github.com/torvalds/linux/blob/3516bd729358a2a9b090c1905bd2a3fa926e24c6/arch/sparc/include/asm/cache.h#L19
//
// All others are assumed to have 64-byte cache line size.
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm64ec",
    target_arch = "powerpc64",
    target_arch = "arm",
    target_arch = "mips",
    target_arch = "mips32r6",
    target_arch = "mips64",
    target_arch = "mips64r6",
    target_arch = "sparc",
    target_arch = "hexagon",
    target_arch = "m68k",
    target_arch = "s390x",
)))]
const CACHE_LINE_SIZE: usize = 64;

struct Chunk<T: DynAlloc + ?Sized> {
    ptr: NonNull<u8>,
    _marker: PhantomData<T>,
}

impl<T: DynAlloc + ?Sized> Chunk<T> {
    unsafe fn new<B: ArenaBackend>(
        backend: &mut B,
        chunk_id: u32,
        item_size: usize,
        item_align: usize,
        chunk_len: usize,
    ) -> Self {
        let layout = unsafe {
            Layout::from_size_align_unchecked(
                item_size * chunk_len,
                item_align.max(CACHE_LINE_SIZE),
            )
        };
        let ptr = unsafe { backend.alloc(chunk_id, layout) };

        unsafe {
            ptr.write_bytes(0, item_size * chunk_len);
        }

        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    unsafe fn get_raw(&self, item_size: usize, index: usize) -> *mut u8 {
        unsafe { self.ptr.as_ptr().add(item_size * index) }
    }

    unsafe fn get_ref(
        &self,
        item_size: usize,
        index: usize,
        metadata: <T as Pointee>::Metadata,
    ) -> &T {
        unsafe { &*ptr::from_raw_parts(self.ptr.as_ptr().add(item_size * index), metadata) }
    }

    unsafe fn get_mut(
        &mut self,
        item_size: usize,
        index: usize,
        metadata: <T as Pointee>::Metadata,
    ) -> &mut T {
        unsafe {
            &mut *(ptr::from_raw_parts_mut(self.ptr.as_ptr().add(item_size * index), metadata))
        }
    }

    unsafe fn init(&self, item_size: usize, index: usize, metadata: T::Metadata, args: T::Args) {
        unsafe {
            T::new_at(self.get_raw(item_size, index), metadata, args);
        }
    }
}

unsafe impl<T: Send + DynAlloc + ?Sized> Send for Chunk<T> {}
unsafe impl<T: Sync + DynAlloc + ?Sized> Sync for Chunk<T> {}

fn align_up(size: usize, alignment: usize) -> usize {
    debug_assert!(alignment != 0, "Alignment must be non-zero");
    debug_assert!(
        alignment.is_power_of_two(),
        "Alignment must be a power of two"
    );

    let mask = alignment - 1;
    if size == 0 { 0 } else { (size + mask) & !mask }
}

pub trait DynAlloc {
    type Metadata: Clone + Copy; // Shared object metadata
    type Args; // Per-object initialization data

    const ALIGN: usize; // Required object alignment

    // Required methods
    fn size(metadata: Self::Metadata) -> usize;
    fn ptr_metadata(metadata: Self::Metadata) -> <Self as Pointee>::Metadata;
    unsafe fn new_at(ptr: *mut u8, metadata: Self::Metadata, args: Self::Args);

    // Provided method
    #[inline(always)]
    fn size_aligned(metadata: Self::Metadata) -> usize {
        let size = Self::size(metadata);
        align_up(size, Self::ALIGN)
    }
}

pub struct Arena<T: DynAlloc + ?Sized, B> {
    chunks: Vec<Chunk<T>>,
    chunk_len: usize,
    metadata: T::Metadata,
    backend: B,
}

impl<T: DynAlloc + ?Sized, B: ArenaBackend> Arena<T, B> {
    pub fn new(chunk_size: usize, metadata: T::Metadata, backend: B) -> Self {
        Self {
            chunks: Vec::new(),
            chunk_len: chunk_size,
            metadata,
            backend,
        }
    }

    pub fn alloc(&mut self, index: u32, args: T::Args) -> Handle<T> {
        let chunk_index = index as usize / self.chunk_len;
        let offset = index as usize % self.chunk_len;

        if chunk_index >= self.chunks.len() {
            while chunk_index >= self.chunks.len() {
                self.chunks.push(unsafe {
                    Chunk::new(
                        &mut self.backend,
                        self.chunks.len() as u32,
                        T::size_aligned(self.metadata),
                        T::ALIGN,
                        self.chunk_len,
                    )
                });
            }
        }

        let chunk = &self.chunks[chunk_index];
        unsafe {
            chunk.init(T::size_aligned(self.metadata), offset, self.metadata, args);
        }

        Handle::new(index)
    }

    #[inline(always)]
    fn split_handle(&self, handle: Handle<T>) -> (usize, usize) {
        let index = *handle as usize;
        (index / self.chunk_len, index % self.chunk_len)
    }

    pub fn clear(&mut self, len: u32) {
        let chunks = mem::take(&mut self.chunks); // Take ownership of the chunks

        let len = len as usize;

        if len == 0 {
            return; // No objects allocated
        }

        if chunks.is_empty() {
            return;
        }

        let item_size = T::size_aligned(self.metadata);
        let item_align = T::ALIGN;

        // Drop each allocated object in reverse order (from last to first)
        for i in (0..len).rev() {
            let chunk_index = i / self.chunk_len;
            let offset = i % self.chunk_len;
            let chunk = &chunks[chunk_index];
            let ptr = unsafe { chunk.get_raw(item_size, offset) };
            let ptr_to_t: *mut T =
                ptr::from_raw_parts_mut(ptr as *mut (), T::ptr_metadata(self.metadata));
            unsafe {
                ptr::drop_in_place(ptr_to_t);
            }
        }

        // Deallocate each chunk
        for chunk in chunks {
            let layout = Layout::from_size_align(item_size * self.chunk_len, item_align)
                .expect("Invalid layout");
            unsafe {
                alloc::alloc::dealloc(chunk.ptr.as_ptr(), layout);
            }
        }
    }

    pub fn flush(&mut self) {
        self.backend.flush();
    }
}

impl<T: DynAlloc + ?Sized, B: ArenaBackend> Index<Handle<T>> for Arena<T, B> {
    type Output = T;

    fn index(&self, handle: Handle<T>) -> &Self::Output {
        let (chunk_index, offset) = self.split_handle(handle);
        let chunk = &self.chunks[chunk_index];
        unsafe {
            chunk.get_ref(
                T::size_aligned(self.metadata),
                offset,
                T::ptr_metadata(self.metadata),
            )
        }
    }
}

impl<T: DynAlloc + ?Sized, B: ArenaBackend> IndexMut<Handle<T>> for Arena<T, B> {
    fn index_mut(&mut self, handle: Handle<T>) -> &mut Self::Output {
        let (chunk_index, offset) = self.split_handle(handle);
        let chunk = &mut self.chunks[chunk_index];
        unsafe {
            chunk.get_mut(
                T::size_aligned(self.metadata),
                offset,
                T::ptr_metadata(self.metadata),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::ArenaBackendMemory;

    use super::*;
    use core::ptr;
    use core::sync::atomic::{AtomicUsize, Ordering};

    // Simple test struct
    #[derive(Debug, PartialEq, Eq)]
    struct TestStruct {
        value: u32,
    }

    impl Default for TestStruct {
        fn default() -> Self {
            TestStruct { value: 42 }
        }
    }

    impl DynAlloc for TestStruct {
        type Metadata = ();
        type Args = u32;

        const ALIGN: usize = align_of::<Self>();

        fn size(_metadata: Self::Metadata) -> usize {
            size_of::<Self>()
        }

        fn ptr_metadata(_metadata: Self::Metadata) -> <Self as Pointee>::Metadata {}

        unsafe fn new_at(ptr: *mut u8, _metadata: (), args: Self::Args) {
            unsafe {
                ptr::write(ptr as *mut Self, Self { value: args });
            }
        }
    }

    // Struct for drop testing
    #[allow(unused)]
    struct DropTest {
        id: u32,
    }

    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    impl DropTest {
        fn new(id: u32) -> Self {
            DropTest { id }
        }
    }

    impl Drop for DropTest {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Default for DropTest {
        fn default() -> Self {
            Self::new(0)
        }
    }

    impl DynAlloc for DropTest {
        type Metadata = ();
        type Args = u32;

        const ALIGN: usize = align_of::<Self>();

        fn size(_metadata: Self::Metadata) -> usize {
            size_of::<Self>()
        }

        fn ptr_metadata(_metadata: Self::Metadata) -> <Self as Pointee>::Metadata {}

        unsafe fn new_at(ptr: *mut u8, _metadata: (), args: Self::Args) {
            unsafe {
                ptr::write(ptr as *mut Self, Self::new(args));
            }
        }
    }

    #[test]
    fn basic_allocation() {
        let mut arena =
            Arena::<TestStruct, ArenaBackendMemory>::new(2, (), ArenaBackendMemory::default());
        let handle1 = arena.alloc(0, 10);
        let handle2 = arena.alloc(1, 20);

        assert_eq!(arena[handle1].value, 10);
        assert_eq!(arena[handle2].value, 20);
    }

    #[test]
    fn chunk_expansion() {
        let mut arena =
            Arena::<TestStruct, ArenaBackendMemory>::new(1, (), ArenaBackendMemory::default()); // Small chunk size
        let handle1 = arena.alloc(0, 1);
        let handle2 = arena.alloc(1, 2); // Should trigger new chunk

        assert_eq!(arena[handle1].value, 1);
        assert_eq!(arena[handle2].value, 2);
    }

    #[test]
    fn clear_operation_and_drop_arena() {
        let mut arena =
            Arena::<DropTest, ArenaBackendMemory>::new(2, (), ArenaBackendMemory::default());
        let _ = arena.alloc(0, 1);
        let _ = arena.alloc(1, 2);

        DROP_COUNT.store(0, Ordering::SeqCst);
        arena.clear(2);

        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 2);

        DROP_COUNT.store(0, Ordering::SeqCst);
        {
            let mut arena =
                Arena::<DropTest, ArenaBackendMemory>::new(2, (), ArenaBackendMemory::default());
            let _ = arena.alloc(0, 1);
            let _ = arena.alloc(1, 2);
        } // Arena dropped here, allocated objects still stay in memory, need to manually clear the arena before dropping

        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn large_allocation() {
        let mut arena =
            Arena::<TestStruct, ArenaBackendMemory>::new(100, (), ArenaBackendMemory::default());
        for i in 0..1000 {
            let handle = arena.alloc(i, i);
            assert_eq!(arena[handle].value, i);
        }
    }
}
