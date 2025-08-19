use alloc::vec::Vec;

// heap only
pub trait MemProject {
    type State;

    fn mem_project(state: Self::State) -> u64;
}

impl<T: MemProject> MemProject for Vec<T> {
    type State = (T::State, usize);

    fn mem_project((el_state, len): Self::State) -> u64 {
        let capacity = projected_capacity(len);

        (len as u64 * T::mem_project(el_state)) + capacity as u64 * size_of::<T>() as u64
    }
}

fn projected_capacity(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let mut v = len - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    if usize::BITS > 32 {
        v |= v >> 32;
    }
    v + 1
}
