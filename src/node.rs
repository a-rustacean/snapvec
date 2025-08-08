use core::{cmp::Ordering, mem, ptr::Pointee};

use crate::{
    arena::{Arena, DynAlloc},
    backend::ArenaBackend,
    handle::Handle,
    metric::DistanceMetric,
    storage::QuantVec,
};

pub type Node0Handle = Handle<Node0>;
pub type Node1Handle = Handle<Node1>;
pub type VecHandle = Handle<QuantVec>;

#[repr(C, align(4))]
pub struct Node0 {
    pub neighbors: Neighbors<Self>,
}

#[repr(C, align(4))]
pub struct Node1 {
    pub vec: VecHandle,
    pub neighbors: Neighbors<Self>,
}

pub trait Node: DynAlloc {
    fn neighbors(&mut self) -> &mut Neighbors<Self>;
}

impl Node for Node0 {
    fn neighbors(&mut self) -> &mut Neighbors<Self> {
        &mut self.neighbors
    }
}

impl Node for Node1 {
    fn neighbors(&mut self) -> &mut Neighbors<Self> {
        &mut self.neighbors
    }
}

#[repr(C, align(4))]
pub struct Neighbors<N: ?Sized> {
    pub(crate) neighbors_full: bool,
    pub(crate) lowest_index: u16,
    pub(crate) neighbors: [Neighbor<N>],
}

impl<N: ?Sized> Neighbors<N> {
    pub fn neighbors(&self) -> &[Neighbor<N>] {
        if self.neighbors_full {
            &self.neighbors
        } else {
            &self.neighbors[..(self.lowest_index as usize)]
        }
    }
}

impl<N: ?Sized + DynAlloc + Node> Neighbors<N> {
    pub fn insert_neighbor<B: ArenaBackend>(
        arena: &mut Arena<N, B>,
        self_handle: Handle<N>,
        distance_metric: &DistanceMetric,
        node: Handle<N>,
        score: f32,
    ) -> bool {
        let self_ = arena[self_handle].neighbors();
        if self_.neighbors_full {
            if distance_metric.cmp_score(score, self_.neighbors[self_.lowest_index as usize].score)
                == Ordering::Greater
            {
                let old_neighbor = mem::replace(
                    &mut self_.neighbors[self_.lowest_index as usize],
                    Neighbor { node, score },
                );
                self_.recompute_lowest_index(distance_metric);
                arena[old_neighbor.node]
                    .neighbors()
                    .remove_neighbor(self_handle);
                true
            } else {
                false
            }
        } else {
            self_.neighbors[self_.lowest_index as usize] = Neighbor { node, score };
            self_.lowest_index += 1;
            if self_.lowest_index as usize == self_.neighbors.len() {
                self_.neighbors_full = true;
                self_.recompute_lowest_index(distance_metric);
            }
            true
        }
    }

    fn recompute_lowest_index(&mut self, distance_metric: &DistanceMetric) {
        let mut lowest_index = 0;
        let mut lowest_score = distance_metric.max_value();

        for i in 0..(self.neighbors.len() as u16) {
            let neighbor = &self.neighbors[i as usize];
            if distance_metric.cmp_score(neighbor.score, lowest_score) == Ordering::Less {
                lowest_score = neighbor.score;
                lowest_index = i;
            }
        }

        self.lowest_index = lowest_index;
    }

    fn remove_neighbor(&mut self, handle: Handle<N>) {
        let len = if self.neighbors_full {
            self.neighbors.len()
        } else {
            self.lowest_index as usize
        };
        for i in 0..len {
            if self.neighbors[i].node == handle {
                self.neighbors[i] = self.neighbors[len - 1];
                self.lowest_index = len as u16 - 1;
                self.neighbors_full = false;
                break;
            }
        }
    }
}

#[repr(C, align(4))]
pub struct Neighbor<N: ?Sized> {
    pub node: Handle<N>,
    pub score: f32,
}

impl<N: ?Sized> Clone for Neighbor<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N: ?Sized> Copy for Neighbor<N> {}

impl DynAlloc for Node0 {
    type Metadata = u16;
    type Args = ();

    const ALIGN: usize = 4;

    fn size(m: u16) -> usize {
        4 + size_of::<Neighbor<Self>>() * m as usize
    }

    fn ptr_metadata(m: u16) -> <Self as Pointee>::Metadata {
        m as usize
    }

    unsafe fn new_at(_: *mut u8, _: u16, _: ()) {}
}

impl DynAlloc for Node1 {
    type Metadata = u16;
    type Args = VecHandle;

    const ALIGN: usize = 4;

    fn size(m: u16) -> usize {
        8 + size_of::<Neighbor<Self>>() * m as usize
    }

    fn ptr_metadata(m: u16) -> <Self as Pointee>::Metadata {
        m as usize
    }

    unsafe fn new_at(ptr: *mut u8, _: u16, vec: VecHandle) {
        unsafe { (ptr as *mut VecHandle).write(vec) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arena::Arena, backend::ArenaBackendMemory};

    #[test]
    fn test_node1_allocation() {
        let metadata: u16 = 5; // Number of neighbors
        let mut arena =
            Arena::<Node1, ArenaBackendMemory>::new(16, metadata, ArenaBackendMemory::default());
        let dummy_child_handle = Handle::new(u32::MAX);

        // Allocate a Node
        let node_handle = arena.alloc(0, dummy_child_handle);
        let node = &arena[node_handle];

        // Verify neighbors initialization
        let neighbors = &node.neighbors;
        assert!(!neighbors.neighbors_full);
        assert_eq!(neighbors.lowest_index, 0);
        assert_eq!(neighbors.neighbors.len(), metadata as usize);

        for neighbor in &neighbors.neighbors {
            assert_eq!(*neighbor.node, 0);
            assert_eq!(neighbor.score, 0.0);
        }
    }
}
