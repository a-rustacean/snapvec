use core::{alloc::Layout, mem, num::NonZeroU32, ops::Range, ptr};

use alloc::{
    alloc::{alloc, dealloc, handle_alloc_error},
    boxed::Box,
    vec::Vec,
};
use binary_heap_plus::BinaryHeap;

use crate::{
    NodeId,
    arena::{Arena, DynAlloc},
    backend::{
        ArenaBackend, ArenaBackendMemory, GraphBackend, GraphOptionsBackend,
        GraphOptionsBackendMemory,
    },
    fixedset::FixedSet,
    handle::Handle,
    level_sampler,
    mem_project::MemProject,
    metric::{DistanceMetric, DistanceMetricKind, magnitude_f32},
    node::{Neighbors, Node0, Node0Handle, Node1, Node1Handle, VecHandle},
    storage::{QuantVec, Quantization, RawVec},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub m: u16,
    pub m0: u16,
    pub dims: u16,
    pub max_level: u8,
    pub quantization: Quantization,
    pub metric: DistanceMetricKind,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            dims: 0,
            max_level: 5,
            quantization: Quantization::default(),
            metric: DistanceMetricKind::default(),
        }
    }
}

// 1. creates levels [0, max_level] (both inclusive).
// 2. for level 0 `Node0` is used, `Node1` or all above levels.
// 3. a `Handle<Node0>` can be used to index into raw vec arena, and quant vec arena, and vice-versa, the three sets are bijective.
// 4. the level distribution is calculated by the `level_sampler::sample` function, which uses derterministic logic.
// 5. to get a child of a node in the level [2, max_level], we simply have to subtract 1 from the handle, for level 1, the `vec` is the child handle (due to [3]).
pub struct Graph<A: ArenaBackend = ArenaBackendMemory, G = GraphOptionsBackendMemory> {
    options: Options,
    distance_metric: DistanceMetric,
    nodes0_arena: Arena<Node0, A>,
    nodes1_arena: Arena<Node1, A>,
    pub(crate) raw_vec_arena: Arena<RawVec, A>,
    quant_vec_arena: Arena<QuantVec, A>,
    top_level_root_node: Node1Handle,
    pub(crate) id_counter: u32,
    backend: G,
}

#[repr(C, align(4))]
#[derive(Debug)]
pub struct InternalSearchResult<T: ?Sized> {
    pub node: Handle<T>,
    pub score: f32,
}

impl<T: ?Sized> Clone for InternalSearchResult<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for InternalSearchResult<T> {}

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub node: NodeId,
    pub score: f32,
}

impl Graph<ArenaBackendMemory, GraphOptionsBackendMemory> {
    pub fn new(options: Options) -> Self {
        Self::new_with_backend(options, GraphBackend::default())
    }
}

impl<A: ArenaBackend, G: GraphOptionsBackend> Graph<A, G> {
    pub fn new_with_backend(options: Options, backend: GraphBackend<A, G>) -> Self {
        let mut nodes0_arena = Arena::new(1000, options.m0, backend.nodes0);
        let mut nodes1_arena = Arena::new(1000, options.m, backend.nodes1);
        let mut raw_vec_arena = Arena::new(1000, options.dims, backend.raw_vec);
        let mut quant_vec_arena = Arena::new(
            1000,
            (options.quantization, options.dims),
            backend.quant_vec,
        );

        let root_vec_raw: Box<[f32]> = unsafe {
            let mut vec = Box::new_uninit_slice(options.dims as usize);
            vec.write_filled(1.0);
            vec.assume_init()
        };

        raw_vec_arena.alloc(0, root_vec_raw.as_ptr());
        quant_vec_arena.alloc(0, (root_vec_raw.as_ptr(), -1.0..1.0));

        let node0_handle = nodes0_arena.alloc(0, ());

        let mut prev_node = node0_handle.cast();

        for i in 0..options.max_level {
            let node_handle = nodes1_arena.alloc(i as u32, Handle::new(0));
            prev_node = node_handle;
        }

        Self {
            distance_metric: DistanceMetric::new(options.metric, options.quantization),
            options,
            nodes1_arena,
            nodes0_arena,
            raw_vec_arena,
            quant_vec_arena,
            top_level_root_node: prev_node,
            id_counter: 1,
            backend: backend.options,
        }
    }

    pub fn insert(&mut self, vec: &[f32], ef: u16) -> NodeId {
        let id = self.id_counter;
        self.id_counter += 1;
        let vec_handle = self.raw_vec_arena.alloc(id, vec.as_ptr());

        self.index_node(vec_handle, ef, -1.0..1.0);

        NodeId(unsafe { NonZeroU32::new_unchecked(id) })
    }

    pub(crate) fn index_node(&mut self, vec_handle: Handle<RawVec>, ef: u16, range: Range<f32>) {
        self.quant_vec_arena.alloc(
            *vec_handle,
            (self.raw_vec_arena[vec_handle].vec.as_ptr(), range),
        );

        let (node_insert_start_index, max_level) =
            level_sampler::sample(self.options.max_level, *vec_handle);

        self.index_levels(
            vec_handle.cast(),
            node_insert_start_index + (max_level as u32) - 1,
            self.top_level_root_node,
            self.options.max_level,
            max_level,
            ef,
        );
    }

    fn index_levels(
        &mut self,
        vec_handle: VecHandle,
        mut node_insert_index: u32,
        mut entry_node: Node1Handle,
        mut current_level: u8,
        max_level: u8,
        ef: u16,
    ) {
        while current_level > max_level {
            let results = self.search_level1(
                entry_node,
                &self.quant_vec_arena[vec_handle.cast()],
                ef,
                1,
                true,
            );
            let child = if current_level == 1 {
                self.nodes1_arena[results[0].node].vec.cast()
            } else {
                Handle::new(*results[0].node - 1)
            };

            entry_node = child;
            current_level -= 1;
        }

        while current_level >= 1 {
            let results = self.search_level1(
                entry_node,
                &self.quant_vec_arena[vec_handle.cast()],
                ef,
                self.options.m * 2,
                true,
            );

            let child = if current_level == 1 {
                self.nodes1_arena[results[0].node].vec.cast()
            } else {
                Handle::new(*results[0].node - 1)
            };

            self.create_node1(vec_handle, node_insert_index, results);

            entry_node = child;
            current_level -= 1;
            node_insert_index -= 1;
        }

        self.index_level0(vec_handle, entry_node.cast(), ef);
    }

    fn index_level0(&mut self, vec_handle: VecHandle, entry_node: Node0Handle, ef: u16) {
        let results = self.search_level0(
            entry_node,
            &self.quant_vec_arena[vec_handle.cast()],
            ef,
            self.options.m0 * 2,
            true,
        );
        self.create_node0(vec_handle, results);
    }

    fn create_node1(
        &mut self,
        vec_handle: VecHandle,
        node_insert_index: u32,
        results: Box<[InternalSearchResult<Node1>]>,
    ) {
        let node_handle = self.nodes1_arena.alloc(node_insert_index, vec_handle);

        let mut num_links_created = 0;

        for result in results {
            let link_created = Neighbors::insert_neighbor(
                &mut self.nodes1_arena,
                result.node,
                &self.distance_metric,
                node_handle,
                result.score,
            );
            if link_created {
                num_links_created += 1;
                Neighbors::insert_neighbor(
                    &mut self.nodes1_arena,
                    node_handle,
                    &self.distance_metric,
                    result.node,
                    result.score,
                );
                if num_links_created == self.options.m {
                    break;
                }
            }
        }
    }

    fn create_node0(&mut self, vec_handle: VecHandle, results: Box<[InternalSearchResult<Node0>]>) {
        let node_handle = self.nodes0_arena.alloc(*vec_handle, ());

        let mut num_links_created = 0;

        for result in results {
            let link_created = Neighbors::insert_neighbor(
                &mut self.nodes0_arena,
                result.node,
                &self.distance_metric,
                node_handle,
                result.score,
            );
            if link_created {
                num_links_created += 1;
                Neighbors::insert_neighbor(
                    &mut self.nodes0_arena,
                    node_handle,
                    &self.distance_metric,
                    result.node,
                    result.score,
                );
                if num_links_created == self.options.m0 {
                    break;
                }
            }
        }
    }

    pub fn search_quantized(&self, query: &[f32], ef: u16, top_k: u16) -> Box<[SearchResult]> {
        self.search_quantized_inner(query, ef, top_k, -1.0..1.0)
    }

    pub(crate) fn search_quantized_inner(
        &self,
        query: &[f32],
        ef: u16,
        top_k: u16,
        range: Range<f32>,
    ) -> Box<[SearchResult]> {
        let (query, ptr, layout): (&QuantVec, *mut u8, Layout) = unsafe {
            let metadata = (self.options.quantization, self.options.dims);
            let size = QuantVec::size_aligned(metadata);
            let layout = Layout::from_size_align_unchecked(size, QuantVec::ALIGN);
            let ptr = alloc(layout);
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            QuantVec::new_at(ptr, metadata, (query.as_ptr(), range));
            let query = &*ptr::from_raw_parts(ptr, QuantVec::ptr_metadata(metadata));
            (query, ptr, layout)
        };
        let mut entry_node = self.top_level_root_node;

        for level in (1..=self.options.max_level).rev() {
            let results = self.search_level1(entry_node, query, ef, 1, true);
            let child = if level == 1 {
                self.nodes1_arena[results[0].node].vec.cast()
            } else {
                Handle::new(*results[0].node - 1)
            };
            entry_node = child;
        }

        let entry_node = entry_node.cast();

        let results = self.search_level0(entry_node, query, ef, top_k, false);

        unsafe {
            dealloc(ptr, layout);
        }

        unsafe { mem::transmute(results) }
    }

    pub fn search(&self, query: &[f32], ef: u16, top_k: u16) -> Box<[SearchResult]> {
        self.search_inner(query, ef, top_k, -1.0..1.0)
    }

    pub(crate) fn search_inner(
        &self,
        query: &[f32],
        ef: u16,
        top_k: u16,
        range: Range<f32>,
    ) -> Box<[SearchResult]> {
        debug_assert!((0..8192).contains(&top_k));
        let mag_query = magnitude_f32(query);
        let results_quantized = self.search_quantized_inner(query, ef, top_k * 8, range);
        if self.options.quantization == Quantization::F32 {
            return results_quantized;
        }
        let results_quantized =
            unsafe { mem::transmute::<Box<[SearchResult]>, Box<[(u32, f32)]>>(results_quantized) };
        let query = unsafe { mem::transmute::<&[f32], &RawVec>(query) };
        let mut results = Vec::with_capacity(results_quantized.len());
        for (node_id, _) in results_quantized {
            let vec_handle = Handle::new(node_id);
            let vec = &self.raw_vec_arena[vec_handle];
            let mag_vec = magnitude_f32(&vec.vec);
            let score = self
                .distance_metric
                .calculate_raw(query, mag_query, vec, mag_vec);
            results.push((node_id, score));
        }

        let top_k = top_k as usize;

        if results.len() > top_k {
            results.select_nth_unstable_by(top_k, |a, b| self.distance_metric.cmp_score(b.1, a.1));
            results.truncate(top_k);
        }

        results.sort_unstable_by(|a, b| self.distance_metric.cmp_score(b.1, a.1));

        unsafe {
            mem::transmute::<Box<[(u32, f32)]>, Box<[SearchResult]>>(results.into_boxed_slice())
        }
    }

    fn search_level1(
        &self,
        entry_node: Node1Handle,
        query: &QuantVec,
        ef: u16,
        top_k: u16,
        include_root: bool,
    ) -> Box<[InternalSearchResult<Node1>]> {
        let mut candidate_queue = BinaryHeap::new_by(|a: &InternalSearchResult<Node1>, b| {
            self.distance_metric.cmp_score(a.score, b.score)
        });
        let mut results = Vec::new();
        let mut set = FixedSet::new(self.options.m);

        let node = &self.nodes1_arena[entry_node];
        let vec = &self.quant_vec_arena[node.vec];

        let score = self.distance_metric.calculate(query, vec);

        set.insert(*entry_node);
        candidate_queue.push(InternalSearchResult {
            node: entry_node,
            score,
        });

        let mut nodes_visisted = 0;

        while let Some(entry) = candidate_queue.pop() {
            if nodes_visisted >= ef {
                break;
            }

            nodes_visisted += 1;
            if include_root || *entry.node != 0 {
                results.push(entry);
            }

            let node = &self.nodes1_arena[entry.node];

            for neighbor in node.neighbors.neighbors() {
                if !set.is_member(*neighbor.node) {
                    let neighbors_node = &self.nodes1_arena[neighbor.node];
                    let neighbor_vec = &self.quant_vec_arena[neighbors_node.vec];
                    let score = self.distance_metric.calculate(query, neighbor_vec);

                    set.insert(*neighbor.node);
                    candidate_queue.push(InternalSearchResult {
                        node: neighbor.node,
                        score,
                    });
                }
            }
        }

        let top_k = top_k as usize;

        if results.len() > top_k {
            results.select_nth_unstable_by(top_k, |a, b| {
                self.distance_metric.cmp_score(b.score, a.score)
            });
            results.truncate(top_k);
        }

        results.sort_unstable_by(|a, b| self.distance_metric.cmp_score(b.score, a.score));

        results.into_boxed_slice()
    }

    fn search_level0(
        &self,
        entry_node: Node0Handle,
        query: &QuantVec,
        ef: u16,
        top_k: u16,
        include_root: bool,
    ) -> Box<[InternalSearchResult<Node0>]> {
        let mut candidate_queue = BinaryHeap::new_by(|a: &InternalSearchResult<Node0>, b| {
            self.distance_metric.cmp_score(a.score, b.score)
        });
        let mut results = Vec::new();
        let mut set = FixedSet::new(self.options.m0);

        let vec = &self.quant_vec_arena[entry_node.cast()];

        let score = self.distance_metric.calculate(query, vec);

        set.insert(*entry_node);
        candidate_queue.push(InternalSearchResult {
            node: entry_node,
            score,
        });

        let mut nodes_visisted = 0;

        while let Some(entry) = candidate_queue.pop() {
            if nodes_visisted >= ef {
                break;
            }

            nodes_visisted += 1;
            if include_root || *entry.node != 0 {
                results.push(entry);
            }

            let node = &self.nodes0_arena[entry.node];

            for neighbor in node.neighbors.neighbors() {
                if !set.is_member(*neighbor.node) {
                    let neighbor_vec = &self.quant_vec_arena[neighbor.node.cast()];
                    let score = self.distance_metric.calculate(query, neighbor_vec);

                    set.insert(*neighbor.node);
                    candidate_queue.push(InternalSearchResult {
                        node: neighbor.node,
                        score,
                    });
                }
            }
        }

        let top_k = top_k as usize;

        if results.len() > top_k {
            results.select_nth_unstable_by(top_k, |a, b| {
                self.distance_metric.cmp_score(b.score, a.score)
            });
            results.truncate(top_k);
        }

        results.sort_unstable_by(|a, b| self.distance_metric.cmp_score(b.score, a.score));

        results.into_boxed_slice()
    }

    pub fn save(&mut self) {
        self.backend.save_options(&self.options);
        self.backend.save_id_counter(self.id_counter);
    }

    pub fn flush(&mut self) {
        self.raw_vec_arena.flush();
        self.quant_vec_arena.flush();
        self.nodes0_arena.flush();
        self.nodes1_arena.flush();
        self.backend.flush();
    }

    pub fn project_memory_usage(&self, num_vectors: u32) -> u64 {
        Self::mem_project((self.options.clone(), num_vectors)) + size_of::<Self>() as u64
    }
}

impl<A: ArenaBackend, G> Drop for Graph<A, G> {
    fn drop(&mut self) {
        self.nodes0_arena.clear(self.id_counter);
        let (last_idx, last_level) = level_sampler::sample(self.options.max_level, self.id_counter);
        self.nodes1_arena.clear(last_idx + last_level as u32);
        self.quant_vec_arena.clear(self.id_counter);
        self.raw_vec_arena.clear(self.id_counter);
    }
}

impl<A: ArenaBackend, G: GraphOptionsBackend> MemProject for Graph<A, G> {
    // (options, number of vectors)
    type State = (Options, u32);

    fn mem_project((options, num_vectors): Self::State) -> u64 {
        let num_vectors = num_vectors + 1; // include root

        let (last_idx, last_level) = level_sampler::sample(options.max_level, num_vectors);
        let node1_arena_len = last_idx as u32 + last_level as u32;
        // node0 arena
        <Arena<Node0, A>>::mem_project((1000, num_vectors , options.m0))
        // node1 arena
        + <Arena<Node1, A>>::mem_project((1000, node1_arena_len, options.m))
        // raw vec arena
        + <Arena<RawVec, A>>::mem_project((1000, num_vectors , options.dims))
        // quant vec arena
        + <Arena<QuantVec, A>>::mem_project((1000, num_vectors , (options.quantization, options.dims)))
    }
}
