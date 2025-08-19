use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use snapvec::{DistanceMetricKind, Graph, Options, Quantization};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Main entry point for vector search benchmarking
fn main() -> Result<()> {
    // ==============================
    // Configuration
    // ==============================
    const FILE_PATH: &str = "dataset.f32";
    const DIMS: u16 = 768;
    const NUM_VECTORS: usize = 1_000_000;
    const NUM_QUERIES: usize = 100_000;
    const TOP_K: u16 = 5;

    // HNSW Configuration
    const M: u16 = 16;
    const M0: u16 = 32;
    const MAX_LEVEL: u8 = 10;
    const EF_CONSTRUCTION: u16 = 64;
    const EF_SEARCH: u16 = 64;
    const QUANTIZATION: Quantization = Quantization::I8;

    let mut rng = StdRng::from_seed([1; 32]);

    let options = Options {
        m: M,
        m0: M0,
        dims: DIMS,
        max_level: MAX_LEVEL,
        quantization: QUANTIZATION,
        metric: DistanceMetricKind::Cos,
    };

    let mut graph = Graph::new(options);
    let graph_memory_usage = graph.project_memory_usage(NUM_VECTORS as u32);
    let corpus_memory_usage = NUM_VECTORS as u64 * DIMS as u64 * 4;

    println!("\n================================================");
    println!("  Estimated memory usage:");
    println!(
        "    Corpus: {:.2} GiB",
        corpus_memory_usage as f32 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "     Graph: {:.2} GiB",
        graph_memory_usage as f32 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "     Total: {:.2} GiB",
        (corpus_memory_usage + graph_memory_usage) as f32 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("================================================\n");

    // ==============================
    // Data Loading
    // ==============================
    let vectors = load_vectors(FILE_PATH, NUM_VECTORS, DIMS)?;

    // ==============================
    // Query Generation & Ground Truth
    // ==============================
    let (queries, brute_force_results) =
        generate_queries_and_ground_truth(&vectors, NUM_QUERIES, TOP_K, &mut rng);

    // ==============================
    // Index Construction
    // ==============================
    build_index(&vectors, &mut graph, EF_CONSTRUCTION, NUM_VECTORS);

    drop(vectors);

    // ==============================
    // Search & Evaluation
    // ==============================
    let hnsw_results = search_queries(&queries, &graph, EF_SEARCH, TOP_K);
    let recall_percent = calculate_recall(&brute_force_results, &hnsw_results, TOP_K);
    let qps = measure_throughput(&queries, &graph, EF_SEARCH, TOP_K, NUM_QUERIES);

    // ==============================
    // Results Reporting
    // ==============================
    print_results(recall_percent, qps);

    Ok(())
}

/// Load vectors from binary file into memory-mapped array
fn load_vectors(file_path: &str, num_vectors: usize, dims: u16) -> Result<Array2<f32>> {
    let progress = ProgressBar::new(100).with_style(progress_style("Loading vectors"));
    let file = File::open(Path::new(file_path)).context("Failed to open file")?;

    // Memory map file for efficient access
    let mmap = unsafe { Mmap::map(&file).context("Memory mapping failed")? };
    let expected_bytes = num_vectors * dims as usize * std::mem::size_of::<f32>();

    // Validate file size
    if mmap.len() != expected_bytes {
        anyhow::bail!(
            "File size mismatch: expected {} bytes, found {}",
            expected_bytes,
            mmap.len()
        );
    }

    // Convert bytes to f32 slice (zero-copy)
    let float_slice: &[f32] = bytemuck::cast_slice(&mmap);
    progress.inc(50); // Simulate mid-load progress

    // Create 2D array [num_vectors, vector_dim]
    let vectors = Array2::from_shape_vec((num_vectors, dims as usize), float_slice.to_vec())
        .context("Shape mismatch during array conversion")?;

    progress.finish_with_message("✓ Vectors loaded successfully!");
    Ok(vectors)
}

/// Generate perturbed queries and compute ground truth results
fn generate_queries_and_ground_truth(
    data: &Array2<f32>,
    num_queries: usize,
    top_k: u16,
    rng: &mut impl Rng,
) -> (Array2<f32>, Vec<Vec<(usize, f32)>>) {
    let pb_queries =
        ProgressBar::new(num_queries as u64).with_style(progress_style("Generating queries"));

    // Randomly select and perturb queries
    let query_indices = sample(rng, data.nrows(), num_queries).into_vec();
    let mut queries = Array2::<f32>::zeros((num_queries, data.ncols()));

    for (i, &idx) in query_indices.iter().enumerate() {
        let original = data.row(idx);
        let mut perturbed = Array1::zeros(data.ncols());

        for j in 0..data.ncols() {
            perturbed[j] = (original[j] + rng.random_range(-0.1..0.1)).clamp(-1.0, 1.0);
        }
        queries.row_mut(i).assign(&perturbed);
        pb_queries.inc(1);
    }
    pb_queries.finish_with_message("✓ Queries generated successfully!");

    // Precompute corpus norms in parallel
    let pb_norms =
        ProgressBar::new(data.nrows() as u64).with_style(progress_style("Computing corpus norms"));
    let corpus_norms = compute_corpus_norms(data, &pb_norms);
    pb_norms.finish_with_message("✓ Corpus norms computed successfully!");

    // Compute ground truth results
    let pb_search = ProgressBar::new(num_queries as u64)
        .with_style(progress_style("Computing brute force results"));
    let results = brute_force_search(&queries, data, &corpus_norms, top_k, &pb_search);
    pb_search.finish_with_message("✓ Brute force computation complete!");

    (queries, results)
}

/// Compute L2 norms for all vectors in corpus
fn compute_corpus_norms<S>(data: &ArrayBase<S, Ix2>, progress: &ProgressBar) -> Array1<f32>
where
    S: Data<Elem = f32>,
{
    let counter = Arc::new(AtomicUsize::new(0));
    let progress_clone = progress.clone();

    data.axis_iter(Axis(0))
        .par_bridge()
        .map(|row| {
            let norm = row.dot(&row).sqrt();
            let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
            progress_clone.set_position(count as u64);
            norm
        })
        .collect::<Vec<_>>()
        .into()
}

/// Brute-force search for top-k nearest neighbors
fn brute_force_search(
    queries: &Array2<f32>,
    corpus: &Array2<f32>,
    corpus_norms: &Array1<f32>,
    top_k: u16,
    progress: &ProgressBar,
) -> Vec<Vec<(usize, f32)>> {
    let mut all_results = Vec::with_capacity(queries.nrows());
    let batch_size = calculate_batch_size(queries.nrows());

    for batch_start in (0..queries.nrows()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(queries.nrows());
        let batch = queries.slice(ndarray::s![batch_start..batch_end, ..]);

        // Compute query norms for current batch
        let query_norms: Vec<f32> = batch
            .axis_iter(Axis(0))
            .map(|row| row.dot(&row).sqrt())
            .collect();

        // Compute dot products for current batch
        let dot_products = batch.dot(&corpus.t());

        // Process each query in batch
        let batch_results: Vec<_> = (0..batch.nrows())
            .into_par_iter()
            .map(|i| {
                let mut heap = BinaryHeap::with_capacity(top_k as usize + 1);
                let q_norm = query_norms[i];

                for (j, (&dot, &c_norm)) in dot_products
                    .row(i)
                    .iter()
                    .zip(corpus_norms.iter())
                    .enumerate()
                {
                    let cos_sim = if dot == 0.0 || q_norm == 0.0 || c_norm == 0.0 {
                        0.0
                    } else {
                        dot / (q_norm * c_norm)
                    };

                    // Maintain top-k in max-heap using reverse ordering
                    heap.push(Reverse((OrderedFloat(cos_sim), j)));
                    if heap.len() > top_k as usize {
                        heap.pop();
                    }
                }

                // Convert to descending order
                let mut results: Vec<_> = heap
                    .into_sorted_vec()
                    .into_iter()
                    .map(|Reverse((score, idx))| (idx, score.into_inner()))
                    .collect();
                results.reverse();
                results
            })
            .collect();

        all_results.extend(batch_results);
        progress.set_position(batch_end as u64);
    }

    all_results
}

/// Build HNSW index from vectors
fn build_index(vectors: &Array2<f32>, graph: &mut Graph, ef_construction: u16, total: usize) {
    let progress = ProgressBar::new(total as u64).with_style(progress_style("Indexing vectors"));
    let counter = Arc::new(AtomicUsize::new(0));
    let progress_clone = progress.clone();

    vectors.axis_iter(Axis(0)).for_each(|vector| {
        graph.insert(vector.as_slice().unwrap(), ef_construction);
        let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_multiple_of(1_000) {
            progress_clone.set_position(count as u64);
        }
    });

    progress.finish_with_message(format!(
        "✓ Indexed {} vectors in {:?}!",
        total,
        progress.elapsed()
    ));
}

/// Search all queries using HNSW index
fn search_queries(
    queries: &Array2<f32>,
    graph: &Graph,
    ef_search: u16,
    top_k: u16,
) -> Vec<Box<[snapvec::SearchResult]>> {
    queries
        .axis_iter(Axis(0))
        .map(|query| graph.search(query.as_slice().unwrap(), ef_search, top_k))
        .collect()
}

/// Calculate recall percentage between HNSW and ground truth results
fn calculate_recall(
    ground_truth: &[Vec<(usize, f32)>],
    hnsw_results: &[Box<[snapvec::SearchResult]>],
    top_k: u16,
) -> f32 {
    let total_matches = ground_truth
        .iter()
        .zip(hnsw_results)
        .map(|(gt, results)| {
            let gt_set: HashSet<u32> = gt.iter().map(|(id, _)| *id as u32).collect();
            let result_set: HashSet<u32> = results
                .iter()
                .map(|res| res.node.0.get() - 1) // Convert to zero-based index
                .collect();
            gt_set.intersection(&result_set).count()
        })
        .sum::<usize>();

    let perfect_matches = top_k as f32 * ground_truth.len() as f32;
    (total_matches as f32 / perfect_matches) * 100.0
}

/// Measure queries per second (QPS)
fn measure_throughput(
    queries: &Array2<f32>,
    graph: &Graph,
    ef_search: u16,
    top_k: u16,
    total_queries: usize,
) -> f32 {
    let progress = ProgressBar::new(total_queries as u64)
        .with_style(progress_style("Evaluating search performance"));
    let counter = Arc::new(AtomicUsize::new(0));
    let progress_clone = progress.clone();

    let start = Instant::now();
    queries.axis_iter(Axis(0)).par_bridge().for_each(|query| {
        black_box(graph.search(query.as_slice().unwrap(), ef_search, top_k));
        let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_multiple_of(1_000) {
            progress_clone.set_position(count as u64);
        }
    });

    let elapsed = start.elapsed();
    progress.finish_with_message("✓ Performance evaluation complete!");

    total_queries as f32 / elapsed.as_secs_f32()
}

/// Print final benchmark results
fn print_results(recall: f32, qps: f32) {
    println!("\n=== BENCHMARK RESULTS ===");
    println!("Recall: {:.2}%", recall);
    println!("QPS (Queries Per Second): {:.2}", qps);
    println!("=========================");
}

/// Calculate batch size for memory-efficient processing
fn calculate_batch_size(query_count: usize) -> usize {
    match query_count {
        0..=1000 => query_count,
        1001..=10_000 => 500,
        _ => 200,
    }
}

/// Create consistent progress bar styling
fn progress_style(task: &str) -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(&format!(
            "{{spinner:.green}} [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] \
             {{pos}}/{{len}} {} ({{percent}}%) ETA: {{eta}} [{{per_sec}}]",
            task
        ))
        .unwrap()
        .progress_chars("#>-")
}
