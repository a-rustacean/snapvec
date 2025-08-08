use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use ndarray::{Array1, Array2, Axis};
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

fn main() -> Result<()> {
    // Configuration - match these to your Python script settings
    let file_path = "dataset.f32";
    let dims: u16 = 768;
    let num_vectors: usize = 1_000_000;
    let num_queries = 100_000;
    let top_k = 5;

    // hnsw config
    let m = 16;
    let m0 = 32;
    let max_level = 10;
    let ef_construction = 64;
    let ef_search = 64;
    let quantization = Quantization::I8;

    let mut rng = StdRng::from_seed([1; 32]);

    // Memory-map the file for zero-copy access
    let file = File::open(Path::new(file_path)).context("Failed to open file")?;
    let mmap = unsafe { Mmap::map(&file) }.context("Memory mapping failed")?;

    // Validate file size
    let expected_bytes = num_vectors * dims as usize * std::mem::size_of::<f32>();
    if mmap.len() != expected_bytes {
        anyhow::bail!(
            "File size mismatch: expected {} bytes, found {}",
            expected_bytes,
            mmap.len()
        );
    }

    // Interpret the bytes as f32 array (zero-copy)
    let float_slice: &[f32] = bytemuck::cast_slice(&mmap);

    // Create progress bar for loading vectors
    let pb_load = ProgressBar::new(100);
    pb_load.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] Loading vectors ({percent}%) ETA: {eta}")
            .unwrap()
            .progress_chars("#>-"),
    );

    // Simulate progress for loading
    for i in 1..=100 {
        pb_load.set_position(i);
        if i == 50 {
            // Reshape to 2D array [num_vectors, vector_dim] at halfway point
            let _vectors_temp =
                Array2::from_shape_vec((num_vectors, dims as usize), float_slice.to_vec())
                    .context("Shape mismatch during array conversion")?;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Actually create the vectors array
    let vectors = Array2::from_shape_vec((num_vectors, dims as usize), float_slice.to_vec())
        .context("Shape mismatch during array conversion")?;

    pb_load.finish_with_message("✓ Vectors loaded successfully!");

    let (queries, brute_force_results) =
        top_closest_vectors(&vectors, num_queries, top_k, &mut rng);

    let options = Options {
        m,
        m0,
        dims,
        max_level,
        quantization,
        metric: DistanceMetricKind::Cos,
    };

    let mut graph = Graph::new(options);

    // Create progress bar for indexing
    let pb_index = ProgressBar::new(num_vectors as u64);
    pb_index.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Indexing vectors ({percent}%) ETA: {eta} [{per_sec}]")
            .unwrap()
            .progress_chars("#>-"),
    );

    let indexing_start = Instant::now();
    let mut indexed_counter = 0;
    let pb_clone = pb_index.clone();

    vectors.axis_iter(Axis(0)).for_each(|vector| {
        graph.insert(vector.as_slice().unwrap(), ef_construction);
        indexed_counter += 1;
        if indexed_counter % 1_000 == 0 {
            pb_clone.set_position(indexed_counter);
        }
    });

    pb_index.finish_with_message(format!(
        "✓ Indexed {} vectors in {:?}!",
        num_vectors,
        indexing_start.elapsed()
    ));

    let hnsw_results = queries
        .axis_iter(Axis(0))
        .map(|query| graph.search(query.as_slice().unwrap(), ef_search, top_k as u16))
        .collect::<Vec<_>>();

    let mut total_recall = 0;

    for (brute_force_results, hnsw_results) in brute_force_results.into_iter().zip(hnsw_results) {
        let brute_force_ids =
            HashSet::<u32>::from_iter(brute_force_results.into_iter().map(|(id, _)| id as u32));
        let hnsw_ids = HashSet::<u32>::from_iter(
            hnsw_results
                .into_iter()
                .map(|result| result.node.0.get() - 1),
        );
        total_recall += brute_force_ids.intersection(&hnsw_ids).count();
    }

    let perfect_recall = top_k * num_queries;

    let recall_percent = (total_recall as f32 / perfect_recall as f32) * 100.0;

    // Create progress bar for performance evaluation
    let pb_perf = ProgressBar::new(num_queries as u64);
    pb_perf.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Evaluating search performance ({percent}%) [{per_sec}]")
            .unwrap()
            .progress_chars("#>-"),
    );

    let start = Instant::now();
    let perf_counter = Arc::new(AtomicUsize::new(0));
    let pb_perf_clone = pb_perf.clone();

    queries.axis_iter(Axis(0)).par_bridge().for_each(|query| {
        black_box(graph.search(query.as_slice().unwrap(), ef_search, top_k as u16));
        let count = perf_counter.fetch_add(1, Ordering::Relaxed) + 1;
        pb_perf_clone.set_position(count as u64);
    });

    let elapsed = start.elapsed();
    let qps = num_queries as f32 / elapsed.as_secs_f32();

    pb_perf.finish_with_message("✓ Performance evaluation complete!");

    // Print final results clearly
    println!();
    println!("=== BENCHMARK RESULTS ===");
    println!("Recall: {:.2}%", recall_percent);
    println!("QPS (Queries Per Second): {:.2}", qps);
    println!("=========================");

    Ok(())
}

fn top_closest_vectors(
    data: &Array2<f32>,
    num_queries: usize,
    top_k: usize,
    rng: &mut impl Rng,
) -> (Array2<f32>, Vec<Vec<(usize, f32)>>) {
    // Progress bar for generating queries
    let pb_queries = ProgressBar::new(num_queries as u64);
    pb_queries.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Generating queries ({percent}%) ETA: {eta}")
            .unwrap()
            .progress_chars("#>-"),
    );

    // 1. Randomly select query indices from corpus
    let query_indices = sample(rng, data.nrows(), num_queries).into_vec();
    let mut queries = Array2::<f32>::zeros((num_queries, data.ncols()));

    // 2. Perturb selected queries and clamp to [-1, 1]
    for (i, &idx) in query_indices.iter().enumerate() {
        let original = data.row(idx);
        let mut perturbed = Array1::zeros(data.ncols());

        for j in 0..data.ncols() {
            // Add Gaussian noise (stddev=0.1) and clamp values
            perturbed[j] = (original[j] + rng.random_range(-0.1..0.1)).clamp(-1.0, 1.0);
        }
        queries.row_mut(i).assign(&perturbed);
        pb_queries.set_position((i + 1) as u64);
    }
    pb_queries.finish_with_message("✓ Queries generated successfully!");

    // Progress bar for computing corpus norms
    let pb_norms = ProgressBar::new(data.nrows() as u64);
    pb_norms.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Computing corpus norms ({percent}%) ETA: {eta} [{per_sec}]")
            .unwrap()
            .progress_chars("#>-"),
    );

    // 3. Precompute norms for all vectors in corpus (once)
    let norms_counter = Arc::new(AtomicUsize::new(0));
    let pb_norms_clone = pb_norms.clone();
    let corpus_norms = data
        .axis_iter(Axis(0))
        .par_bridge()
        .map(|row| {
            let norm = row.dot(&row).sqrt();
            let count = norms_counter.fetch_add(1, Ordering::Relaxed) + 1;
            pb_norms_clone.set_position(count as u64);
            norm
        })
        .collect::<Vec<_>>();
    let corpus_norms = Array1::from_vec(corpus_norms);
    pb_norms.finish_with_message("✓ Corpus norms computed successfully!");

    // Progress bar for brute force search
    let pb_search = ProgressBar::new(num_queries as u64);
    pb_search.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Computing brute force results ({percent}%) ETA: {eta} [{per_sec}]")
            .unwrap()
            .progress_chars("#>-"),
    );

    // 4. Process queries in batches to avoid memory explosion
    // Dynamically adjust batch size based on available memory and query count
    let batch_size = if num_queries <= 1000 {
        num_queries // Process all at once for small query sets
    } else if num_queries <= 10000 {
        500 // Smaller batches for medium query sets
    } else {
        200 // Very small batches for large query sets to minimize memory usage
    };
    let mut all_results = Vec::with_capacity(num_queries);

    for batch_start in (0..num_queries).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_queries);
        let current_batch_size = batch_end - batch_start;

        // Extract current batch of queries
        let batch_queries = queries.slice(ndarray::s![batch_start..batch_end, ..]);

        // Compute norms for current batch
        let batch_query_norms = batch_queries
            .axis_iter(Axis(0))
            .map(|row| row.dot(&row).sqrt())
            .collect::<Vec<f32>>();

        // Compute dot products for current batch only
        let batch_dot_products = batch_queries.dot(&data.t());

        // Find top-k matches for each query in current batch
        let batch_results = (0..current_batch_size)
            .into_par_iter()
            .map(|i| {
                // Use MAX-heap to keep highest cosine similarities
                let mut heap: BinaryHeap<(Reverse<OrderedFloat<f32>>, usize)> =
                    BinaryHeap::with_capacity(top_k + 1);
                let q_norm = batch_query_norms[i];

                for (j, (&dot, &c_norm)) in batch_dot_products
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

                    // Use max-heap for cosine similarity (higher is better)
                    // Keep the LOWEST scoring items to maintain top-k highest
                    if heap.len() < top_k {
                        heap.push((Reverse(OrderedFloat(cos_sim)), j));
                    } else {
                        let entry = (Reverse(OrderedFloat(cos_sim)), j);
                        // If this similarity is higher than the lowest in our top-k
                        if entry.0.0 > heap.peek().unwrap().0.0 {
                            heap.pop();
                            heap.push(entry);
                        }
                    }
                }

                // Convert to sorted vec (highest similarity first)
                let mut results: Vec<_> = heap.into_iter().collect();
                results.sort_by(|a, b| b.0.0.partial_cmp(&a.0.0).unwrap()); // Descending order
                results
                    .into_iter()
                    .map(|(score, idx)| (idx, score.0.0))
                    .collect()
            })
            .collect::<Vec<_>>();

        // Add batch results to overall results
        all_results.extend(batch_results);

        // Update progress bar
        pb_search.set_position(batch_end as u64);
    }

    pb_search.finish_with_message("✓ Brute force computation complete!");

    (queries, all_results)
}
