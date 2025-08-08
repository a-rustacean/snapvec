from datasets import load_dataset
import numpy as np
import os

# Configuration
DATASET_NAME = "Sreenath/million-text-embeddings"
SPLIT = "train"
VECTOR_FIELD = "embedding"
VECTOR_SIZE = 768
TARGET_ROWS = 1_000_000
OUTPUT_FILE = "dataset.f32"
BATCH_SIZE = 1000

# Load dataset with streaming for memory efficiency
dataset = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)
batched_dataset = dataset.batch(BATCH_SIZE)

with open(OUTPUT_FILE, "wb") as f:
    vectors_written = 0
    
    for batch in batched_dataset:
        if vectors_written >= TARGET_ROWS:
            break
            
        # Get vectors and convert to float32
        vectors = np.array(batch[VECTOR_FIELD], dtype=np.float32)
        
        # Ensure proper 2D shape [batch_size, VECTOR_SIZE]
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # Verify vector dimensions
        if vectors.shape[1] != VECTOR_SIZE:
            print(f"Vector size mismatch: {vectors.shape[1]} ≠ {VECTOR_SIZE}")
            continue
        
        # Calculate how many vectors we can write without exceeding target
        remaining = TARGET_ROWS - vectors_written
        if remaining < len(vectors):
            vectors = vectors[:remaining]
        
        # Write to binary file
        f.write(vectors.tobytes())
        vectors_written += len(vectors)
        print(f"Saved {vectors_written}/{TARGET_ROWS} vectors", end="\r")

# Verify final output
expected_size = TARGET_ROWS * VECTOR_SIZE * 4  # 4 bytes per float32
actual_size = os.path.getsize(OUTPUT_FILE)

if actual_size == expected_size:
    print(f"\nSuccess! Saved {vectors_written} vectors")
    print(f"File size: {actual_size / (1024**3):.2f} GB")
else:
    print(f"\nSize mismatch: Expected {expected_size} bytes, got {actual_size}")
    print(f"Saved {vectors_written} vectors instead of {TARGET_ROWS}")
