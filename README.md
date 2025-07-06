# Python Implementation of MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings

This Python implementation was created to make the FDE algorithm more accessible while maintaining complete fidelity to the original C++ implementation. Every function and parameter has been carefully mapped to ensure identical behavior.

## What is FDE?

Fixed-Dimensional Encoding (FDE) solves a fundamental problem in modern search systems: how to efficiently search through billions of documents when each document is represented by hundreds of vectors (as in ColBERT-style models).

### The Problem
- **Traditional search**: Document = 1 vector → Fast but inaccurate
- **Modern multi-vector search**: Document = 100s of vectors → Accurate but extremely slow

### The FDE Solution
FDE transforms multiple vectors into a single fixed-size vector while preserving the similarity relationships. The magic is that the dot product between two FDE vectors approximates the original Chamfer similarity between the multi-vector sets.

### Running Guide
```
$ uv run main.py

2025-07-06 13:10:09,942 - INFO - Using device: mps
2025-07-06 13:10:09,942 - INFO - Loading dataset from Hugging Face Hub: 'zeta-alpha-ai/NanoFiQA2018'...
2025-07-06 13:10:18,583 - INFO - Dataset loaded: 4598 documents, 50 queries.
2025-07-06 13:10:18,583 - INFO - Initializing retrieval models...
2025-07-06 13:10:20,095 - INFO - --- PHASE 1: INDEXING ---
2025-07-06 13:10:20,096 - INFO - [ColbertFdeRetriever] Generating native multi-vector embeddings...
ColBERT documents embeddings: 100%|██████████| 144/144 [01:05<00:00,  2.21it/s]
2025-07-06 13:11:25,420 - INFO - [ColbertFdeRetriever] Generating FDEs from ColBERT embeddings in BATCH mode...
2025-07-06 13:11:25,420 - INFO - [FDE Batch] Starting batch FDE generation for 4598 documents
2025-07-06 13:11:25,420 - INFO - [FDE Batch] Using identity projection (dim=128)
2025-07-06 13:11:25,420 - INFO - [FDE Batch] Configuration: 20 repetitions, 128 partitions, projection_dim=128
2025-07-06 13:11:25,422 - INFO - [FDE Batch] Total vectors: 1177088, avg per doc: 256.0
2025-07-06 13:11:25,627 - INFO - [FDE Batch] Concatenation completed in 0.205s
2025-07-06 13:11:25,627 - INFO - [FDE Batch] Output FDE dimension: 327680
2025-07-06 13:11:25,627 - INFO - [FDE Batch] Processing repetition 1/20
2025-07-06 13:11:33,469 - INFO - [FDE Batch] Repetition timing breakdown:
2025-07-06 13:11:33,469 - INFO -   - SimHash: 0.037s
2025-07-06 13:11:33,469 - INFO -   - Projection: 0.000s
2025-07-06 13:11:33,469 - INFO -   - Partition indices: 0.019s
2025-07-06 13:11:33,469 - INFO -   - Aggregation: 2.101s
2025-07-06 13:11:33,469 - INFO -   - Averaging: 5.655s
2025-07-06 13:11:33,469 - INFO -   - Filled 462482 empty partitions
2025-07-06 13:12:04,662 - INFO - [FDE Batch] Processing repetition 6/20
2025-07-06 13:12:43,054 - INFO - [FDE Batch] Processing repetition 11/20
2025-07-06 13:13:22,420 - INFO - [FDE Batch] Processing repetition 16/20
2025-07-06 13:14:01,083 - INFO - [FDE Batch] Batch generation completed in 155.663s
2025-07-06 13:14:01,083 - INFO - [FDE Batch] Average time per document: 33.85ms
2025-07-06 13:14:01,083 - INFO - [FDE Batch] Throughput: 29.5 docs/sec
2025-07-06 13:14:01,083 - INFO - [FDE Batch] Output shape: (4598, 327680)
2025-07-06 13:14:01,188 - INFO - '2. ColBERT + FDE' indexing finished in 221.09 seconds.
2025-07-06 13:14:01,188 - INFO - --- PHASE 2: SEARCH & EVALUATION ---
2025-07-06 13:14:01,188 - INFO - Running search for '2. ColBERT + FDE' on 50 queries...
ColBERT queries embeddings: 100%|██████████| 1/1 [00:00<00:00,  2.44it/s]
ColBERT queries embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.13it/s]
ColBERT queries embeddings: 100%|██████████| 1/1 [00:00<00:00, 45.49it/s]
=====================================================================================
                                    FINAL REPORT                                     
(Dataset: zeta-alpha-ai/NanoFiQA2018)
=====================================================================================
Retriever                 | Indexing Time (s)    | Avg Query Time (ms)    | Recall@10 
-------------------------------------------------------------------------------------
1. ColBERT (Native)       | 82.31                | 1618.29                | 0.7000    
=====================================================================================
Retriever                 | Indexing Time (s)    | Avg Query Time (ms)    | Recall@10 
-------------------------------------------------------------------------------------
2. ColBERT + FDE          | 221.09               | 189.97                 | 0.6400    
=====================================================================================
2025-07-06 13:14:10,688 - INFO - '2. ColBERT + FDE' search finished. Avg query time: 189.97 ms.

Process finished with exit code 0
```

## Detailed Implementation Guide

### 1. Configuration Classes

#### EncodingType Enum
```python
class EncodingType(Enum):
    DEFAULT_SUM = 0    # For queries: sum vectors in each partition
    AVERAGE = 1        # For documents: average vectors in each partition
```
**C++ Mapping**: Directly corresponds to `FixedDimensionalEncodingConfig::EncodingType` in the proto file.

#### ProjectionType Enum
```python
class ProjectionType(Enum):
    DEFAULT_IDENTITY = 0    # No dimensionality reduction
    AMS_SKETCH = 1         # Use AMS sketch for reduction
```
**C++ Mapping**: Maps to `FixedDimensionalEncodingConfig::ProjectionType`.

#### FixedDimensionalEncodingConfig
```python
@dataclass
class FixedDimensionalEncodingConfig:
    dimension: int = 128                      # Original vector dimension
    num_repetitions: int = 10                # Number of independent runs
    num_simhash_projections: int = 6         # Controls partition granularity
    seed: int = 42                           # Random seed
    encoding_type: EncodingType = DEFAULT_SUM
    projection_type: ProjectionType = DEFAULT_IDENTITY
    projection_dimension: Optional[int] = None
    fill_empty_partitions: bool = False
    final_projection_dimension: Optional[int] = None
```
**C++ Mapping**: Direct equivalent of `FixedDimensionalEncodingConfig` message in the proto file.

### 2. Internal Helper Functions

#### Gray Code Functions
```python
def _append_to_gray_code(gray_code: int, bit: bool) -> int:
    return (gray_code << 1) + (int(bit) ^ (gray_code & 1))
```
**C++ Mapping**: Exact implementation of `internal::AppendToGrayCode()`.

```python
def _gray_code_to_binary(num: int) -> int:
    mask = num >> 1
    while mask != 0:
        num = num ^ mask
        mask >>= 1
    return num
```
**C++ Mapping**: Equivalent to `internal::GrayCodeToBinary()`. The C++ version uses `num ^ (num >> 1)`, while Python uses a loop for clarity.

#### Random Matrix Generators

```python
def _simhash_matrix_from_seed(dimension: int, num_projections: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dimension, num_projections))
```
**C++ Mapping**: Maps to `internal::SimHashMatrixFromSeed()`. Uses Gaussian distribution for LSH.

```python
def _ams_projection_matrix_from_seed(dimension: int, projection_dim: int, seed: int):
    # Creates sparse random matrix with one non-zero per row
```
**C++ Mapping**: Corresponds to `internal::AMSProjectionMatrixFromSeed()`.

#### Partition Index Calculation
```python
def _simhash_partition_index_gray(sketch_vector: np.ndarray) -> int:
    partition_index = 0
    for val in sketch_vector:
        partition_index = _append_to_gray_code(partition_index, val > 0)
    return partition_index
```
**C++ Mapping**: Direct implementation of `internal::SimHashPartitionIndex()`.

### 3. Core Algorithm

The `_generate_fde_internal()` function implements the main FDE generation logic:

```python
def _generate_fde_internal(point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig):
    # Step 1: Validate inputs (matches C++ parameter validation)
    # Step 2: Calculate dimensions
    # Step 3: For each repetition:
    #   - Apply SimHash for space partitioning
    #   - Apply optional dimensionality reduction
    #   - Aggregate vectors by partition
    #   - Apply averaging for document FDE
    # Step 4: Optional final projection
```

**C++ Mapping**: This function combines the logic from both `GenerateQueryFixedDimensionalEncoding()` and `GenerateDocumentFixedDimensionalEncoding()` in the C++ code.

### 4. Public API

```python
def generate_query_fde(point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig):
    # Forces encoding_type to DEFAULT_SUM
```
**C++ Mapping**: Equivalent to `GenerateQueryFixedDimensionalEncoding()`.

```python
def generate_document_fde(point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig):
    # Forces encoding_type to AVERAGE
```
**C++ Mapping**: Equivalent to `GenerateDocumentFixedDimensionalEncoding()`.

```python
def generate_fde(point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig):
    # Routes based on config.encoding_type
```
**C++ Mapping**: Equivalent to `GenerateFixedDimensionalEncoding()`.

## C++ to Python Mapping

### Key Differences and Similarities

| Feature | C++ Implementation | Python Implementation | Notes |
|---------|-------------------|----------------------|--------|
| **Matrix Operations** | Eigen library | NumPy | Functionally equivalent |
| **Memory Management** | Manual with Eigen::Map | Automatic (NumPy) | Python is simpler |
| **Gray Code Conversion** | `num ^ (num >> 1)` | While loop | Both produce same result |
| **Error Handling** | absl::Status | Python exceptions | Different style, same checks |
| **Random Number Generation** | std::mt19937 | np.random.default_rng | Same distributions |
| **Configuration** | Protocol Buffers | dataclass | Same fields and defaults |

### Exact Function Mappings

| C++ Function | Python Function | Purpose |
|--------------|-----------------|---------|
| `GenerateFixedDimensionalEncoding()` | `generate_fde()` | Top-level routing function |
| `GenerateQueryFixedDimensionalEncoding()` | `generate_query_fde()` | Query FDE generation |
| `GenerateDocumentFixedDimensionalEncoding()` | `generate_document_fde()` | Document FDE generation |
| `internal::SimHashPartitionIndex()` | `_simhash_partition_index_gray()` | Partition assignment |
| `internal::DistanceToSimHashPartition()` | `_distance_to_simhash_partition()` | Hamming distance calculation |
| `internal::ApplyCountSketchToVector()` | `_apply_count_sketch_to_vector()` | Final projection |

## Usage Examples

### Basic Usage

```python
import numpy as np
from fde_generator import FixedDimensionalEncodingConfig, generate_query_fde, generate_document_fde

# 1. Create configuration
config = FixedDimensionalEncodingConfig(
    dimension=128,              # Vector dimension
    num_repetitions=10,         # Number of independent partitionings
    num_simhash_projections=6,  # Creates 2^6 = 64 partitions
    seed=42
)

# 2. Prepare data
# Query: 32 vectors of 128 dimensions each
query_vectors = np.random.randn(32, 128).astype(np.float32)

# Document: 80 vectors of 128 dimensions each  
doc_vectors = np.random.randn(80, 128).astype(np.float32)

# 3. Generate FDEs
query_fde = generate_query_fde(query_vectors, config)
doc_fde = generate_document_fde(doc_vectors, config)

# 4. Compute similarity (approximates Chamfer similarity)
similarity_score = np.dot(query_fde, doc_fde)
print(f"Similarity: {similarity_score}")
```

### Advanced Usage with Dimensionality Reduction

```python
from fde_generator import ProjectionType, replace

# Use AMS Sketch for internal projection
config_with_projection = replace(
    config,
    projection_type=ProjectionType.AMS_SKETCH,
    projection_dimension=16  # Reduce from 128 to 16 dimensions
)

# Use Count Sketch for final projection
config_with_final_projection = replace(
    config,
    final_projection_dimension=1024  # Final FDE will be 1024 dimensions
)
```

## Algorithm Walkthrough

### Step-by-Step Process

1. **Input**: Multiple vectors representing a document/query
   - Example: 32 vectors of 128 dimensions each

2. **Space Partitioning** (per repetition):
   - Apply SimHash: Multiply by random Gaussian matrix
   - Convert to partition indices using Gray Code
   - Creates 2^k_sim partitions (e.g., 64 partitions)

3. **Vector Aggregation**:
   - **For Queries**: Sum all vectors in each partition
   - **For Documents**: Average all vectors in each partition

4. **Repetition**:
   - Repeat steps 2-3 with different random seeds
   - Concatenate results from all repetitions

5. **Output**: Single FDE vector
   - Dimension: `num_repetitions × num_partitions × projection_dim`

### Why It Works

The key insight is that FDE preserves the local structure of the vector space through LSH (Locality Sensitive Hashing). Vectors that are close in the original space are likely to:
1. End up in the same partition
2. Contribute to the same parts of the FDE vector
3. Produce high dot products when their FDEs are compared

## Performance Characteristics

- **FDE Generation Time**: O(n × d × r × k) where:
  - n = number of vectors
  - d = vector dimension
  - r = number of repetitions
  - k = number of SimHash projections

- **Search Time**: O(1) using standard MIPS libraries
- **Memory**: Configurable via projection dimensions

## References

- **Original Paper**: [MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings](https://arxiv.org/pdf/2405.19504)
- **C++ Implementation**: [Google Graph Mining Repository](https://github.com/google/graph-mining/tree/main/sketching/point_cloud)
- **Blog Post**: [MUVERA: Making multi-vector retrieval as fast as single-vector search](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/)

## Contributing

Contributions are welcome! Please ensure any changes maintain compatibility with the C++ implementation.

## License

This implementation follows the same Apache 2.0 license as the original C++ code.
