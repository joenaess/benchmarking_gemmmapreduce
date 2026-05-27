# ⚡ PyTorch Attention Showdown: Triton vs. CUDA C++ vs. SDPA

This benchmarking suite compares the performance, efficiency, and memory scaling of three different PyTorch attention implementations:
1. **GeMMMapReduce (CUDA C++)**: A custom CUDA C++ PyTorch extension compiled locally with a newly added **Dynamic JIT Autotuner**.
2. **flashreduce (Triton)**: A high-performance code-generated Triton implementation.
3. **PyTorch SDPA**: PyTorch's native, highly-optimized `scaled_dot_product_attention`.

---

## 📊 Latest Benchmark Results

Conducted with microsecond-accurate GPU timing using CUDA Events and proper warmup phases (batch size `B = 1`, hidden dimension `D = 128`, data type `bfloat16`, running on an **NVIDIA RTX 4080**).

| Sequence Length | C++ Time (ms) | Triton Time (ms) | SDPA Time (ms) | C++ Peak Mem (MB) | Triton Peak Mem (MB) | SDPA Peak Mem (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | 2.06 | 0.28 | 0.77 | **17.25** | 18.76 | 21.62 |
| **1024** | 3.88 | 0.27 | **0.24** | **18.25** | 21.27 | 35.00 |
| **2048** | 13.85 | 0.67 | **0.64** | **20.26** | 26.29 | 85.75 |
| **4096** | 29.39 | **2.25** | 3.83 | **24.27** | 36.33 | 283.25 |
| **8192** | 60.30 | **8.80** | 14.52 | **32.28** | 56.41 | 1062.25 |

---

## 🏆 Key Takeaways & Deep-Dive Analysis

### 🚀 The C++ Evolution: Dynamic JIT Autotuning ⚙️
We have successfully implemented a **Dynamic JIT Autotuner** for `GeMMMapReduce`.
- **How it Works**: The autotuner queries the current GPU's physical hardware constraints at runtime (via `torch.cuda.get_device_properties`). It retrieves the block-level shared memory ceiling (`shared_memory_per_block`) to filter and select viable candidate tile sizes (e.g., `32`, `64`), preventing illegal shared memory limits on low-memory or consumer cards. 
- **The Performance Leap**: By switching to a JIT compilation strategy, the compiler (`nvcc`) is invoked dynamically at runtime with target compilation optimizations:
  ```bash
  extra_cflags=["-O3", "-std=c++20"]
  extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++20", "-DTILE_SIZE=size"]
  ```
  Adding `--use_fast_math` and compiling specifically for the local GPU architecture unlocked major compiler loop-unrolling and register optimizations.
- **The Results**: The C++ implementation experienced a massive speedup across the board!
  - At **8k sequence length**, C++ runtime dropped from **76.97 ms** to **60.30 ms** (a **21.6% performance increase**!).
  - At **4k sequence length**, C++ runtime dropped from **38.23 ms** to **29.39 ms** (a **23.1% performance increase**!).

### 👑 Why Triton Still Takes the Crown
Even with JIT compilation and hardware-guided block screening, Triton remains faster at larger sequences, executing the 8k sequence in **8.80 ms** compared to the autotuned C++ at **60.30 ms**.

- **Multi-parameter Autotuning**: While our C++ autotuner evaluates block tile sizes (selecting `32` for consumer limits under 49,152 bytes), Triton's autotuner evaluates **a matrix of multiple hyperparameters** at runtime:
  - Block size variations along multiple dimensions (e.g. `l_block`, `r_block`).
  - Varying **warp allocations** (e.g., `4` or `8` warps) to optimize register utilization.
  - Number of **pipelining stages** (`num_stages`), which maximizes compute-memory overlap.
  - Number of thread **shards** for parallel reduction.
- **Tiling Complexity**: Triton's code generator yields highly-optimized thread tile layouts and shared memory access patterns dynamically, ensuring maximum SM saturation and memory coalescing that manual static loops struggle to match.

### 💾 Memory Scaling Analysis
- **PyTorch SDPA**: Extremely fast at low sequences due to low dispatch overhead, but memory scales quadratically, reaching a massive **1,062.25 MB (1.06 GB)** at sequence length 8192.
- **GeMMMapReduce (C++)**: Excels in memory efficiency, peaking at only **32.28 MB** (a spectacular **33x reduction** compared to SDPA!).
- **flashreduce (Triton)**: Maintains superb memory scaling, peaking at **56.41 MB** at sequence length 8192 (an **18.8x reduction** compared to SDPA!).

---

## 🛠️ Environment Setup & Running

This repository leverages `uv` for blazing-fast virtual environment management and local dependency linking.

### Prerequisites
- Python 3.10+
- CUDA Toolkit installed (for compiling the custom CUDA C++ extension)
- `uv` installed (`pip install uv` or via standalone installer)

### Setup
Initialize the environment and install dependencies along with the editable extensions:
```bash
# Initialize venv
uv venv
source .venv/bin/activate

# Install requirements
uv pip install torch triton

# Link local extensions in editable mode
uv pip install -e /home/jonas/Projects/GeMMMapReduce-main/GeMMMapReduce-main
uv pip install -e /home/jonas/Projects/flashreduce
```

### Run Benchmarks
Run the benchmark script to print the comparison table:
```bash
python benchmark_attention.py
```
