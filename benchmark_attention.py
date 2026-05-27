import torch
import gc
import sys

# Import the custom CUDA C++ implementation
try:
    from gemmmapreduce.attention_cuda_wrapper import GeMMMrAttention
except ImportError:
    try:
        from gemmmapreduce import GeMMMrAttention
    except ImportError:
        print("Error: Could not import GeMMMrAttention from gemmmapreduce package.")
        sys.exit(1)

# Import the Triton implementation
try:
    from flashreduce.examples.attention import _kernel as flashreduce_kernel
except ImportError:
    print("Error: Could not import flashreduce attention kernel.")
    sys.exit(1)

# --- CONFIGURATION ---
B = 1  # Batch Size
D = 128  # Hidden Dimension
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192]
DTYPE = torch.bfloat16
DEVICE = "cuda"
NUM_ITERS = 10


def get_dummy_data(N):
    Q = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    K = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    V = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    dO = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)
    return Q, K, V, dO


# --- WRAPPER FUNCTIONS ---
def run_pytorch_sdpa(Q, K, V, dO):
    # SDPA natively works with (B, N, D)
    out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    out.backward(dO)
    return out


def run_cuda_gemmmr(Q, K, V, dO):
    # GeMMMrAttention expects (N, D). Since B=1, we squeeze the batch dimension.
    # PyTorch's autograd automatically propagates gradients back through the squeeze.
    q_2d = Q.squeeze(0)
    k_2d = K.squeeze(0)
    v_2d = V.squeeze(0)
    do_2d = dO.squeeze(0)
    out = GeMMMrAttention.apply(q_2d, k_2d, v_2d)
    out.backward(do_2d)
    return out


def run_triton_flashreduce(Q, K, V, dO):
    # flashreduce_kernel expects (N, D). Since B=1, we squeeze the batch dimension.
    q_2d = Q.squeeze(0)
    k_2d = K.squeeze(0)
    v_2d = V.squeeze(0)
    do_2d = dO.squeeze(0)
    (out,) = flashreduce_kernel(q_2d, k_2d, v_2d)
    out.backward(do_2d)
    return out


def benchmark_function(func, Q, K, V, dO, num_iters=NUM_ITERS):
    # WARMUP
    for _ in range(3):
        func(Q, K, V, dO)
        Q.grad, K.grad, V.grad = None, None, None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # RUN AND TIME
    start_event.record()
    for _ in range(num_iters):
        func(Q, K, V, dO)
        Q.grad, K.grad, V.grad = None, None, None
    end_event.record()

    torch.cuda.synchronize()

    avg_time_ms = start_event.elapsed_time(end_event) / num_iters
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return avg_time_ms, peak_mem_mb


if __name__ == "__main__":
    print(
        "| Seq Len | C++ Time (ms) | Triton Time (ms) | SDPA Time (ms) | C++ Mem (MB) | Triton Mem (MB) | SDPA Mem (MB) |"
    )
    print("|---|---|---|---|---|---|---|")

    for N in SEQ_LENGTHS:
        results = {
            "cuda": ("OOM", "OOM"),
            "triton": ("OOM", "OOM"),
            "sdpa": ("OOM", "OOM"),
        }

        # 1. Native SDPA Benchmark
        try:
            Q, K, V, dO = get_dummy_data(N)
            t, m = benchmark_function(run_pytorch_sdpa, Q, K, V, dO)
            results["sdpa"] = (f"{t:.2f}", f"{m:.2f}")
        except Exception as e:
            pass
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        # 2. C++ GeMMMapReduce Benchmark
        # 2. C++ GeMMMapReduce Benchmark
        # Temporarily removing try/except to see the real error!
        Q, K, V, dO = get_dummy_data(N)
        t, m = benchmark_function(run_cuda_gemmmr, Q, K, V, dO)
        results["cuda"] = (f"{t:.2f}", f"{m:.2f}")
        torch.cuda.empty_cache()
        # try:
        #     Q, K, V, dO = get_dummy_data(N)
        #     t, m = benchmark_function(run_cuda_gemmmr, Q, K, V, dO)
        #     results["cuda"] = (f"{t:.2f}", f"{m:.2f}")
        # except Exception as e:
        #     pass
        # finally:
        #     torch.cuda.empty_cache()
        #     gc.collect()

        # 3. Triton Flashreduce Benchmark
        try:
            Q, K, V, dO = get_dummy_data(N)
            t, m = benchmark_function(run_triton_flashreduce, Q, K, V, dO)
            results["triton"] = (f"{t:.2f}", f"{m:.2f}")
        except Exception as e:
            pass
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        print(
            f"| {N} | {results['cuda'][0]} | {results['triton'][0]} | {results['sdpa'][0]} | {results['cuda'][1]} | {results['triton'][1]} | {results['sdpa'][1]} |"
        )
