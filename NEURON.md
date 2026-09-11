# Neuron pair codec in llama.cpp

A fork of llama.cpp carrying a weight and KV-cache format from the NeuronAI project.
Branch `neuron-codec`, rebased against `upstream/master`.

## What it is

A pair of weights `(a, b)` is stored as a magnitude and a direction:

    r     = sqrt(a^2 + b^2)   on a LOG ladder, 2^m levels
    theta = atan2(b, a)       on a uniform circle, 2^k levels, k = m+1

Log space for the magnitude is the whole point and is **not** interchangeable with a
linear scale. It gives constant RELATIVE error, so a small weight takes a proportionally
small error rather than the same absolute one as a large one. Measured against groupwise
affine at matched bits, that is worth **39% less perplexity damage** — and it is invisible
to a Frobenius norm, which is dominated by the largest weights. Do not "simplify" it back
to a linear scale on the strength of a weight-error number.

The ladder is piecewise linear through three fp16 anchors per block — `lo`, `mid`, `hi` —
in log space. `mid` lets it BEND: `lo` and `hi` are min and max, both tail statistics, so
one outlier otherwise spends every level covering ground the block never occupies.

## Layouts

`k = m+1` always, so `k + m = 2m+1` bits per pair — odd by construction. A pair never
fills whole bytes but a block always does, and `vec_dot` unpacks bits as Q5_K does.
Embeddings take the layout one notch finer than the weights.

| type | bits/pair | block bytes | bits/value |
|---|---|---|---|
| `NEURON_M1` | 3 | 30 | 1.875 |
| `NEURON_M3` | 7 | 62 | 3.875 |
| `NEURON_M5` | 11 | 94 | 5.875 |
| `NEURON_M7` | 15 | 126 | 7.875 |
| `NEURON_M8` | 17 | 158 | 9.875 |

`QK_NEURON = 128` values (64 pairs). Two constraints fix it: ggml blocks a ROW and needs
`ne[0] % blck_size == 0` (rows of 2560 and 9728 exist, ruling out 1024), and a KV cache
type must divide `n_embd_head_k` — 128 — because a block spanning several attention heads
would cover all their magnitude scales with one set of anchors. 512 passed the first test
and was rejected by the second.

## Usage

    # convert and quantise
    python convert_hf_to_gguf.py <hf-model> --outfile model-f16.gguf --outtype f16
    ./build/bin/llama-quantize model-f16.gguf model-neuron.gguf NEURON_M5 8

    # serve, with a compressed KV cache
    ./build/bin/llama-server -m model-neuron.gguf -fa 1 -ctk neuron_m5 -ctv neuron_m5

## Measured — Qwen3-4B-Instruct-2507, M4, Metal

Weights:

| model | size | pp128 | tg32 |
|---|---|---|---|
| F16 | 8.22 GiB | 335.6 t/s | 11.92 t/s |
| NEURON_M5 | **3.05 GiB** | 253.7 t/s | **16.36 t/s** |

1.37x faster generation at 2.7x smaller; prompt processing reaches 76% of f16.

KV cache, ctx 8192 (re-measured after the corruption fix below):

| type | bits/val | size | vs f16 | tg64 |
|---|---|---|---|---|
| f16 | 16.000 | 1.12 GB | 1.00x | 16.27 |
| q8_0 | 8.500 | 0.60 GB | 1.88x | 15.86 |
| **neuron_m5** | 5.875 | **0.41 GB** | **2.72x** | **15.83** |
| neuron_m3 | 3.875 | 0.27 GB | 4.13x | — |

**Correction.** Commits `82d2840a0` and `adfccec00` quote KV figures — "2.4x slower",
then "1.1% speed cost" — that were measured on a CORRUPT cache, before `3facddf8f`. The
real cost is **2.7%**. They were taken through `llama-bench`, which reports throughput and
never inspects output, so a model emitting nonsense benchmarks perfectly happily. The
weight numbers are unaffected: weights never pass through `set_rows`, and both the CPU and
Metal paths were verified to produce correct text.

## Integration points

A new quantised type touches more than the type-traits table. Each of these aborts or
corrupts rather than failing gracefully:

| where | what |
|---|---|
| `ggml.h` | enum slot |
| `ggml-common.h` | block struct + `static_assert` on its size |
| `ggml.c` | traits entry, and a case in `ggml_quantize_chunk` returning `nrows*row_size` |
| `ggml-quants.c` | reference quantise/dequantise, and a `ggml_validate_row_data` case |
| `ggml-cpu/quants.c` | `vec_dot` and a `from_float` wrapper |
| `ggml-cpu/ggml-cpu.c` | CPU traits entry |
| `ggml-cpu/ops.cpp` | join the type to the quantised group in `get_rows`, or embedding lookup aborts |
| `ggml-metal/kernels/mul_mv.metal` | GEMV kernel — note the kernels take an `args_t` TEMPLATE |
| `ggml-metal/kernels/mul_mm.metal` | GEMM instantiation |
| `ggml-metal/kernels/dequantize.h` | 4x4 tile dequantiser, shared by mul_mm, get_rows and FA |
| `ggml-metal/kernels/quantize.h/.metal` | KV encoder + `set_rows`/`get_rows` |
| `ggml-metal/kernels/fa.metal` | flash-attention conversion kernel |
| `ggml-metal-device.cpp` | both mul_mv `nsg`/`nr0` switches |
| `ggml-metal-device.m` | `supports_op` for SET_ROWS **and** FLASH_ATTN_EXT |
| `ggml-metal-ops.cpp` | `use_kv_f16` — must be unconditional for types with no direct FA kernel |
| `llama.h`, `llama-quant.cpp`, `quantize.cpp` | ftype plumbing |
| `common/arg.cpp` **and** `llama-bench.cpp` | KV type lists, parsed separately |

## Traps found the hard way

**`kernel_set_rows_q32` hardcodes a 32-value source stride.** The `32` is the block size,
not `f32`. Every quantised KV type upstream ships has a 32-value block. With 128, every
block after the first encodes the wrong floats and the cache is silently garbage. Use the
`QK`-templated `kernel_set_rows_q`, as TQ2_0 does.

**`use_kv_f16` returns false below `ne[1] < 32`.** For k-quants that picks the faster of
two working routes. For a type with only the conversion route it is a cliff — the direct
`kernel_flash_attn_ext_<type>_dk*_dv*` does not exist and the pipeline resolves nil.

**A rejected `supports_op` for FLASH_ATTN_EXT disables FA for the whole model**, not just
the cache, and shows up as a large unexplained slowdown rather than an error.

**mul_mv lane striding must not assume the block size.** Striding within a block
(`g = tiisg*8; g < QK_NEURON/2`) left 24 of 32 lanes idle when the block shrank to 128.
Stride flat over the row's groups.

## Not done

- **CPU `vec_dot` is ~12x behind q5_K** and does not vectorise: NEON has no gather, and
  the angle table is 256 bytes at k=6, far past `vtbl`. The GPU gathers natively. The CPU
  path is a correctness fallback.
- `llama_model_ftype_name` has no entries, so models report "unknown, may not work".
- No CUDA/Vulkan kernels. Metal and scalar CPU only.
- `m1`/`m2` exceed `test-quantize-fns` thresholds calibrated for 4-bit types and need
  mapping to the low-bit constants, as Q2_K and the ternary types are.
