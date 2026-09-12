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

`k = m + 1.5` — half a bit more angle than magnitude. A fractional bit needs a
non-power-of-two alphabet, so the angles of TWO pairs are coded jointly in `2m+3` bits:
`2^(2m+3) = (2^(m+1.5))^2`, exactly `m+1.5` bits each. Those states are the checkerboard
sublattice of an `A x A` grid, `A = 2^(m+2)`, keeping `i+j` even. Its second moment matches
a uniform `2^(m+1.5)`-level quantiser, so the angle floor is `1/sqrt(2)` below what `k = m+1`
reaches, for a flat +0.25 bits/value.

    pack     v = (i << (m+1)) | (j >> 1)
    unpack   i = v >> (m+1),  j = 2*(v & (2^(m+1) - 1)) + (i & 1)

The group is 16 pairs: `16*m` bits of magnitude = `2m` bytes, then `8*(2m+3)` bits of angle
= `2m+3` bytes. Neither field straddles the other, and the group is `4m+3` bytes — always
ODD, so `gcd(width, 8) = 1` survives at every level. That property is why `k = m+1` was
mandatory before, and it is preserved.

| type | bits/pair | group bytes | block bytes | bits/value |
|---|---|---|---|---|
| `NEURON_M1` | 3.5 | 7 | 34 | 2.125 |
| `NEURON_M2` | 5.5 | 11 | 50 | 3.125 |
| `NEURON_M3` | 7.5 | 15 | 66 | 4.125 |
| `NEURON_M4` | 9.5 | 19 | 82 | 5.125 |
| `NEURON_M5` | 11.5 | 23 | 98 | 6.125 |
| `NEURON_M6` | 13.5 | 27 | 114 | 7.125 |
| `NEURON_M7` | 15.5 | 31 | 130 | 8.125 |
| `NEURON_M8` | 17.5 | 35 | 146 | 9.125 |

`QK_NEURON = 128` values (64 pairs) = 4 groups. Two constraints fix it: ggml blocks a ROW
and needs `ne[0] % blck_size == 0` (rows of 2560 and 9728 exist, ruling out 1024), and a KV
cache type must divide `n_embd_head_k` — 128 — because a block spanning several attention
heads would cover all their magnitude scales with one set of anchors. 512 passed the first
test and was rejected by the second.

A tensor whose `ne[0]` does not divide 128 cannot carry the codec at all — the codec pairs
adjacent values, which is what ties it to the block. Those fall back per
`tensor_type_fallback`: m1-m3 to Q4_0, m4 to Q5_0, m5 to Q5_1, m6-m8 to Q8_0, and to F16 if
the 32-block type does not fit either. Qwen3.5's vision `ffn_down` is 4304 wide and takes
the F16 path.

### Encoder

Round-to-nearest picks the pair of best codes; the best PAIR of codes is a different thing,
because a slightly worse angle can admit a much better magnitude. `neuron_refit_block`
alternates:

- **E** — reassign both codes together over 2 candidate angles x 3 candidate levels, under
  the checkerboard parity constraint that couples each pair to its neighbour.
- **M1** — per-block least-squares scale, folded into the stored anchors. The
  reconstruction is biased outward because `E[cos(angle error)] < 1`.

Objective is plain Frobenius; weighting by `r^-2` was measured at 27.06 -> 30.19 perplexity
at 4 bits. ENCODER-ONLY — the format, both decoders and every kernel are untouched. Its
value scales with how much damage there is to undo: -1.7% perplexity at m3, 0.0% at m5.

## Usage

    # convert and quantise
    python convert_hf_to_gguf.py <hf-model> --outfile model-f16.gguf --outtype f16
    ./build/bin/llama-quantize model-f16.gguf model-neuron.gguf NEURON_M5 8

    # serve, with a compressed KV cache
    ./build/bin/llama-server -m model-neuron.gguf -fa 1 -ctk neuron_m5 -ctv neuron_m5

## Measured — Qwen3-4B-Instruct-2507, M4, Metal

*Taken at `k = m+1`, before the format change. Sizes and rates differ from the table above.*

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

## Measured — Qwen3.5-9B, M4, Metal

Weights and vision tower both through the codec. Perplexity on a fixed 311-chunk corpus at
`n_ctx=512`; only ratios within this table mean anything, absolute values are corpus- and
model-specific.

| build | weights | vision | total | BPW | PPL | vs Q5_K_M |
|---|---|---|---|---|---|---|
| `NEURON_M3` | 4.72 GiB | 0.42 GiB | **5.14 GiB** | 4.40 | 8.9095 | +4.12% |
| Q5_K_M | 6.19 GiB | 0.50 GiB | 6.69 GiB | 5.77 | 8.5569 | — |
| `NEURON_M5` | 6.62 GiB | 0.49 GiB | 7.11 GiB | 6.18 | **8.4592** | **-1.14%** |
| f16 | 17.14 GiB | 0.86 GiB | 18.00 GiB | 16.00 | — | — |

**Damage falls steeply with model size.** The same comparison against Q5_K_M at two scales:

| | 4B | 9B |
|---|---|---|
| m3 | +10.93% | +4.08% |
| m5 | +0.99% | **-1.21%** |

m5 crosses from losing to winning; m3's penalty falls by 63%. At 9B the codec has a
position on both sides of Q5_K_M — 23% smaller and 4% worse, or 6% larger and 1.2% better —
where at 4B it only had the worse half.

Throughput at 9B, steady state (b>=8), matched back-to-back sweeps:

| build | tg64 | pp512 @b=128 |
|---|---|---|
| Q5_K_M | 10.8 | 110.66 |
| `NEURON_M5` | 10.3 | 89.95 |

m5 is 4.6% behind on generation and 18.7% on prompt while carrying 7% more bytes on a
bandwidth-bound kernel — slower is the physically expected result at this size ratio.

The vision tower compresses to 9.24 BPW at m5 and 7.84 at m3, not to the weights' rates:
its 4304-wide `ffn_down` tensors divide by no usable block size and stay f16 at every
level, so ~30% of it is fixed cost. Q5_K_M's own projector lands at 9.39 BPW — marginally
larger than the codec's — because 1152-wide `attn_qkv` misses its 256 block too.

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
- **Prompt processing trails `k = m+1`** at m5 — 162.56 against 196.38 t/s at 4B b=128.
  Halving the tile dequantiser's duplicate angle reads recovered 7% of it; the rest has no
  verified cause. m3 is within 4% because it stages a joint table the larger layouts cannot.
- The joint angle table only fits for m <= 3 (4 KB); at m5 it would be 64 KB and at m8 4 MB,
  because an entry holds trig for two INDEPENDENT indices and one load needs the full cross
  product. m4+ stage the grid instead and pay two loads per pair-couple.
- **Perplexity and task completion can disagree.** At 9B m5 beats Q5_K_M on perplexity, then
  failed an OCR transcription Q5_K_M passed — and succeeded on retry with nothing changed.
  Reasoning length varied 10.9x between two identical requests, so single-sample latency or
  reliability comparisons of a thinking model measure almost nothing.
- No CUDA/Vulkan kernels. Metal and scalar CPU only.
- `m1`/`m2` exceed `test-quantize-fns` thresholds calibrated for 4-bit types and need
  mapping to the low-bit constants, as Q2_K and the ternary types are.
