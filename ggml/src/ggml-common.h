#ifndef GGML_COMMON_DECL

#if defined(GGML_COMMON_DECL_C)
#include <stdint.h>

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_CPP)
#include <cstdint>

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

// std-c++ allow anonymous unions but some compiler warn on it
#define GGML_COMMON_AGGR_U data
// std-c++ do not allow it.
#define GGML_COMMON_AGGR_S data

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_METAL)
#include <metal_stdlib>

typedef half  ggml_half;
typedef half2 ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_CUDA)
#if defined(GGML_COMMON_DECL_MUSA)
#include <musa_fp16.h>
#else
#include <cuda_fp16.h>
#endif
#include <cstdint>

typedef half  ggml_half;
typedef half2 ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S data

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_HIP)
#include <hip/hip_fp16.h>
#include <cstdint>

typedef half  ggml_half;
typedef half2 ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S data

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_SYCL)
#include <sycl/half_type.hpp>
#include <cstdint>

typedef sycl::half  ggml_half;
typedef sycl::half2 ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S data

#define GGML_COMMON_DECL
#endif

#if defined(GGML_COMMON_DECL)

#ifndef __cplusplus
#ifndef static_assert
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
#define static_assert(cond, msg) _Static_assert(cond, msg)
#else
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif
#endif
#endif // __cplusplus

// QK = number of values after dequantization
// QK_K = super-block size

#define QK_K 256
#define K_SCALE_SIZE 12

#if defined(GGML_COMMON_DECL_CUDA) || defined(GGML_COMMON_DECL_HIP) || defined(GGML_COMMON_DECL_SYCL)
// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QI1_0 (QK1_0 / 32)
#define QR1_0 1

#define QI2_0 (QK2_0 / 32)
#define QR2_0 1


#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QR4_0 2

#define QI4_1 (QK4_1 / (4 * QR4_1))
#define QR4_1 2

#define QI_MXFP4 (QK_MXFP4 / (4 * QR_MXFP4))
#define QR_MXFP4 2

#define QI_NVFP4 (QK_NVFP4 / (4 * QR_NVFP4))
#define QR_NVFP4 2

#define QI5_0 (QK5_0 / (4 * QR5_0))
#define QR5_0 2

#define QI5_1 (QK5_1 / (4 * QR5_1))
#define QR5_1 2

#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_0 1

#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define QI2_K (QK_K / (4*QR2_K))
#define QR2_K 4

#define QI3_K (QK_K / (4*QR3_K))
#define QR3_K 4

#define QI4_K (QK_K / (4*QR4_K))
#define QR4_K 2

#define QI5_K (QK_K / (4*QR5_K))
#define QR5_K 2

#define QI6_K (QK_K / (4*QR6_K))
#define QR6_K 2

#define QI2_XXS (QK_K / (4*QR2_XXS))
#define QR2_XXS 4

#define QI2_XS (QK_K / (4*QR2_XS))
#define QR2_XS 4

#define QI2_S (QK_K / (4*QR2_S))
#define QR2_S 4

#define QI3_XXS (QK_K / (4*QR3_XXS))
#define QR3_XXS 4

#define QI3_XS (QK_K / (4*QR3_XS))
#define QR3_XS 4

#define QI1_S (QK_K / (4*QR1_S))
#define QR1_S 8

#define QI1_M (QK_K / (4*QR1_M))
#define QR1_M 8

#define QI4_NL (QK4_NL / (4*QR4_NL))
#define QR4_NL 2

#define QI4_XS (QK_K / (4*QR4_XS))
#define QR4_XS 2

#define QI3_S (QK_K / (4*QR3_S))
#define QR3_S 4

#endif // GGML_COMMON_DECL_CUDA || GGML_COMMON_DECL_HIP

#ifdef _MSC_VER
#define GGML_EXTENSION
#else // _MSC_VER
#define GGML_EXTENSION __extension__
#endif // _MSC_VER

#define QK1_0 128
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK1_0 / 8]; // bits / quants
} block_q1_0;
static_assert(sizeof(block_q1_0) == sizeof(ggml_half) + QK1_0 / 8, "wrong q1_0 block size/padding");

#define QK2_0 64
typedef struct {
    ggml_half d;              // delta (scale)
    uint8_t qs[QK2_0 / 4];   // 2 bits per element
} block_q2_0;
static_assert(sizeof(block_q2_0) == sizeof(ggml_half) + QK2_0 / 4, "wrong q2_0 block size/padding");

// Neuron pair codec ---------------------------------------------------------------------
//
// 128 values (64 pairs) per block.
//
// Two constraints, and 128 is the largest size meeting both. ggml blocks a ROW and needs
// ne[0] % blck_size == 0, which rules out 1024 (rows of 2560 and 9728 exist). And a KV
// cache type must divide n_embd_head_k -- 128 on every model here -- because a block that
// spans several attention heads would have to cover all their magnitude scales with one
// set of anchors, which is precisely the failure per-block ranges exist to prevent.
//
// 512 worked for weights and was rejected by the KV cache for exactly that reason. Going
// to 128 costs 0.375 bits/value in anchors instead of 0.094, about 5% on a file, and buys
// one block size that serves weights and KV alike -- plus a tighter fit per block, since
// each set of anchors now covers a quarter as many values.
//
// Three fp16 anchors define a piecewise-linear ladder in LOG magnitude through lo, mid and
// hi. `mid` is what lets the ladder BEND rather than only slide and stretch: lo and hi are
// min and max, both tail statistics, and one outlier in a block otherwise spends every
// level covering ground the rest of the block never occupies.
//
// Codes are joint (k+m)-bit values, angle in the high bits and magnitude in the low:
//     code = acode * 2^m + mcode
// packed little-endian across the block, as the odd-width k-quants do.
#define QK_NEURON 128

// k = m + 1.5 at every level. A fractional bit needs a non-power-of-two alphabet, so the
// angles of TWO pairs are coded jointly in 2m+3 bits -- 2^(2m+3) = (2^(m+1.5))^2, i.e.
// exactly m+1.5 bits each. Those states are the checkerboard sublattice of an A x A grid,
// A = 2^(m+2), keeping i+j even: A^2/2 = 2^(2m+3). Its second moment matches a uniform
// 2^(m+1.5)-level quantiser, so the angle floor is pi/(2^(m+1.5) * sqrt(3)) -- a factor
// 1/sqrt(2) below what k = m+1 can reach, for a flat +0.25 bits/weight.
//
//   pack     v = (i << (m+1)) | (j >> 1)
//   unpack   i = v >> (m+1),  j = 2*(v & (2^(m+1) - 1)) + (i & 1)
//
// 16 pairs is the natural group: 16 magnitudes * m bits = 2m bytes, then 8 joint angles *
// (2m+3) bits = 2m+3 bytes. Neither field ever straddles the other and the group is
// 4m+3 bytes -- always ODD, so gcd(width, 8) = 1 survives at every level, which is the
// property that made k = m+1 mandatory in the first place.
// NEURON_KX is 2*(k - m): the angle offset in half-bits, and the single knob for the
// magnitude/angle split. It must be ODD, because a half-integer k is what the two-pair
// checkerboard codes; KX = 3 is k = m+1.5, KX = 1 is k = m+0.5, KX = -1 puts the angle
// BELOW the magnitude. Everything below derives from it, so trying a different split is a
// one-number change rather than an edit across five files.
//
//   k   = m + KX/2
//   A   = 2^(k + 0.5)          grid before the checkerboard halving
//   AW  = 2k                   joint code width over two pairs
//   GBY = 4m + KX              bytes per 16-pair group (2m magnitude + 2m+KX angle)
#define NEURON_KX        1
#define NEURON_GRP       16                     /* pairs per group                      */
#define NEURON_AK(LP)    ((LP) + (NEURON_KX + 1)/2)
#define NEURON_GBY(LP)   (4*(LP) + NEURON_KX)   /* bytes per group                      */
#define NEURON_AOFF(LP)  (2*(LP))               /* angle field offset within the group  */
#define NEURON_AW(LP)    (2*(LP) + NEURON_KX)   /* joint angle code width, in bits      */
#define NEURON_ANG(LP)   (1 << NEURON_AK(LP))   /* grid A, before the checkerboard halving */
#define NEURON_JSH(LP)   (NEURON_AK(LP) - 1)    /* unpack shift                         */
#define NEURON_ANGTAB    1024                   /* constant trig table: the m8 grid     */

#define NEURON_BLOCK(LP)                                                          \
    typedef struct {                                                              \
        ggml_half lo;                                                             \
        ggml_half mid;                                                            \
        ggml_half hi;                                                             \
        uint8_t   qs[(QK_NEURON / 32) * NEURON_GBY(LP)];                          \
    } block_neuron_m##LP;                                                         \
    static_assert(sizeof(block_neuron_m##LP) ==                                   \
                  3 * sizeof(ggml_half) + (QK_NEURON / 32) * NEURON_GBY(LP),      \
                  "wrong block_neuron_m" #LP " size/padding");

NEURON_BLOCK(1)
NEURON_BLOCK(2)
NEURON_BLOCK(3)
NEURON_BLOCK(4)
NEURON_BLOCK(5)
NEURON_BLOCK(6)
NEURON_BLOCK(7)
NEURON_BLOCK(8)

// ---------------------------------------------------------------------------------------
// neuron_v5 -- the d=2 vector-quantised geometry, 5.125 bits/value.
//
// The polar family codes a pair as (r, theta). Measured against a fitted 2-D codebook at
// matched rate (experiments/d_sweep.json) that grid trails by 0.28-0.43 bits, and trails
// a plain d=1 Lloyd scalar quantiser by 0.23-0.28, because theta is uniform to 0.000188
// bits/pair -- half the payload buys an incompressible quantity at full price. So v5 drops
// the parameterisation and stores a literal index into a codebook of 1024 points in the
// pair plane.
//
// One code per pair, 10 bits, 4 codes per 5 bytes: code t sits at bit 10*t, so its shift
// is always even and it never straddles more than two bytes.
//   c = (qs[b] | qs[b+1] << 8) >> sh & 0x3FF     b = 10t >> 3, sh = 10t & 7
//
// One fp16 absmax-seeded scale per block replaces lo/mid/hi: 0.125 bits/value of side
// information instead of 0.375, which is a quarter of the total advantage on its own. The
// bent log ladder needed three anchors; a codebook needs one number.
//
// The codebook is a compile-time constant (kNeuronVQ5, below) because a ggml type has no
// per-tensor side channel. Fitting it per tensor instead would buy 0.009 bits
// (experiments/global_codebook.json) -- after per-block normalisation the pair
// distribution is very nearly universal, so one table serves every tensor and every model.
// B is bits per PAIR, so bits/value is B/2 + 0.125 for the scale, and the suffix is that
// rounded down:
//
//   v4   B=8    K=256     4.125 bpv    one byte per pair -- no bit extraction at all
//   v5   B=10   K=1024    5.125 bpv
//   v6   B=12   K=4096    6.125 bpv
//
// Code p sits at bit B*p, so for B=10 and B=12 the shift is always even and a code spans
// exactly two bytes, never three. For B=8 it is a plain byte index and the second read
// must not happen -- qs is only QK_NEURON/2 bytes wide.
#define NEURON_VQ4_BITS  8
#define NEURON_VQ4_K     256
#define NEURON_VQ5_BITS  10
#define NEURON_VQ5_K     1024
#define NEURON_VQ6_BITS  12
#define NEURON_VQ6_K     4096

// One fp16 scale across 128 values is coarse: it is set by whichever outlier is largest, so
// every quiet region of the block is coded against a scale far too big for it. Each block
// therefore carries a 4-bit multiplier per 16-value sub-block, indexing kNeuronVQSB.
//
// Granularity is priced, not copied. MSE with per-sub-block scales over MSE with one scale
// is mean_s (a_s/A)^2, so the error ratio is its square root at ~6.02 dB/bit, against one
// 4-bit index per sub-block (experiments/fit_subscale.py):
//
//     sub-block   bits/val   err ratio   gain      NET
//       8 values     0.500      0.6579   +0.604   +0.104
//      16 values     0.250      0.7539   +0.408   +0.158   <- chosen
//      32 values     0.125      0.8430   +0.246   +0.121   (Q_K's granularity)
//      64 values     0.062      0.9250   +0.112   +0.050
//
// And 0.25 bits spent here buys 0.408 bits of error reduction where the same 0.25 spent on
// a larger codebook buys ~0.25, the d=2 curve running at ~5.7 dB/bit -- about 1.6x more
// bit-efficient, which is the actual argument for spending bits here rather than there.
#define NEURON_VQ_SBV   16                              /* values per sub-block         */
#define NEURON_VQ_NSB   (QK_NEURON / NEURON_VQ_SBV)     /* sub-blocks per block: 8      */

// How finely the sub-block scale must be resolved depends on the codebook it is paired
// with, so the index width is per type rather than family-wide. Measured share of squared
// error carried by the 4-bit grid's 1.761% relative error:
//
//     v4   codebook 7.861%   grid  4.8% of error energy   -> a finer index is net NEGATIVE
//     v5   codebook 3.356%   grid 21.6%                   -> net +0.038 bits, not worth it
//     v6   codebook 1.645%   grid 53.4%                   -> the grid is the binding limit
//
// For v6 a 6-bit index drops the grid to 0.503% and the total from 2.410% to 1.720%,
// -28.6% error for +0.125 bits/value: net +0.375 bits. 8 bits only reaches 1.649% because
// the codebook becomes the floor again, so 6 is where it saturates.
#define NEURON_VQ4_SBB  4
#define NEURON_VQ5_SBB  4
#define NEURON_VQ6_SBB  6
#define NEURON_VQ4_SBT  kNeuronVQSB                     /* 16 multipliers               */
#define NEURON_VQ5_SBT  kNeuronVQSB
#define NEURON_VQ6_SBT  kNeuronVQSB64                   /* 64 multipliers               */
#define NEURON_VQ_SBBY(SFX_BITS) (NEURON_VQ_NSB * (SFX_BITS) / 8)

#define NEURON_V_BLOCK(SFX)                                                       \
    typedef struct {                                                              \
        ggml_half d;                                 /* block scale            */ \
        uint8_t   sb[NEURON_VQ_SBBY(NEURON_VQ##SFX##_SBB)];                       \
        uint8_t   qs[(QK_NEURON / 2) * NEURON_VQ##SFX##_BITS / 8];                \
    } block_neuron_v##SFX;                                                        \
    static_assert(sizeof(block_neuron_v##SFX) == sizeof(ggml_half) +              \
                  NEURON_VQ_SBBY(NEURON_VQ##SFX##_SBB) +                          \
                  (QK_NEURON / 2) * NEURON_VQ##SFX##_BITS / 8,                    \
                  "wrong block_neuron_v" #SFX " size/padding");
// Pair p lives in sub-block s = p / (NEURON_VQ_SBV/2) = p/8, whose index sits at bit s*SBB
// of sb[]:  v = sb[b] | (sb[b+1] << 8) if (bit&7)+SBB > 8;  idx = (v >> (bit&7)) & (2^SBB-1).
// The guarded second byte matters: at SBB=4 it is never needed and reading it would run off
// a 4-byte sb[], while at SBB=6 it is needed for sub-blocks 1,2,5,6 and the highest byte
// touched is 5 of 6. Spelled out at each use rather than shared as a helper, because this
// header is also compiled as Metal, where a pointer parameter needs an address-space
// qualifier and a generic one will not compile.

NEURON_V_BLOCK(4)
NEURON_V_BLOCK(5)
NEURON_V_BLOCK(6)

#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_half) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK_MXFP4 32
typedef struct {
    uint8_t e; // E8M0
    uint8_t qs[QK_MXFP4/2];
} block_mxfp4;
static_assert(sizeof(block_mxfp4) == sizeof(uint8_t) + QK_MXFP4/2, "wrong mxfp4 block size/padding");

#define QK_NVFP4 64
#define QK_NVFP4_SUB 16  // sub-block size for per-group scales
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)
    uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values (32 bytes)
} block_nvfp4;
static_assert(sizeof(block_nvfp4) == sizeof(uint8_t)*(QK_NVFP4/QK_NVFP4_SUB) + QK_NVFP4/2, "wrong nvfp4 block size/padding");

#define QK5_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_half) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_half) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d; // delta
            ggml_half s; // d * sum(qs[i])
        } GGML_COMMON_AGGR_S;
        ggml_half2 ds;
    } GGML_COMMON_AGGR_U;
    int8_t qs[QK8_1]; // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1, "wrong q8_1 block size/padding");

//
// Ternary quantization
//

// 1.6875 bpw
typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // 5 elements per byte (3^5 = 243 < 256)
    uint8_t qh[QK_K/64]; // 4 elements per byte
    ggml_half d;
} block_tq1_0;
static_assert(sizeof(block_tq1_0) == sizeof(ggml_half) + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5, "wrong tq1_0 block size/padding");

// 2.0625 bpw
typedef struct {
    uint8_t qs[QK_K/4]; // 2 bits per element
    ggml_half d;
} block_tq2_0;
static_assert(sizeof(block_tq2_0) == sizeof(ggml_half) + QK_K / 4, "wrong tq2_0 block size/padding");

//
// Super-block quantization structures
//

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_half) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight
typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");

// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");

// This is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");

// (Almost) "true" 2-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 2.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(ggml_half) + QK_K/8*sizeof(uint16_t), "wrong iq2_xxs block size/padding");

// 2.3125 bpw quants
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
} block_iq2_xs;
static_assert(sizeof(block_iq2_xs) == sizeof(ggml_half) + QK_K/8*sizeof(uint16_t) + QK_K/32, "wrong iq2_xs block size/padding");

// 2.5625 bpw quants
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t scales[QK_K/32];
} block_iq2_s;
static_assert(sizeof(block_iq2_s) == sizeof(ggml_half) + QK_K/4 + QK_K/16, "wrong iq2_s block size/padding");

// (Almost) "true" 3-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 3.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    ggml_half d;
    uint8_t qs[3*QK_K/8];
} block_iq3_xxs;
static_assert(sizeof(block_iq3_xxs) == sizeof(ggml_half) + 3*(QK_K/8), "wrong iq3_xxs block size/padding");

// 3.4375 bpw
#define IQ3S_N_SCALE QK_K/64
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
static_assert(sizeof(block_iq3_s) == sizeof(ggml_half) + 13*(QK_K/32) + IQ3S_N_SCALE, "wrong iq3_s block size/padding");

// 1.5625 bpw
typedef struct {
    ggml_half d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;
static_assert(sizeof(block_iq1_s) == sizeof(ggml_half) + QK_K/8 + QK_K/16, "wrong iq1_s block size/padding");

// 1.75 bpw
typedef struct {
    uint8_t  qs[QK_K/8];      // grid index, low 8 bits
    uint8_t  qh[QK_K/16];     // grid index, high 3 bits + grid shift bit (for two groups of 8)
    uint8_t  scales[QK_K/32]; // 3-bit block scales (4-bit if QK_K == 64)
} block_iq1_m;
static_assert(sizeof(block_iq1_m) == QK_K/8 + QK_K/16 + QK_K/32, "wrong iq1_m block size/padding");

// Used by IQ1_M quants
typedef union {
    ggml_half f16;
    uint16_t  u16;
} iq1m_scale_t;

// Non-linear quants
#define QK4_NL 32
typedef struct {
    ggml_half d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2, "wrong iq4_nl block size/padding");

typedef struct {
    ggml_half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;
static_assert(sizeof(block_iq4_xs) == sizeof(ggml_half) + sizeof(uint16_t) + QK_K/64 + QK_K/2, "wrong iq4_xs block size/padding");

#endif // GGML_COMMON_DECL
#endif // GGML_COMMON_DECL

////////////////////////////////////////////////////////////////////////////////

#ifndef GGML_COMMON_IMPL

#if defined(GGML_COMMON_IMPL_C)
#include <stdint.h>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };

#define GGML_COMMON_IMPL
#elif defined(GGML_COMMON_IMPL_CPP)
#include <cstdint>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };

#define GGML_COMMON_IMPL
#elif defined(GGML_COMMON_IMPL_METAL)
#include <metal_stdlib>

#define GGML_TABLE_BEGIN(type, name, size) static const constant type name[size] = {
#define GGML_TABLE_END() };

#define GGML_COMMON_IMPL
#elif defined(GGML_COMMON_IMPL_CUDA) || defined(GGML_COMMON_IMPL_HIP) || defined(GGML_COMMON_IMPL_MUSA)
#include <cstdint>

#define GGML_TABLE_BEGIN(type, name, size) static const __device__ type name[size] = {
#define GGML_TABLE_END() };

#define GGML_COMMON_IMPL
#elif defined(GGML_COMMON_IMPL_SYCL)

#include <cstdint>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };

#define GGML_COMMON_IMPL
#endif

#if defined(GGML_COMMON_IMPL)

// The 64-level variant, for v6. Its codebook resolves 1.645% while the 16-level grid
// carries 1.761%, so more than half of v6's squared error was the scale rather than the
// codeword. At 64 levels the grid drops to 0.503%. Fitted identically: level 0 pinned to
// 1.0, the rest 1-D Lloyd in log space on the same ratio distribution.
GGML_TABLE_BEGIN(float, kNeuronVQSB64, 64)
    +1.0000000f, +0.9880516f, +0.9707765f, +0.9546325f, +0.9394543f, +0.9254030f, +0.9122313f, +0.8996946f, +0.8878583f, +0.8763236f, +0.8652702f, +0.8547704f, +0.8451930f, +0.8359646f, +0.8270454f, +0.8182786f, +0.8098052f, +0.8013983f, +0.7932720f, +0.7852620f, +0.7774255f, +0.7698001f, +0.7621784f, +0.7545241f, +0.7473072f, +0.7403655f, +0.7333952f, +0.7262845f, +0.7192319f, +0.7122412f, +0.7052374f, +0.6983470f, +0.6914963f, +0.6846121f, +0.6776644f, +0.6707796f, +0.6638127f, +0.6568347f, +0.6498680f, +0.6428695f, +0.6358926f, +0.6288605f, +0.6216339f, +0.6143019f, +0.6066570f, +0.5990283f, +0.5911396f, +0.5831997f, +0.5751225f, +0.5667161f, +0.5577198f, +0.5480989f, +0.5378115f, +0.5264323f, +0.5134985f, +0.4983489f, +0.4799054f, +0.4568993f, +0.4278959f, +0.3914923f, +0.3436285f, +0.2768318f, +0.1803031f, +0.0746950f,
GGML_TABLE_END()

// Sub-block scale multipliers, ratios of sub-block absmax to block absmax. Level 0 is
// pinned to exactly 1.0 because by construction some sub-block attains the block
// absmax -- that level is forced by the definition, not chosen. The other 15 are 1-D
// Lloyd levels on the empirical ratio distribution, fitted in LOG space (constant
// relative resolution is what a positive quantity of unknown scale wants). Mean
// relative error of the grid: 1.62%. See experiments/fit_subscale.py.
GGML_TABLE_BEGIN(float, kNeuronVQSB, 16)
    +1.0000000f, +0.9686971f, +0.9174675f, +0.8703421f, +0.8274198f, +0.7865809f, +0.7469853f, +0.7069893f, +0.6659136f, +0.6228875f, +0.5759767f, +0.5239401f, +0.4643680f, +0.3922645f, +0.2931716f, +0.1260537f,
GGML_TABLE_END()

// The neuron_v6 codebook: 4096 points in the pair plane, same fit as kNeuronVQ4/5 --
// 800k absmax-normalised pairs with their sign orbit from all 252 layer tensors of
// Qwen3-4B, magnitude-weighted at alpha=1.5, seed 0, ordered by radius. Held out from
// Qwen3-0.6B. See experiments/fit_vq.py.
GGML_TABLE_BEGIN(float, kNeuronVQ6, 8192)
    +0.0017851f, +0.0038767f, -0.0035023f, -0.0043234f, +0.0055438f, -0.0036881f, -0.0074789f, +0.0027507f,
    +0.0105188f, +0.0061177f, -0.0061930f, +0.0109771f, -0.0001361f, -0.0135637f, -0.0131033f, -0.0063056f,
    +0.0103127f, -0.0111000f, +0.0029606f, +0.0155042f, +0.0176132f, -0.0019187f, -0.0177820f, +0.0040112f,
    -0.0111923f, -0.0167977f, +0.0104305f, -0.0222339f, -0.0177203f, +0.0170614f, -0.0086364f, +0.0234673f,
    +0.0150093f, +0.0203201f, +0.0231143f, +0.0113585f, -0.0019503f, -0.0257266f, +0.0238043f, -0.0123249f,
    -0.0283021f, -0.0051034f, +0.0033783f, +0.0287882f, -0.0273892f, +0.0108440f, -0.0246279f, -0.0173887f,
    +0.0305277f, +0.0029178f, -0.0169872f, -0.0285076f, +0.0222142f, -0.0272325f, +0.0290561f, +0.0241418f,
    -0.0080452f, +0.0370438f, +0.0104439f, -0.0364703f, -0.0054135f, -0.0378244f, -0.0200436f, +0.0329058f,
    +0.0149720f, +0.0362955f, -0.0387460f, +0.0069819f, +0.0388065f, -0.0073064f, -0.0314687f, +0.0258247f,
    +0.0345632f, -0.0222482f, +0.0407462f, +0.0121157f, -0.0428919f, -0.0076458f, -0.0386965f, -0.0220444f,
    +0.0020135f, +0.0468343f, -0.0319618f, -0.0344803f, -0.0178782f, -0.0442635f, +0.0275634f, -0.0397547f,
    -0.0450690f, +0.0195756f, +0.0030190f, -0.0490820f, +0.0312106f, +0.0399654f, -0.0297604f, +0.0422955f,
    +0.0446818f, +0.0286027f, +0.0531678f, -0.0004977f, +0.0505206f, -0.0170791f, -0.0154664f, +0.0512850f,
    -0.0535043f, +0.0069622f, +0.0440659f, -0.0337596f, +0.0206371f, +0.0515335f, +0.0207929f, -0.0530505f,
    -0.0453187f, +0.0353949f, -0.0559570f, -0.0167834f, -0.0087597f, -0.0588383f, +0.0575574f, +0.0153597f,
    -0.0325267f, -0.0501065f, -0.0498181f, -0.0346019f, +0.0098199f, +0.0609144f, -0.0632438f, -0.0041664f,
    -0.0591032f, +0.0233041f, +0.0080502f, -0.0635229f, -0.0054376f, +0.0639922f, +0.0397318f, -0.0505813f,
    -0.0311014f, +0.0563148f, +0.0489952f, +0.0443045f, +0.0618395f, -0.0266107f, +0.0673498f, -0.0096440f,
    -0.0216065f, -0.0647103f, +0.0373337f, +0.0589635f, -0.0487803f, -0.0500681f, +0.0697772f, +0.0062310f,
    -0.0471859f, +0.0523391f, +0.0580561f, -0.0419185f, +0.0641010f, +0.0320907f, -0.0205079f, +0.0688689f,
    -0.0711240f, +0.0126201f, -0.0607392f, +0.0404799f, -0.0662917f, -0.0318926f, +0.0238630f, +0.0697310f,
    +0.0354543f, -0.0653888f, +0.0236701f, -0.0724833f, -0.0063070f, -0.0763125f, -0.0752509f, -0.0155818f,
    -0.0387978f, -0.0665425f, +0.0078199f, +0.0771686f, -0.0378910f, +0.0702208f, +0.0531775f, -0.0595841f,
    +0.0100902f, -0.0797897f, +0.0784694f, -0.0215096f, -0.0662648f, -0.0479150f, -0.0818676f, +0.0004824f,
    +0.0663569f, +0.0489745f, +0.0798294f, +0.0207568f, -0.0078754f, +0.0823488f, -0.0764504f, +0.0319208f,
    -0.0215711f, -0.0813957f, +0.0842046f, -0.0009788f, +0.0553512f, +0.0636228f, +0.0757995f, -0.0383164f,
    -0.0552910f, +0.0654280f, -0.0577774f, -0.0643660f, -0.0835786f, -0.0296821f, -0.0692288f, +0.0560685f,
    +0.0219227f, +0.0871159f, +0.0708003f, -0.0553576f, +0.0390775f, +0.0809399f, -0.0232176f, +0.0868690f,
    -0.0887240f, +0.0179423f, +0.0482274f, -0.0767444f, -0.0362287f, -0.0850555f, +0.0830701f, +0.0414013f,
    +0.0932886f, -0.0142689f, +0.0028589f, -0.0949657f, -0.0940571f, -0.0154272f, +0.0219229f, -0.0929506f,
    -0.0829812f, +0.0476360f, -0.0848302f, -0.0455952f, -0.0383032f, +0.0886128f, +0.0378940f, -0.0892734f,
    -0.0529015f, -0.0816204f, +0.0960184f, +0.0157646f, +0.0061509f, +0.0976217f, -0.0524730f, +0.0827994f,
    -0.0147615f, -0.0969563f, +0.0661675f, -0.0725621f, +0.0928826f, -0.0320215f, +0.0752634f, +0.0635340f,
    +0.0548052f, +0.0821098f, -0.0754662f, -0.0638201f, -0.0997208f, +0.0027901f, +0.0883591f, -0.0502491f,
    -0.0704470f, +0.0744511f, -0.0973103f, +0.0338904f, -0.0109273f, +0.1031106f, +0.1037768f, -0.0003861f,
    +0.0988299f, +0.0357233f, +0.0289538f, +0.1012659f, -0.0303461f, -0.1025603f, -0.1027920f, -0.0330712f,
    +0.0710671f, +0.0814652f, +0.0564215f, -0.0925779f, -0.0307387f, +0.1046470f, -0.0715893f, -0.0830945f,
    +0.0838290f, -0.0708334f, +0.0935389f, +0.0581331f, -0.1084783f, +0.0194753f, -0.0867212f, +0.0686357f,
    -0.0473473f, -0.1002972f, +0.0468838f, +0.1005188f, -0.0943271f, -0.0599280f, +0.0338533f, -0.1066617f,
    +0.0003765f, -0.1119904f, +0.1106100f, -0.0178712f, -0.1120533f, -0.0098750f, -0.0995676f, +0.0532484f,
    -0.0656503f, +0.0922153f, +0.0158265f, -0.1125494f, -0.0507429f, +0.1028462f, +0.1091799f, -0.0351824f,
    -0.1048262f, -0.0475640f, +0.1157692f, +0.0094709f, +0.0170180f, +0.1150263f, -0.0888798f, -0.0751517f,
    +0.0752174f, -0.0888820f, +0.1134796f, +0.0277442f, +0.0999771f, -0.0629917f, +0.0907121f, +0.0760587f,
    -0.0175999f, -0.1170944f, -0.0035210f, +0.1193491f, -0.0660566f, -0.0995694f, +0.1090516f, -0.0494478f,
    +0.0656283f, +0.1002132f, +0.1095508f, +0.0508709f, -0.1151445f, +0.0399845f, -0.1194192f, -0.0249297f,
    -0.1217877f, +0.0071681f, -0.0827955f, +0.0899531f, +0.0502390f, -0.1121294f, +0.0373037f, +0.1176783f,
    -0.0214213f, +0.1226707f, -0.0400569f, -0.1190004f, +0.0832803f, +0.0946336f, +0.0691576f, -0.1069844f,
    +0.1269988f, -0.0115998f, -0.0414014f, +0.1208122f, -0.1086447f, +0.0678321f, -0.0914278f, -0.0909125f,
    -0.1055611f, -0.0747121f, -0.0699221f, +0.1088384f, -0.1270118f, +0.0258740f, +0.0930781f, -0.0902594f,
    -0.0584526f, -0.1157644f, +0.1263211f, -0.0301747f, +0.0342740f, -0.1253059f, +0.0592373f, +0.1160678f,
    -0.1158025f, -0.0603522f, +0.0026568f, -0.1309178f, -0.1028478f, +0.0817337f, +0.1051200f, -0.0791520f,
    -0.1241652f, -0.0436894f, +0.1106744f, +0.0723935f, +0.0196357f, -0.1308386f, +0.1310109f, +0.0205007f,
    -0.1326251f, -0.0093519f, +0.1268863f, +0.0411267f, -0.1212323f, +0.0558629f, +0.0093410f, +0.1332201f,
    +0.1336206f, +0.0030895f, -0.0272950f, -0.1317457f, +0.1011898f, +0.0895441f, -0.0842899f, -0.1059865f,
    +0.1191127f, -0.0652604f, +0.1279670f, -0.0468525f, -0.0596251f, +0.1234873f, +0.0268392f, +0.1346977f,
    +0.0890026f, -0.1055048f, -0.0985692f, +0.0976046f, -0.0117182f, -0.1383829f, +0.0493746f, +0.1309260f,
    +0.0835895f, +0.1129725f, +0.0493777f, -0.1316132f, +0.0651197f, -0.1246566f, -0.1377052f, -0.0296603f,
    -0.1408541f, +0.0089534f, +0.1278950f, +0.0601645f, -0.0094875f, +0.1417250f, -0.0309249f, +0.1389388f,
    -0.0892393f, +0.1110437f, -0.0764290f, -0.1208515f, -0.1122918f, -0.0914402f, +0.1426951f, -0.0253786f,
    -0.1383305f, +0.0450435f, -0.0439005f, -0.1392753f, -0.0598910f, -0.1338064f, -0.1245312f, -0.0778799f,
    +0.0722723f, +0.1281111f, -0.1252380f, +0.0775591f, -0.1340878f, -0.0611430f, +0.1441656f, +0.0324622f,
    +0.1028786f, +0.1070022f, +0.1209270f, -0.0864050f, -0.0508926f, +0.1399784f, -0.1461279f, +0.0295003f,
    +0.1098184f, -0.1011602f, +0.0838466f, -0.1238416f, +0.1496101f, -0.0084107f, -0.0800993f, +0.1266938f,
    +0.0244556f, -0.1483860f, +0.1503705f, +0.0123964f, -0.1044314f, -0.1089558f, -0.1506803f, -0.0146643f,
    +0.0087535f, +0.1512970f, +0.1186405f, +0.0944685f, +0.1405429f, -0.0587910f, +0.0405264f, +0.1470230f,
    -0.1186642f, +0.0960393f, -0.1461518f, -0.0469550f, +0.0071177f, -0.1533510f, -0.1389831f, +0.0654868f,
    +0.1485976f, -0.0408848f, +0.1312261f, +0.0811978f, +0.0408101f, -0.1490348f, -0.0302378f, -0.1516864f,
    +0.1352563f, -0.0752922f, +0.1481991f, +0.0509293f, -0.1083256f, +0.1134947f, +0.1020647f, -0.1195931f,
    -0.0953723f, -0.1251841f, +0.0602616f, +0.1455550f, +0.0595056f, -0.1463608f, -0.0094555f, -0.1579692f,
    -0.1583539f, -0.0008076f, +0.0266199f, +0.1570049f, -0.0782804f, -0.1391826f, -0.0703700f, +0.1434040f,
    -0.0245846f, +0.1584005f, +0.1453653f, +0.0682610f, +0.0759422f, -0.1417075f, -0.1598324f, +0.0181168f,
    +0.0900113f, +0.1336139f, -0.1582240f, -0.0319896f, -0.0412035f, +0.1564579f, -0.0067878f, +0.1626980f,
    +0.1050929f, +0.1246164f, -0.1227802f, -0.1074666f, -0.0989476f, +0.1298848f, -0.1424994f, -0.0811411f,
    -0.0551217f, -0.1546610f, +0.1627934f, -0.0260871f, +0.1362490f, -0.0931728f, +0.1621238f, +0.0358711f,
    -0.1385719f, +0.0915332f, +0.1662925f, +0.0036696f, +0.1650943f, +0.0217198f, +0.0781321f, +0.1471043f,
    -0.1352713f, -0.0974371f, -0.1564076f, +0.0579388f, -0.1539027f, -0.0649317f, +0.0247026f, -0.1658606f,
    -0.1629969f, +0.0408678f, +0.1232073f, +0.1148664f, +0.0938616f, -0.1415707f, +0.1609885f, -0.0552557f,
    +0.1548529f, -0.0713224f, +0.1198467f, -0.1214889f, +0.0499484f, +0.1634062f, +0.0130902f, +0.1704069f,
    +0.1316278f, -0.1091409f, -0.1532014f, +0.0769505f, -0.0913136f, +0.1452094f, -0.0582629f, +0.1616912f,
    -0.1714029f, -0.0152778f, -0.0428209f, -0.1674523f, +0.1387843f, +0.1036431f, -0.0228898f, -0.1717252f,
    -0.1184419f, +0.1267353f, -0.1193626f, -0.1264102f, -0.0730444f, -0.1577858f, +0.1735560f, -0.0124306f,
    +0.0580963f, -0.1644554f, -0.1330052f, +0.1129727f, -0.1673122f, -0.0505153f, -0.1066221f, -0.1388090f,
    -0.0931766f, -0.1484706f, -0.1752945f, +0.0069861f, +0.1525343f, +0.0870980f, +0.1717408f, -0.0410935f,
    +0.0429148f, -0.1713273f, +0.1689704f, +0.0520356f, +0.1533063f, -0.0892880f, +0.0793586f, -0.1589448f,
    +0.0111259f, -0.1774688f, +0.1633866f, +0.0705932f, -0.0064226f, -0.1779892f, -0.0797541f, +0.1594945f,
    +0.1105577f, -0.1399761f, -0.1762614f, +0.0274005f, +0.0356427f, +0.1750989f, +0.0703636f, +0.1656831f,
    +0.1068077f, +0.1454663f, -0.0211164f, +0.1793373f, -0.0406138f, +0.1770871f, -0.1604492f, -0.0852983f,
    -0.1798653f, -0.0328750f, -0.1717166f, +0.0628464f, +0.1822116f, +0.0181014f, -0.0022203f, +0.1834202f,
    +0.0929536f, +0.1585599f, +0.1223100f, +0.1373409f, -0.1374490f, -0.1226082f, -0.0594532f, -0.1746655f,
    +0.1514310f, -0.1071997f, -0.1142375f, +0.1461950f, -0.1521953f, +0.1062068f, +0.0297150f, -0.1832649f,
    -0.1619828f, +0.0931774f, +0.1850494f, -0.0297916f, -0.0374841f, -0.1839482f, -0.1748014f, -0.0688581f,
    -0.1541609f, -0.1076374f, +0.0204645f, +0.1872467f, +0.1766363f, -0.0663364f, -0.1889056f, -0.0035019f,
    +0.1857253f, +0.0347000f, +0.1007938f, -0.1600314f, +0.1891181f, +0.0032906f, +0.1389632f, +0.1284448f,
    +0.1706564f, -0.0817873f, -0.1834845f, +0.0489056f, +0.1424756f, -0.1257628f, -0.1344507f, +0.1349497f,
    +0.1295881f, -0.1397652f, +0.0558659f, +0.1823038f, -0.0724102f, +0.1765170f, -0.1209693f, -0.1476944f,
    -0.1026116f, +0.1612646f, +0.0729600f, -0.1771007f, -0.1740334f, +0.0808347f, -0.0909667f, -0.1700926f,
    +0.1835091f, +0.0619239f, +0.1659203f, +0.1000627f, +0.1933423f, -0.0133862f, +0.1865649f, -0.0528193f,
    -0.0596209f, +0.1849181f, -0.1873558f, -0.0522763f, +0.1559983f, +0.1162254f, -0.0204214f, -0.1937807f,
    -0.1941552f, -0.0196437f, +0.1174358f, -0.1560054f, -0.0767456f, -0.1797723f, -0.1072569f, -0.1634187f,
    -0.0000509f, -0.1960120f, -0.1933416f, +0.0328720f, -0.1513784f, +0.1251728f, +0.1691830f, -0.1003201f,
    +0.1786212f, +0.0840827f, +0.0864041f, +0.1779333f, -0.1977199f, +0.0149098f, +0.0194960f, -0.1973762f,
    -0.0918973f, +0.1760261f, +0.0606757f, -0.1893306f, -0.1732086f, -0.0984412f, -0.0562990f, -0.1913061f,
    +0.0938094f, -0.1761348f, +0.0386501f, +0.1961812f, -0.0292933f, +0.1980645f, +0.0070945f, +0.2001454f,
    +0.0436379f, -0.1955635f, -0.1878766f, +0.0700452f, +0.1949706f, +0.0480252f, +0.1235766f, +0.1584681f,
    +0.1086626f, +0.1701006f, -0.1399843f, -0.1464574f, -0.0104917f, +0.2025580f, +0.2020374f, +0.0185961f,
    -0.1557207f, -0.1308356f, -0.1215249f, +0.1632875f, +0.1994997f, -0.0414276f, +0.1639010f, -0.1212889f,
    -0.2009337f, -0.0367350f, -0.0474615f, +0.1992590f, -0.1868244f, -0.0852088f, +0.0721474f, +0.1922581f,
    +0.1413377f, +0.1504298f, -0.0387219f, -0.2028442f, -0.1367829f, +0.1547267f, -0.1708606f, +0.1167573f,
    +0.2059021f, -0.0214651f, +0.1867772f, -0.0897756f, -0.1798158f, +0.1031576f, -0.1962986f, -0.0674368f,
    +0.1572083f, +0.1362977f, +0.1340928f, -0.1592329f, +0.1481029f, -0.1472204f, +0.1959164f, -0.0732323f,
    -0.2091693f, -0.0020008f, -0.1725136f, -0.1185024f, -0.1270565f, -0.1668054f, -0.2046681f, +0.0463107f,
    -0.1535793f, +0.1431848f, +0.1142497f, -0.1762215f, +0.2100215f, -0.0006321f, -0.1909521f, +0.0883180f,
    +0.0214642f, +0.2101842f, -0.0826701f, +0.1945418f, +0.0562543f, +0.2039945f, -0.1117822f, +0.1797049f,
    +0.1754997f, +0.1188388f, +0.0852405f, -0.1942637f, +0.1974890f, +0.0785118f, +0.2096190f, +0.0374520f,
    +0.1805084f, -0.1131342f, -0.0083651f, -0.2128994f, +0.1627732f, -0.1376090f, -0.2030335f, +0.0665044f,
    -0.0958866f, -0.1913353f, -0.0763637f, -0.2000970f, +0.1875291f, +0.1040362f, -0.0676646f, +0.2035802f,
    +0.0134977f, -0.2144788f, +0.2060796f, +0.0612757f, +0.0338918f, -0.2124389f, -0.1120985f, -0.1837470f,
    +0.1020358f, +0.1898917f, -0.2139781f, +0.0265238f, -0.2090496f, -0.0530057f, +0.2085680f, -0.0580393f,
    -0.0555797f, -0.2096740f, -0.2162235f, -0.0210810f, -0.1709037f, +0.1350431f, -0.1003022f, +0.1946732f,
    +0.0740402f, -0.2070313f, -0.0204661f, +0.2191154f, +0.1329905f, +0.1754834f, +0.0862816f, +0.2026358f,
    -0.0280792f, -0.2184843f, +0.0393474f, +0.2169303f, -0.1900934f, -0.1119861f, -0.1609798f, -0.1511746f,
    +0.2180401f, -0.0371844f, +0.1322633f, -0.1774979f, +0.1053286f, -0.1950181f, +0.2209968f, +0.0197722f,
    +0.0008333f, +0.2220785f, +0.0590936f, -0.2142203f, -0.1295945f, +0.1806085f, -0.1474617f, -0.1667320f,
    -0.0405241f, +0.2191890f, -0.1770077f, -0.1373188f, +0.1196322f, +0.1897887f, -0.1578249f, +0.1595442f,
    +0.1976406f, -0.1066997f, -0.2108348f, -0.0776472f, -0.2023716f, -0.0977474f, -0.2082664f, +0.0845686f,
    -0.2249778f, +0.0099372f, +0.1773974f, +0.1389678f, +0.2249485f, -0.0179349f, -0.1992272f, +0.1063813f,
    +0.2072621f, -0.0904716f, +0.2047831f, +0.0968048f, +0.1841853f, -0.1318833f, -0.2185781f, +0.0596370f,
    -0.1334799f, -0.1836065f, -0.1465565f, +0.1739140f, +0.1518192f, +0.1693849f, +0.1506670f, -0.1706558f,
    +0.1653687f, +0.1564648f, -0.1909213f, +0.1247653f, -0.2245624f, -0.0408006f, +0.0718795f, +0.2168817f,
    +0.2227909f, +0.0515711f, -0.0609267f, +0.2204636f, -0.2253538f, +0.0417008f, +0.1653316f, -0.1589030f,
    +0.0084306f, -0.2297158f, +0.2300381f, +0.0017963f, +0.2180939f, -0.0741767f, -0.0830048f, +0.2151621f,
    -0.0711603f, -0.2194421f, +0.0475540f, -0.2258450f, +0.0217189f, +0.2298514f, +0.1933704f, +0.1265150f,
    -0.0461429f, -0.2264724f, +0.2199125f, +0.0732318f, +0.1265344f, -0.1942611f, +0.0951531f, -0.2128843f,
    -0.1211797f, +0.1992736f, +0.0287361f, -0.2315915f, +0.0539656f, +0.2272732f, -0.2335151f, -0.0072559f,
    -0.0108554f, -0.2341180f, +0.2279256f, -0.0553712f, -0.1031062f, +0.2109456f, -0.1237249f, -0.1996824f,
    -0.0922942f, -0.2160537f, -0.1959039f, -0.1297062f, +0.1822385f, -0.1485409f, -0.2268076f, -0.0622213f,
    +0.2006708f, -0.1238295f, +0.2334736f, +0.0348055f, -0.1878680f, +0.1438848f, -0.1763039f, +0.1586175f,
    +0.1107185f, +0.2097731f, -0.1094588f, -0.2110663f, -0.2367431f, +0.0264258f, +0.2073919f, +0.1172518f,
    -0.0305965f, +0.2368142f, -0.2262245f, +0.0764366f, +0.0952611f, +0.2198058f, -0.1664314f, -0.1724227f,
    +0.1173856f, -0.2096868f, +0.2373470f, -0.0381613f, -0.0140845f, +0.2399837f, +0.0391544f, +0.2374054f,
    -0.1821290f, -0.1574568f, -0.2205950f, -0.0965488f, +0.0812428f, -0.2267786f, +0.2234680f, +0.0901396f,
    -0.2192517f, +0.1003826f, -0.2398559f, -0.0269877f, -0.1521969f, -0.1875219f, +0.1479760f, -0.1909288f,
    -0.2115468f, -0.1166568f, -0.1451895f, +0.1933796f, -0.0484342f, +0.2369406f, +0.0044249f, +0.2419590f,
    -0.0286731f, -0.2404451f, +0.1516476f, +0.1891988f, +0.1349945f, +0.2019376f, -0.1674480f, +0.1761509f,
    +0.2429782f, +0.0186891f, -0.2117971f, +0.1214878f, +0.2434715f, -0.0184613f, +0.2183067f, -0.1097243f,
    +0.0658144f, -0.2362353f, +0.1870462f, +0.1591857f, -0.2380531f, +0.0605679f, +0.2369377f, +0.0653846f,
    +0.2280021f, -0.0925030f, -0.2319600f, -0.0822914f, -0.0815896f, +0.2330902f, -0.2466888f, +0.0119776f,
    +0.2470278f, +0.0013913f, +0.0457466f, -0.2429922f, +0.0686922f, +0.2378099f, +0.2006575f, -0.1453052f,
    +0.0027935f, -0.2478843f, +0.2007761f, +0.1454213f, -0.0662455f, +0.2390275f, +0.1845362f, -0.1659621f,
    -0.0646574f, -0.2396803f, -0.1985522f, -0.1493987f, +0.1727794f, +0.1786085f, +0.2386421f, -0.0699715f,
    -0.1002095f, +0.2276218f, +0.1691689f, -0.1823917f, -0.2046853f, +0.1416519f, +0.2237732f, +0.1090290f,
    -0.1201944f, +0.2185188f, -0.0463805f, -0.2453123f, -0.2457660f, +0.0443278f, +0.0228661f, -0.2489451f,
    -0.0840289f, -0.2356509f, +0.0866132f, +0.2348400f, +0.0225558f, +0.2496864f, +0.1056208f, -0.2273779f,
    -0.1440945f, -0.2052726f, -0.2463293f, -0.0497675f, +0.2470669f, +0.0486663f, +0.1415224f, -0.2091022f,
    -0.2525356f, -0.0055694f, -0.1397623f, +0.2111554f, -0.1956987f, +0.1609435f, -0.1648919f, +0.1924499f,
    +0.2161552f, +0.1350693f, -0.2160566f, -0.1358307f, -0.1306896f, -0.2192147f, +0.2187741f, -0.1315637f,
    +0.1267401f, +0.2220655f, +0.2507721f, -0.0509068f, -0.1035846f, -0.2340559f, +0.0531293f, +0.2509082f,
    -0.2376370f, +0.0985965f, +0.2435763f, +0.0833769f, -0.2445971f, +0.0812151f, -0.0336365f, +0.2557144f,
    -0.1846338f, -0.1810008f, -0.1864648f, +0.1793066f, -0.2356998f, -0.1067147f, +0.1093506f, +0.2349682f,
    -0.0144004f, -0.2587793f, +0.1601433f, -0.2039713f, +0.2577483f, +0.0298654f, -0.2585093f, +0.0298804f,
    +0.2456641f, -0.0862105f, -0.0102952f, +0.2603101f, +0.1268403f, -0.2278231f, +0.0943965f, -0.2431212f,
    -0.2508642f, -0.0719603f, -0.1701933f, -0.1978758f, +0.2591037f, -0.0314672f, -0.2327234f, +0.1187220f,
    -0.1551146f, +0.2102892f, +0.1528334f, +0.2120543f, +0.2403990f, +0.1031150f, +0.0366290f, +0.2594562f,
    +0.2619805f, -0.0102657f, +0.2396726f, -0.1064034f, +0.0380573f, -0.2595330f, +0.0099686f, +0.2622291f,
    -0.0558818f, +0.2565784f, -0.2617715f, -0.0215746f, -0.2603969f, -0.0383763f, -0.2238304f, +0.1390253f,
    -0.2009706f, -0.1706830f, -0.0347354f, -0.2614456f, -0.2554717f, +0.0658888f, +0.2059224f, -0.1649626f,
    +0.2324921f, +0.1249189f, +0.1699771f, +0.2020488f, -0.1206709f, -0.2351095f, +0.0734881f, +0.2539032f,
    +0.0073149f, -0.2646076f, -0.0973909f, +0.2462445f, +0.2647267f, +0.0098974f, +0.2090954f, +0.1633877f,
    -0.2335621f, -0.1260399f, +0.1952162f, +0.1799865f, +0.0609785f, -0.2585934f, +0.1915820f, -0.1841962f,
    +0.0799217f, -0.2541620f, +0.2198623f, -0.1506844f, -0.2152687f, +0.1575281f, -0.1341920f, +0.2305630f,
    +0.2367911f, -0.1230241f, -0.1163781f, +0.2402094f, -0.2169439f, -0.1555624f, -0.0571414f, -0.2608387f,
    +0.2580575f, +0.0689352f, -0.0791833f, +0.2557286f, -0.0774849f, -0.2567629f, +0.2596329f, -0.0680016f,
    -0.2680523f, +0.0136337f, +0.0944470f, +0.2513941f, +0.2649204f, +0.0472111f, -0.1608651f, -0.2160198f,
    -0.2527049f, -0.0932300f, +0.1506013f, -0.2246200f, -0.0976468f, -0.2531027f, -0.1880854f, +0.1955281f,
    -0.2666623f, +0.0504116f, +0.1861028f, +0.1981163f, -0.2067550f, +0.1771460f, +0.1137487f, -0.2473930f,
    +0.1470990f, +0.2296910f, -0.2727978f, -0.0035321f, +0.1838586f, -0.2015621f, -0.2668782f, -0.0590408f,
    -0.1756292f, +0.2098570f, +0.2265081f, +0.1549010f, +0.0256167f, -0.2735830f, -0.0253450f, +0.2741274f,
    -0.1488121f, -0.2317446f, +0.2593980f, +0.0956275f, -0.1875276f, -0.2033125f, -0.2583635f, +0.0999817f,
    +0.0269557f, +0.2758511f, +0.2730499f, -0.0476074f, +0.2759858f, +0.0285021f, +0.1126296f, +0.2536952f,
    +0.2530287f, +0.1145732f, +0.2578691f, -0.1035484f, +0.1321702f, +0.2446368f, +0.1395389f, -0.2405261f,
    -0.2535846f, -0.1145465f, -0.2768049f, +0.0339374f, +0.0502803f, +0.2743682f, -0.2525907f, +0.1186631f,
    -0.1505110f, +0.2350392f, +0.2397182f, -0.1432362f, +0.0867024f, +0.2656121f, +0.2783042f, -0.0250162f,
    +0.1736849f, -0.2196526f, -0.2039106f, -0.1919585f, -0.0480282f, +0.2761787f, -0.2672675f, +0.0845681f,
    +0.2803175f, -0.0050750f, +0.2406547f, +0.1439953f, -0.0680073f, +0.2720755f, -0.0258679f, -0.2792738f,
    -0.2396269f, -0.1459922f, +0.0682613f, +0.2723394f, -0.2800680f, -0.0227968f, +0.0485281f, -0.2769842f,
    -0.0045320f, -0.2813370f, +0.2143653f, -0.1824037f, -0.1661566f, +0.2272529f, -0.1376024f, -0.2456893f,
    -0.2456640f, +0.1378846f, +0.2679770f, -0.0878183f, +0.0044841f, +0.2821487f, -0.1012687f, +0.2640166f,
    -0.2197287f, -0.1787421f, +0.1020540f, -0.2645234f, -0.2357661f, +0.1579993f, -0.1195126f, -0.2576784f,
    -0.0483748f, -0.2799076f, -0.2763143f, +0.0662858f, +0.2311064f, -0.1656710f, +0.2777006f, +0.0612543f,
    +0.2186855f, +0.1818638f, +0.2842281f, +0.0129603f, -0.2325148f, -0.1641237f, +0.0674552f, -0.2765638f,
    -0.2254104f, +0.1745746f, +0.2564324f, -0.1246991f, -0.2821786f, -0.0418022f, -0.2735377f, -0.0814005f,
    -0.1356415f, +0.2511939f, +0.2729502f, +0.0841184f, +0.1710020f, +0.2294709f, +0.2546850f, +0.1313363f,
    +0.0150728f, -0.2862036f, -0.1202963f, +0.2602314f, +0.0836830f, -0.2744229f, +0.2055216f, +0.2003880f,
    -0.1819643f, -0.2221752f, -0.2870251f, +0.0132297f, +0.2056030f, -0.2008968f, -0.2124328f, +0.1940681f,
    +0.2795577f, -0.0685847f, -0.1978655f, +0.2095879f, -0.0696770f, -0.2797066f, -0.0150616f, +0.2880781f,
    +0.1278414f, -0.2586078f, +0.1891528f, +0.2179061f, +0.1617880f, -0.2401695f, -0.0882693f, -0.2758658f,
    -0.1698922f, -0.2361331f, +0.2878348f, +0.0431494f, -0.2594512f, -0.1320476f, +0.1545337f, +0.2470139f,
    -0.0875585f, +0.2779214f, -0.2719894f, -0.1047880f, -0.1875876f, +0.2237137f, -0.1072534f, -0.2716194f,
    -0.2868359f, -0.0596279f, +0.1079583f, +0.2725161f, +0.1273387f, +0.2641523f, +0.2730364f, +0.1085725f,
    -0.2939448f, -0.0098947f, -0.2899891f, +0.0526615f, +0.2367773f, +0.1755530f, +0.0870707f, +0.2820236f,
    +0.2571416f, -0.1469836f, +0.2000749f, -0.2186530f, +0.0214084f, +0.2956808f, -0.2727112f, +0.1168205f,
    -0.0157649f, -0.2963112f, +0.0427459f, +0.2937947f, +0.2955712f, -0.0328028f, -0.2023110f, -0.2181668f,
    +0.1808135f, -0.2365666f, -0.2961863f, +0.0323403f, +0.2934537f, -0.0534970f, +0.0355605f, -0.2964357f,
    +0.2760865f, -0.1136639f, -0.0360284f, +0.2965619f, +0.1503119f, -0.2581983f, -0.2213230f, -0.2007361f,
    +0.2889343f, +0.0780144f, -0.2557685f, +0.1556970f, +0.2992250f, -0.0117991f, +0.2502964f, +0.1644146f,
    -0.2662183f, +0.1374251f, -0.1567418f, -0.2555311f, +0.2505540f, -0.1647058f, -0.1667739f, +0.2495218f,
    -0.1494970f, +0.2602857f, -0.1408684f, -0.2652138f, -0.2835499f, +0.0991661f, +0.0658484f, +0.2932721f,
    -0.2499796f, -0.1671919f, +0.2996698f, +0.0263354f, +0.2867387f, -0.0911185f, +0.1186520f, -0.2766684f,
    -0.2994106f, -0.0317744f, -0.1105051f, +0.2801617f, -0.1821761f, +0.2400575f, +0.2624899f, +0.1486978f,
    -0.2617395f, -0.1503616f, -0.2359045f, +0.1883694f, -0.2914744f, +0.0787769f, -0.0410452f, -0.2992489f,
    +0.0981315f, -0.2859449f, +0.2962451f, +0.0604661f, -0.2394394f, -0.1852133f, +0.1477928f, +0.2642047f,
    +0.2738432f, +0.1294205f, +0.2383078f, -0.1873501f, +0.0579888f, -0.2976237f, -0.0579624f, +0.2978175f,
    -0.2142228f, +0.2148992f, +0.0094034f, -0.3033227f, +0.2234953f, +0.2055634f, +0.3038727f, +0.0059891f,
    +0.2250558f, -0.2046798f, -0.0018456f, +0.3042104f, -0.0774069f, +0.2946218f, +0.2890492f, +0.0987256f,
    +0.2738334f, -0.1359287f, -0.2504265f, +0.1754271f, -0.2904068f, -0.0958474f, +0.0789632f, -0.2958016f,
    -0.0629204f, -0.2998672f, -0.1320875f, +0.2764939f, -0.3064641f, +0.0046907f, +0.1383767f, -0.2735977f,
    +0.1736640f, +0.2532467f, -0.2981887f, -0.0758412f, -0.1254133f, -0.2811457f, -0.2301200f, +0.2045625f,
    -0.2038379f, +0.2308113f, -0.1753082f, -0.2537611f, -0.2825274f, -0.1239644f, -0.0843242f, -0.2970509f,
    +0.2096571f, +0.2268393f, +0.1929025f, +0.2413163f, -0.1932493f, -0.2410573f, +0.3016600f, -0.0706058f,
    +0.1275615f, +0.2833500f, -0.1054458f, -0.2924508f, +0.1053275f, +0.2931508f, +0.2421073f, +0.1964667f,
    +0.2029200f, -0.2369012f, +0.0836689f, +0.3005916f, -0.0977932f, +0.2965223f, -0.3079836f, -0.0518510f,
    +0.1713799f, -0.2612628f, +0.3094688f, +0.0484353f, -0.2243912f, -0.2189790f, -0.3069434f, +0.0648961f,
    -0.3131473f, +0.0219806f, -0.2860226f, +0.1303452f, -0.3141115f, -0.0150188f, -0.0216593f, +0.3138617f,
    +0.0507650f, +0.3107022f, -0.0052717f, -0.3149100f, +0.3033445f, +0.0848634f, +0.3142391f, -0.0228977f,
    +0.2916272f, +0.1202541f, -0.3124550f, +0.0438415f, +0.2954617f, -0.1108600f, +0.2592414f, -0.1803883f,
    -0.0262499f, -0.3147810f, +0.3128984f, -0.0452043f, +0.1899601f, -0.2530575f, +0.0275899f, -0.3153897f,
    -0.2410999f, -0.2058102f, +0.0141668f, +0.3167119f, -0.2812362f, -0.1464681f, +0.1156503f, -0.2953555f,
    +0.2727572f, -0.1620361f, -0.2744372f, +0.1610113f, -0.1691223f, +0.2698041f, -0.2685722f, -0.1711470f,
    -0.2145006f, -0.2361256f, +0.2830844f, +0.1480418f, -0.1526783f, +0.2808161f, +0.2210325f, -0.2311819f,
    -0.3061798f, +0.0927001f, +0.3065451f, -0.0927936f, +0.1565474f, -0.2795775f, +0.3188419f, +0.0341326f,
    -0.1207699f, +0.2971982f, -0.2996999f, +0.1149544f, -0.1508854f, -0.2838007f, +0.1350541f, -0.2917652f,
    -0.1864964f, +0.2620314f, +0.2731985f, +0.1699032f, -0.2597710f, -0.1900918f, -0.2037069f, +0.2493975f,
    +0.2946283f, -0.1303329f, +0.0515448f, -0.3181449f, -0.3019769f, -0.1128670f, +0.1500125f, +0.2854452f,
    +0.2625542f, +0.1872625f, +0.3221357f, +0.0153798f, +0.3224753f, -0.0056238f, -0.2542686f, +0.1985449f,
    -0.0392396f, +0.3203396f, +0.3154053f, +0.0687682f, +0.2330890f, +0.2234456f, +0.1697414f, +0.2750715f,
    -0.1698881f, -0.2753651f, +0.0318854f, +0.3221030f, -0.0529521f, -0.3194197f, -0.3220184f, -0.0348482f,
    -0.2343497f, +0.2238237f, +0.0917177f, -0.3109486f, -0.0781004f, +0.3148802f, -0.2888293f, +0.1479497f,
    -0.1890718f, -0.2640145f, -0.0759212f, -0.3158664f, +0.0722334f, +0.3168317f, +0.2889406f, -0.1488651f,
    -0.2221087f, +0.2373833f, +0.0725775f, -0.3173925f, +0.3076069f, +0.1073347f, +0.1217362f, +0.3024494f,
    -0.2704870f, +0.1820906f, +0.1917246f, +0.2639463f, +0.2496793f, -0.2104070f, +0.2382662f, -0.2235948f,
    -0.3268787f, +0.0024167f, -0.3139440f, -0.0929610f, -0.0063620f, +0.3276020f, +0.2616487f, -0.1980952f,
    -0.0577592f, +0.3232812f, +0.0105806f, -0.3290457f, +0.2118718f, +0.2525145f, -0.1212233f, -0.3066972f,
    -0.0996181f, -0.3147698f, -0.3224142f, -0.0715527f, +0.0980359f, +0.3159763f, -0.1386401f, -0.3009915f,
    +0.1111820f, -0.3123117f, +0.2114468f, -0.2556656f, -0.3296672f, +0.0376464f, -0.2091618f, -0.2578786f,
    -0.2370176f, -0.2325421f, +0.3304017f, -0.0334216f, +0.2555065f, +0.2127430f, +0.3198942f, +0.0910688f,
    +0.3236540f, -0.0775728f, -0.3005040f, -0.1430732f, -0.0349986f, -0.3311521f, -0.1037275f, +0.3164813f,
    -0.2500025f, +0.2204225f, +0.3019409f, +0.1412124f, -0.1459908f, +0.2998008f, +0.3289071f, -0.0566060f,
    +0.1795086f, -0.2814457f, +0.0390347f, -0.3318371f, -0.2569969f, -0.2135384f, -0.3289340f, +0.0600043f,
    -0.2900154f, -0.1668233f, +0.0513136f, +0.3308947f, +0.2796716f, -0.1841503f, +0.1958458f, -0.2716319f,
    -0.3251171f, +0.0805084f, -0.0130979f, -0.3349905f, -0.3349671f, -0.0162272f, +0.3141689f, -0.1174266f,
    +0.3313272f, +0.0536282f, +0.2320780f, +0.2437793f, -0.3363681f, +0.0177330f, +0.1426400f, +0.3052920f,
    -0.3067968f, +0.1399481f, -0.3331009f, -0.0530779f, +0.3136780f, +0.1242424f, +0.1304083f, -0.3122354f,
    -0.3226107f, +0.1020987f, +0.2940483f, -0.1679908f, -0.3130437f, -0.1292896f, +0.1501211f, -0.3040205f,
    -0.2820559f, -0.1884983f, +0.3376687f, +0.0330372f, +0.3241932f, -0.1000997f, +0.3394296f, +0.0045053f,
    +0.3312747f, +0.0744821f, -0.2958261f, +0.1666945f, -0.1314204f, +0.3131872f, -0.1709952f, +0.2940212f,
    +0.2983142f, +0.1635697f, +0.0126007f, +0.3405971f, -0.2068330f, +0.2709670f, +0.3094351f, -0.1441352f,
    -0.1895861f, +0.2842357f, +0.2344902f, -0.2485553f, -0.3195884f, +0.1212630f, +0.1695802f, -0.2968292f,
    -0.1697482f, -0.2968850f, -0.0710557f, -0.3346864f, +0.1227183f, +0.3195305f, +0.3422706f, -0.0153724f,
    +0.1660573f, +0.2998455f, -0.2230599f, +0.2603219f, -0.0254216f, +0.3423854f, -0.2298138f, -0.2553627f,
    -0.2665740f, +0.2169882f, -0.2748159f, -0.2068923f, +0.1884874f, +0.2878802f, +0.1022193f, -0.3285944f,
    +0.2761185f, +0.2055916f, -0.0829185f, +0.3342959f, -0.2017589f, -0.2793698f, +0.0710927f, +0.3376466f,
    -0.2793546f, +0.2025940f, -0.1865710f, -0.2906791f, -0.2436877f, +0.2450379f, +0.0839179f, -0.3354943f,
    +0.0636521f, -0.3400422f, -0.2920140f, +0.1857485f, -0.1183891f, -0.3263249f, +0.2917062f, +0.1886045f,
    -0.0498857f, -0.3437810f, +0.2530842f, -0.2383283f, -0.3357451f, -0.0903192f, -0.3460805f, -0.0334151f,
    -0.3296367f, -0.1110484f, +0.2552527f, +0.2365057f, -0.1591275f, -0.3099440f, +0.0359081f, +0.3465663f,
    +0.2686580f, -0.2218868f, -0.0951699f, -0.3352065f, -0.3485263f, +0.0022765f, -0.0455561f, +0.3458680f,
    -0.3475466f, +0.0307239f, +0.0221482f, -0.3482377f, +0.3319177f, +0.1085070f, +0.3471852f, -0.0376480f,
    -0.2587325f, -0.2345662f, +0.2834095f, -0.2045214f, +0.2263292f, +0.2663979f, -0.0653744f, +0.3439163f,
    -0.3468987f, +0.0511019f, +0.3500021f, +0.0212463f, +0.2274280f, -0.2668943f, +0.2108041f, +0.2802109f,
    +0.0889766f, +0.3393684f, -0.3094427f, -0.1655894f, +0.1079442f, +0.3339623f, +0.3446507f, -0.0665898f,
    -0.0029035f, -0.3515099f, -0.1221191f, +0.3297681f, +0.3404510f, +0.0891967f, -0.1009602f, +0.3374099f,
    -0.0079066f, +0.3528429f, +0.3226171f, +0.1440194f, +0.0439929f, -0.3507298f, +0.3279354f, -0.1326128f,
    +0.1960899f, -0.2946450f, -0.0295509f, -0.3528842f, +0.2137478f, -0.2823962f, -0.2486817f, -0.2528994f,
    -0.3478966f, -0.0705640f, +0.3434397f, -0.0904992f, -0.1654038f, +0.3143278f, -0.3172786f, +0.1597355f,
    -0.2197184f, -0.2791945f, +0.3016626f, -0.1876895f, +0.3139684f, -0.1670576f, -0.1444867f, -0.3251038f,
    +0.3495023f, +0.0672196f, -0.2648809f, +0.2387807f, -0.3445058f, +0.0935900f, -0.3497194f, +0.0728862f,
    +0.3541836f, +0.0467929f, +0.1228163f, -0.3356963f, +0.1438153f, +0.3276280f, -0.0809673f, -0.3486349f,
    +0.1446603f, -0.3279028f, +0.1657943f, -0.3178751f, -0.3401241f, +0.1137006f, -0.3051539f, -0.1886568f,
    +0.0565848f, +0.3543032f, +0.3407436f, -0.1132756f, -0.3587111f, -0.0186546f, -0.3317575f, +0.1381558f,
    -0.3259388f, -0.1518473f, -0.2154468f, +0.2881357f, +0.2481839f, +0.2605936f, -0.2784798f, -0.2280419f,
    +0.3364215f, +0.1283921f, -0.2941727f, -0.2077793f, +0.3133150f, +0.1778180f, +0.2768292f, +0.2305793f,
    -0.2391345f, +0.2695371f, -0.3119887f, +0.1803651f, -0.3350647f, -0.1330700f, +0.3605997f, -0.0011515f,
    -0.1502501f, +0.3281581f, +0.1842752f, +0.3104075f, -0.1965104f, +0.3035741f, -0.3595013f, -0.0489914f,
    +0.2950812f, +0.2119735f, +0.1630200f, +0.3249309f, -0.2854062f, +0.2254183f, +0.1871338f, -0.3119632f,
    -0.3000610f, +0.2060302f, +0.0222187f, +0.3633643f, +0.1261486f, +0.3414969f, -0.1150460f, -0.3457837f,
    +0.3642695f, -0.0232917f, +0.1025836f, -0.3503462f, -0.0614805f, -0.3598826f, +0.3615346f, -0.0521030f,
    +0.2617335f, -0.2548983f, +0.2474189f, -0.2690753f, -0.2389105f, -0.2766967f, -0.3652831f, +0.0173775f,
    -0.1323602f, -0.3409443f, +0.3285058f, +0.1611128f, -0.1876443f, -0.3142805f, +0.3312585f, -0.1558818f,
    -0.2071865f, -0.3023375f, +0.2897657f, -0.2249780f, +0.2065196f, +0.3037437f, -0.2604349f, +0.2591215f,
    -0.0331968f, +0.3661468f, +0.0585672f, -0.3633198f, -0.3658953f, +0.0394559f, -0.1183455f, +0.3485421f,
    +0.0793945f, +0.3594328f, +0.3603557f, -0.0772173f, +0.1068619f, +0.3529755f, -0.1869263f, +0.3180214f,
    +0.0051896f, +0.3690335f, -0.3574419f, -0.0919650f, +0.3680757f, +0.0326463f, +0.0090199f, -0.3697491f,
    -0.3698506f, -0.0034156f, +0.3553466f, +0.1032484f, -0.0574727f, +0.3657439f, -0.1695893f, -0.3291343f,
    +0.3608711f, +0.0833622f, -0.0992210f, +0.3568524f, -0.2710894f, -0.2530825f, -0.1007261f, -0.3571576f,
    +0.0831262f, -0.3616980f, +0.2375537f, +0.2851719f, +0.0350685f, -0.3696826f, +0.3071340f, -0.2089895f,
    +0.2797893f, -0.2444361f, -0.3273816f, -0.1758347f, -0.1398079f, +0.3444893f, -0.0803135f, +0.3632475f,
    -0.3535389f, -0.1171699f, -0.3676239f, +0.0599281f, -0.0158819f, -0.3723854f, +0.2350488f, -0.2892721f,
    +0.3156489f, +0.1988916f, -0.0407660f, -0.3709517f, +0.3732741f, +0.0154360f, +0.2693657f, +0.2589040f,
    +0.0403760f, +0.3719178f, +0.3258373f, -0.1842902f, +0.1590510f, -0.3389232f, +0.2182627f, -0.3042576f,
    -0.3393773f, +0.1596406f, +0.3481159f, -0.1397233f, -0.0821230f, -0.3662516f, +0.3619525f, -0.1010847f,
    +0.3732306f, +0.0502782f, -0.3754080f, -0.0326925f, +0.3567304f, +0.1228999f, -0.3667473f, +0.0890519f,
    -0.2605496f, -0.2730726f, -0.2370069f, +0.2937541f, -0.3707368f, -0.0713612f, -0.1752044f, +0.3344871f,
    -0.0115933f, +0.3778997f, -0.3548564f, +0.1304613f, -0.2844867f, +0.2490997f, +0.1823583f, -0.3319165f,
    -0.2197971f, +0.3087385f, +0.1841370f, +0.3314193f, -0.3245026f, +0.1961828f, +0.3508866f, +0.1440305f,
    -0.3632698f, +0.1091315f, -0.3146555f, -0.2121710f, +0.3773535f, -0.0406632f, -0.1556001f, -0.3462132f,
    -0.3019282f, -0.2300856f, +0.1416906f, -0.3525632f, -0.3355779f, +0.1786653f, +0.3743877f, +0.0674121f,
    -0.3263724f, -0.1956091f, +0.1515156f, +0.3490936f, -0.2285402f, -0.3045776f, -0.3531846f, -0.1428903f,
    +0.2061642f, -0.3206353f, +0.2272868f, +0.3060868f, +0.1239170f, -0.3607411f, -0.2912028f, -0.2467214f,
    +0.3347137f, +0.1841088f, +0.3820359f, -0.0039011f, -0.2560562f, +0.2838358f, +0.0982218f, +0.3696225f,
    +0.1309377f, +0.3599953f, +0.3025741f, +0.2351422f, +0.3622237f, -0.1257408f, +0.0606626f, +0.3789637f,
    -0.3184244f, +0.2146984f, -0.1378403f, -0.3585455f, +0.3790959f, -0.0638999f, +0.2887627f, +0.2541968f,
    +0.2058945f, +0.3251303f, +0.3470772f, -0.1666388f, -0.3079503f, +0.2311210f, +0.2602274f, +0.2840037f,
    -0.3483180f, -0.1652870f, -0.3789086f, +0.0733057f, +0.3187845f, +0.2178996f, -0.3822843f, -0.0549022f,
    -0.3861971f, +0.0053559f, -0.3546644f, +0.1535177f, +0.0680214f, -0.3804759f, -0.1658577f, +0.3492825f,
    +0.2761297f, -0.2709910f, -0.1239541f, +0.3668050f, -0.3862831f, +0.0294192f, +0.2581134f, -0.2891952f,
    +0.0182294f, +0.3872418f, +0.3511046f, +0.1648837f, +0.3282788f, -0.2068227f, -0.2496122f, -0.2970580f,
    -0.1195767f, -0.3696573f, +0.3783585f, +0.0899368f, -0.2142632f, -0.3246042f, -0.1938385f, -0.3372316f,
    -0.2766716f, +0.2736540f, +0.3105744f, -0.2346157f, +0.0094836f, -0.3894105f, +0.3801251f, -0.0857787f,
    +0.1078466f, -0.3746813f, -0.0973417f, -0.3775729f, -0.3896217f, -0.0163305f, -0.3871514f, +0.0485378f,
    +0.3892840f, -0.0270963f, -0.1443634f, +0.3625592f, +0.0271394f, -0.3894134f, -0.3798374f, -0.0907432f,
    -0.0557789f, -0.3869389f, -0.0756077f, -0.3837478f, -0.0302831f, +0.3900225f, +0.1744056f, +0.3501963f,
    -0.0538935f, +0.3876384f, -0.2107446f, +0.3298282f, -0.3747638f, -0.1134008f, +0.1743135f, -0.3507513f,
    +0.3902712f, +0.0355134f, +0.0830481f, +0.3834962f, -0.1025457f, +0.3788385f, +0.2971109f, -0.2572852f,
    +0.0492296f, -0.3900567f, -0.0116625f, -0.3931300f, +0.3773220f, +0.1113828f, -0.2876660f, -0.2685882f,
    -0.0780598f, +0.3857549f, +0.2402259f, -0.3122774f, -0.3461589f, -0.1885369f, +0.3944442f, +0.0140554f,
    -0.0333985f, -0.3933825f, +0.1195356f, +0.3763476f, +0.0884359f, -0.3849927f, -0.1760984f, -0.3542739f,
    -0.1965822f, +0.3437565f, +0.0403464f, +0.3947416f, +0.3812406f, -0.1111152f, +0.2511337f, +0.3082331f,
    +0.3679624f, -0.1513274f, +0.2009144f, -0.3435609f, -0.2473756f, +0.3118697f, +0.3943137f, +0.0549014f,
    -0.3849401f, +0.1019594f, +0.3483132f, -0.1938475f, -0.3744522f, -0.1377100f, +0.1978505f, +0.3464835f,
    -0.3728147f, +0.1423445f, -0.0052023f, +0.3991235f, +0.3424603f, +0.2059735f, +0.3964589f, -0.0505894f,
    -0.3976829f, -0.0409972f, -0.3372436f, -0.2152080f, +0.3760912f, +0.1371583f, +0.1617407f, -0.3662662f,
    -0.3100705f, +0.2534183f, -0.3481390f, +0.1979911f, -0.2745193f, -0.2918823f, +0.2871707f, +0.2795130f,
    -0.3091583f, -0.2554998f, +0.2274564f, +0.3304052f, -0.3823219f, +0.1215811f, +0.2227126f, -0.3337888f,
    -0.3240227f, -0.2368232f, +0.3557467f, +0.1863013f, +0.3301754f, -0.2290672f, +0.4021220f, -0.0077226f,
    +0.1650227f, +0.3672189f, +0.1447435f, +0.3756851f, -0.1584161f, -0.3701389f, -0.3599316f, +0.1805946f,
    +0.3955770f, +0.0756191f, -0.2982061f, +0.2707460f, +0.3082639f, +0.2595231f, -0.3390349f, +0.2180863f,
    -0.3985083f, +0.0623467f, -0.2342419f, +0.3285001f, -0.3948016f, +0.0832754f, -0.4037317f, -0.0000445f,
    -0.3981664f, -0.0673380f, -0.1635282f, +0.3696051f, -0.2702329f, +0.3005462f, +0.1443420f, -0.3779727f,
    +0.0664524f, +0.3991933f, -0.3281568f, +0.2375785f, -0.2415328f, -0.3254223f, +0.1032855f, +0.3922183f,
    -0.1389309f, -0.3811482f, +0.3247215f, +0.2448955f, +0.2804246f, -0.2946409f, +0.3661997f, -0.1774388f,
    -0.1158009f, -0.3901746f, -0.4066229f, +0.0217246f, -0.3734744f, -0.1623714f, +0.3744819f, +0.1602226f,
    +0.0727976f, -0.4013309f, +0.1889594f, -0.3616114f, +0.3847831f, -0.1361918f, +0.1249010f, -0.3888012f,
    +0.4021074f, -0.0720480f, -0.3731765f, +0.1663130f, -0.1238968f, +0.3896505f, +0.3401239f, +0.2271778f,
    -0.4084178f, -0.0240428f, +0.2745821f, +0.3033044f, -0.0948163f, -0.3981678f, +0.0067146f, -0.4095027f,
    +0.2996349f, -0.2796214f, -0.1861019f, +0.3655413f, +0.0185345f, +0.4097844f, +0.3207355f, -0.2558556f,
    -0.2645228f, -0.3138375f, -0.2167652f, -0.3486103f, +0.3992044f, +0.0963540f, +0.2632982f, -0.3153445f,
    -0.4087975f, +0.0410582f, -0.0714564f, -0.4047506f, -0.0989038f, +0.3989518f, -0.1471477f, +0.3838841f,
    -0.4009703f, -0.0913035f, +0.4108139f, -0.0314564f, -0.3966886f, -0.1115372f, +0.4011553f, -0.0945765f,
    +0.3507660f, -0.2169327f, +0.0308464f, -0.4113218f, +0.1016588f, -0.3998313f, -0.1964376f, -0.3634428f,
    -0.0245796f, +0.4126218f, +0.4125368f, +0.0278120f, -0.0714611f, +0.4074639f, +0.3951823f, +0.1228484f,
    -0.2933144f, +0.2919550f, -0.0188010f, -0.4138017f, +0.1254256f, +0.3949005f, -0.3708854f, -0.1849109f,
    -0.0477573f, +0.4120693f, +0.1934356f, +0.3674553f, +0.4110599f, +0.0633978f, +0.2501190f, -0.3323433f,
    +0.2508003f, +0.3318869f, -0.0528303f, -0.4127522f, -0.3607014f, -0.2075045f, -0.3014685f, -0.2870437f,
    +0.2179977f, +0.3551337f, -0.2598697f, +0.3264786f, +0.4171779f, +0.0092299f, +0.0530111f, +0.4140701f,
    -0.3955292f, -0.1339605f, -0.3925922f, +0.1426715f, -0.2069589f, +0.3632507f, +0.0875250f, +0.4088662f,
    +0.0527279f, -0.4148443f, +0.4012606f, -0.1187645f, -0.2249162f, +0.3534454f, -0.3490512f, -0.2324462f,
    +0.3119015f, +0.2819116f, +0.3876443f, -0.1630480f, -0.4074234f, +0.1047437f, +0.3784370f, +0.1840371f,
    -0.1775993f, -0.3817284f, +0.4186068f, +0.0467016f, +0.1628754f, +0.3890559f, +0.3704129f, -0.2017140f,
    -0.3360550f, -0.2549722f, -0.3212625f, -0.2734137f, -0.4219908f, -0.0086481f, -0.4186847f, -0.0544406f,
    -0.4141996f, +0.0818643f, -0.2393598f, -0.3478859f, +0.3673987f, +0.2081900f, -0.3663337f, +0.2100922f,
    +0.2170263f, -0.3629432f, +0.1697448f, -0.3877071f, -0.4188524f, +0.0611157f, -0.3311616f, +0.2637554f,
    +0.4235910f, -0.0109378f, +0.3964849f, +0.1498176f, +0.2396876f, -0.3496470f, -0.2903905f, -0.3089373f,
    -0.0008678f, +0.4240212f, -0.2462728f, +0.3459209f, +0.3470953f, -0.2447716f, -0.4178372f, -0.0780696f,
    +0.4215080f, -0.0559742f, +0.2983559f, +0.3031434f, -0.0373770f, -0.4237817f, +0.0354493f, +0.4239491f,
    +0.1823276f, +0.3844697f, -0.4242118f, -0.0337219f, -0.1567356f, -0.3956405f, -0.3180536f, +0.2828262f,
    -0.4254670f, +0.0121786f, -0.4068843f, +0.1251698f, +0.1924250f, -0.3797508f, -0.1322804f, -0.4047373f,
    -0.3460261f, +0.2483625f, -0.2895075f, +0.3126520f, -0.3813179f, +0.1902976f, -0.3956994f, -0.1582895f,
    +0.3470412f, +0.2475736f, -0.3927222f, +0.1660863f, -0.1749137f, +0.3889037f, +0.4190446f, +0.0802794f,
    -0.3587508f, +0.2316261f, +0.0746332f, -0.4206171f, +0.3015640f, -0.3033313f, +0.3207751f, -0.2830834f,
    +0.2744360f, +0.3282549f, +0.3340862f, +0.2674690f, +0.1489586f, +0.4013086f, +0.1103653f, +0.4138792f,
    +0.2859528f, -0.3189233f, +0.2420599f, +0.3543103f, -0.1109111f, -0.4146442f, +0.1220057f, -0.4119627f,
    +0.1495416f, -0.4028554f, -0.1392565f, +0.4068320f, -0.1177615f, +0.4138952f, +0.3632873f, +0.2306518f,
    -0.2784497f, +0.3281366f, -0.0933255f, +0.4201391f, -0.0045225f, -0.4304005f, +0.0972698f, -0.4193506f,
    +0.3887098f, -0.1858353f, -0.0883495f, -0.4223565f, +0.4163955f, +0.1134453f, -0.2151175f, -0.3744548f,
    +0.0728417f, +0.4258119f, +0.4241658f, -0.0824068f, -0.4308356f, +0.0352878f, +0.4182035f, -0.1112216f,
    +0.3402789f, -0.2677518f, -0.4161650f, -0.1196439f, -0.2646661f, -0.3428636f, +0.3975005f, +0.1721780f,
    +0.0166776f, -0.4330609f, +0.4105284f, -0.1390799f, +0.4130834f, +0.1315039f, -0.2803623f, -0.3308547f,
    +0.3688382f, -0.2285243f, -0.3938482f, -0.1825495f, +0.1333265f, +0.4137591f, -0.1587253f, +0.4053957f,
    +0.2130312f, +0.3805164f, +0.4354815f, -0.0315588f, -0.3101687f, +0.3073083f, -0.1955973f, -0.3906578f,
    +0.0405523f, -0.4350427f, +0.2764866f, -0.3384806f, -0.0656018f, +0.4321450f, -0.0186073f, +0.4370852f,
    -0.3747611f, -0.2257333f, -0.0411700f, +0.4356725f, +0.4337637f, +0.0612676f, +0.4377978f, +0.0158359f,
    -0.4275646f, -0.0968399f, -0.2353318f, -0.3699422f, -0.2197398f, +0.3798268f, +0.0185087f, +0.4386934f,
    -0.3892506f, -0.2034185f, +0.4287905f, +0.0955230f, +0.4376760f, +0.0384764f, -0.1980291f, +0.3925785f,
    -0.0686537f, -0.4347748f, -0.4365655f, +0.0568217f, -0.3622173f, -0.2503729f, -0.4156218f, -0.1460028f,
    -0.4138863f, +0.1519068f, -0.4334303f, +0.0816614f, +0.0959944f, +0.4306147f, -0.4411532f, -0.0192054f,
    +0.2977096f, +0.3264582f, -0.3303703f, -0.2936529f, +0.4101932f, -0.1649209f, +0.3306810f, +0.2934710f,
    +0.0632680f, -0.4379491f, -0.4306965f, +0.1021792f, +0.2346458f, +0.3753583f, +0.2674633f, +0.3527265f,
    -0.0237810f, -0.4421246f, +0.4428774f, -0.0061048f, -0.3159484f, -0.3106202f, -0.4408765f, -0.0440822f,
    -0.2457236f, +0.3688870f, -0.2672825f, +0.3537945f, +0.0536398f, +0.4401989f, -0.4376808f, -0.0715604f,
    +0.4167949f, +0.1522369f, +0.2453595f, -0.3697838f, +0.2153074f, -0.3887395f, -0.4444265f, +0.0033577f,
    +0.1746892f, +0.4089399f, +0.1968293f, -0.3989261f, -0.1497337f, -0.4189102f, -0.4055272f, +0.1829837f,
    +0.3919227f, -0.2107717f, -0.3929115f, +0.2091982f, +0.3995024f, +0.1965201f, -0.3486736f, -0.2781222f,
    +0.1773903f, -0.4095194f, +0.3217871f, -0.3095013f, +0.1976727f, +0.4003337f, +0.2673161f, -0.3580819f,
    +0.4360501f, -0.0980394f, -0.3397722f, +0.2903670f, +0.4413677f, -0.0704819f, -0.3824412f, +0.2318439f,
    +0.3915544f, +0.2167207f, -0.4297870f, +0.1262014f, +0.0895077f, -0.4391135f, -0.0486263f, -0.4456511f,
    +0.1383343f, -0.4264521f, -0.1277395f, -0.4299069f, +0.4460313f, -0.0496065f, -0.3688512f, +0.2557321f,
    -0.3041316f, -0.3304390f, -0.3556189f, +0.2743044f, -0.1344432f, +0.4288660f, +0.3669397f, +0.2595430f,
    -0.2621790f, -0.3657093f, +0.1249962f, +0.4326424f, -0.1816199f, +0.4122660f, -0.1753539f, -0.4150458f,
    +0.1162029f, -0.4357386f, -0.0912196f, +0.4417569f, +0.3403518f, -0.2960812f, -0.4506934f, +0.0247519f,
    +0.3213782f, +0.3172008f, +0.3711421f, -0.2575693f, -0.2916668f, +0.3451152f, +0.3044885f, -0.3340027f,
    +0.1513904f, +0.4261131f, -0.1040982f, -0.4401112f, +0.4454556f, +0.0795657f, +0.3846768f, +0.2384069f,
    -0.4189769f, -0.1713593f, -0.1148331f, +0.4381673f, +0.4359668f, -0.1239371f, -0.3319520f, +0.3089729f,
    +0.4116716f, -0.1910860f, +0.1595908f, -0.4250845f, -0.3091756f, +0.3325405f, +0.3553295f, +0.2828012f,
    -0.0000178f, -0.4543510f, -0.0012852f, +0.4543991f, +0.3593130f, -0.2782998f, +0.2361188f, -0.3885707f,
    +0.0749581f, +0.4487793f, +0.4423050f, +0.1069514f, +0.3890453f, -0.2367883f, +0.0278088f, -0.4550012f,
    -0.2245705f, -0.3968523f, -0.4424783f, -0.1121586f, +0.4317715f, -0.1482241f, -0.2380460f, +0.3896684f,
    +0.4218688f, +0.1748741f, -0.2031525f, -0.4090628f, +0.4558267f, +0.0295466f, +0.4373138f, +0.1331852f,
    +0.2920440f, +0.3517358f, -0.1588271f, +0.4287460f, -0.4550201f, +0.0458730f, -0.2907965f, -0.3537825f,
    -0.3707696f, -0.2692723f, +0.4548006f, +0.0569505f, -0.4012144f, -0.2217475f, +0.0378609f, +0.4568973f,
    -0.3868552f, -0.2463211f, -0.4383812f, -0.1355845f, -0.4550714f, -0.0593975f, +0.4592899f, -0.0199552f,
    -0.4547018f, +0.0695394f, +0.2196079f, +0.4042950f, -0.2509197f, -0.3862834f, +0.0993943f, +0.4502636f,
    +0.4612286f, +0.0062327f, -0.4600670f, -0.0354330f, -0.4525384f, -0.0903719f, -0.4270096f, +0.1750265f,
    +0.2621889f, +0.3803422f, -0.0286523f, +0.4610974f, +0.1756545f, +0.4279555f, -0.4197138f, -0.1953875f,
    -0.4534152f, +0.0941588f, -0.0534418f, +0.4600576f, -0.4630696f, -0.0129183f, -0.4374154f, +0.1533489f,
    -0.4487856f, +0.1160834f, -0.0847607f, -0.4557440f, -0.4174948f, +0.2022900f, -0.2218116f, +0.4077508f,
    +0.0490379f, -0.4616034f, -0.0319369f, -0.4638641f, +0.4326717f, -0.1713449f, +0.0734203f, -0.4595390f,
    +0.2970009f, -0.3584589f, +0.4391474f, +0.1552377f, +0.2427921f, +0.3975221f, +0.4569457f, -0.0911519f,
    -0.3503627f, -0.3075868f, -0.2672342f, +0.3820928f, -0.3323355f, -0.3274470f, -0.4666051f, +0.0072816f,
    +0.2012247f, -0.4212186f, +0.3509241f, +0.3081567f, +0.4120365f, -0.2199157f, +0.4650410f, -0.0434193f,
    -0.4387653f, -0.1603422f, -0.2853378f, +0.3698767f, +0.2676808f, -0.3832687f, -0.1543741f, -0.4415014f,
    +0.3267089f, -0.3349372f, -0.3881485f, +0.2615534f, -0.0781313f, +0.4615113f, -0.2015808f, +0.4225095f,
    +0.4545911f, -0.1131797f, +0.0170670f, +0.4684332f, +0.1020934f, -0.4575264f, +0.4640944f, -0.0662547f,
    -0.0630443f, -0.4645712f, +0.3165775f, +0.3458289f, -0.3301800f, +0.3330834f, +0.1178902f, +0.4541625f,
    +0.4268187f, +0.1960896f, -0.4119246f, +0.2266086f, -0.3753966f, +0.2830961f, +0.1828995f, -0.4333675f,
    +0.3929515f, +0.2585608f, -0.4011934f, +0.2461920f, +0.2243966f, -0.4138483f, +0.4170181f, +0.2185722f,
    +0.4561528f, +0.1197521f, -0.1319668f, -0.4532219f, -0.3626116f, +0.3025004f, -0.3185843f, -0.3486341f,
    -0.3717474f, -0.2916819f, +0.0583576f, +0.4689690f, -0.1825959f, +0.4360733f, +0.3469128f, -0.3212184f,
    +0.2019105f, +0.4275750f, +0.1468221f, -0.4496376f, +0.4628558f, +0.0975537f, +0.1380731f, +0.4526816f,
    +0.2871731f, +0.3764729f, +0.0826267f, +0.4662349f, +0.3785003f, -0.2849750f, +0.3949684f, -0.2616804f,
    -0.4532570f, +0.1379842f, +0.3634979f, -0.3039219f, -0.2853139f, -0.3785009f, +0.4718861f, +0.0449318f,
    -0.1992835f, -0.4301870f, -0.1136999f, +0.4604936f, -0.1409527f, +0.4529393f, +0.4529208f, -0.1411321f,
    +0.4685165f, +0.0745438f, +0.1250742f, -0.4578122f, +0.3399400f, +0.3312129f, -0.3063592f, +0.3625037f,
    +0.0156602f, -0.4746674f, +0.3835998f, +0.2800810f, -0.4690948f, -0.0753193f, +0.1579715f, +0.4483367f,
    -0.4224617f, -0.2184639f, -0.1096169f, -0.4628485f, -0.4749272f, +0.0283993f, +0.2531388f, -0.4028792f,
    -0.3922888f, -0.2697184f, -0.0075556f, -0.4762017f, +0.4330680f, -0.1982903f, -0.2544242f, +0.4029770f,
    +0.4121430f, +0.2421434f, +0.4115860f, -0.2439730f, +0.4784704f, +0.0212391f, -0.2702118f, -0.3957881f,
    -0.1785414f, -0.4450002f, -0.4661908f, -0.1140375f, -0.2266787f, -0.4232419f, -0.4777048f, +0.0494414f,
    -0.4121733f, -0.2467338f, -0.3546775f, +0.3243403f, -0.4781490f, -0.0512464f, -0.4717560f, +0.0942633f,
    +0.1665193f, -0.4514464f, +0.2917160f, -0.3830003f, +0.4483230f, +0.1757426f, -0.2491811f, -0.4122120f,
    -0.0113969f, +0.4821163f, -0.4399989f, +0.1980220f, +0.4605362f, +0.1440508f, +0.1818068f, +0.4470435f,
    +0.3772345f, +0.3013109f, -0.3084815f, -0.3714203f, -0.4612604f, -0.1428314f, -0.1654645f, +0.4536828f,
    +0.4821297f, -0.0298261f, +0.2293635f, +0.4251692f, +0.3230199f, -0.3598658f, +0.4835647f, -0.0066748f,
    +0.1049318f, +0.4722350f, -0.4787058f, +0.0698622f, -0.4464328f, -0.1866199f, +0.3116421f, +0.3705832f,
    -0.4840923f, -0.0269125f, +0.2574283f, +0.4118773f, -0.4711743f, +0.1197186f, +0.0866354f, -0.4783676f,
    +0.0601434f, -0.4824375f, -0.3276389f, +0.3593291f, +0.0344019f, -0.4853197f, -0.3563975f, -0.3314222f,
    +0.4811687f, -0.0807361f, +0.0352458f, +0.4866469f, +0.4772817f, -0.1015446f, -0.2224546f, +0.4344099f,
    +0.4337874f, -0.2239828f, -0.4543352f, +0.1789219f, +0.4718953f, -0.1257318f, -0.0662710f, +0.4843740f,
    +0.4605234f, -0.1645263f, +0.4858730f, -0.0575420f, -0.0386690f, +0.4877560f, -0.0539303f, -0.4863190f,
    +0.4751163f, +0.1170454f, -0.0270944f, -0.4886473f, -0.2972625f, +0.3888880f, -0.4037871f, +0.2768961f,
    +0.4530612f, -0.1857797f, -0.4807994f, -0.0931329f, -0.4608487f, -0.1660750f, -0.0955923f, +0.4805374f,
    +0.3630218f, +0.3291350f, -0.4898918f, +0.0140371f, +0.2123810f, -0.4417850f, +0.0115723f, +0.4905434f,
    -0.0851552f, -0.4834898f, -0.4647631f, +0.1587671f, -0.4358633f, +0.2265084f, -0.4203668f, +0.2545736f,
    +0.4832551f, +0.0913908f, -0.1536398f, -0.4675281f, +0.4880659f, +0.0635497f, +0.4416365f, +0.2173288f,
    -0.3770320f, -0.3166320f, +0.3387192f, +0.3574368f, -0.2505487f, +0.4247728f, -0.4931517f, -0.0076152f,
    +0.4092689f, +0.2754635f, -0.3423084f, -0.3552857f, +0.1066499f, -0.4819298f, +0.1271774f, +0.4769787f,
    -0.3927686f, +0.2992626f, +0.2834080f, +0.4046732f, +0.4014208f, -0.2882170f, +0.3494324f, -0.3502409f,
    -0.4479260f, -0.2103855f, +0.1893199f, -0.4572736f, +0.3692981f, -0.3295232f, +0.2410979f, -0.4324431f,
    +0.4938950f, +0.0395740f, -0.1263754f, -0.4791096f, +0.0604299f, +0.4918531f, +0.2816747f, -0.4078428f,
    -0.2797760f, +0.4091752f, -0.3962880f, -0.2980394f, +0.0834320f, +0.4888828f, +0.1498802f, -0.4728433f,
    -0.3519323f, +0.3500587f, -0.1966607f, +0.4563391f, +0.1505020f, +0.4739577f, -0.4928005f, -0.0683437f,
    +0.1285298f, -0.4807263f, -0.1225734f, +0.4824384f, -0.4131203f, -0.2777697f, +0.3151668f, -0.3854465f,
    +0.4977377f, +0.0125745f, +0.0040639f, -0.4980173f, +0.4565471f, +0.1996942f, -0.4401071f, -0.2344539f,
    +0.3888844f, -0.3124752f, +0.4198132f, -0.2697337f, +0.2635317f, -0.4241726f, -0.2008486f, -0.4578595f,
    -0.1470588f, +0.4778731f, +0.1725874f, +0.4693685f, +0.2001006f, +0.4585526f, -0.3826824f, +0.3224139f,
    -0.2280353f, -0.4454217f, +0.4740975f, +0.1604529f, +0.4546931f, -0.2096801f, +0.4790556f, -0.1480159f,
    -0.4835368f, -0.1330115f, -0.5004800f, +0.0370315f, -0.4944635f, +0.0858676f, -0.3311861f, -0.3776155f,
    -0.2940691f, -0.4072888f, +0.2267939f, +0.4492033f, -0.2724183f, -0.4231881f, +0.4424670f, +0.2400104f,
    +0.2514919f, +0.4362659f, -0.4826984f, +0.1436344f, -0.3214076f, +0.3879490f, -0.5020638f, -0.0461656f,
    -0.4583926f, +0.2099686f, +0.3890345f, +0.3208568f, +0.4313347f, +0.2617577f, +0.4365219f, -0.2534582f,
    +0.4864095f, +0.1349277f, -0.2524834f, -0.4372500f, -0.1805411f, -0.4715561f, -0.4938294f, +0.1060549f,
    +0.5038234f, -0.0443168f, -0.4330632f, -0.2615902f, +0.5059048f, -0.0172107f, +0.0713361f, -0.5011809f,
    +0.3073897f, +0.4028704f, +0.3315805f, +0.3839584f, +0.3417180f, -0.3749927f, +0.0471326f, -0.5052876f,
    +0.4093150f, +0.3008522f, -0.1763147f, +0.4767171f, +0.1061529f, +0.4970790f, -0.3142433f, -0.3997165f,
    -0.0021833f, +0.5087541f, +0.3619238f, +0.3577022f, -0.3443062f, +0.3751913f, +0.1753090f, -0.4781095f,
    -0.5053048f, +0.0632743f, +0.2763592f, +0.4279905f, -0.0824305f, +0.5031700f, -0.1035421f, -0.4996040f,
    -0.4978591f, -0.1118532f, +0.2265497f, -0.4572316f, -0.3637992f, -0.3580832f, -0.0182000f, -0.5101504f,
    -0.2212819f, +0.4600293f, +0.0229537f, -0.5100226f, -0.0556081f, +0.5076803f, +0.4530449f, -0.2358134f,
    +0.4987455f, +0.1105933f, +0.4768479f, +0.1836067f, -0.3778594f, +0.3440566f, -0.2461668f, +0.4480651f,
    +0.4964740f, -0.1226924f, -0.0270324f, +0.5114068f, -0.5121483f, +0.0075973f, -0.4788627f, -0.1819467f,
    -0.4192872f, +0.2944253f, -0.4333974f, +0.2735236f, +0.0255174f, +0.5120277f, -0.5122734f, -0.0205894f,
    -0.4881950f, -0.1568606f, -0.0463757f, -0.5106771f, -0.4457594f, +0.2540541f, +0.5099411f, +0.0566865f,
    +0.4765617f, -0.1904204f, +0.5039126f, -0.0978091f, +0.5062191f, +0.0854279f, -0.3823453f, -0.3430985f,
    +0.0510932f, +0.5117217f, -0.0757309f, -0.5088637f, +0.5094661f, -0.0715922f, +0.5137195f, +0.0317408f,
    +0.3064310f, -0.4136456f, -0.4864010f, +0.1688680f, +0.0925065f, -0.5067675f, -0.4721462f, -0.2068233f,
    -0.2731982f, +0.4371518f, -0.1471338f, -0.4941300f, -0.3088666f, +0.4133414f, +0.3905598f, -0.3378700f,
    -0.5005572f, +0.1277746f, +0.2052677f, -0.4740978f, +0.1141483f, -0.5041292f, +0.3753123f, -0.3554161f,
    -0.4604913f, +0.2348629f, +0.1432383f, -0.4971798f, +0.1851392f, +0.4831471f, +0.3855424f, +0.3452948f,
    +0.4905801f, -0.1667198f, -0.4800195f, +0.1951644f, -0.4027904f, -0.3260549f, +0.4684325f, +0.2220781f,
    +0.4131918f, -0.3132075f, -0.5105265f, -0.0914351f, -0.2213859f, -0.4691289f, -0.4215056f, -0.3034610f,
    +0.0811669f, +0.5130891f, -0.4104046f, +0.3185154f, -0.1692555f, -0.4912590f, -0.1088176f, +0.5082787f,
    +0.1563681f, +0.4959950f, +0.1323182f, +0.5029773f, +0.2158567f, +0.4734060f, +0.5203654f, +0.0062203f,
    +0.4285757f, -0.2969271f, +0.2677732f, -0.4474984f, -0.3555563f, -0.3817914f, -0.4667832f, -0.2330324f,
    +0.4992850f, +0.1531057f, +0.2906070f, -0.4340961f, +0.4361988f, +0.2887072f, -0.1375902f, +0.5049827f,
    -0.5064373f, -0.1342532f, -0.5222301f, +0.0436045f, -0.3720143f, +0.3696983f, -0.4559277f, -0.2592657f,
    -0.2962145f, +0.4328689f, +0.3346088f, -0.4042011f, +0.4432722f, -0.2808380f, +0.3580538f, +0.3836947f,
    +0.3667294f, -0.3755540f, -0.1270623f, -0.5098889f, +0.4773211f, -0.2202988f, +0.2502525f, -0.4625098f,
    +0.2490034f, +0.4632552f, +0.5245619f, -0.0400425f, -0.5249909f, -0.0439302f, -0.5224316f, -0.0698589f,
    -0.5198531f, +0.0870289f, -0.2963851f, -0.4361551f, -0.2739746f, -0.4509670f, -0.2482172f, -0.4657171f,
    -0.4428732f, -0.2870272f, -0.2054331f, -0.4864383f, +0.4554215f, +0.2674539f, +0.4866819f, +0.2052968f,
    -0.2072970f, +0.4858675f, -0.3366697f, +0.4070599f, -0.3390602f, -0.4053180f, +0.0547048f, -0.5256094f,
    +0.1686899f, -0.5009479f, +0.5088358f, -0.1436186f, +0.2740315f, +0.4523272f, +0.0028765f, -0.5290403f,
    +0.4692935f, +0.2442517f, +0.5136116f, +0.1272528f, +0.1132066f, +0.5169982f, +0.2985684f, +0.4371020f,
    -0.1756587f, +0.5004036f, +0.3382207f, +0.4089884f, -0.5304840f, +0.0207826f, +0.4997142f, +0.1799214f,
    -0.0542137f, +0.5286457f, +0.4601302f, -0.2663684f, +0.4148524f, +0.3326854f, -0.5206015f, +0.1100066f,
    -0.4079762f, +0.3419025f, -0.5084428f, +0.1576756f, -0.4825135f, +0.2251081f, +0.4740949f, -0.2428284f,
    +0.3590633f, -0.3938389f, +0.5331672f, -0.0141548f, +0.1319512f, -0.5171441f, -0.3616264f, +0.3925521f,
    -0.2658393f, +0.4628928f, +0.5331129f, +0.0278647f, +0.5286978f, +0.0746245f, -0.5339145f, -0.0053822f,
    +0.2249140f, -0.4844441f, +0.4151684f, -0.3363872f, -0.0103897f, +0.5345194f, +0.4942099f, -0.2040864f,
    +0.0309601f, -0.5341325f, -0.0819774f, +0.5288607f, -0.4982380f, -0.1955504f, +0.0802757f, -0.5294070f,
    +0.5257264f, +0.1018992f, -0.3215963f, -0.4284859f, +0.0128357f, +0.5356278f, -0.4345683f, +0.3134252f,
    +0.1985624f, +0.4977585f, -0.3889917f, -0.3686814f, +0.5333232f, +0.0538396f, -0.0248491f, -0.5357185f,
    -0.2412465f, +0.4794479f, -0.5117044f, -0.1656178f, +0.5251804f, -0.1164748f, -0.4503834f, +0.2947924f,
    +0.1962442f, -0.5013937f, +0.3228835f, +0.4312633f, +0.3244981f, -0.4301028f, +0.5349536f, -0.0641399f,
    -0.1933286f, -0.5030439f, +0.2378592f, +0.4836421f, -0.5352951f, +0.0647839f, -0.4760284f, +0.2536471f,
    +0.5318136f, -0.0903124f, -0.5279228f, -0.1117706f, +0.4375057f, +0.3160934f, -0.0516702f, -0.5375216f,
    -0.0343554f, +0.5389966f, +0.0689932f, +0.5356700f, +0.3913688f, +0.3722027f, +0.0393105f, +0.5389612f,
    +0.1473036f, +0.5199520f, -0.5397486f, -0.0269630f, -0.1042754f, -0.5306165f, -0.4092644f, -0.3534890f,
    +0.1085569f, -0.5300025f, -0.5235479f, +0.1373551f, -0.0779952f, -0.5356452f, +0.5076548f, -0.1878868f,
    -0.1554941f, -0.5185510f, -0.3224215f, +0.4352231f, -0.4270383f, -0.3346107f, +0.4058405f, -0.3603050f,
    -0.4938468f, -0.2260262f, +0.0962415f, +0.5345595f, -0.4672726f, +0.2769030f, -0.1545834f, +0.5214260f,
    -0.5107066f, +0.1873205f, -0.4021178f, +0.3664467f, -0.4424011f, -0.3171542f, -0.3776959f, -0.3921445f,
    +0.3902485f, -0.3796794f, +0.1732591f, +0.5166234f, -0.1198511f, +0.5319985f, +0.4398169f, -0.3225072f,
    -0.5033078f, +0.2101773f, +0.5197856f, -0.1662408f, -0.2328562f, -0.4937210f, +0.5462175f, +0.0090345f,
    -0.4839144f, -0.2541875f, +0.4956706f, +0.2309108f, +0.2905620f, -0.4632994f, +0.1537455f, -0.5248485f,
    -0.5289775f, -0.1391385f, +0.3116124f, -0.4497729f, +0.5207957f, +0.1689308f, +0.4135218f, +0.3589459f,
    +0.5282210f, +0.1476298f, +0.2500790f, -0.4887034f, -0.5457637f, -0.0613437f, -0.2989737f, +0.4609058f,
    +0.4566143f, -0.3055590f, +0.4821488f, -0.2634377f, +0.2732139f, -0.4767355f, +0.5489150f, -0.0361972f,
    +0.3632575f, -0.4133701f, -0.1789809f, -0.5213539f, -0.3639655f, -0.4143470f, -0.5501157f, +0.0411546f,
    +0.3840644f, +0.3960851f, -0.5444499f, -0.0898144f, +0.4626741f, +0.3007471f, -0.3057348f, -0.4594666f,
    -0.2594032f, -0.4874178f, +0.3655017f, +0.4138873f, +0.5055493f, -0.2227912f, +0.1249308f, +0.5383890f,
    -0.4764100f, -0.2804239f, -0.4373323f, +0.3386652f, -0.5460464f, +0.0889607f, +0.4889178f, +0.2590321f,
    +0.4728420f, -0.2878588f, -0.3638128f, +0.4175200f, +0.2209099f, +0.5079150f, +0.5144395f, +0.2057800f,
    -0.3912612f, +0.3925008f, +0.5530990f, +0.0374879f, -0.1344232f, -0.5379667f, +0.5368636f, -0.1389661f,
    -0.2832524f, -0.4767880f, +0.2641232f, +0.4878432f, +0.0150941f, -0.5551652f, -0.5552559f, +0.0124495f,
    +0.5424802f, +0.1195370f, +0.3097775f, +0.4612529f, +0.4797951f, +0.2806298f, +0.2860005f, +0.4766717f,
    +0.3482367f, -0.4335900f, +0.4368902f, -0.3444044f, -0.2063244f, +0.5167531f, +0.5494408f, +0.0878833f,
    -0.3474957f, -0.4346108f, -0.2307555f, +0.5068770f, -0.5440103f, +0.1194390f, -0.2779356f, +0.4831198f,
    +0.1793310f, -0.5278892f, +0.4414143f, +0.3412873f, -0.0091282f, -0.5579087f, +0.0613483f, -0.5547039f,
    -0.1807406f, +0.5280827f, -0.5244704f, -0.1914829f, -0.4662845f, -0.3071654f, +0.2238996f, -0.5116978f,
    +0.1353611f, -0.5425920f, -0.5341350f, +0.1664044f, -0.0933452f, +0.5520229f, -0.3471940f, +0.4394088f,
    +0.1993480f, +0.5236350f, +0.5602062f, -0.0113536f, -0.2142246f, -0.5180203f, -0.4083429f, -0.3841556f,
    -0.0642370f, +0.5571829f, -0.4918954f, +0.2695867f, +0.0538225f, +0.5584537f, -0.0400564f, -0.5596848f,
    -0.5596229f, -0.0412034f, +0.0906079f, -0.5537924f, +0.3890584f, -0.4043989f, +0.5579768f, +0.0633374f,
    -0.5364038f, -0.1662823f, +0.5034695f, -0.2489800f, -0.0069406f, +0.5617200f, -0.5068990f, +0.2422913f,
    +0.3508599f, +0.4396945f, -0.4297944f, -0.3629829f, +0.1504529f, +0.5423262f, -0.4615580f, +0.3225696f,
    +0.5605956f, -0.0578682f, -0.3295970f, -0.4576416f, -0.5202582f, -0.2197431f, +0.2468898f, +0.5079534f,
    +0.0235904f, +0.5642945f, -0.1465584f, +0.5454704f, +0.5315873f, +0.1916181f, +0.0813768f, +0.5592560f,
    +0.5592139f, -0.0823793f, -0.5655023f, -0.0131835f, -0.0356983f, +0.5647902f, -0.4530638f, -0.3399201f,
    -0.3274803f, +0.4621962f, -0.4338695f, +0.3642058f, -0.3923546f, -0.4089389f, +0.5562062f, -0.1096753f,
    +0.1170095f, -0.5550041f, +0.0368495f, -0.5661787f, -0.0692538f, -0.5633297f, +0.5319170f, -0.1980689f,
    +0.4141139f, -0.3883738f, -0.5490218f, +0.1449110f, +0.3116860f, -0.4747014f, -0.2582006f, +0.5058935f,
    -0.5098217f, -0.2505915f, -0.4782001f, +0.3067830f, -0.2482731f, -0.5112709f, +0.4337989f, -0.3674790f,
    +0.2046311f, -0.5312480f, +0.5451210f, -0.1656977f, +0.5694212f, +0.0219479f, -0.5573304f, -0.1188789f,
    +0.4660008f, +0.3282622f, -0.0956457f, -0.5619402f, -0.5669034f, +0.0641132f, +0.2514439f, -0.5127974f,
    +0.5226765f, +0.2312010f, +0.3406166f, -0.4592245f, +0.4988502f, -0.2794872f, -0.5273782f, +0.2211471f,
    -0.5671978f, -0.0734982f, -0.1204988f, +0.5592757f, +0.4362055f, +0.3704488f, +0.1091259f, +0.5617865f,
    -0.1612163f, -0.5491686f, +0.5538571f, +0.1450289f, +0.1787832f, +0.5440264f, +0.4165775f, +0.3932790f,
    -0.5008000f, -0.2784367f, +0.3787813f, -0.4299590f, -0.3901756f, +0.4202380f, +0.5281292f, -0.2240525f,
    -0.1220608f, -0.5607014f, +0.1603122f, -0.5511362f, +0.4694441f, -0.3303238f, -0.5400318f, +0.1950986f,
    +0.3338843f, +0.4671941f, +0.3985789f, +0.4135519f, +0.5731868f, -0.0377245f, -0.5561060f, -0.1449333f,
    +0.5477559f, +0.1738817f, -0.5649210f, +0.1067232f, +0.4841113f, -0.3105220f, -0.5740068f, +0.0364607f,
    -0.4201029f, +0.3929482f, +0.2786163f, -0.5036602f, +0.5161548f, +0.2563668f, -0.3078713f, +0.4872588f,
    +0.5676971f, +0.1020364f, -0.1945071f, -0.5433724f, -0.2801092f, -0.5059814f, -0.3765387f, -0.4403113f,
    +0.1344399f, +0.5637978f, -0.3240473f, -0.4806304f, +0.5072396f, +0.2817605f, +0.4603066f, -0.3533062f,
    -0.4643828f, +0.3479935f, +0.0090178f, -0.5802709f, +0.5807613f, +0.0029045f, +0.2829905f, +0.5071634f,
    +0.2307455f, +0.5330546f, +0.5638772f, -0.1400820f, -0.4995614f, +0.2967827f, -0.3035549f, -0.4955908f,
    -0.1739473f, +0.5546585f, +0.4935491f, +0.3078444f, -0.5170997f, +0.2674296f, -0.5814357f, -0.0346275f,
    +0.2023460f, +0.5462363f, +0.5808844f, +0.0450493f, +0.3839799f, +0.4382418f, -0.3519176f, -0.4647183f,
    +0.5780758f, +0.0774161f, -0.5750593f, -0.0985377f, -0.2048133f, +0.5465997f, -0.2379018f, -0.5330377f,
    -0.4167570f, -0.4088469f, -0.5778137f, +0.0839099f, +0.0807310f, -0.5783960f, +0.3120241f, +0.4936912f,
    -0.2376242f, +0.5340773f, +0.3707054f, -0.4520214f, -0.0204807f, -0.5842812f, +0.5709389f, +0.1258879f,
    -0.5521728f, -0.1928169f, -0.0864865f, +0.5786749f, -0.2870622f, +0.5098574f, -0.3767861f, +0.4478393f,
    -0.4362717f, -0.3902951f, +0.4145337f, -0.4135364f, -0.4979733f, -0.3084175f, +0.3037938f, -0.5008419f,
    -0.4831015f, -0.3323890f, +0.0584496f, +0.5836056f, -0.5865971f, +0.0142574f, +0.1096771f, -0.5765361f,
    +0.2381069f, -0.5365723f, +0.0019467f, +0.5870950f, -0.5715915f, +0.1343243f, -0.5632271f, +0.1667787f,
    -0.0530163f, +0.5858078f, +0.5322741f, -0.2509130f, +0.1628881f, +0.5656360f, -0.4557487f, -0.3726537f,
    -0.4129794f, +0.4195758f, -0.3569229f, +0.4681915f, -0.0530920f, -0.5863779f, +0.0534764f, -0.5864492f,
    +0.5853162f, -0.0648846f, +0.1892131f, -0.5578313f, +0.5822769f, -0.0915840f, +0.5576245f, -0.1913782f,
    +0.2609903f, +0.5286957f, -0.5372818f, -0.2430149f, +0.4670297f, +0.3605578f, +0.0861836f, +0.5837188f,
    +0.5476228f, +0.2200128f, -0.3999932f, -0.4349200f, -0.5661456f, -0.1703614f, +0.1369876f, -0.5754308f,
    -0.5494610f, -0.2197651f, +0.5916072f, -0.0213242f, +0.4382066f, -0.3980368f, -0.4732340f, -0.3563765f,
    +0.5688565f, +0.1664479f, -0.0272029f, +0.5924136f, -0.1163129f, +0.5817399f, -0.1474345f, +0.5749018f,
    +0.5093434f, -0.3047703f, +0.4031943f, -0.4361002f, -0.4494209f, +0.3883662f, +0.5258577f, -0.2763813f,
    -0.1464691f, -0.5760382f, +0.0317233f, +0.5935960f, -0.3358544f, +0.4905934f, +0.3331987f, -0.4924875f,
    -0.2222255f, -0.5516450f, +0.4889555f, +0.3388155f, -0.0865724f, -0.5888212f, -0.5316448f, -0.2688769f,
    +0.3696011f, +0.4672790f, -0.5416812f, +0.2483699f, -0.5923988f, -0.0665031f, +0.5837896f, -0.1212326f,
    -0.5964717f, -0.0103903f, +0.5515368f, -0.2282906f, -0.2677392f, -0.5335546f, +0.5728814f, -0.1699832f,
    -0.5548959f, +0.2217913f, +0.4617330f, -0.3801310f, +0.4543871f, +0.3889431f, -0.5960414f, +0.0519280f,
    +0.2676814f, -0.5353686f, -0.4944547f, +0.3376124f, +0.0277346f, -0.5981861f, +0.5639070f, +0.2020805f,
    +0.4169885f, +0.4300722f, -0.2675200f, +0.5362125f, -0.4732180f, +0.3694877f, -0.5916870f, +0.1031995f,
    +0.3638144f, -0.4779096f, +0.5467912f, +0.2487256f, +0.1130125f, +0.5900779f, +0.2231562f, +0.5579815f,
    -0.5714108f, +0.1874560f, -0.1794701f, -0.5740905f, +0.3472816f, +0.4912183f, -0.5252632f, -0.2933773f,
    -0.1147694f, -0.5907131f, +0.2180805f, -0.5609796f, -0.4039861f, +0.4462863f, -0.5836300f, -0.1492899f,
    +0.6017353f, +0.0300226f, -0.3764766f, -0.4703972f, +0.1454146f, +0.5847459f, -0.3250581f, -0.5074456f,
    +0.1649002f, -0.5796835f, -0.5907464f, -0.1194257f, +0.6009806f, -0.0454738f, +0.5928766f, +0.1090959f,
    +0.1907554f, +0.5722106f, -0.5352674f, +0.2783981f, +0.6038021f, +0.0051442f, +0.4841425f, -0.3610778f,
    -0.3129854f, +0.5176643f, +0.5013930f, -0.3385027f, +0.6017218f, +0.0629202f, +0.5261252f, +0.2988521f,
    -0.3526751f, -0.4920692f, +0.5862053f, +0.1525005f, -0.2965036f, -0.5285647f, -0.4397670f, +0.4172485f,
    +0.1083148f, -0.5969087f, +0.5418459f, +0.2736685f, -0.5199348f, +0.3134681f, +0.2952974f, +0.5305405f,
    -0.0719136f, +0.6035506f, -0.5945216f, +0.1282624f, +0.5141295f, +0.3249679f, +0.4425661f, +0.4173934f,
    -0.4417128f, -0.4187132f, -0.6075895f, -0.0387305f, -0.2042777f, +0.5738981f, +0.0742509f, -0.6046561f,
    -0.2077984f, -0.5726936f, -0.0003645f, -0.6095099f, +0.2962965f, -0.5332259f, +0.3880869f, -0.4707297f,
    -0.0312487f, -0.6097388f, -0.2371599f, +0.5634536f, -0.1740974f, +0.5861955f, -0.3824619f, +0.4775388f,
    -0.5734359f, -0.2135351f, -0.5118237f, -0.3358809f, -0.6120343f, +0.0308998f, +0.3285738f, +0.5173974f,
    -0.4825040f, -0.3779690f, -0.4649091f, -0.3996129f, +0.5748286f, -0.2131212f, -0.4250774f, -0.4417818f,
    -0.0987637f, +0.6051073f, +0.6072008f, +0.0879895f, -0.6090615f, +0.0782680f, +0.5953000f, -0.1511364f,
    +0.4365019f, -0.4321844f, -0.1283015f, +0.6009195f, +0.6076838f, -0.0913242f, +0.4042133f, +0.4631039f,
    +0.2486073f, +0.5623356f, -0.2513712f, -0.5611291f, +0.0631442f, +0.6117097f, -0.6081796f, -0.0926966f,
    +0.0169310f, +0.6149840f, -0.5951751f, +0.1561884f, -0.0110314f, +0.6152686f, +0.2467011f, -0.5638597f,
    +0.3593068f, -0.5002345f, -0.5012048f, -0.3586800f, +0.5017081f, +0.3580190f, +0.5603991f, -0.2566362f,
    -0.0674157f, -0.6128287f, -0.3428243f, +0.5125519f, +0.4118576f, -0.4589550f, +0.0472952f, -0.6148928f,
    +0.3200887f, -0.5271378f, +0.2724521f, +0.5535519f, +0.1680747f, +0.5938957f, +0.6135783f, -0.0673395f,
    -0.6172857f, +0.0057390f, +0.5888770f, +0.1853038f, -0.5632173f, -0.2529096f, +0.6029898f, +0.1337272f,
    +0.6172385f, -0.0299708f, +0.5267777f, -0.3234051f, +0.4638043f, -0.4088347f, -0.5305852f, -0.3182438f,
    +0.0898536f, +0.6122468f, +0.1982240f, -0.5862498f, +0.1328258f, +0.6046381f, +0.5519723f, -0.2808096f,
    -0.5890624f, -0.1914151f, +0.4863628f, +0.3839782f, -0.4079531f, -0.4666133f, +0.5410169f, -0.3032928f,
    +0.5731665f, +0.2373182f, +0.1438804f, -0.6036633f, -0.0456667f, +0.6191059f, -0.5816006f, +0.2171201f,
    +0.2169053f, +0.5817592f, -0.1340297f, -0.6068102f, +0.6105068f, -0.1172554f, -0.2982948f, +0.5459229f,
    -0.5452096f, +0.2998216f, +0.3435265f, -0.5193177f, +0.6227613f, -0.0047519f, +0.5925549f, -0.1921957f,
    -0.4352266f, +0.4462002f, -0.6203532f, -0.0623984f, -0.6145443f, +0.1082785f, +0.6221093f, +0.0492794f,
    -0.1663783f, -0.6019291f, +0.3862484f, +0.4914122f, +0.4301636f, +0.4535756f, -0.4675603f, +0.4154121f,
    -0.3460267f, -0.5212533f, -0.3710338f, +0.5045929f, +0.0959154f, -0.6193855f, -0.1519412f, +0.6083657f,
    -0.2857286f, -0.5581984f, +0.2802809f, -0.5612996f, -0.5064726f, +0.3706124f, +0.4750244f, +0.4104896f,
    -0.5613625f, -0.2815844f, -0.4878290f, +0.3958023f, +0.5506722f, +0.3025167f, -0.6289335f, -0.0169345f,
    +0.5925572f, +0.2116329f, -0.4140439f, +0.4739510f, -0.2738793f, +0.5666572f, +0.4908236f, -0.3940412f,
    -0.3902130f, -0.4940022f, -0.6271778f, +0.0545167f, -0.6154116f, -0.1336954f, +0.5807604f, -0.2439366f,
    +0.1932055f, +0.5996478f, -0.3229900f, -0.5409480f, +0.5702553f, +0.2682003f, -0.5786301f, +0.2498038f,
    +0.1782743f, -0.6046989f, +0.3643180f, +0.5148356f, -0.5272913f, +0.3473875f, +0.2275458f, -0.5890173f,
    -0.5687332f, +0.2746794f, -0.1008130f, -0.6235785f, -0.6010190f, +0.1947916f, -0.6098794f, -0.1649913f,
    -0.4595258f, -0.4350951f, -0.3703510f, -0.5131550f, +0.0420163f, +0.6315936f, +0.5129515f, -0.3710468f,
    +0.3919533f, -0.4974158f, +0.6330497f, +0.0198949f, +0.1137191f, +0.6231325f, -0.2014639f, -0.6005625f,
    +0.6248720f, +0.1089424f, +0.5379444f, +0.3361570f, -0.0713785f, +0.6305212f, -0.5876117f, -0.2402306f,
    +0.0684326f, -0.6312268f, -0.1983932f, +0.6031432f, -0.4997991f, -0.3916439f, +0.2972695f, +0.5614063f,
    +0.0186218f, -0.6352590f, +0.6135272f, +0.1660260f, +0.6314387f, +0.0764826f, -0.2365990f, -0.5904945f,
    +0.1234862f, -0.6240879f, -0.3293545f, +0.5445501f, +0.3241934f, +0.5478085f, +0.6356937f, -0.0488789f,
    -0.0106703f, -0.6375260f, +0.4397363f, -0.4619220f, -0.2278577f, +0.5957600f, +0.4634114f, -0.4383159f,
    -0.0422100f, -0.6367398f, +0.4596269f, +0.4426977f, -0.5572163f, -0.3110568f, +0.6226721f, -0.1408442f,
    -0.4420152f, -0.4610993f, +0.4197054f, -0.4818906f, +0.1560749f, +0.6197832f, -0.1215010f, +0.6274967f,
    -0.6235204f, +0.1405718f, +0.6340208f, -0.0847225f, -0.5500194f, +0.3267974f, +0.3720990f, -0.5204712f,
    +0.3086470f, -0.5612199f, -0.4867894f, -0.4164499f, +0.5379595f, -0.3478873f, +0.0108295f, +0.6409280f,
    -0.4023434f, +0.4995257f, +0.2412067f, +0.5947775f, +0.6023022f, -0.2241834f, -0.6415852f, -0.0421309f,
    +0.5807530f, -0.2762232f, +0.5163684f, +0.3838081f, -0.6356027f, -0.1004738f, +0.6196318f, -0.1736593f,
    -0.4635164f, +0.4467130f, -0.2651578f, -0.5865949f, -0.6379371f, +0.0865163f, -0.5329478f, -0.3612559f,
    -0.6436344f, +0.0253599f, -0.1734018f, +0.6205387f, -0.3588176f, +0.5351745f, +0.2674541f, -0.5866020f,
    +0.1584666f, -0.6252871f, +0.4165924f, +0.4925607f, -0.6070459f, -0.2184241f, +0.2653172f, +0.5881511f,
    -0.2559279f, +0.5930968f, -0.5482051f, -0.3417224f, +0.3383809f, -0.5503829f, +0.6461104f, -0.0209521f,
    +0.5965520f, +0.2496229f, +0.5357334f, +0.3629617f, -0.6227088f, +0.1765176f, +0.3563912f, +0.5405530f,
    +0.6328242f, +0.1376711f, -0.1539402f, -0.6293439f, -0.0196882f, +0.6476929f, +0.4888041f, -0.4261238f,
    -0.0991717f, +0.6412952f, +0.0715564f, +0.6456415f, -0.4432078f, +0.4751913f, -0.0736612f, -0.6459398f,
    +0.0434175f, -0.6488484f, +0.5751048f, -0.3044353f, -0.6086673f, +0.2302093f, +0.2123494f, -0.6153433f,
    -0.3594877f, -0.5428688f, +0.5633998f, +0.3266930f, -0.3166488f, -0.5697085f, +0.6504904f, +0.0453056f,
    +0.5624929f, -0.3301186f, +0.6414788f, -0.1189823f, -0.5269184f, -0.3856552f, -0.5775257f, +0.3047130f,
    +0.2196564f, +0.6149505f, -0.6494001f, -0.0732195f, +0.0983995f, +0.6460951f, -0.6237467f, -0.1951889f,
    -0.6432697f, +0.1158532f, +0.6225423f, +0.1992325f, +0.1846547f, +0.6273886f, +0.2491240f, -0.6048118f,
    -0.4153917f, -0.5053123f, -0.3149136f, +0.5738658f, -0.1300899f, -0.6416878f, -0.5042122f, +0.4177825f,
    +0.6232199f, -0.2011134f, +0.5824664f, +0.2995675f, -0.1848233f, -0.6288703f, -0.5208692f, +0.3979273f,
    +0.5151427f, -0.4057502f, +0.1126558f, -0.6461959f, -0.5981333f, -0.2695462f, -0.6564578f, -0.0113706f,
    +0.0816117f, -0.6516165f, -0.4401410f, -0.4874429f, +0.1318948f, +0.6435767f, -0.3897727f, -0.5289525f,
    -0.4871265f, -0.4411015f, +0.6498241f, +0.0979413f, +0.4495298f, +0.4798815f, -0.4892526f, +0.4394885f,
    -0.2881565f, +0.5912851f, +0.3976847f, +0.5241641f, +0.5113402f, +0.4140918f, -0.0460671f, +0.6565424f,
    -0.6449088f, -0.1314423f, -0.3952112f, +0.5263470f, +0.4916592f, +0.4376546f, -0.2929600f, -0.5895851f,
    +0.2920083f, +0.5902532f, +0.6180052f, +0.2281362f, +0.3679107f, -0.5465897f, +0.6591769f, +0.0137530f,
    +0.6084296f, -0.2542593f, -0.5392672f, +0.3817357f, -0.6582419f, +0.0588249f, -0.1531122f, +0.6431873f,
    -0.5591767f, +0.3527825f, -0.2235642f, -0.6228011f, -0.4278128f, +0.5048956f, +0.3980775f, -0.5288862f,
    +0.4237039f, -0.5086300f, +0.6010769f, +0.2774535f, +0.2969518f, -0.5917594f, +0.1865587f, -0.6362986f,
    -0.0744010f, +0.6589581f, +0.6601198f, -0.0647906f, +0.3197717f, +0.5811878f, -0.5907933f, -0.3018729f,
    -0.5195403f, -0.4127088f, +0.5404497f, -0.3856053f, +0.4759755f, -0.4636909f, -0.4766974f, -0.4631370f,
    +0.6415636f, +0.1746434f, -0.2213856f, +0.6271372f, -0.6435410f, -0.1689039f, -0.3483306f, -0.5668791f,
    +0.1605162f, +0.6457195f, +0.0407335f, +0.6647338f, -0.1030058f, -0.6581116f, +0.6619607f, +0.0750186f,
    -0.0159404f, -0.6661093f, -0.6017916f, +0.2861500f, +0.6474182f, -0.1588036f, -0.6324247f, +0.2108346f,
    -0.1925778f, +0.6385494f, -0.3514569f, +0.5668539f, +0.1513054f, -0.6502037f, -0.1241817f, +0.6560279f,
    +0.6607319f, -0.0975779f, +0.5378910f, +0.3960138f, -0.2517826f, -0.6188512f, +0.4255455f, +0.5151481f,
    -0.4769210f, +0.4680793f, -0.6464842f, +0.1699964f, +0.0164405f, -0.6686592f, -0.6157107f, +0.2614269f,
    +0.5659754f, +0.3575172f, +0.3499449f, +0.5711395f, +0.3341098f, -0.5810145f, +0.4811084f, +0.4674917f,
    -0.6264623f, -0.2407453f, -0.6696532f, -0.0453945f, +0.6099860f, -0.2801859f, -0.0508325f, -0.6695045f,
    -0.4108777f, -0.5314170f, +0.4582860f, -0.4911994f, -0.5859350f, +0.3292277f, +0.6307976f, -0.2329831f,
    +0.2477324f, +0.6255500f, +0.0099120f, +0.6728086f, -0.3832600f, +0.5535994f, -0.2495976f, +0.6262550f,
    -0.6588120f, +0.1430539f, +0.0607153f, -0.6715606f, +0.6733770f, -0.0375000f, +0.3794200f, +0.5580121f,
    +0.6628826f, +0.1274612f, -0.6749901f, +0.0170032f, -0.5862630f, -0.3351085f, -0.5685277f, -0.3645187f,
    -0.6674004f, -0.1060665f, +0.2449354f, -0.6303772f, +0.5704687f, -0.3639109f, +0.5895717f, +0.3327499f,
    -0.3799335f, -0.5604312f, +0.6772162f, -0.0092679f, +0.5110961f, -0.4447327f, -0.1656725f, -0.6585389f,
    -0.5173217f, -0.4407626f, +0.6286131f, +0.2586508f, -0.6743911f, +0.0887014f, +0.0984757f, -0.6730825f,
    +0.6054316f, -0.3105015f, +0.2081686f, +0.6478133f, +0.2118291f, -0.6466637f, +0.2872860f, +0.6168739f,
    -0.2872485f, -0.6170415f, +0.5916325f, -0.3374866f, +0.6628629f, +0.1586993f, -0.0995296f, +0.6747175f,
    -0.4733827f, -0.4909790f, -0.2815186f, +0.6213555f, +0.1299865f, -0.6700985f, +0.6483665f, -0.2142103f,
    +0.2824883f, -0.6219657f, +0.6516210f, +0.2053531f, -0.4574015f, +0.5075738f, +0.6703649f, -0.1323497f,
    +0.1099769f, +0.6744119f, -0.3381733f, -0.5939052f, +0.6826950f, +0.0359530f, +0.6576495f, -0.1876220f,
    -0.6503850f, -0.2119378f, +0.6118307f, +0.3065664f, -0.6800499f, -0.0781986f, -0.5582149f, -0.3962953f,
    -0.5649765f, +0.3874558f, -0.3179859f, +0.6071811f, -0.2073389f, -0.6533275f, -0.6690960f, -0.1489639f,
    -0.4537684f, -0.5139060f, -0.6836795f, +0.0510697f, -0.5215310f, +0.4450152f, -0.3176975f, -0.6078463f,
    +0.0730469f, +0.6820551f, -0.5424423f, +0.4203590f, +0.5418963f, -0.4216958f, +0.4621776f, +0.5078288f,
    +0.3689363f, -0.5795796f, +0.6840990f, +0.0653967f, -0.3440067f, +0.5957484f, -0.6773747f, +0.1208996f,
    +0.4049227f, +0.5564682f, -0.0273554f, +0.6879080f, +0.5666257f, +0.3915968f, +0.3208396f, -0.6095344f,
    +0.6486747f, +0.2323758f, -0.5874551f, +0.3602100f, -0.6306394f, -0.2778232f, -0.4329117f, +0.5364406f,
    -0.1348793f, -0.6761885f, +0.3977662f, -0.5634163f, -0.0804634f, -0.6853369f, -0.2109854f, +0.6571345f,
    -0.6899729f, -0.0167844f, -0.4832345f, +0.4927953f, +0.1410508f, +0.6757362f, +0.5259399f, +0.4478327f,
    -0.0260624f, -0.6909307f, -0.4353370f, -0.5381070f, +0.0366606f, -0.6914620f, +0.4514322f, -0.5250668f,
    +0.4267847f, -0.5455344f, -0.1738594f, +0.6706475f, +0.3214929f, +0.6140346f, -0.5113024f, -0.4690491f,
    +0.6420363f, -0.2637523f, -0.0725475f, +0.6903273f, -0.6672888f, +0.1914451f, -0.4069123f, -0.5626184f,
    +0.4373244f, +0.5394064f, +0.6869017f, +0.1029364f, +0.6917775f, -0.0634668f, +0.1756224f, +0.6721312f,
    +0.5496784f, +0.4248500f, -0.6475348f, +0.2521271f, -0.4158573f, +0.5568998f, -0.2436388f, -0.6510081f,
    -0.1424169f, +0.6804777f, +0.1816280f, -0.6710996f, -0.6191970f, +0.3170466f, +0.6341377f, +0.2866302f,
    -0.6237118f, -0.3093633f, +0.0808546f, -0.6921133f, +0.5712970f, -0.3995602f, -0.5504863f, -0.4283816f,
    -0.3800279f, +0.5851582f, -0.6607840f, +0.2257272f, +0.5049289f, -0.4823701f, -0.3710776f, -0.5915576f,
    +0.6915826f, -0.0993740f, -0.6738950f, -0.1856976f, +0.6801112f, -0.1624568f, -0.5143375f, +0.4746216f,
    +0.5154890f, +0.4740311f, +0.3496775f, +0.6069879f, +0.6147384f, +0.3359453f, -0.6824394f, +0.1601227f,
    +0.6744220f, +0.1916666f, +0.2783704f, +0.6438394f, +0.2434199f, +0.6579372f, +0.0373088f, +0.7005767f,
    -0.6923288f, -0.1193805f, +0.6004004f, +0.3655082f, +0.7031902f, +0.0067400f, -0.7015401f, -0.0499620f,
    +0.0037018f, -0.7033643f, +0.3748395f, +0.5958339f, -0.2807073f, -0.6457433f, -0.6589655f, -0.2489745f,
    +0.6618331f, -0.2412763f, -0.6174150f, -0.3394596f, -0.6152170f, +0.3435683f, -0.1906085f, -0.6786694f,
    +0.1519355f, -0.6883798f, -0.2755312f, +0.6492758f, +0.4813562f, -0.5159897f, +0.1163380f, -0.6961063f,
    +0.3552788f, -0.6102805f, +0.5381055f, -0.4573689f, -0.6441044f, +0.2903137f, +0.2065485f, +0.6757843f,
    -0.2473489f, +0.6619738f, +0.6930446f, +0.1405303f, -0.7071401f, +0.0060003f, +0.6436352f, -0.2929772f,
    -0.1147337f, +0.6978620f, +0.2452082f, -0.6639658f, +0.5977670f, -0.3792046f, +0.2726591f, -0.6539897f,
    -0.1137964f, -0.6996104f, -0.5724885f, +0.4184715f, +0.0025398f, +0.7091470f, +0.7086922f, -0.0299740f,
    +0.6309536f, -0.3248171f, +0.5003932f, +0.5039756f, -0.3091965f, +0.6396480f, -0.0537291f, -0.7085364f,
    +0.1066417f, +0.7030141f, -0.7084426f, +0.0635714f, -0.7106261f, +0.0330318f, -0.5049776f, -0.5012102f,
    -0.5943443f, +0.3916793f, -0.4596093f, +0.5441485f, -0.1932872f, +0.6857854f, +0.6800470f, -0.2128554f,
    -0.6082678f, -0.3713597f, -0.4844939f, +0.5227825f, -0.6938466f, -0.1656311f, +0.6647276f, +0.2593048f,
    -0.3247688f, -0.6356547f, -0.3531079f, -0.6204240f, -0.7077631f, +0.0951479f, -0.1573922f, -0.6966479f,
    -0.5904613f, -0.4020796f, -0.5560511f, +0.4491207f, +0.5677763f, -0.4342550f, -0.5452617f, -0.4625179f,
    +0.2180903f, -0.6809410f, -0.4827462f, -0.5279180f, +0.6207938f, -0.3556738f, +0.6943646f, +0.1745279f,
    +0.7159298f, +0.0489074f, -0.3723283f, +0.6138207f, +0.7042050f, -0.1396589f, +0.0576027f, -0.7159606f,
    +0.4283446f, +0.5767976f, +0.6935573f, -0.1874974f, +0.4041884f, +0.5941930f, +0.6450008f, +0.3176802f,
    +0.5947016f, +0.4047494f, +0.1429615f, +0.7050244f, -0.0556353f, +0.7173968f, +0.3054615f, -0.6516761f,
    -0.4088216f, -0.5926644f, +0.3981873f, -0.6000488f, -0.0264450f, -0.7200639f, +0.4589381f, -0.5555113f,
    +0.4816053f, +0.5361113f, -0.6626467f, -0.2833198f, +0.3145079f, +0.6484789f, -0.2281716f, -0.6841090f,
    +0.3309516f, -0.6407848f, -0.2237819f, +0.6860870f, +0.7183722f, -0.0694555f, -0.3415979f, +0.6359376f,
    -0.7109179f, +0.1256456f, +0.6834824f, +0.2325283f, -0.4638722f, -0.5534596f, -0.4105108f, +0.5942266f,
    -0.6897144f, +0.2145970f, +0.0689103f, +0.7190904f, -0.6868617f, -0.2250343f, -0.5175205f, +0.5054194f,
    +0.6635010f, +0.2892404f, +0.4323506f, -0.5806247f, -0.2587529f, -0.6763574f, -0.7192315f, -0.0866135f,
    +0.0936814f, -0.7186319f, -0.7243986f, -0.0291032f, +0.7201657f, +0.0858873f, -0.3870867f, -0.6142088f,
    -0.4377386f, -0.5799180f, -0.6218013f, +0.3769991f, +0.5595747f, +0.4647124f, +0.1947884f, -0.7009224f,
    -0.1558740f, +0.7109085f, +0.0273940f, -0.7274299f, +0.7200307f, -0.1080947f, -0.7117719f, +0.1552384f,
    +0.1756509f, +0.7070188f, +0.5097026f, -0.5208718f, -0.7044801f, +0.1868280f, -0.0931371f, +0.7233208f,
    +0.4609518f, +0.5654150f, +0.5860084f, +0.4346637f, -0.6543196f, -0.3232224f, +0.2427859f, +0.6883853f,
    -0.3152433f, -0.6584151f, +0.3471833f, +0.6423326f, -0.0280995f, +0.7299774f, +0.6407870f, +0.3508824f,
    -0.0779736f, -0.7269316f, +0.6897883f, -0.2428713f, -0.7153348f, -0.1526056f, +0.7312207f, +0.0197037f,
    -0.5535417f, +0.4781885f, +0.2796246f, +0.6760449f, -0.6896610f, +0.2452283f, -0.4494725f, +0.5778206f,
    -0.5902113f, -0.4333456f, +0.6232489f, +0.3849414f, +0.1309122f, -0.7207709f, -0.6781977f, +0.2775350f,
    -0.5415543f, -0.4938458f, -0.7056997f, -0.1983032f, -0.2781128f, +0.6782402f, +0.6026365f, -0.4181927f,
    +0.7339559f, -0.0180960f, +0.5367399f, -0.5012081f, +0.7037665f, +0.2112752f, +0.6800187f, -0.2796827f,
    +0.7154851f, -0.1719827f, -0.6461771f, +0.3522748f, +0.5433687f, +0.4966924f, -0.6617478f, +0.3235564f,
    -0.7368214f, +0.0032887f, +0.1652767f, -0.7180537f, -0.0063816f, -0.7373905f, +0.0368433f, +0.7365998f,
    +0.3587268f, -0.6449950f, -0.1948020f, +0.7122187f, +0.5666829f, -0.4735530f, -0.2851572f, -0.6815042f,
    -0.1218329f, +0.7286572f, +0.7210822f, +0.1634790f, +0.3819188f, +0.6332367f, -0.7385968f, +0.0381735f,
    +0.2577371f, -0.6934854f, -0.3124475f, +0.6708632f, -0.7297475f, -0.1238679f, +0.0997142f, +0.7335691f,
    -0.6121513f, +0.4164837f, +0.7296928f, +0.1261469f, +0.6700031f, -0.3171915f, -0.5106094f, +0.5376780f,
    -0.5210353f, -0.5275940f, +0.6540236f, -0.3496467f, -0.5921554f, +0.4464980f, -0.7383454f, +0.0708514f,
    -0.1817310f, -0.7191570f, +0.2173775f, +0.7092019f, -0.1453655f, -0.7279645f, -0.1134723f, -0.7347351f,
    -0.6990594f, -0.2546687f, +0.1320100f, +0.7322137f, -0.3794095f, -0.6400571f, -0.2170979f, -0.7119567f,
    -0.2516117f, +0.7005097f, -0.5793724f, -0.4673021f, +0.6266599f, -0.4020699f, -0.6497036f, -0.3645870f,
    +0.7140916f, -0.2137339f, +0.7436371f, -0.0544535f, +0.0051213f, +0.7461752f, -0.4709440f, -0.5798487f,
    +0.4426993f, +0.6018584f, -0.7395586f, +0.1065463f, -0.4998767f, -0.5556913f, -0.7452939f, -0.0566276f,
    -0.6271985f, -0.4071037f, +0.5954120f, -0.4528479f, +0.5289381f, +0.5290162f, -0.3777344f, +0.6460066f,
    +0.2904415f, -0.6896990f, +0.4967189f, -0.5597425f, +0.6941200f, +0.2817198f, +0.0535506f, -0.7473637f,
    -0.0443221f, -0.7483321f, -0.3517621f, -0.6621130f, +0.6432876f, -0.3853397f, +0.3926968f, -0.6389864f,
    -0.4931873f, +0.5657364f, +0.4376643f, -0.6098985f, +0.7388169f, -0.1385210f, -0.6928001f, -0.2918048f,
    +0.5029076f, +0.5589883f, +0.1085402f, -0.7442697f, +0.7503305f, +0.0538856f, -0.4359701f, +0.6132851f,
    +0.2328464f, -0.7157210f, -0.4082326f, +0.6325433f, +0.7468734f, -0.0948856f, -0.6857781f, +0.3108132f,
    +0.6809598f, +0.3213991f, -0.7228149f, +0.2131793f, +0.3279252f, +0.6785285f, -0.5535542f, +0.5120727f,
    -0.0697693f, +0.7528778f, -0.3488398f, +0.6718104f, +0.7127510f, +0.2567101f, -0.7348256f, -0.1845778f,
    +0.6283076f, +0.4234466f, +0.0708174f, +0.7544819f, +0.6728656f, +0.3486062f, -0.2642594f, -0.7103020f,
    -0.4248018f, -0.6278044f, -0.2231379f, +0.7245908f, -0.6458699f, +0.3972349f, +0.7325379f, +0.1962467f,
    -0.7444783f, +0.1451852f, +0.7528944f, +0.0958875f, +0.3317755f, -0.6826534f, -0.3202436f, -0.6888546f,
    +0.7186580f, -0.2461929f, +0.4179672f, +0.6344396f, -0.6825546f, -0.3337434f, +0.1660983f, +0.7414231f,
    +0.2720163f, +0.7094963f, -0.2868830f, +0.7037756f, +0.6544969f, +0.3867912f, -0.4497062f, -0.6130360f,
    +0.4715867f, -0.5966692f, -0.1817213f, +0.7400827f, +0.7622663f, +0.0028929f, +0.0795258f, -0.7581147f,
    +0.6041306f, +0.4651923f, +0.2002190f, -0.7357759f, -0.5935164f, +0.4787798f, -0.7563454f, -0.0987585f,
    +0.0192003f, -0.7626123f, -0.5528288f, -0.5262634f, -0.7631593f, -0.0201248f, +0.4796877f, +0.5951057f,
    +0.7387082f, -0.1974674f, +0.2002384f, +0.7379645f, +0.1575175f, -0.7487029f, -0.7307615f, -0.2282153f,
    +0.5356730f, -0.5472557f, +0.4274492f, -0.6354796f, -0.4068491f, -0.6489673f, -0.1459770f, +0.7519737f,
    -0.0859751f, -0.7613732f, -0.7446525f, +0.1805074f, -0.7519995f, -0.1507226f, -0.0348219f, +0.7662492f,
    -0.7668023f, +0.0199634f, -0.6780617f, +0.3602894f, +0.3667179f, +0.6746997f, +0.5752917f, -0.5088900f,
    -0.5364185f, +0.5501916f, +0.3047851f, +0.7054547f, -0.6235194f, -0.4494552f, +0.7499156f, -0.1688971f,
    +0.7524863f, +0.1573717f, +0.2415262f, +0.7308838f, -0.7134602f, +0.2895882f, -0.7292016f, +0.2473087f,
    -0.4676878f, +0.6123520f, +0.5886555f, +0.4983326f, -0.7695102f, +0.0571485f, -0.2422444f, -0.7326796f,
    -0.6336392f, +0.4408318f, +0.7050874f, -0.3142223f, +0.1097071f, +0.7641457f, -0.2943231f, -0.7141858f,
    +0.7723898f, -0.0344795f, -0.5842239f, -0.5068426f, +0.7377445f, +0.2335615f, -0.0147096f, -0.7739266f,
    +0.6322737f, -0.4468403f, -0.1049369f, +0.7673100f, -0.1703168f, -0.7560464f, +0.6057194f, -0.4834632f,
    -0.7699122f, +0.0890469f, -0.3281414f, +0.7022069f, +0.0397007f, +0.7744191f, -0.4914221f, +0.5999310f,
    +0.7224987f, -0.2819955f, +0.6930144f, -0.3482280f, +0.6757962f, -0.3806731f, +0.3750736f, -0.6790839f,
    -0.2061487f, -0.7485554f, +0.7661918f, -0.1270068f, -0.5093229f, -0.5881003f, +0.0022537f, +0.7787244f,
    -0.7304748f, -0.2699335f, +0.2761448f, -0.7283261f, +0.1289936f, -0.7682894f, -0.6691969f, -0.3993960f,
    -0.1314794f, -0.7684005f, -0.5402068f, -0.5622478f, +0.7174435f, +0.3055588f, +0.3137328f, -0.7140379f,
    +0.5731877f, +0.5290232f, +0.4573209f, +0.6319447f, -0.3855558f, -0.6781503f, -0.5882711f, +0.5127381f,
    +0.7799156f, +0.0297135f, -0.4844087f, -0.6121176f, +0.7774516f, -0.0701776f, -0.7773913f, -0.0709562f,
    -0.4076186f, +0.6657848f, -0.7711734f, +0.1216052f, +0.3957313f, +0.6736940f, -0.0549010f, -0.7794970f,
    +0.1415320f, +0.7695181f, -0.2617431f, +0.7375693f, +0.6621509f, -0.4199614f, -0.7092144f, +0.3344640f,
    -0.6542920f, -0.4322768f, +0.5435604f, +0.5657865f, -0.3771584f, +0.6882694f, +0.7751977f, +0.1230080f,
    +0.7815905f, +0.0735570f, -0.6929326f, -0.3693382f, +0.2351688f, -0.7495126f, +0.5156363f, +0.5926857f,
    +0.6935478f, +0.3698735f, -0.6192144f, -0.4842520f, -0.7254941f, -0.3029718f, +0.6343097f, +0.4646914f,
    +0.7794699f, -0.1036119f, -0.6801610f, +0.3948730f, +0.0527148f, -0.7847212f, -0.6276325f, +0.4747088f,
    -0.2977604f, +0.7284865f, +0.5266976f, -0.5853876f, -0.7605747f, +0.2040783f, +0.5750896f, -0.5390932f,
    -0.5718998f, +0.5426296f, -0.4532862f, +0.6451083f, -0.3497204f, -0.7067143f, +0.7480915f, -0.2517103f,
    -0.5236899f, +0.5907677f, -0.7738042f, +0.1571337f, -0.2313963f, +0.7551395f, -0.7142007f, -0.3380046f,
    -0.4315906f, -0.6626725f, +0.3468510f, +0.7107956f, +0.4212081f, -0.6705945f, -0.7810262f, -0.1310057f,
    +0.4333762f, +0.6629213f, -0.7911670f, -0.0414931f, +0.7775509f, -0.1539499f, -0.7921019f, +0.0326068f,
    +0.6805121f, +0.4076102f, -0.7644984f, -0.2121487f, +0.1854867f, +0.7715343f, -0.2058223f, +0.7670631f,
    -0.0647116f, +0.7919043f, -0.6703667f, +0.4270296f, +0.5041381f, -0.6145552f, -0.7429539f, +0.2853257f,
    +0.4650912f, -0.6460609f, +0.7720574f, +0.1941833f, +0.1677344f, -0.7785657f, -0.1721034f, +0.7779986f,
    +0.6624217f, +0.4430676f, +0.0870628f, +0.7923357f, +0.7971613f, -0.0131946f, -0.7573239f, -0.2495353f,
    -0.7975900f, +0.0006455f, +0.7739920f, -0.1953399f, +0.7503969f, +0.2736437f, -0.7793478f, -0.1749105f,
    +0.7223668f, +0.3436690f, -0.5863935f, -0.5441846f, +0.0952844f, -0.7946500f, +0.7683332f, -0.2241764f,
    +0.7391759f, -0.3082538f, +0.4939341f, +0.6306482f, +0.2216604f, +0.7698163f, +0.2844723f, +0.7491765f,
    -0.3218781f, -0.7339862f, -0.2768864f, -0.7524356f, -0.4645745f, -0.6534742f, +0.0159833f, -0.8021578f,
    -0.5566539f, +0.5779225f, +0.7881434f, +0.1517103f, +0.6261799f, +0.5039737f, -0.8008871f, +0.0688706f,
    +0.2058553f, -0.7771185f, +0.7704087f, +0.2310098f, +0.2534347f, +0.7634437f, -0.7987730f, -0.0988572f,
    -0.4359429f, +0.6771517f, +0.3671607f, -0.7170678f, +0.5648373f, -0.5745440f, -0.2299952f, -0.7724918f,
    -0.7119852f, +0.3787609f, -0.1425750f, +0.7949294f, -0.0242123f, +0.8075116f, +0.7156024f, -0.3753760f,
    -0.5063525f, +0.6306544f, -0.0987314f, -0.8028499f, -0.8028014f, +0.1025847f, +0.6399873f, -0.4954160f,
    +0.6197525f, -0.5225698f, +0.0223233f, +0.8103746f, +0.8041190f, +0.1040605f, +0.7017406f, -0.4065360f,
    -0.5358535f, -0.6090645f, -0.7772986f, +0.2324456f, -0.7867579f, +0.1982459f, -0.7682975f, +0.2608533f,
    +0.8099240f, +0.0549308f, +0.7358437f, -0.3436322f, -0.4951848f, -0.6437061f, -0.0233968f, -0.8118255f,
    -0.3696249f, +0.7237184f, -0.6264609f, +0.5178180f, +0.3290911f, -0.7434419f, +0.8112174f, -0.0557764f,
    -0.6693806f, +0.4620074f, +0.6702644f, -0.4608937f, -0.6219139f, -0.5243999f, +0.3222430f, +0.7469691f,
    -0.7475729f, +0.3212968f, -0.3315503f, +0.7440268f, -0.6610736f, -0.4764158f, -0.1096560f, +0.8074834f,
    -0.4345689f, -0.6893644f, +0.0516032f, -0.8137587f, -0.3870708f, -0.7179631f, +0.1341491f, -0.8045629f,
    -0.5710562f, -0.5826650f, +0.7535195f, +0.3128386f, -0.7443883f, -0.3343826f, -0.1468802f, -0.8027424f,
    +0.6118106f, +0.5407988f, +0.8114474f, -0.0935327f, -0.7632454f, -0.2916442f, +0.8169684f, +0.0150869f,
    +0.5820358f, +0.5735067f, -0.8064612f, +0.1353704f, +0.3918586f, +0.7178301f, +0.4992084f, -0.6480973f,
    +0.4135589f, -0.7061963f, +0.2543694f, -0.7787431f, -0.2933729f, +0.7654750f, +0.4724557f, +0.6699539f,
    +0.0591420f, +0.8177004f, +0.7776671f, -0.2602376f, +0.2929726f, -0.7665033f, +0.1320125f, +0.8099716f,
    -0.7131116f, -0.4069355f, -0.2554389f, +0.7803940f, +0.5398181f, +0.6188162f, +0.1652898f, +0.8045684f,
    -0.4153462f, +0.7088185f, -0.6914194f, -0.4445150f, +0.4373091f, +0.6961890f, -0.1900402f, -0.8000700f,
    +0.8129107f, -0.1271010f, -0.5852915f, +0.5783885f, -0.8217404f, +0.0439189f, -0.0600214f, -0.8210325f,
    +0.7260427f, +0.3880606f, -0.8232889f, -0.0238660f, -0.7946165f, -0.2174417f, -0.7363026f, -0.3714406f,
    -0.8118532f, -0.1473164f, -0.8221105f, -0.0709611f, -0.4899327f, +0.6645954f, -0.2079401f, +0.7992672f,
    +0.6974863f, -0.4426647f, +0.6059542f, -0.5625709f, -0.6109891f, +0.5582123f, +0.8111329f, -0.1655007f,
    -0.6587203f, +0.5014977f, +0.5533697f, -0.6163209f, +0.7746841f, -0.2938706f, -0.5522773f, +0.6176744f,
    +0.8070847f, +0.1885412f, -0.8119239f, +0.1672823f, -0.3632509f, -0.7458905f, -0.7453155f, +0.3650897f,
    +0.6662361f, +0.4950737f, -0.3117407f, -0.7695472f, -0.7149076f, +0.4287553f, +0.1066881f, +0.8269994f,
    -0.6573233f, -0.5132143f, -0.0481077f, +0.8330314f, -0.7923127f, -0.2621064f, +0.4604383f, -0.6964719f,
    +0.7169732f, +0.4282931f, +0.8355184f, -0.0273411f, +0.8109210f, -0.2032517f, +0.2059761f, +0.8103464f,
    -0.5269952f, -0.6498102f, +0.8232643f, +0.1515572f, -0.8370298f, +0.0136671f, -0.2726723f, -0.7915305f,
    -0.6980553f, +0.4635800f, -0.4223756f, -0.7237984f, +0.1831135f, -0.8179412f, +0.5206138f, +0.6569282f,
    -0.4837911f, -0.6846166f, -0.8347700f, +0.0805087f, -0.4688598f, +0.6953368f, +0.0248375f, -0.8384180f,
    +0.7925686f, +0.2759894f, +0.5740650f, +0.6122724f, +0.3736642f, +0.7516472f, +0.8056423f, -0.2365517f,
    +0.2845659f, +0.7900817f, +0.6734333f, -0.5018310f, -0.8158386f, +0.2035357f, -0.8207591f, -0.1827468f,
    +0.2295348f, -0.8093593f, +0.8379345f, +0.0778292f, -0.7777061f, -0.3235100f, +0.2427271f, +0.8068804f,
    +0.3713914f, -0.7566893f, -0.6110678f, -0.5816691f, -0.2301140f, -0.8117172f, -0.5760831f, -0.6168258f,
    -0.0905418f, +0.8393192f, +0.7661939f, -0.3550019f, +0.5306478f, -0.6570076f, +0.0703442f, -0.8418190f,
    +0.7046791f, +0.4663269f, -0.1742831f, +0.8274937f, +0.7544743f, -0.3830493f, +0.7790592f, -0.3306561f,
    +0.4977890f, -0.6846318f, +0.7642180f, +0.3642420f, -0.7912297f, +0.3014008f, -0.8389710f, -0.1145125f,
    -0.6417329f, -0.5526180f, -0.4582735f, -0.7123837f, +0.0028609f, +0.8476042f, +0.1026282f, -0.8414240f,
    -0.1375312f, -0.8369892f, -0.5324457f, +0.6603010f, +0.8473858f, +0.0382072f, -0.2354786f, +0.8150572f,
    +0.8408218f, +0.1163174f, +0.6541506f, +0.5418375f, -0.8070177f, +0.2656364f, -0.7454485f, +0.4080326f,
    +0.8461398f, -0.0829891f, -0.0973597f, -0.8447555f, +0.0448512f, +0.8491679f, +0.8174672f, +0.2353107f,
    +0.6188995f, +0.5838240f, +0.7377964f, -0.4238438f, +0.1502966f, -0.8379769f, -0.8506735f, -0.0478661f,
    -0.2896670f, +0.8018237f, +0.6419762f, -0.5615378f, -0.3285389f, +0.7872142f, +0.5987364f, -0.6082284f,
    -0.3615964f, +0.7731968f, +0.3243577f, +0.7896056f, +0.7875183f, +0.3296093f, +0.3330464f, -0.7861527f,
    -0.6450002f, +0.5595468f, +0.7055209f, -0.4811065f, -0.0449338f, -0.8530698f, -0.3942655f, +0.7579197f,
    -0.8478057f, +0.1176677f, -0.0043798f, -0.8559424f, +0.4233997f, -0.7439247f, +0.8121239f, -0.2704660f,
    +0.4307479f, +0.7398698f, -0.3914664f, -0.7622656f, +0.0818768f, +0.8534296f, +0.2883577f, -0.8075935f,
    -0.7817651f, +0.3524541f, +0.1660623f, +0.8413745f, -0.7244551f, -0.4595484f, -0.7766465f, -0.3657784f,
    +0.6719292f, -0.5346793f, -0.4256954f, +0.7458289f, +0.5017940f, +0.6970047f, -0.5906177f, +0.6238337f,
    -0.8255048f, +0.2378704f, -0.7004272f, -0.4988620f, -0.3525074f, -0.7851157f, -0.1360808f, +0.8503272f,
    -0.4562843f, +0.7303587f, -0.8284907f, -0.2351965f, -0.6958159f, +0.5077415f, -0.6748404f, +0.5353896f,
    -0.7619082f, -0.4028923f, -0.5616428f, -0.6539587f, +0.4676541f, +0.7244220f, -0.5202988f, -0.6884444f,
    -0.8622146f, +0.0426354f, +0.1273917f, +0.8538730f, -0.5166335f, +0.6917810f, -0.8104842f, -0.2994097f,
    -0.8643342f, -0.0149351f, -0.0275947f, +0.8644005f, +0.8652534f, +0.0034531f, +0.8554164f, -0.1314573f,
    -0.6227452f, +0.6017499f, +0.5748796f, -0.6476788f, -0.7238181f, +0.4770687f, -0.5701351f, +0.6533324f,
    -0.8535416f, +0.1530890f, -0.7509148f, -0.4339513f, +0.8453529f, +0.1963648f, +0.7037212f, +0.5087715f,
    +0.8486633f, -0.1841698f, +0.7640906f, +0.4132559f, +0.7318550f, -0.4689884f, +0.5590820f, +0.6669162f,
    -0.1883121f, -0.8499601f, +0.8125523f, -0.3134732f, -0.2658813f, +0.8297657f, -0.8501633f, +0.1924216f,
    -0.6780107f, -0.5483999f, -0.3164677f, -0.8126977f, -0.8683094f, -0.0821673f, -0.2739518f, -0.8282562f,
    -0.8597183f, -0.1507750f, +0.2493944f, -0.8378437f, -0.0664421f, +0.8718961f, -0.8041337f, -0.3443933f,
    -0.4298637f, -0.7619122f, -0.2171351f, +0.8487152f, +0.0439294f, -0.8751293f, +0.7082562f, -0.5162629f,
    +0.2085970f, -0.8516209f, +0.8757710f, -0.0477705f, -0.4982677f, +0.7222458f, +0.8639814f, +0.1540047f,
    +0.6348995f, -0.6068082f, -0.7566106f, +0.4470011f, +0.8402570f, -0.2586110f, -0.4991343f, -0.7241964f,
    +0.4055852f, +0.7809227f, +0.4805655f, -0.7373380f, -0.4671960f, -0.7462999f, +0.0356070f, +0.8808079f,
    +0.3733267f, -0.7987497f, +0.8789173f, +0.0703297f, +0.2846746f, +0.8345277f, +0.8003374f, -0.3704117f,
    +0.6622001f, +0.5827449f, -0.8591577f, -0.2007883f, +0.7570366f, +0.4532990f, +0.2008113f, +0.8602928f,
    +0.1741519f, -0.8665334f, +0.5540689f, -0.6887303f, +0.8233873f, +0.3216725f, +0.1288439f, -0.8747919f,
    -0.6586240f, -0.5900034f, -0.6267615f, -0.6249093f, +0.6288141f, +0.6230168f, +0.8784389f, +0.1106391f,
    +0.3636943f, +0.8072910f, +0.8406006f, +0.2805335f, -0.0697903f, -0.8840720f, -0.8836801f, +0.0766375f,
    +0.2433255f, +0.8529931f, +0.6970321f, +0.5506364f, +0.5990858f, +0.6561837f, +0.4125493f, -0.7871307f,
    -0.8885671f, +0.0193817f, -0.8468421f, -0.2703681f, +0.5201334f, -0.7209867f, -0.2393929f, -0.8570418f,
    -0.8334068f, +0.3127600f, +0.8606011f, -0.2285907f, -0.4271449f, +0.7825758f, -0.3915936f, +0.8017156f,
    +0.7651937f, -0.4592170f, -0.0007596f, +0.8924698f, -0.8514585f, +0.2701593f, -0.1073928f, +0.8870169f,
    +0.6104133f, -0.6529583f, -0.7961045f, +0.4064547f, -0.1830697f, +0.8751002f, -0.3588298f, +0.8188908f,
    -0.1109001f, -0.8872303f, -0.8200158f, +0.3567222f, -0.3099902f, +0.8388363f, +0.7394036f, +0.5036522f,
    +0.7932065f, -0.4140420f, -0.1563904f, -0.8811763f, -0.6056173f, -0.6598775f, -0.0321142f, -0.8951540f,
    -0.6771305f, +0.5870656f, +0.0836190f, -0.8924594f, -0.8887787f, -0.1177160f, +0.0064949f, -0.8968006f,
    +0.6772830f, -0.5887039f, -0.5668533f, -0.6964667f, -0.6215498f, +0.6483147f, -0.5696976f, +0.6943321f,
    +0.1562614f, +0.8848069f, -0.7357698f, -0.5161517f, +0.8118984f, +0.3859387f, +0.5430831f, +0.7171021f,
    -0.4717059f, +0.7660952f, +0.4968358f, +0.7502063f, -0.4086288f, -0.8017092f, +0.3240043f, -0.8395462f,
    +0.8477217f, -0.3032871f, +0.8957872f, -0.0942501f, +0.1045312f, +0.8946812f, -0.7606113f, +0.4833974f,
    -0.2643375f, +0.8618214f, +0.8674859f, +0.2457184f, -0.7705472f, -0.4702502f, -0.8887884f, +0.1603820f,
    -0.3699502f, -0.8239610f, -0.5399196f, +0.7241923f, +0.3202132f, +0.8449705f, -0.0352984f, +0.9032345f,
    -0.7406778f, +0.5181774f, -0.7065992f, -0.5637840f, -0.8968028f, +0.1150913f, -0.5031708f, +0.7515414f,
    +0.0643883f, +0.9022248f, +0.4609843f, -0.7784371f, +0.8899090f, -0.1634339f, -0.9028107f, -0.0630450f,
    -0.6049469f, +0.6737046f, +0.9056216f, -0.0141086f, -0.8391319f, -0.3412331f, -0.5439593f, -0.7247977f,
    +0.2904005f, -0.8586748f, +0.8324436f, -0.3594623f, -0.7203978f, +0.5514761f, +0.7501719f, -0.5113698f,
    +0.7212057f, -0.5521566f, +0.4641986f, +0.7816557f, -0.1484884f, +0.8971347f, -0.9101366f, -0.0156952f,
    -0.8562163f, -0.3091313f, -0.2886568f, -0.8634286f, -0.8793325f, +0.2371948f, +0.2467256f, -0.8780686f,
    -0.6605161f, +0.6299079f, -0.8115364f, -0.4193469f, +0.8912091f, +0.2037793f, +0.9015008f, +0.1519424f,
    +0.9132277f, +0.0442386f, +0.5930791f, -0.6972265f, -0.3401745f, +0.8498326f, -0.4664093f, -0.7885615f,
    -0.8338305f, -0.3801155f, +0.3615923f, -0.8421287f, -0.9017212f, -0.1663813f, +0.6572008f, -0.6403729f,
    +0.8778074f, -0.2683931f, +0.1732783f, -0.9015815f, -0.2296320f, +0.8891471f, +0.1238111f, -0.9101812f,
    -0.8839394f, -0.2523366f, -0.3349966f, -0.8563656f, -0.9182981f, +0.0496504f, +0.5881191f, +0.7072794f,
    -0.1985329f, -0.8988436f, -0.5092317f, -0.7677580f, +0.7787774f, +0.4932281f, +0.0206670f, +0.9221770f,
    +0.1939307f, +0.9025044f, +0.6869309f, +0.6170833f, +0.5164467f, -0.7663245f, -0.9021007f, +0.2005906f,
    +0.6418992f, +0.6651905f, +0.9193289f, +0.1011048f, +0.8508574f, +0.3637639f, -0.1088052f, +0.9195583f,
    +0.4425416f, +0.8136340f, +0.9000492f, -0.2185672f, -0.6878060f, -0.6204522f, -0.4708779f, +0.7979713f,
    -0.0611388f, +0.9250133f, +0.5593415f, -0.7396626f, -0.8030416f, +0.4646352f, +0.4002895f, +0.8373010f,
    +0.2757487f, +0.8863621f, +0.0340819f, -0.9280884f, -0.9045083f, -0.2113234f, -0.4248047f, +0.8260637f,
    +0.7864624f, -0.4945668f, -0.0757241f, -0.9260045f, +0.8720992f, +0.3212741f, -0.4468577f, -0.8152506f,
    +0.2169196f, -0.9042224f, +0.7489492f, +0.5522361f, +0.4025654f, -0.8389659f, +0.8203697f, +0.4398453f,
    -0.3880834f, +0.8475884f, -0.6022915f, -0.7127663f, -0.2422210f, -0.9014977f, +0.8697100f, -0.3410045f,
    +0.8137646f, -0.4596978f, +0.6358362f, -0.6851766f, +0.7245553f, +0.5910820f, +0.8343226f, -0.4230245f,
    +0.9338205f, -0.0589040f, -0.7239699f, +0.5930591f, +0.8934729f, +0.2785273f, -0.3075458f, +0.8844892f,
    -0.1792641f, +0.9192445f, -0.7503219f, -0.5607520f, -0.6531544f, -0.6716441f, -0.6466619f, +0.6780062f,
    -0.1366639f, -0.9274489f, +0.3524523f, +0.8689014f, -0.8333365f, +0.4311135f, +0.7256637f, -0.5949827f,
    +0.1367683f, +0.9290022f, -0.7258040f, -0.5974910f, -0.7818928f, +0.5219492f, +0.0779660f, +0.9371299f,
    -0.5531163f, +0.7605170f, +0.2315331f, +0.9117619f, -0.5150424f, +0.7872910f, -0.8924727f, +0.2991598f,
    -0.4112076f, -0.8470135f, -0.8752497f, +0.3472508f, -0.6190962f, +0.7100689f, -0.9385799f, +0.0859147f,
    +0.9354187f, -0.1252223f, +0.4956360f, -0.8031445f, -0.7824855f, -0.5291368f, +0.0963839f, -0.9396802f,
    -0.2751090f, +0.9037627f, -0.8610545f, +0.3890313f, -0.9452297f, +0.0107955f, -0.5923622f, +0.7380954f,
    -0.9365883f, -0.1362573f, -0.7055629f, +0.6325324f, -0.0226076f, +0.9474335f, -0.9473376f, -0.0369691f,
    +0.9484331f, +0.0147040f, -0.9385877f, +0.1373861f, -0.8998238f, -0.3004300f, -0.9443325f, -0.0917603f,
    +0.7045088f, -0.6357407f, +0.4476058f, -0.8371292f, +0.3391846f, -0.8866230f, -0.5666770f, -0.7620690f,
    -0.0214600f, -0.9494794f, +0.1580363f, -0.9373104f, +0.8989124f, -0.3095722f, +0.7848942f, -0.5365873f,
    +0.5401317f, +0.7833983f, -0.8846596f, -0.3588118f, +0.9275336f, +0.2267640f, +0.6088519f, -0.7361881f,
    +0.9449588f, +0.1418581f, -0.8384999f, -0.4584901f, +0.8750296f, -0.3848186f, -0.3191738f, -0.9011163f,
    +0.0647713f, -0.9537968f, -0.3445826f, +0.8925476f, -0.7681789f, +0.5703548f, +0.9387746f, +0.1848154f,
    -0.4576853f, +0.8407300f, -0.3691935f, -0.8840340f, +0.9403408f, -0.1835280f, +0.2928409f, -0.9129492f,
    +0.3159585f, +0.9062165f, +0.6376392f, +0.7173938f, -0.1404420f, +0.9495251f, -0.2115238f, +0.9367487f,
    -0.0882903f, +0.9567904f, +0.5946357f, +0.7551199f, +0.9575413f, +0.0835114f, -0.8207712f, -0.5024500f,
    +0.0300610f, +0.9620997f, +0.7705101f, -0.5774863f, -0.5419037f, -0.7959549f, +0.4970894f, +0.8248608f,
    -0.2801374f, -0.9216428f, -0.9467730f, -0.1812446f, -0.6851447f, +0.6783141f, +0.7975234f, +0.5417634f,
    +0.4399370f, +0.8582219f, -0.9468458f, +0.1900739f, +0.8267601f, +0.5010884f, +0.2015830f, -0.9455261f,
    -0.1897591f, -0.9482602f, +0.6876783f, +0.6808157f, +0.8724760f, +0.4206910f, +0.1940577f, +0.9493365f,
    -0.9412251f, -0.2315273f, +0.1306745f, -0.9605141f, +0.6803065f, -0.6924421f, -0.8275833f, +0.5081426f,
    -0.4230137f, +0.8743205f, -0.5058987f, +0.8291817f, -0.9400905f, +0.2455620f, -0.8730347f, -0.4267330f,
    -0.5034112f, -0.8317571f, +0.8290294f, -0.5083669f, -0.7064366f, -0.6687278f, +0.2477563f, -0.9410590f,
    -0.4513941f, -0.8624595f, +0.4025990f, -0.8867479f, +0.5599385f, -0.7973264f, +0.7281454f, +0.6476154f,
    -0.6472796f, -0.7301550f, +0.2666236f, +0.9388409f, -0.2404266f, -0.9468201f, +0.4061097f, +0.8890678f,
    -0.6497217f, +0.7308114f, -0.8583651f, +0.4685376f, -0.2497033f, +0.9456857f, +0.9310809f, +0.3019944f,
    -0.6115250f, -0.7645736f, -0.7382476f, -0.6435893f, +0.9128636f, +0.3550734f, -0.3839607f, +0.9021350f,
    -0.1809149f, +0.9645801f, +0.9555621f, -0.2316578f, +0.9434637f, -0.2773211f, +0.8649573f, -0.4679533f,
    -0.5561798f, +0.8128927f, -0.2957096f, +0.9397693f, -0.9425300f, -0.2873008f, +0.6533034f, -0.7385328f,
    +0.8671958f, +0.4697929f, -0.8910568f, +0.4262458f, -0.7637174f, +0.6284441f, +0.9546894f, +0.2609138f,
    -0.4182939f, -0.8988872f, +0.3711993f, +0.9196841f, -0.7383369f, +0.6623795f, +0.8936356f, -0.4316388f,
    +0.5266848f, -0.8411462f, +0.7762985f, +0.6195329f, -0.6260089f, +0.7719944f, +0.3665456f, -0.9243711f,
    -0.7787797f, -0.6185580f, +0.4802148f, -0.8711067f, -0.9347273f, +0.3423558f, -0.9950442f, +0.0477117f,
    +0.9960458f, -0.0224688f, +0.2863006f, -0.9543061f, -0.6997774f, +0.7097780f, +0.0149696f, -0.9966666f,
    -0.9536015f, +0.2924147f, -0.9977435f, +0.0195588f, -0.0193974f, -0.9979820f, -0.0850858f, -0.9945909f,
    -0.0506580f, -0.9969429f, -0.0112614f, +0.9982423f, +0.9967998f, -0.0548946f, +0.9983990f, +0.0083617f,
    -0.5996448f, +0.7983851f, -0.9985452f, -0.0092589f, -0.9963413f, -0.0695899f, -0.7993816f, +0.5989714f,
    -0.0489615f, +0.9977803f, +0.9982340f, +0.0429661f, +0.0231658f, +0.9989948f, -0.9984974f, -0.0414839f,
    +0.6119843f, -0.7904927f, -0.9972028f, +0.0707346f, +0.0543745f, -0.9982604f, +0.0548399f, +0.9982804f,
    +0.9960486f, -0.0871582f, +0.0890464f, +0.9961504f, +0.7572362f, -0.6537566f, +0.9980006f, +0.0761172f,
    -0.9967293f, +0.0945572f, -0.9207160f, +0.3933287f, +0.0897064f, -0.9973905f, +0.7191122f, -0.6970010f,
    -0.0821982f, +0.9984124f, -0.3631090f, -0.9337376f, -0.9952324f, +0.1172992f, -0.9969717f, -0.1027833f,
    -0.8261462f, -0.5679369f, -0.9388810f, -0.3528067f, +0.9961391f, -0.1176558f, -0.1223065f, -0.9956287f,
    +0.9965831f, +0.1151101f, +0.1258325f, +0.9957311f, -0.1182617f, +0.9974359f, -0.4844403f, +0.8807178f,
    -0.9952812f, +0.1406290f, +0.3346450f, -0.9482567f, -0.3118102f, -0.9560581f, -0.9962811f, -0.1370570f,
    +0.9275450f, -0.3889763f, +0.9467573f, -0.3404635f, +0.8202598f, +0.5827387f, -0.8334822f, +0.5638569f,
    +0.9952009f, -0.1490277f, -0.6996202f, -0.7236264f, +0.7922429f, -0.6211505f, +0.1264004f, -0.9989487f,
    -0.3472383f, +0.9453136f, -0.1597781f, -0.9948866f, +0.1630998f, +0.9948534f, +0.9971573f, +0.1497455f,
    -0.8688239f, -0.5120954f, +0.3226203f, +0.9555399f, +0.8308378f, -0.5722852f, +0.1603277f, -0.9976145f,
    -0.9975871f, +0.1618711f, -0.1524546f, +0.9994351f, +0.6974406f, +0.7327846f, -0.9279634f, -0.4047672f,
    -0.9977867f, -0.1733005f, +0.4899666f, +0.8864954f, +0.5367002f, +0.8598535f, +0.6600276f, +0.7693226f,
    +0.9979934f, -0.1810242f, -0.5764082f, -0.8346009f, +0.9971274f, +0.1858879f, -0.5334013f, +0.8635013f,
    +0.9362697f, +0.3935352f, -0.9987141f, +0.1845910f, +0.6201069f, +0.8046814f, +0.1921119f, -0.9980749f,
    -0.9954270f, +0.2058859f, -0.8988339f, +0.4749352f, -0.1906576f, +0.9989287f, -0.1977336f, -0.9980287f,
    +0.8686870f, -0.5301698f, +0.9957585f, -0.2104243f, +0.4162019f, -0.9295479f, -0.9965761f, -0.2104249f,
    -0.4149456f, +0.9303873f, +0.2050974f, +0.9979464f, -0.6881752f, +0.7518591f, +0.8661814f, +0.5389937f,
    +0.7440298f, +0.6980833f, +0.9951148f, +0.2251619f, -0.8718101f, +0.5300580f, +0.5840932f, +0.8370469f,
    +0.2228928f, -0.9969630f, -0.9953403f, +0.2312372f, +0.6981964f, -0.7470499f, -0.2262552f, +0.9973422f,
    -0.4774811f, -0.9051313f, -0.2344171f, -0.9976550f, -0.9960099f, -0.2456180f, +0.2456576f, +0.9968326f,
    -0.7506592f, -0.7025700f, +0.9989738f, -0.2512522f, +0.2544274f, -0.9981699f, -0.9980775f, +0.2568477f,
    -0.2597182f, +0.9980587f, +0.5249135f, -0.8881202f, -0.5264871f, -0.8872961f, +0.9980103f, +0.2646431f,
    +0.4474107f, +0.9308626f, -0.2710804f, -0.9979395f, -0.7560898f, +0.7055668f, -0.9957597f, -0.2798730f,
    -0.6269955f, -0.8235036f, -0.7909008f, +0.6694724f, -0.9231536f, -0.4707245f, -0.9630259f, +0.3829092f,
    -0.6712084f, -0.7897541f, -0.9970617f, +0.2855880f, +0.9319909f, +0.4552575f, +0.2872773f, +0.9968221f,
    +0.9965016f, -0.2911279f, -0.5869736f, +0.8568082f, +0.6604271f, -0.8026977f, -0.2950222f, +0.9968882f,
    +0.2934378f, -0.9978594f, +0.4672900f, -0.9304027f, +0.9964430f, +0.3058532f, +0.7970197f, +0.6722826f,
    -0.8352247f, +0.6251760f, -0.4262835f, -0.9526659f, -0.4677038f, +0.9339129f, -0.9972062f, -0.3125112f,
    +0.6044295f, -0.8526458f, -0.3094211f, -0.9986553f, +0.7610669f, -0.7168257f, -0.9973331f, +0.3140728f,
    -0.8457710f, -0.6149632f, -0.8043061f, -0.6703998f, +0.9967912f, -0.3298068f, +0.9457211f, -0.4562984f,
    -0.9464263f, +0.4556643f, +0.8465245f, +0.6228582f, +0.3300959f, +0.9980944f, -0.6716866f, +0.8087885f,
    +0.9224069f, +0.5053234f, -0.3374342f, +0.9971111f, +0.3377773f, -0.9973829f, +0.9259869f, -0.5028761f,
    +0.9964256f, +0.3428801f, -0.9959652f, +0.3446162f, -0.6311934f, +0.8455442f, -0.9971726f, -0.3475830f,
    -0.3523804f, -0.9960245f, +0.5317918f, +0.9140394f, +0.8503978f, -0.6288007f, -0.5866886f, -0.8815489f,
    +0.8225155f, -0.6704186f, +0.8806242f, -0.5922543f, +0.3694432f, +0.9955447f, +0.9957210f, -0.3699816f,
    +0.9950395f, +0.3745111f, -0.7548227f, +0.7489741f, +0.6754632f, +0.8252335f, -0.9139365f, -0.5512263f,
    +0.3801605f, -0.9973316f, -0.9380860f, +0.5095024f, -0.3790909f, +0.9983034f, -0.8839698f, +0.5995291f,
    -0.5190364f, +0.9357044f, -0.9959181f, -0.3919841f, -0.7275031f, -0.7850870f, -0.9992573f, +0.3837152f,
    +0.4096128f, +0.9897895f, +0.7558683f, +0.7599475f, +0.6061437f, +0.8841262f, +0.7187943f, +0.7964067f,
    -0.3955624f, -0.9972535f, +0.9243295f, -0.5500289f, +0.9973264f, +0.4049627f, +0.9958540f, -0.4091865f,
    +0.5235007f, -0.9433507f, +0.5740609f, -0.9139366f, -0.5691788f, +0.9170141f, +0.9226412f, +0.5636988f,
    -0.4219511f, +0.9959417f, -0.8520844f, +0.6670114f, -0.7328414f, +0.7973207f, -0.5285783f, -0.9454666f,
    -0.9977589f, +0.4217194f, -0.7934548f, -0.7378033f, +0.4277600f, -0.9964129f, +0.7344754f, -0.8006862f,
    -0.9946058f, -0.4380784f, -0.4354384f, -0.9978613f, +0.9005206f, +0.6127486f, +0.9946512f, +0.4440148f,
    +0.8024712f, +0.7367803f, +0.9936971f, -0.4488969f, -0.8122507f, +0.7280697f, -0.9334821f, +0.5671371f,
    +0.8204587f, -0.7216612f, +0.4521681f, +0.9965257f, -0.9057648f, -0.6141668f, -0.9939604f, +0.4579073f,
    +0.6946707f, -0.8512779f, -0.4642599f, +0.9963198f, -0.6502576f, -0.8867229f, -0.4767275f, -0.9915156f,
    +0.5869472f, +0.9330087f, -0.9925357f, -0.4812292f, +0.4796630f, -0.9960829f, -0.5967533f, -0.9314386f,
    +0.9940623f, +0.4883220f, +0.7917032f, -0.7748987f, +0.5016059f, +0.9878224f, +0.9437593f, -0.5816783f,
    +0.8561673f, +0.7060751f, +0.9956772f, -0.4910340f, +0.6710539f, +0.8865514f, -0.8715525f, -0.6919830f,
    -0.9960120f, +0.4983156f, -0.7154675f, -0.8539934f, +0.6522807f, -0.9070873f, -0.5065701f, +0.9961941f,
    -0.6785163f, +0.8906910f, -0.6366563f, +0.9266634f, -0.9944385f, -0.5273496f, -0.7767788f, -0.8156111f,
    +0.9351164f, -0.6322332f, +0.9920576f, -0.5393392f, +0.9956906f, +0.5329087f, -0.5335899f, -0.9962856f,
    +0.5431476f, +0.9918513f, -0.9218832f, +0.6600046f, -0.9933337f, +0.5483198f, +0.5414529f, -0.9974306f,
    +0.9127990f, -0.6775145f, -0.7453491f, +0.8587109f, -0.8290756f, -0.7799734f, +0.9075742f, +0.6894931f,
    -0.5552178f, +0.9957367f, +0.6406130f, +0.9449303f, +0.7909036f, +0.8245621f, +0.7320194f, +0.8781233f,
    -0.8124416f, +0.8089982f, -0.9889181f, -0.5847584f, -0.8857421f, +0.7341269f, -0.8835096f, -0.7412981f,
    +0.5878035f, +0.9936983f, +0.9910342f, +0.5937663f, +0.9968024f, -0.5876311f, +0.5911041f, -0.9949980f,
    -0.5981730f, -0.9944639f, +0.9000710f, -0.7367601f, +0.8154873f, -0.8304484f, -0.9389176f, -0.6894902f,
    -0.9906180f, +0.6163189f, +0.8690115f, +0.7793108f, -0.6102763f, +0.9954281f, -0.7360321f, -0.9148750f,
    +0.6245595f, +0.9958566f, +0.7254984f, -0.9257979f, +0.6390314f, -0.9890027f, +0.8740547f, -0.7893417f,
    +0.7803063f, -0.8864580f, -0.6570300f, -0.9836772f, -0.9931620f, -0.6463186f, +0.9962831f, -0.6424773f,
    -0.8141840f, -0.8665285f, -0.9452282f, +0.7219116f, +0.9895673f, +0.6691787f, -0.8882060f, +0.7993616f,
    -0.8794126f, -0.8112231f, -0.6671611f, +0.9946812f, +0.6761580f, +0.9894978f, -0.9971902f, +0.6748670f,
    -0.7324099f, +0.9611204f, -0.7981041f, +0.9088948f, +0.6955252f, -0.9934003f, +0.9856844f, -0.7064275f,
    -0.9977736f, -0.6957873f, -0.7124797f, -0.9931890f, +0.8487936f, +0.8811694f, +0.7218114f, +0.9953319f,
    +0.7616612f, +0.9768904f, -0.8644747f, -0.8893580f, -0.9993937f, +0.7367626f, -0.9879332f, -0.7525699f,
    +0.9857590f, +0.7588065f, -0.7634153f, -0.9856504f, +0.8934920f, -0.8704771f, +0.9908703f, -0.7616319f,
    -0.8934043f, +0.8747116f, +0.7751005f, -0.9867684f, -0.7934866f, +0.9920945f, -0.9888957f, +0.8066753f,
    +0.9832231f, -0.8180810f, -0.9867878f, -0.8153759f, -0.8400570f, -0.9843009f, +0.9819871f, +0.8432844f,
    +0.8428259f, +0.9869555f, -0.9813844f, -0.8831383f, +0.8857568f, -0.9829018f, -0.8964107f, +0.9805712f,
    +0.9860990f, -0.9217185f, -0.9881734f, +0.9214768f, +0.9563189f, +0.9607177f, -0.9447461f, -0.9766368f,
GGML_TABLE_END()

// The neuron_v4 codebook: 256 points in the pair plane, interleaved x,y. Same fit as
// kNeuronVQ5 below -- 800k absmax-normalised pairs with their sign orbit, drawn from all
// 252 layer weight tensors of Qwen3-4B, ordered by radius. See experiments/fit_vq.py.
// Held out from Qwen3-0.6B, the model it is first measured on.
GGML_TABLE_BEGIN(float, kNeuronVQ4, 512)
    +0.0085214f, -0.0103627f, -0.0074475f, +0.0188862f, -0.0282919f, -0.0048543f, +0.0236809f, +0.0195744f,
    -0.0108775f, -0.0320677f, +0.0418858f, -0.0119011f, +0.0234397f, -0.0461208f, -0.0423791f, +0.0304908f,
    -0.0043569f, +0.0538559f, -0.0487382f, -0.0425486f, +0.0657025f, +0.0184173f, +0.0381955f, +0.0583791f,
    -0.0722811f, -0.0038850f, -0.0094090f, -0.0737405f, +0.0680977f, -0.0445222f, -0.0443925f, +0.0791174f,
    -0.0872751f, +0.0450065f, +0.0434224f, -0.0920584f, +0.0067930f, +0.1022556f, +0.1049778f, -0.0092418f,
    +0.0893813f, +0.0586268f, -0.0552201f, -0.0921759f, -0.0989466f, -0.0503112f, +0.0590521f, +0.1071753f,
    -0.0074064f, -0.1247155f, +0.0981522f, -0.0824133f, -0.1310678f, +0.0022535f, -0.0910295f, +0.1051782f,
    +0.1380167f, +0.0412617f, -0.0396249f, +0.1391295f, +0.1396342f, -0.0502911f, +0.0427461f, -0.1472624f,
    -0.1094652f, -0.1075658f, -0.1398851f, +0.0662323f, +0.1195000f, +0.1062557f, -0.0582835f, -0.1524605f,
    -0.1548687f, -0.0583050f, +0.0205430f, +0.1651312f, +0.0993397f, -0.1357959f, +0.1771994f, -0.0052788f,
    +0.0812149f, +0.1637200f, +0.0027273f, -0.1920768f, -0.1922826f, +0.0312941f, +0.1599829f, -0.1127481f,
    -0.1500385f, +0.1309802f, -0.0977414f, +0.1735584f, -0.2030600f, -0.0226667f, +0.2027187f, +0.0517295f,
    -0.0299117f, +0.2080404f, -0.1208591f, -0.1729421f, +0.2031754f, -0.0659500f, +0.0839041f, -0.1965386f,
    +0.1828810f, +0.1128124f, -0.1760946f, -0.1258853f, -0.0635072f, -0.2186946f, +0.1481615f, +0.1730524f,
    -0.2099179f, +0.0897565f, +0.1540380f, -0.1799083f, -0.2282787f, -0.0857733f, +0.0321158f, +0.2452226f,
    +0.0964455f, +0.2285533f, +0.2522329f, -0.0121753f, +0.0540277f, -0.2521026f, -0.0882342f, +0.2478132f,
    -0.0096843f, -0.2643024f, -0.2152624f, +0.1551310f, -0.1670654f, +0.2095478f, +0.2252265f, -0.1502956f,
    -0.1895989f, -0.1956110f, -0.2723573f, -0.0241348f, -0.2722062f, +0.0444271f, +0.2720643f, +0.0549331f,
    +0.2645683f, -0.0867560f, -0.1338360f, -0.2493549f, +0.2556118f, +0.1241210f, +0.1353397f, -0.2501358f,
    +0.2217327f, +0.1878906f, -0.0227441f, +0.2971203f, -0.2606230f, -0.1534076f, +0.1700822f, +0.2520969f,
    -0.0699110f, -0.2978741f, -0.2874098f, +0.1187927f, +0.2121497f, -0.2315053f, -0.1555477f, +0.2839409f,
    +0.1035293f, +0.3075553f, -0.3141108f, -0.0844753f, +0.0992199f, -0.3183910f, -0.2634127f, +0.2063048f,
    +0.3316039f, -0.0496639f, +0.0247843f, -0.3349244f, +0.3385888f, +0.0261877f, -0.2506462f, -0.2363732f,
    +0.3182726f, -0.1385617f, -0.0978366f, +0.3374905f, -0.3514962f, +0.0013252f, +0.3345426f, +0.1102378f,
    +0.0430736f, +0.3523376f, -0.1997718f, -0.2947673f, +0.2893046f, -0.2132194f, +0.1836909f, -0.3179316f,
    -0.2371800f, +0.2805778f, -0.1295919f, -0.3442203f, +0.2569788f, +0.2655955f, +0.3143505f, +0.2003923f,
    -0.3717949f, +0.0845598f, -0.0449166f, -0.3806487f, -0.3550703f, -0.1522060f, +0.1906846f, +0.3403163f,
    -0.3511162f, +0.1732863f, -0.0336308f, +0.3947259f, -0.3280972f, -0.2269799f, +0.2674839f, -0.3036343f,
    -0.1965235f, +0.3642520f, -0.4077250f, -0.0719999f, +0.4024685f, -0.1057538f, +0.1302250f, -0.3974154f,
    +0.4182538f, -0.0139669f, -0.3314383f, +0.2643387f, +0.0385073f, -0.4239933f, +0.4217179f, +0.0781295f,
    +0.3854890f, -0.2006382f, +0.1306504f, +0.4162377f, -0.2117822f, -0.3835018f, -0.2983756f, -0.3219055f,
    +0.4029478f, +0.1741409f, -0.1297919f, +0.4322467f, +0.2873538f, +0.3508283f, -0.4543310f, +0.0174289f,
    -0.1274137f, -0.4376748f, +0.0404419f, +0.4579675f, +0.3581241f, -0.2902724f, +0.2306784f, -0.4017527f,
    +0.3664353f, +0.2913655f, -0.3035467f, +0.3586289f, -0.4474902f, -0.1703357f, -0.0446117f, -0.4825722f,
    -0.4417915f, +0.2030806f, -0.4751765f, +0.1133135f, -0.4014608f, -0.2814757f, +0.2373927f, +0.4421277f,
    +0.3264087f, -0.3854334f, +0.5000378f, -0.0754615f, +0.1198647f, -0.4917616f, -0.0559696f, +0.5031812f,
    +0.4814618f, -0.1804980f, -0.4106950f, +0.3093828f, -0.2468799f, +0.4531111f, +0.5152636f, +0.0302509f,
    -0.5140986f, -0.0691467f, -0.2954142f, -0.4310121f, +0.4623199f, +0.2511117f, +0.5133756f, +0.1468165f,
    -0.2105444f, -0.4952056f, +0.0355778f, -0.5402184f, +0.4615022f, -0.2895437f, +0.2183530f, -0.4997082f,
    +0.1124205f, +0.5403942f, -0.3882689f, -0.3965896f, +0.3668461f, +0.4205840f, -0.5034708f, -0.2541265f,
    -0.1793808f, +0.5415158f, -0.1142454f, -0.5688870f, -0.3637539f, +0.4522247f, -0.5802324f, +0.0440380f,
    +0.4313085f, -0.4016615f, +0.2110175f, +0.5548010f, +0.0133624f, +0.5952150f, -0.5768549f, -0.1682383f,
    -0.5355381f, +0.2778357f, +0.3437701f, -0.4971443f, -0.5848423f, +0.1698489f, +0.4814536f, +0.3752563f,
    +0.5932634f, -0.1546021f, -0.4655649f, +0.3992450f, -0.4949035f, -0.3682630f, +0.6188181f, -0.0470319f,
    -0.1031661f, +0.6231015f, +0.6275713f, +0.0717028f, +0.1344568f, -0.6181171f, -0.3400891f, -0.5374485f,
    +0.3299155f, +0.5452181f, +0.5710626f, +0.2883892f, -0.0106015f, -0.6400163f, +0.5806268f, -0.2706710f,
    -0.6419488f, -0.0624897f, -0.3113593f, +0.5723833f, -0.2359121f, -0.6137070f, +0.2718352f, -0.6033866f,
    +0.6359544f, +0.1877384f, +0.5569190f, -0.4002109f, -0.6252122f, -0.2984244f, -0.4672088f, -0.5161439f,
    +0.4735410f, +0.5165356f, +0.0973540f, +0.7051937f, -0.5934581f, +0.4012923f, +0.4812228f, -0.5343958f,
    -0.4802913f, +0.5384948f, -0.7221473f, +0.1141025f, +0.2436505f, +0.6992700f, -0.2274413f, +0.7070997f,
    -0.1122860f, -0.7352782f, -0.5919820f, -0.4541820f, +0.7319474f, -0.1844645f, -0.0541265f, +0.7585913f,
    +0.7600259f, -0.0275518f, -0.7077445f, +0.2810722f, -0.7364563f, -0.1978261f, +0.0604739f, -0.7609809f,
    -0.7713764f, -0.0295557f, +0.2302250f, -0.7455492f, +0.7733245f, +0.1357259f, +0.6418620f, +0.4582520f,
    +0.3980181f, -0.6813768f, +0.7247923f, +0.3127046f, -0.2673854f, -0.7467001f, +0.7135490f, -0.3464886f,
    +0.4111186f, +0.6894252f, -0.4246031f, -0.6815467f, -0.4031822f, +0.7020566f, -0.7383438f, -0.4070008f,
    +0.6695739f, -0.5164131f, -0.7014700f, +0.4940970f, +0.5971268f, +0.6333750f, -0.6249038f, -0.6187592f,
    -0.5947129f, +0.6625736f, +0.5850632f, -0.6799822f, +0.1731645f, +0.9061710f, -0.0227303f, -0.9354665f,
    -0.9124477f, +0.2241877f, -0.1895215f, +0.9229159f, +0.0028992f, +0.9425840f, -0.9464406f, +0.0569812f,
    +0.9413974f, -0.1200189f, -0.9439777f, -0.1272662f, +0.1560025f, -0.9407831f, +0.9539340f, +0.0635031f,
    -0.1919242f, -0.9368621f, +0.9202584f, -0.3043292f, +0.9394706f, +0.2617530f, +0.3415847f, -0.9154246f,
    -0.9266452f, -0.3108146f, +0.3608063f, +0.9084183f, -0.3806626f, -0.9158460f, -0.3881356f, +0.9128651f,
    -0.9123176f, +0.4223240f, +0.8909844f, +0.4687683f, +0.8991819f, -0.5187830f, -0.5876437f, -0.8683028f,
    -0.9049390f, -0.5296957f, +0.5374418f, -0.9012963f, +0.5980064f, +0.8796631f, -0.6395192f, +0.8893213f,
    +0.8445984f, +0.7067246f, -0.8658654f, +0.6867218f, +0.8172473f, -0.7924799f, -0.8401572f, -0.7750302f,
GGML_TABLE_END()

// The neuron_v5 codebook: 1024 points in the pair plane, interleaved x,y.
// Fitted by Lloyd iteration on 800k absmax-normalised weight pairs drawn from all 252
// layer weight tensors of Qwen3-4B, carrying each pair's sign orbit because the weight
// distribution is symmetric about zero. Ordered by radius. See experiments/fit_vq5.py.
// Held out from Qwen3-0.6B, the model it is first measured on.
GGML_TABLE_BEGIN(float, kNeuronVQ5, 2048)
    +0.0024361f, -0.0045525f, -0.0035632f, +0.0111712f, -0.0129358f, -0.0034853f, +0.0117496f, +0.0080765f,
    -0.0010199f, -0.0195416f, +0.0171212f, -0.0116606f, -0.0168520f, +0.0182211f, +0.0053522f, +0.0285729f,
    -0.0218182f, -0.0207715f, +0.0311287f, +0.0014299f, -0.0319874f, +0.0016124f, +0.0286983f, +0.0218236f,
    +0.0175723f, -0.0329301f, -0.0090556f, -0.0400870f, -0.0194379f, +0.0404513f, -0.0404069f, +0.0247328f,
    -0.0444626f, -0.0182350f, +0.0440493f, -0.0199727f, +0.0039208f, +0.0530773f, +0.0283939f, +0.0468157f,
    +0.0551754f, +0.0063663f, -0.0594620f, +0.0032570f, -0.0424260f, -0.0430721f, +0.0104510f, -0.0605344f,
    +0.0360937f, -0.0505859f, +0.0536923f, +0.0339681f, -0.0205881f, -0.0628064f, -0.0216469f, +0.0670302f,
    -0.0459573f, +0.0552786f, -0.0656950f, +0.0306808f, -0.0713867f, -0.0254569f, +0.0758080f, -0.0152576f,
    +0.0652853f, -0.0418667f, +0.0044100f, +0.0812286f, +0.0329973f, +0.0752307f, +0.0825853f, +0.0133874f,
    +0.0600998f, +0.0628671f, -0.0022357f, -0.0877263f, +0.0554131f, -0.0705828f, -0.0549016f, -0.0714588f,
    -0.0919431f, +0.0019120f, +0.0830727f, +0.0408795f, +0.0303821f, -0.0875959f, -0.0743822f, +0.0606467f,
    -0.0800835f, -0.0538948f, -0.0328056f, -0.0911867f, -0.0937444f, +0.0326861f, -0.0288428f, +0.0954639f,
    -0.0568595f, +0.0880788f, +0.0988050f, -0.0382700f, +0.1081118f, -0.0053565f, -0.1053889f, -0.0282992f,
    +0.0872354f, -0.0677887f, +0.0261180f, +0.1081858f, -0.0064489f, +0.1138262f, +0.0554090f, +0.1014272f,
    +0.0680163f, -0.0971067f, +0.0128369f, -0.1180849f, +0.1012795f, +0.0637228f, +0.1165336f, +0.0296956f,
    +0.0824046f, +0.0880026f, -0.0217086f, -0.1192687f, -0.0881635f, -0.0880167f, -0.1096897f, +0.0607936f,
    -0.0611466f, -0.1110456f, -0.1252575f, +0.0294303f, -0.0923036f, +0.0898074f, +0.0467085f, -0.1215627f,
    -0.1161141f, -0.0593336f, -0.1305751f, -0.0021573f, -0.0439924f, +0.1259765f, +0.1325965f, -0.0351851f,
    +0.1227308f, -0.0690897f, +0.1427490f, -0.0015270f, +0.0172441f, +0.1417949f, -0.0775865f, +0.1207197f,
    +0.1041151f, -0.0993108f, -0.1444661f, -0.0328102f, -0.0176414f, +0.1478042f, +0.1352009f, +0.0646659f,
    +0.0854741f, +0.1239610f, -0.0421215f, -0.1454965f, -0.0050764f, -0.1514407f, +0.0540203f, +0.1415844f,
    +0.1163626f, +0.0983484f, +0.0848565f, -0.1290705f, -0.1240252f, -0.0922522f, +0.0282821f, -0.1534460f,
    +0.1542306f, +0.0290760f, -0.1319965f, +0.0866228f, -0.1490132f, +0.0574044f, -0.1017113f, -0.1256883f,
    -0.1608474f, +0.0228814f, -0.1154137f, +0.1183806f, -0.0743005f, -0.1479320f, -0.1556733f, -0.0657813f,
    +0.0651309f, -0.1569039f, -0.0661057f, +0.1584241f, +0.1686929f, -0.0343778f, -0.1731591f, -0.0091946f,
    +0.1611298f, -0.0688618f, +0.1436471f, -0.1013271f, -0.0344097f, +0.1764982f, +0.0385069f, +0.1760580f,
    +0.0023175f, +0.1811382f, +0.1230689f, -0.1335267f, +0.1194360f, +0.1371083f, +0.1822972f, -0.0008870f,
    -0.0306507f, -0.1799969f, +0.1721523f, +0.0608842f, +0.1531152f, +0.1031693f, -0.1039866f, +0.1543082f,
    +0.0899172f, +0.1640961f, -0.1401972f, -0.1243608f, +0.0044585f, -0.1903835f, +0.0436216f, -0.1865602f,
    -0.1865633f, +0.0455608f, -0.1552815f, +0.1141876f, -0.1884393f, -0.0434045f, -0.1683669f, -0.1009239f,
    +0.1059389f, -0.1660840f, -0.1794605f, +0.0818708f, -0.1027135f, -0.1704139f, -0.0663220f, -0.1881705f,
    +0.1976778f, +0.0307863f, -0.1419953f, +0.1489897f, +0.0716704f, +0.1933841f, +0.2057564f, -0.0352970f,
    -0.2088494f, +0.0184658f, -0.1368226f, -0.1592938f, +0.1880052f, +0.0947858f, +0.1982877f, -0.0734188f,
    +0.1563027f, +0.1424363f, +0.1620198f, -0.1387978f, +0.1845092f, -0.1078659f, -0.1995636f, -0.0781982f,
    -0.0612147f, +0.2054288f, +0.0856092f, -0.1968113f, -0.0943340f, +0.1949296f, -0.2162137f, -0.0167081f,
    -0.0263196f, +0.2164257f, +0.1307853f, +0.1766458f, +0.0089348f, +0.2199809f, +0.0430151f, +0.2162464f,
    -0.0362730f, -0.2183422f, +0.2133743f, +0.0627764f, +0.2229083f, -0.0003338f, +0.0265666f, -0.2213732f,
    -0.1927405f, +0.1162906f, +0.1436019f, -0.1733871f, -0.1763418f, -0.1405414f, -0.1293522f, +0.1855029f,
    -0.2206477f, +0.0614884f, -0.1031988f, -0.2098946f, +0.0632050f, -0.2267348f, +0.1944744f, +0.1327224f,
    -0.2292015f, -0.0544097f, -0.1819221f, +0.1503575f, +0.1089560f, +0.2120120f, -0.2090779f, -0.1161821f,
    -0.0051830f, -0.2396246f, -0.1399131f, -0.1983250f, -0.0731383f, -0.2324063f, +0.2426699f, +0.0321728f,
    +0.1296825f, -0.2081760f, -0.2244016f, +0.1017704f, +0.2365981f, -0.0691570f, +0.2267424f, +0.0997558f,
    +0.1752865f, +0.1772673f, +0.2035455f, -0.1448629f, +0.2480349f, -0.0323756f, +0.0759325f, +0.2385941f,
    -0.2517221f, +0.0069125f, +0.2268710f, -0.1094635f, -0.1718898f, +0.1851725f, -0.1777940f, -0.1796246f,
    -0.0842347f, +0.2388961f, +0.1823751f, -0.1794671f, -0.0101869f, +0.2560426f, -0.0492579f, +0.2516164f,
    +0.0329180f, +0.2566144f, +0.1031254f, -0.2374004f, -0.1208610f, +0.2302576f, -0.2567042f, +0.0431466f,
    +0.1533683f, +0.2114722f, -0.2600209f, -0.0310150f, -0.2462323f, -0.0901328f, +0.0326068f, -0.2605179f,
    -0.2206312f, +0.1474533f, -0.2155082f, -0.1555447f, +0.2569678f, +0.0695206f, -0.0413298f, -0.2633778f,
    -0.1533797f, +0.2191639f, +0.2697545f, +0.0024953f, -0.2583252f, +0.0838521f, +0.2133289f, +0.1731135f,
    -0.1196816f, -0.2482466f, +0.2389448f, +0.1388534f, +0.0720387f, -0.2681941f, +0.1736652f, -0.2183518f,
    -0.1557023f, -0.2355207f, -0.0011792f, -0.2827492f, -0.0869747f, -0.2694301f, +0.1123887f, +0.2600605f,
    -0.2146978f, +0.1881115f, -0.2535344f, -0.1312395f, +0.2665391f, -0.1023045f, +0.2794993f, -0.0631745f,
    +0.2445326f, -0.1496512f, -0.2576062f, +0.1275411f, -0.1888138f, -0.2171585f, +0.1474356f, +0.2472704f,
    +0.2673202f, +0.1092703f, +0.0755087f, +0.2800381f, +0.2875611f, +0.0395942f, +0.1492818f, -0.2504598f,
    +0.2222474f, -0.1888554f, -0.2855006f, -0.0610912f, +0.1967095f, +0.2175858f, +0.0047953f, +0.2940624f,
    -0.2949142f, -0.0090805f, -0.1918352f, +0.2242680f, -0.2227333f, -0.1951717f, -0.0355236f, +0.2940581f,
    -0.1142898f, +0.2735156f, -0.0769848f, +0.2866140f, +0.2971821f, -0.0285661f, -0.3002685f, +0.0308722f,
    +0.1154474f, -0.2791625f, +0.0443885f, +0.2998853f, -0.2971024f, +0.0703907f, -0.1566148f, +0.2627679f,
    -0.2897536f, -0.1038285f, -0.2591234f, +0.1677794f, +0.0419033f, -0.3059123f, +0.2992562f, +0.0767675f,
    -0.0697611f, -0.3024870f, +0.2563596f, +0.1757581f, -0.2590376f, -0.1727052f, -0.0326521f, -0.3122464f,
    -0.2937363f, +0.1113652f, +0.2167274f, -0.2292446f, +0.3155476f, +0.0070746f, +0.2850357f, -0.1401061f,
    -0.1216863f, -0.2942579f, -0.1579924f, -0.2784064f, +0.0831856f, -0.3097431f, +0.2401845f, +0.2130578f,
    +0.1883770f, +0.2605346f, +0.2873251f, +0.1500352f, -0.1946208f, -0.2601079f, +0.1085197f, +0.3063831f,
    +0.2670652f, -0.1866955f, +0.1519371f, +0.2890591f, +0.0058756f, -0.3266962f, +0.3074640f, +0.1135768f,
    -0.2544121f, +0.2066625f, -0.3262098f, -0.0355123f, +0.3112582f, -0.1050770f, -0.2299261f, +0.2349517f,
    +0.3246860f, -0.0665498f, -0.2313257f, -0.2379065f, -0.2976205f, -0.1468973f, +0.1978307f, -0.2681356f,
    -0.2004342f, +0.2676157f, +0.1587207f, -0.2942842f, -0.0622270f, +0.3291276f, +0.0215813f, +0.3348426f,
    -0.0185939f, +0.3358836f, +0.3335938f, +0.0457884f, -0.3378253f, +0.0107123f, -0.1398256f, +0.3077981f,
    -0.3282531f, -0.0818752f, -0.3037411f, +0.1521393f, +0.0697225f, +0.3346858f, -0.1036813f, +0.3268323f,
    -0.2672715f, -0.2154272f, +0.3430530f, -0.0267444f, +0.2346977f, +0.2552991f, -0.1064312f, -0.3303908f,
    -0.3428764f, +0.0626343f, +0.2638751f, -0.2277372f, -0.3321654f, +0.1060636f, +0.1299402f, -0.3257096f,
    -0.0689933f, -0.3461166f, +0.3423792f, +0.0873250f, -0.1789361f, +0.3055574f, +0.2910181f, +0.2026352f,
    -0.2935989f, +0.1989731f, +0.0473577f, -0.3515649f, +0.3091899f, -0.1745329f, -0.3055897f, -0.1927791f,
    +0.0914876f, -0.3495718f, -0.3389563f, -0.1253679f, -0.0306404f, -0.3608975f, +0.3622495f, +0.0117161f,
    -0.1869420f, -0.3112288f, +0.1487313f, +0.3335290f, +0.3382560f, +0.1380752f, +0.3373351f, -0.1416666f,
    +0.2514529f, -0.2664346f, +0.1900960f, +0.3131941f, -0.2295130f, -0.2857402f, -0.2438357f, +0.2750706f,
    -0.3645305f, -0.0533794f, -0.1479571f, -0.3383865f, +0.2776796f, +0.2449887f, +0.1954653f, -0.3148561f,
    -0.2778387f, +0.2461839f, +0.3258329f, +0.1782472f, +0.0080361f, -0.3714109f, +0.1106276f, +0.3554331f,
    +0.0358242f, +0.3716566f, +0.2268117f, +0.2967892f, +0.3688798f, -0.0614809f, -0.0517890f, +0.3704377f,
    -0.2696727f, -0.2603266f, +0.3617451f, -0.1035134f, -0.3753900f, +0.0340291f, +0.3079057f, -0.2179392f,
    -0.1425316f, +0.3525666f, -0.3811929f, -0.0126500f, -0.3530607f, +0.1471786f, -0.3433218f, -0.1692962f,
    -0.2219191f, +0.3127401f, -0.0075232f, +0.3843054f, -0.3353912f, +0.1898596f, +0.3809228f, +0.0621400f,
    -0.3752557f, -0.0967019f, -0.0976062f, +0.3761831f, -0.3756141f, +0.1055515f, -0.1075367f, -0.3751889f,
    +0.2386522f, -0.3090835f, -0.3098409f, -0.2396196f, +0.1687974f, -0.3547567f, +0.3758015f, +0.1180218f,
    +0.0784158f, +0.3872814f, +0.1251083f, -0.3754325f, +0.3960965f, -0.0252114f, +0.3495058f, -0.1896919f,
    -0.1867367f, +0.3513247f, -0.0620065f, -0.3965377f, -0.3229756f, +0.2387441f, +0.0756493f, -0.3948190f,
    +0.4016369f, +0.0218913f, +0.3263562f, +0.2357184f, -0.3971094f, +0.0675099f, -0.2297474f, -0.3319291f,
    -0.1896527f, -0.3576008f, +0.2720538f, +0.3017172f, -0.2829675f, +0.2919391f, +0.3086137f, -0.2654542f,
    +0.1967612f, +0.3574816f, +0.1565585f, +0.3776720f, +0.0315484f, -0.4081428f, -0.0168426f, -0.4096112f,
    +0.3796650f, -0.1552470f, -0.2743935f, -0.3055969f, +0.3769442f, +0.1654171f, -0.3507347f, -0.2167743f,
    -0.4082554f, -0.0603163f, -0.3873862f, -0.1443721f, +0.3593986f, +0.2079875f, +0.3096512f, +0.2776595f,
    +0.4097542f, -0.0727514f, +0.0354760f, +0.4151390f, -0.4183129f, +0.0168211f, +0.2879396f, -0.3041890f,
    +0.2171362f, -0.3583482f, +0.2404076f, +0.3436498f, -0.0583183f, +0.4165033f, -0.1488317f, -0.3938905f,
    -0.1410917f, +0.3971875f, -0.2614693f, +0.3326634f, +0.3537838f, -0.2351382f, -0.3166593f, -0.2835977f,
    +0.4139316f, +0.0986090f, +0.4094898f, -0.1204291f, +0.1251292f, +0.4081540f, -0.3833033f, +0.1883579f,
    -0.2303128f, +0.3638184f, -0.4045186f, +0.1482234f, -0.3245074f, +0.2850767f, -0.3660302f, +0.2293517f,
    -0.0063898f, +0.4329087f, -0.1039141f, -0.4239426f, +0.1536074f, -0.4089287f, -0.4363149f, -0.0255928f,
    -0.3938590f, -0.1897048f, -0.4245850f, -0.1091275f, -0.4259713f, +0.1050936f, +0.4361480f, +0.0525073f,
    +0.2641068f, -0.3514464f, +0.4406675f, +0.0007349f, +0.1098023f, -0.4281074f, -0.1066715f, +0.4303629f,
    +0.0797325f, +0.4367557f, -0.1889002f, +0.4021412f, -0.3593621f, -0.2616408f, -0.2025366f, -0.3974034f,
    +0.3966614f, -0.2044576f, -0.0504603f, -0.4450276f, +0.4225923f, +0.1496913f, -0.2421641f, -0.3783877f,
    -0.2792613f, -0.3531729f, +0.0593655f, -0.4463574f, -0.4475443f, +0.0547973f, +0.4499302f, -0.0449285f,
    +0.1979673f, -0.4066379f, -0.3066986f, +0.3342185f, +0.1788902f, +0.4177639f, +0.3548476f, +0.2848201f,
    +0.3804853f, +0.2505328f, +0.4076453f, +0.2036737f, +0.0060684f, -0.4567169f, +0.2216720f, +0.4012600f,
    +0.2934838f, +0.3525491f, +0.3258990f, +0.3232051f, +0.3674336f, -0.2792366f, +0.4292785f, -0.1695909f,
    -0.3218777f, -0.3309160f, +0.4538571f, -0.0939999f, +0.3415535f, -0.3137403f, -0.0398057f, +0.4631855f,
    -0.3735201f, +0.2773091f, -0.4591776f, -0.0771112f, -0.1760013f, -0.4311344f, -0.4394932f, -0.1555028f,
    -0.4022586f, -0.2362738f, +0.0425793f, +0.4653048f, +0.3107851f, -0.3506062f, +0.4587921f, +0.1024036f,
    -0.1596601f, +0.4421670f, -0.2351695f, +0.4088266f, +0.2677420f, +0.3887038f, -0.4729490f, +0.0132183f,
    +0.2503507f, -0.4016470f, +0.1203283f, +0.4578036f, -0.2818832f, +0.3808424f, -0.4336233f, +0.1915627f,
    -0.3675168f, -0.3060206f, -0.4136010f, +0.2421019f, -0.1375223f, -0.4593027f, -0.4578425f, +0.1447193f,
    -0.3565062f, +0.3220736f, +0.4105126f, -0.2517002f, -0.4729841f, +0.0959750f, -0.0836319f, +0.4758498f,
    +0.4832265f, +0.0149612f, +0.4645470f, -0.1374986f, -0.0902059f, -0.4767780f, +0.1614429f, -0.4600263f,
    +0.0018303f, +0.4890039f, -0.4875904f, -0.0389817f, +0.1067141f, -0.4788052f, +0.4872678f, +0.0588425f,
    +0.1665188f, +0.4619354f, -0.4462298f, -0.2058594f, -0.0387979f, -0.4937690f, -0.2329205f, -0.4372981f,
    +0.4556132f, +0.1961998f, +0.4952254f, -0.0311933f, -0.2769780f, -0.4121891f, -0.3198895f, -0.3812668f,
    -0.3371890f, +0.3682642f, +0.4769425f, +0.1479673f, +0.4364944f, +0.2443752f, +0.0536253f, -0.4976358f,
    -0.4852832f, -0.1248587f, +0.4522500f, -0.2166760f, -0.4149318f, -0.2820264f, -0.1288330f, +0.4851406f,
    +0.3820744f, +0.3272389f, +0.2206151f, -0.4541037f, +0.3037764f, -0.4033979f, -0.2180104f, +0.4560916f,
    +0.3483070f, +0.3668911f, +0.4154572f, +0.2890392f, +0.0870262f, +0.4987490f, +0.5008538f, -0.0844621f,
    -0.3656929f, -0.3539473f, -0.5065113f, +0.0556670f, +0.2177369f, +0.4617743f, +0.0072315f, -0.5107573f,
    +0.2637416f, +0.4383005f, -0.1918365f, -0.4747222f, -0.2730549f, +0.4346709f, -0.0488176f, +0.5141518f,
    -0.4642047f, +0.2277159f, -0.4263823f, +0.2926019f, +0.3890032f, -0.3426933f, +0.4204793f, -0.3037781f,
    -0.1796057f, +0.4879393f, -0.4910536f, -0.1726086f, +0.3538146f, -0.3818595f, +0.3172625f, +0.4142604f,
    +0.5112745f, +0.1087227f, +0.4895067f, -0.1843669f, -0.5159676f, -0.0876934f, -0.5235328f, -0.0050603f,
    -0.4007916f, +0.3371881f, -0.4914454f, +0.1812372f, +0.0445787f, +0.5231947f, +0.1386317f, +0.5072352f,
    -0.4594540f, -0.2560217f, -0.3258963f, +0.4151831f, +0.2771088f, -0.4500659f, -0.5149720f, +0.1261313f,
    -0.1401952f, -0.5128025f, -0.4204510f, -0.3303157f, -0.0869053f, -0.5293201f, +0.5376657f, +0.0144164f,
    +0.5204521f, -0.1381570f, +0.1376568f, -0.5209103f, -0.3241561f, -0.4322148f, +0.4707814f, -0.2673884f,
    +0.5066325f, +0.1917776f, -0.2780893f, -0.4650617f, +0.1949843f, -0.5061774f, -0.0019143f, +0.5433095f,
    -0.3771114f, +0.3914993f, +0.5398619f, +0.0649601f, -0.1033615f, +0.5364421f, +0.0817977f, -0.5407050f,
    -0.3692268f, -0.4037840f, -0.4981501f, -0.2266241f, +0.4939055f, +0.2383703f, +0.5475073f, -0.0414572f,
    +0.4725134f, +0.2805910f, -0.5481667f, -0.0499860f, -0.4778370f, +0.2742624f, -0.0324365f, -0.5511249f,
    +0.1955681f, +0.5172868f, -0.2365207f, -0.5006959f, +0.4111483f, +0.3727637f, -0.2659178f, +0.4889421f,
    +0.3754074f, +0.4113055f, +0.4490608f, +0.3302763f, +0.5362611f, +0.1534828f, -0.4664792f, -0.3065693f,
    +0.0265967f, -0.5594621f, -0.5533870f, +0.0893046f, -0.5594998f, +0.0373808f, -0.3116042f, +0.4662945f,
    +0.5528830f, -0.0950732f, -0.4149767f, -0.3780454f, -0.1623849f, +0.5377560f, -0.2213622f, +0.5176908f,
    +0.3082889f, +0.4721977f, -0.5439960f, -0.1494944f, +0.3508815f, -0.4421177f, +0.2548519f, -0.5057097f,
    +0.0930555f, +0.5590708f, +0.2626700f, +0.5030820f, -0.1925394f, -0.5339528f, -0.5224640f, +0.2241585f,
    +0.5188388f, -0.2330914f, -0.4609999f, +0.3370027f, -0.0490936f, +0.5698577f, +0.1444792f, +0.5555112f,
    -0.5651834f, -0.1026128f, -0.5389313f, -0.2006201f, +0.4067373f, -0.4066859f, +0.4777741f, -0.3223096f,
    -0.5495098f, +0.1741156f, +0.5686334f, +0.1076095f, +0.4502096f, -0.3650534f, -0.4378372f, +0.3819786f,
    +0.5517583f, -0.1840184f, +0.3174497f, -0.4883777f, -0.3627007f, +0.4567625f, +0.3614612f, +0.4612057f,
    +0.0481536f, +0.5840151f, -0.5862416f, -0.0135087f, -0.5180727f, -0.2754098f, -0.1355954f, -0.5709167f,
    -0.3292695f, -0.4868289f, -0.0798883f, -0.5851396f, +0.1818581f, -0.5637126f, +0.1197862f, -0.5801019f,
    +0.5568318f, +0.2070001f, -0.3754006f, -0.4606838f, -0.2829711f, -0.5230679f, +0.5940236f, +0.0477805f,
    -0.4692052f, -0.3680072f, -0.4161214f, +0.4298933f, +0.5989545f, -0.0076810f, +0.5172549f, +0.3041900f,
    -0.4186248f, -0.4299698f, -0.5074999f, +0.3203907f, -0.0036688f, +0.6015730f, -0.0928239f, +0.5946357f,
    -0.5371410f, +0.2734181f, -0.5872629f, +0.1366894f, +0.2366914f, +0.5554301f, +0.0613794f, -0.6007259f,
    +0.6020907f, -0.0678115f, +0.5281323f, -0.2977476f, +0.5493940f, +0.2581604f, -0.3231873f, +0.5142349f,
    +0.4704137f, +0.3843191f, +0.5931535f, -0.1365211f, -0.1493760f, +0.5911437f, +0.4254769f, +0.4369014f,
    -0.0088132f, -0.6105199f, -0.5151092f, -0.3297769f, -0.6079269f, -0.0682146f, +0.2424696f, -0.5624125f,
    -0.2692479f, +0.5507869f, +0.5924512f, +0.1604832f, -0.2127964f, +0.5762966f, +0.1894463f, +0.5880251f,
    -0.6184521f, +0.0308395f, -0.2478669f, -0.5691355f, +0.3416345f, +0.5195904f, -0.6165858f, +0.0880413f,
    +0.5698301f, -0.2527646f, -0.1866686f, -0.5948330f, -0.5714329f, -0.2500827f, +0.4085814f, -0.4716723f,
    -0.5809079f, +0.2292047f, -0.5921633f, -0.1995647f, -0.6093274f, -0.1396131f, +0.1350813f, +0.6141866f,
    +0.3080365f, -0.5501074f, +0.5221202f, +0.3536453f, +0.6238445f, +0.0985838f, +0.4673843f, -0.4267684f,
    -0.4048005f, +0.4868018f, +0.2947664f, +0.5623284f, -0.4742325f, -0.4272862f, -0.4809205f, +0.4198467f,
    -0.3331006f, -0.5464748f, +0.1597323f, -0.6216200f, +0.0805623f, +0.6368487f, +0.4034343f, +0.4996417f,
    +0.3696049f, -0.5251645f, -0.5194895f, +0.3805013f, +0.6113106f, -0.2030555f, -0.0468363f, +0.6433015f,
    +0.5264535f, -0.3738950f, -0.0633893f, -0.6432478f, -0.3907695f, -0.5154305f, -0.6199820f, +0.1929851f,
    +0.6089922f, +0.2256950f, -0.6498899f, -0.0293145f, -0.3753121f, +0.5324913f, -0.4400104f, -0.4805281f,
    +0.1052169f, -0.6430846f, -0.5205367f, -0.3928925f, +0.6459609f, -0.0946228f, -0.1170158f, +0.6424379f,
    +0.0193717f, +0.6535038f, -0.5648005f, +0.3324018f, +0.6549745f, -0.0312944f, -0.1265456f, -0.6437158f,
    -0.5904708f, +0.2862627f, -0.4567448f, +0.4713537f, -0.5844760f, -0.3002704f, +0.6565107f, +0.0384308f,
    +0.4842498f, +0.4467059f, +0.0373693f, -0.6593316f, +0.2187126f, -0.6235461f, +0.5887251f, +0.3008435f,
    -0.3219838f, +0.5776932f, -0.5590280f, -0.3556781f, +0.2496442f, +0.6145365f, -0.1982394f, +0.6365241f,
    -0.6603200f, -0.0938897f, +0.5858376f, -0.3201634f, +0.6492016f, +0.1593086f, -0.2617124f, +0.6156122f,
    -0.6705498f, +0.0605356f, -0.6424996f, -0.2043585f, -0.6599051f, +0.1462265f, -0.3056897f, -0.6052587f,
    +0.6601256f, -0.1550169f, +0.2882036f, -0.6147726f, -0.2443665f, -0.6339346f, +0.6282810f, -0.2640520f,
    +0.5392382f, +0.4170986f, +0.3526441f, +0.5856860f, +0.6764230f, +0.1023737f, +0.1999797f, +0.6558246f,
    -0.1884710f, -0.6593287f, +0.4644778f, +0.5105662f, +0.4327326f, -0.5380657f, +0.5331173f, -0.4408785f,
    -0.0191296f, -0.6916068f, -0.6429692f, +0.2589713f, +0.4066995f, +0.5613841f, +0.4847170f, -0.4961150f,
    -0.6766260f, -0.1541805f, -0.4405483f, +0.5365734f, +0.5904176f, +0.3656724f, -0.3790132f, -0.5829127f,
    -0.6955609f, +0.0074492f, +0.1391046f, +0.6828051f, +0.3581415f, -0.5987499f, -0.1554840f, +0.6801757f,
    -0.5346528f, -0.4503737f, -0.5443490f, +0.4401774f, -0.4968077f, -0.4942543f, -0.6471237f, -0.2692820f,
    -0.5876794f, +0.3824511f, -0.0830324f, +0.6964533f, -0.4489035f, -0.5440000f, +0.1798118f, -0.6836357f,
    -0.3838051f, +0.5939420f, +0.6721937f, +0.2197527f, -0.5073454f, +0.4932315f, -0.6158143f, -0.3491237f,
    +0.3170299f, +0.6333649f, +0.0676090f, +0.7051060f, +0.6531930f, +0.2768486f, -0.0119401f, +0.7098094f,
    +0.1030640f, -0.7024623f, +0.7123244f, -0.0029560f, +0.7086112f, -0.0861195f, +0.6048445f, -0.3802214f,
    -0.7067509f, +0.1050847f, -0.0869165f, -0.7129500f, +0.6830088f, -0.2235392f, -0.7158967f, -0.0706157f,
    -0.5885039f, -0.4139932f, -0.3231501f, +0.6436322f, -0.6884447f, +0.2121743f, -0.6421949f, +0.3277680f,
    +0.5305791f, +0.4970925f, +0.2598464f, -0.6834265f, +0.7268196f, +0.0812290f, +0.0360087f, -0.7326186f,
    +0.7137918f, +0.1702077f, +0.6547986f, +0.3336651f, +0.6636386f, -0.3160166f, -0.2669785f, +0.6867564f,
    -0.2087686f, +0.7088237f, -0.1550484f, -0.7234299f, -0.7062410f, -0.2207191f, +0.7206940f, -0.1677172f,
    +0.2747648f, +0.6890622f, +0.4300502f, -0.6046913f, -0.3655578f, -0.6472673f, -0.2956743f, -0.6855479f,
    -0.6700970f, -0.3317271f, -0.5038834f, +0.5538855f, +0.4718840f, +0.5814660f, -0.4427026f, -0.6042859f,
    +0.6096844f, +0.4360457f, +0.5940768f, -0.4572272f, -0.4477939f, +0.6019200f, +0.3324139f, -0.6762226f,
    -0.2316587f, -0.7176856f, -0.1314393f, +0.7428352f, -0.7547507f, -0.0225133f, +0.2100897f, +0.7257633f,
    -0.5635236f, -0.5035501f, -0.7546263f, +0.0460130f, +0.4075153f, +0.6376388f, -0.7381012f, +0.1711851f,
    -0.5150371f, -0.5571371f, -0.6111772f, +0.4496871f, -0.7449260f, -0.1501774f, +0.1296586f, +0.7497688f,
    -0.6531426f, +0.3905864f, +0.1498358f, -0.7485733f, +0.5041573f, -0.5733936f, -0.7100274f, +0.2850059f,
    +0.5626397f, -0.5214041f, -0.3922690f, +0.6592854f, +0.0354470f, +0.7664564f, +0.7674068f, +0.0370155f,
    +0.7686077f, -0.0419686f, -0.5769104f, +0.5109767f, -0.0405354f, -0.7704167f, -0.0547883f, +0.7700419f,
    +0.7276727f, +0.2620679f, -0.6619200f, -0.4088131f, +0.2235563f, -0.7459652f, +0.7305589f, -0.2800582f,
    -0.7740725f, +0.1153279f, -0.7275584f, -0.2894265f, +0.3593029f, +0.6962288f, +0.6773020f, +0.3967761f,
    +0.7752417f, -0.1262675f, +0.6556803f, -0.4347868f, +0.4099758f, -0.6715338f, -0.6234348f, -0.4799670f,
    +0.6922354f, -0.3765616f, +0.5998512f, +0.5126399f, +0.7770584f, +0.1388791f, -0.3319184f, +0.7179612f,
    +0.0818408f, -0.7877158f, -0.7923604f, -0.0887646f, +0.5479912f, +0.5795385f, -0.4441467f, -0.6654761f,
    -0.7231218f, +0.3482253f, -0.5556486f, +0.5822591f, -0.2072667f, +0.7794929f, -0.1239481f, -0.7975548f,
    -0.7726619f, +0.2421936f, +0.7331297f, +0.3441817f, -0.3724055f, -0.7194626f, -0.5263480f, -0.6195741f,
    +0.7869539f, -0.2074904f, -0.5895199f, -0.5625862f, -0.6795651f, +0.4502974f, -0.5134759f, +0.6351320f,
    -0.7822304f, -0.2353133f, -0.7307186f, -0.3659253f, -0.2138071f, -0.7913404f, +0.2986521f, +0.7635010f,
    -0.4651279f, +0.6757940f, +0.3136985f, -0.7582407f, +0.0168374f, -0.8206497f, +0.7951692f, +0.2070395f,
    +0.2153556f, +0.7952456f, +0.4988573f, -0.6568599f, +0.4983711f, +0.6578847f, +0.8208862f, +0.0895318f,
    -0.2806628f, +0.7766369f, -0.8274465f, -0.0071318f, -0.1302415f, +0.8181843f, -0.3013027f, -0.7728102f,
    +0.1251930f, +0.8209841f, +0.8265899f, -0.0819878f, -0.8282028f, +0.0792479f, +0.6472071f, -0.5257107f,
    +0.8343109f, +0.0060401f, +0.6773854f, +0.4870948f, -0.8151810f, +0.1780871f, +0.1602272f, -0.8201793f,
    -0.4098495f, +0.7293232f, -0.7297612f, +0.4102153f, -0.6585861f, +0.5186408f, -0.8227027f, -0.1624265f,
    +0.5823881f, -0.6057151f, +0.0390128f, +0.8399293f, -0.0492247f, +0.8395304f, +0.4403885f, +0.7183779f,
    -0.6903558f, -0.4944302f, -0.6240504f, +0.5761606f, +0.8001109f, +0.2945281f, +0.7701542f, -0.3664786f,
    +0.2436736f, -0.8176383f, +0.4048201f, -0.7511870f, +0.8094945f, -0.2883519f, -0.7993648f, +0.3197365f,
    -0.4574475f, -0.7304736f, +0.7307084f, -0.4587705f, -0.7450884f, -0.4427638f, +0.7440123f, +0.4466916f,
    -0.8050087f, -0.3266426f, +0.3791975f, +0.7834532f, +0.6581160f, +0.5717213f, -0.5310854f, -0.6913726f,
    -0.0692835f, -0.8724636f, -0.5919664f, +0.6467111f, -0.6667407f, -0.5716887f, -0.3643787f, +0.7999027f,
    +0.6060220f, +0.6404957f, +0.0972904f, -0.8765379f, -0.8822453f, -0.0774186f, -0.6146032f, -0.6407862f,
    -0.7457439f, +0.4849051f, +0.8764336f, -0.1552555f, -0.3862916f, -0.8022184f, -0.2193615f, +0.8653546f,
    -0.1663972f, -0.8783931f, +0.8093618f, +0.3845276f, +0.8823679f, +0.1624766f, -0.8651584f, +0.2481551f,
    -0.4965194f, +0.7510185f, +0.0120952f, -0.9007600f, -0.9001184f, +0.0434373f, +0.8999661f, -0.0474410f,
    +0.5047419f, -0.7473397f, +0.9002064f, +0.0698053f, -0.8921132f, +0.1528542f, +0.2853571f, +0.8594576f,
    +0.1924196f, +0.8850988f, -0.2605157f, -0.8684930f, -0.8080477f, +0.4119121f, -0.0159142f, +0.9071479f,
    -0.1310043f, +0.9005243f, +0.6675038f, -0.6187724f, +0.0946450f, +0.9060204f, +0.5912086f, -0.6950297f,
    -0.8736846f, -0.2635498f, +0.8785041f, -0.2497408f, -0.5581169f, +0.7232275f, +0.5216928f, +0.7546194f,
    +0.3098496f, -0.8646511f, -0.9037760f, -0.1697373f, +0.1959183f, -0.8997416f, -0.3115442f, +0.8671960f,
    +0.8840920f, +0.2602737f, +0.7479016f, +0.5433701f, +0.7444031f, -0.5482733f, -0.4438028f, +0.8148065f,
    -0.4632129f, -0.8046650f, -0.7583340f, -0.5358876f, -0.6832047f, +0.6295144f, +0.3917605f, -0.8442057f,
    -0.7395392f, +0.5655778f, -0.8380300f, -0.4211845f, +0.6015402f, +0.7220433f, -0.3497197f, -0.8794083f,
    -0.8843726f, +0.3376022f, -0.5388274f, -0.7805851f, +0.8755536f, -0.3648177f, +0.4648516f, +0.8291232f,
    -0.6111445f, -0.7285293f, -0.6419417f, +0.7069208f, +0.8361576f, -0.4620076f, +0.8872122f, +0.3579843f,
    +0.3792312f, +0.8793520f, -0.8959889f, -0.3582828f, -0.7547551f, -0.6181142f, +0.8523171f, +0.4786253f,
    +0.7270306f, +0.6553103f, +0.4906181f, -0.8496716f, -0.7103988f, -0.6812648f, -0.8397788f, +0.5156603f,
    -0.8872350f, +0.4322805f, -0.4104562f, +0.8977518f, -0.9882981f, -0.0344881f, +0.9893247f, +0.0094388f,
    -0.8521242f, -0.5046932f, -0.0450430f, -0.9896201f, +0.6770949f, -0.7236059f, +0.1192893f, -0.9842341f,
    +0.0453498f, +0.9917771f, +0.0385310f, -0.9922286f, -0.0331763f, +0.9931705f, -0.9933886f, +0.0471682f,
    +0.9872521f, -0.1227469f, +0.9936745f, -0.0573017f, -0.1320985f, -0.9867291f, -0.1113858f, +0.9894928f,
    -0.9890981f, -0.1153084f, +0.9933295f, +0.0776391f, -0.9894915f, +0.1280985f, +0.9876567f, +0.1525908f,
    +0.1343039f, +0.9911650f, -0.4682204f, -0.8844823f, -0.1939309f, +0.9839375f, -0.5277652f, +0.8533627f,
    -0.6211832f, +0.7901094f, +0.9865716f, -0.1924095f, -0.9886428f, -0.1943209f, -0.2220367f, -0.9833463f,
    +0.7681906f, -0.6551118f, +0.9845671f, +0.2252553f, -0.9872243f, +0.2171700f, +0.2220562f, +0.9873286f,
    +0.2037255f, -0.9923806f, +0.6031507f, -0.8154258f, -0.7745835f, +0.6553162f, -0.2713896f, +0.9830117f,
    +0.9870027f, -0.2651592f, +0.2914540f, -0.9825205f, -0.9863437f, -0.2791628f, +0.3065091f, +0.9829413f,
    +0.9883939f, +0.2970866f, +0.8451408f, +0.5937594f, -0.7200384f, +0.7405153f, +0.8563747f, -0.5781524f,
    -0.9871804f, +0.3112979f, -0.3168432f, -0.9872803f, +0.5971950f, +0.8542361f, +0.6967390f, +0.7756870f,
    +0.9865466f, -0.3447451f, -0.3504501f, +0.9850587f, +0.3822059f, -0.9750156f, -0.6940877f, -0.7870535f,
    -0.5997635f, -0.8657183f, -0.8638299f, -0.6057245f, -0.9903726f, -0.3669571f, +0.9880455f, +0.3757233f,
    -0.8667820f, +0.6174700f, +0.4042474f, +0.9883737f, -0.9884764f, +0.4101280f, +0.9802066f, -0.4337381f,
    -0.4236525f, -0.9862565f, -0.9771566f, -0.4565516f, +0.9788690f, +0.4634466f, +0.5044816f, +0.9586782f,
    -0.4422544f, +0.9891341f, +0.4779977f, -0.9777919f, -0.6398648f, +0.8825701f, -0.9756752f, +0.5092181f,
    -0.8332741f, -0.7229068f, +0.9722787f, -0.5245732f, +0.8387917f, +0.7197183f, +0.7339427f, -0.8313786f,
    -0.5292935f, +0.9750075f, +0.8357673f, -0.7470663f, +0.9716807f, +0.5631371f, -0.9788474f, -0.5566355f,
    +0.5897797f, -0.9647853f, -0.5600806f, -0.9826826f, -0.7591529f, +0.8424618f, -0.8504958f, +0.7592480f,
    +0.9653354f, -0.6290697f, -0.7825680f, -0.8466023f, -0.9797512f, +0.6085486f, +0.6145102f, +0.9791595f,
    +0.7907359f, +0.8493837f, -0.6295238f, +0.9895397f, -0.9781751f, -0.6833395f, +0.9772800f, +0.6889105f,
    -0.7048766f, -0.9760795f, +0.7371566f, -0.9598892f, -0.9732065f, +0.7252908f, -0.7440614f, +0.9621292f,
    +0.9672187f, -0.7462835f, +0.7579523f, +0.9779361f, -0.9728731f, -0.8307916f, -0.9448507f, +0.8785052f,
    -0.8629266f, +0.9686990f, +0.9153012f, -0.9206548f, +0.9663150f, +0.8732710f, -0.8756098f, -0.9694053f,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint8_t, kmask_iq2xs, 8)
    1, 2, 4, 8, 16, 32, 64, 128
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint8_t, ksigns_iq2xs, 128)
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint64_t, ksigns64, 128)
    0x0000000000000000, 0xff000000000000ff, 0xff0000000000ff00, 0x000000000000ffff,
    0xff00000000ff0000, 0x0000000000ff00ff, 0x0000000000ffff00, 0xff00000000ffffff,
    0xff000000ff000000, 0x00000000ff0000ff, 0x00000000ff00ff00, 0xff000000ff00ffff,
    0x00000000ffff0000, 0xff000000ffff00ff, 0xff000000ffffff00, 0x00000000ffffffff,
    0xff0000ff00000000, 0x000000ff000000ff, 0x000000ff0000ff00, 0xff0000ff0000ffff,
    0x000000ff00ff0000, 0xff0000ff00ff00ff, 0xff0000ff00ffff00, 0x000000ff00ffffff,
    0x000000ffff000000, 0xff0000ffff0000ff, 0xff0000ffff00ff00, 0x000000ffff00ffff,
    0xff0000ffffff0000, 0x000000ffffff00ff, 0x000000ffffffff00, 0xff0000ffffffffff,
    0xff00ff0000000000, 0x0000ff00000000ff, 0x0000ff000000ff00, 0xff00ff000000ffff,
    0x0000ff0000ff0000, 0xff00ff0000ff00ff, 0xff00ff0000ffff00, 0x0000ff0000ffffff,
    0x0000ff00ff000000, 0xff00ff00ff0000ff, 0xff00ff00ff00ff00, 0x0000ff00ff00ffff,
    0xff00ff00ffff0000, 0x0000ff00ffff00ff, 0x0000ff00ffffff00, 0xff00ff00ffffffff,
    0x0000ffff00000000, 0xff00ffff000000ff, 0xff00ffff0000ff00, 0x0000ffff0000ffff,
    0xff00ffff00ff0000, 0x0000ffff00ff00ff, 0x0000ffff00ffff00, 0xff00ffff00ffffff,
    0xff00ffffff000000, 0x0000ffffff0000ff, 0x0000ffffff00ff00, 0xff00ffffff00ffff,
    0x0000ffffffff0000, 0xff00ffffffff00ff, 0xff00ffffffffff00, 0x0000ffffffffffff,
    0xffff000000000000, 0x00ff0000000000ff, 0x00ff00000000ff00, 0xffff00000000ffff,
    0x00ff000000ff0000, 0xffff000000ff00ff, 0xffff000000ffff00, 0x00ff000000ffffff,
    0x00ff0000ff000000, 0xffff0000ff0000ff, 0xffff0000ff00ff00, 0x00ff0000ff00ffff,
    0xffff0000ffff0000, 0x00ff0000ffff00ff, 0x00ff0000ffffff00, 0xffff0000ffffffff,
    0x00ff00ff00000000, 0xffff00ff000000ff, 0xffff00ff0000ff00, 0x00ff00ff0000ffff,
    0xffff00ff00ff0000, 0x00ff00ff00ff00ff, 0x00ff00ff00ffff00, 0xffff00ff00ffffff,
    0xffff00ffff000000, 0x00ff00ffff0000ff, 0x00ff00ffff00ff00, 0xffff00ffff00ffff,
    0x00ff00ffffff0000, 0xffff00ffffff00ff, 0xffff00ffffffff00, 0x00ff00ffffffffff,
    0x00ffff0000000000, 0xffffff00000000ff, 0xffffff000000ff00, 0x00ffff000000ffff,
    0xffffff0000ff0000, 0x00ffff0000ff00ff, 0x00ffff0000ffff00, 0xffffff0000ffffff,
    0xffffff00ff000000, 0x00ffff00ff0000ff, 0x00ffff00ff00ff00, 0xffffff00ff00ffff,
    0x00ffff00ffff0000, 0xffffff00ffff00ff, 0xffffff00ffffff00, 0x00ffff00ffffffff,
    0xffffffff00000000, 0x00ffffff000000ff, 0x00ffffff0000ff00, 0xffffffff0000ffff,
    0x00ffffff00ff0000, 0xffffffff00ff00ff, 0xffffffff00ffff00, 0x00ffffff00ffffff,
    0x00ffffffff000000, 0xffffffffff0000ff, 0xffffffffff00ff00, 0x00ffffffff00ffff,
    0xffffffffffff0000, 0x00ffffffffff00ff, 0x00ffffffffffff00, 0xffffffffffffffff,
GGML_TABLE_END()


GGML_TABLE_BEGIN(uint64_t, iq2xxs_grid, 256)
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint64_t, iq2xs_grid, 512)
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint64_t, iq2s_grid, 1024)
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint32_t, iq3xxs_grid, 256)
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
GGML_TABLE_END()

// TODO: fix name to kvalues_iq4_nl
GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
GGML_TABLE_END()

// e2m1 values (doubled), shared by MXFP4 and NVFP4
// ref: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
GGML_TABLE_BEGIN(int8_t, kvalues_fp4, 16)
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
GGML_TABLE_END()
#define kvalues_mxfp4 kvalues_fp4

#define NGRID_IQ1S 2048
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f
#if defined(GGML_COMMON_IMPL_C) || defined(GGML_COMMON_IMPL_CPP)
GGML_TABLE_BEGIN(uint64_t, iq1s_grid, NGRID_IQ1S)
    0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff,
    0xffffffffffff0101, 0xffffffffff00ff00, 0xffffffffff000000, 0xffffffffff01ffff,
    0xffffffffff01ff01, 0xffffffffff0101ff, 0xffffffffff010101, 0xffffffff00ff0000,
    0xffffffff0000ff00, 0xffffffff000000ff, 0xffffffff00000001, 0xffffffff00010000,
    0xffffffff01ffffff, 0xffffffff01ffff01, 0xffffffff01ff01ff, 0xffffffff01ff0101,
    0xffffffff01000000, 0xffffffff0101ffff, 0xffffffff0101ff01, 0xffffffff010101ff,
    0xffffffff01010101, 0xffffff00ffff00ff, 0xffffff00ffff0000, 0xffffff00ff00ff00,
    0xffffff00ff0000ff, 0xffffff00ff000001, 0xffffff00ff000100, 0xffffff00ff000101,
    0xffffff00ff010000, 0xffffff0000ffff00, 0xffffff0000ff0001, 0xffffff0000ff0100,
    0xffffff000000ff01, 0xffffff0000000000, 0xffffff0000000101, 0xffffff000001ff00,
    0xffffff00000100ff, 0xffffff0000010001, 0xffffff00000101ff, 0xffffff0001ff0000,
    0xffffff000100ff00, 0xffffff00010000ff, 0xffffff0001000001, 0xffffff0001010000,
    0xffffff01ffffffff, 0xffffff01ffffff01, 0xffffff01ffff01ff, 0xffffff01ffff0101,
    0xffffff01ff000000, 0xffffff01ff01ffff, 0xffffff01ff01ff01, 0xffffff01ff0101ff,
    0xffffff01ff010101, 0xffffff0100ff0000, 0xffffff010000ff00, 0xffffff0100000100,
    0xffffff01000100ff, 0xffffff0100010100, 0xffffff0101ffffff, 0xffffff0101ffff01,
    0xffffff0101ff01ff, 0xffffff0101ff0101, 0xffffff010100ff00, 0xffffff0101000000,
    0xffffff0101000100, 0xffffff010101ffff, 0xffffff010101ff01, 0xffffff01010101ff,
    0xffffff0101010101, 0xffff00ffff00ff00, 0xffff00ffff0000ff, 0xffff00ffff000001,
    0xffff00ffff010000, 0xffff00ff00ffff00, 0xffff00ff00ff0100, 0xffff00ff00000000,
    0xffff00ff00000101, 0xffff00ff000100ff, 0xffff00ff00010000, 0xffff00ff0100ff00,
    0xffff00ff01000100, 0xffff00ff01010000, 0xffff0000ffffff00, 0xffff0000ffff00ff,
    0xffff0000ffff0000, 0xffff0000ffff0001, 0xffff0000ff000000, 0xffff0000ff0001ff,
    0xffff0000ff000101, 0xffff0000ff010100, 0xffff000000ffffff, 0xffff000000ff0000,
    0xffff000000ff0101, 0xffff00000000ffff, 0xffff00000000ff00, 0xffff0000000000ff,
    0xffff000000000000, 0xffff000000000001, 0xffff000000000100, 0xffff00000001ffff,
    0xffff00000001ff01, 0xffff000000010000, 0xffff0000000101ff, 0xffff000000010101,
    0xffff000001ffff00, 0xffff00000100ff00, 0xffff000001000000, 0xffff0000010001ff,
    0xffff000001000101, 0xffff00000101ff00, 0xffff0000010100ff, 0xffff000001010000,
    0xffff000001010001, 0xffff000001010100, 0xffff0001ff0000ff, 0xffff0001ff000100,
    0xffff000100ffff00, 0xffff000100ff00ff, 0xffff00010000ffff, 0xffff00010000ff01,
    0xffff000100000000, 0xffff0001000001ff, 0xffff00010001ffff, 0xffff00010001ff00,
    0xffff000100010001, 0xffff000100010100, 0xffff000101ff0000, 0xffff00010100ff00,
    0xffff0001010000ff, 0xffff000101000100, 0xffff01ffffffffff, 0xffff01ffffffff01,
    0xffff01ffffff01ff, 0xffff01ffffff0101, 0xffff01ffff000000, 0xffff01ffff01ffff,
    0xffff01ffff01ff01, 0xffff01ffff0101ff, 0xffff01ffff010101, 0xffff01ff00ff0000,
    0xffff01ff0000ff00, 0xffff01ff00000001, 0xffff01ff00010000, 0xffff01ff01ffffff,
    0xffff01ff01ffff01, 0xffff01ff01ff01ff, 0xffff01ff01ff0101, 0xffff01ff01000000,
    0xffff01ff0101ffff, 0xffff01ff0101ff01, 0xffff01ff010101ff, 0xffff01ff01010101,
    0xffff0100ffff0000, 0xffff0100ff00ff00, 0xffff0100ff0000ff, 0xffff0100ff000100,
    0xffff0100ff0100ff, 0xffff0100ff010000, 0xffff010000ffff00, 0xffff01000000ffff,
    0xffff01000000ff00, 0xffff010000000000, 0xffff01000001ff00, 0xffff0100000100ff,
    0xffff010000010100, 0xffff01000100ff00, 0xffff0100010000ff, 0xffff010001000001,
    0xffff010001000100, 0xffff010001010000, 0xffff0101ffffffff, 0xffff0101ffffff01,
    0xffff0101ffff01ff, 0xffff0101ffff0101, 0xffff0101ff000000, 0xffff0101ff01ffff,
    0xffff0101ff01ff01, 0xffff0101ff0101ff, 0xffff0101ff010101, 0xffff010100ff0000,
    0xffff01010000ff00, 0xffff010100000100, 0xffff01010001ff00, 0xffff010100010000,
    0xffff010101ffffff, 0xffff010101ffff01, 0xffff010101ff0000, 0xffff010101ff01ff,
    0xffff010101ff0101, 0xffff010101000000, 0xffff01010101ffff, 0xffff01010101ff01,
    0xffff0101010101ff, 0xffff010101010101, 0xff00ffffff00ffff, 0xff00ffffff00ff00,
    0xff00ffffff0000ff, 0xff00ffffff000100, 0xff00ffffff0100ff, 0xff00ffffff010000,
    0xff00ffff00ffff00, 0xff00ffff00ff00ff, 0xff00ffff0000ffff, 0xff00ffff00000000,
    0xff00ffff000001ff, 0xff00ffff0001ff00, 0xff00ffff000100ff, 0xff00ffff00010000,
    0xff00ffff00010100, 0xff00ffff0100ff00, 0xff00ffff010000ff, 0xff00ffff01000001,
    0xff00ffff0101ff00, 0xff00ffff01010000, 0xff00ff00ffffff00, 0xff00ff00ffff00ff,
    0xff00ff00ffff0001, 0xff00ff00ffff0100, 0xff00ff00ff00ffff, 0xff00ff00ff00ff01,
    0xff00ff00ff000000, 0xff00ff00ff0001ff, 0xff00ff00ff01ff00, 0xff00ff00ff0100ff,
    0xff00ff00ff010100, 0xff00ff0000ff0000, 0xff00ff0000ff0101, 0xff00ff000000ffff,
    0xff00ff000000ff00, 0xff00ff000000ff01, 0xff00ff00000000ff, 0xff00ff0000000000,
    0xff00ff0000000001, 0xff00ff0000000100, 0xff00ff000001ffff, 0xff00ff0000010000,
    0xff00ff0001ff00ff, 0xff00ff000100ff01, 0xff00ff0001000000, 0xff00ff000101ff00,
    0xff00ff00010100ff, 0xff00ff01ff00ff00, 0xff00ff01ff0000ff, 0xff00ff01ff000001,
    0xff00ff01ff010000, 0xff00ff0100ffffff, 0xff00ff0100ff0001, 0xff00ff0100ff0100,
    0xff00ff010000ff01, 0xff00ff0100000000, 0xff00ff01000001ff, 0xff00ff0100000101,
    0xff00ff01000100ff, 0xff00ff0100010001, 0xff00ff0101ff0000, 0xff00ff010100ff00,
    0xff00ff01010000ff, 0xff00ff0101000001, 0xff00ff0101010000, 0xff0000ffffffff00,
    0xff0000ffffff0001, 0xff0000ffffff0100, 0xff0000ffff0000ff, 0xff0000ffff000000,
    0xff0000ffff0001ff, 0xff0000ffff000100, 0xff0000ffff01ff00, 0xff0000ffff010001,
    0xff0000ff00ffff00, 0xff0000ff00ff0000, 0xff0000ff00ff0001, 0xff0000ff00ff01ff,
    0xff0000ff00ff0101, 0xff0000ff0000ff00, 0xff0000ff000000ff, 0xff0000ff00000000,
    0xff0000ff00000001, 0xff0000ff00000100, 0xff0000ff0001ff01, 0xff0000ff00010000,
    0xff0000ff000101ff, 0xff0000ff01ff00ff, 0xff0000ff01ff0100, 0xff0000ff0100ffff,
    0xff0000ff010000ff, 0xff0000ff01000000, 0xff0000ff010001ff, 0xff0000ff01000100,
    0xff0000ff01000101, 0xff0000ff0101ff00, 0xff0000ff010100ff, 0xff0000ff01010000,
    0xff0000ff01010100, 0xff000000ffffff01, 0xff000000ffff0000, 0xff000000ffff0101,
    0xff000000ff00ff00, 0xff000000ff0000ff, 0xff000000ff000000, 0xff000000ff000001,
    0xff000000ff000100, 0xff000000ff01ffff, 0xff000000ff01ff01, 0xff000000ff010000,
    0xff000000ff0101ff, 0xff000000ff010101, 0xff00000000ffff00, 0xff00000000ff00ff,
    0xff00000000ff0000, 0xff00000000ff0001, 0xff0000000000ff00, 0xff0000000000ff01,
    0xff000000000000ff, 0xff00000000000000, 0xff00000000000001, 0xff00000000000100,
    0xff00000000000101, 0xff0000000001ff00, 0xff000000000100ff, 0xff00000000010000,
    0xff00000000010001, 0xff00000000010100, 0xff00000001ffffff, 0xff00000001ffff01,
    0xff00000001ff00ff, 0xff00000001ff0000, 0xff00000001ff01ff, 0xff00000001ff0101,
    0xff0000000100ffff, 0xff0000000100ff00, 0xff000000010000ff, 0xff00000001000000,
    0xff00000001000001, 0xff00000001000100, 0xff00000001000101, 0xff0000000101ffff,
    0xff0000000101ff01, 0xff00000001010000, 0xff000001ffffff00, 0xff000001ffff00ff,
    0xff000001ffff0000, 0xff000001ffff0001, 0xff000001ff000000, 0xff000001ff000001,
    0xff000001ff0001ff, 0xff000001ff000101, 0xff000001ff01ff00, 0xff000001ff010001,
    0xff00000100ffffff, 0xff00000100ffff01, 0xff00000100ff00ff, 0xff00000100ff0000,
    0xff00000100ff01ff, 0xff00000100ff0101, 0xff0000010000ff00, 0xff00000100000000,
    0xff00000100000001, 0xff000001000001ff, 0xff00000100000100, 0xff0000010001ff00,
    0xff000001000100ff, 0xff00000100010000, 0xff000001000101ff, 0xff00000100010100,
    0xff00000100010101, 0xff00000101ff0001, 0xff00000101ff0101, 0xff0000010100ff01,
    0xff00000101000000, 0xff000001010100ff, 0xff00000101010100, 0xff0001ffff00ff00,
    0xff0001ffff000001, 0xff0001ffff010000, 0xff0001ff00ffff00, 0xff0001ff00ff00ff,
    0xff0001ff00ff0001, 0xff0001ff00ff0100, 0xff0001ff0000ffff, 0xff0001ff00000000,
    0xff0001ff000001ff, 0xff0001ff00000101, 0xff0001ff0001ffff, 0xff0001ff0001ff00,
    0xff0001ff000100ff, 0xff0001ff00010001, 0xff0001ff00010100, 0xff0001ff01ff0000,
    0xff0001ff0100ff00, 0xff0001ff010000ff, 0xff0001ff01010000, 0xff000100ff00ffff,
    0xff000100ff00ff01, 0xff000100ff000000, 0xff000100ff000101, 0xff000100ff01ff00,
    0xff000100ff010000, 0xff00010000ffff01, 0xff00010000ff00ff, 0xff00010000ff0000,
    0xff00010000ff01ff, 0xff0001000000ff00, 0xff000100000000ff, 0xff00010000000000,
    0xff00010000000001, 0xff00010000000100, 0xff00010000000101, 0xff0001000001ffff,
    0xff00010000010000, 0xff00010000010101, 0xff00010001ff0100, 0xff0001000100ff00,
    0xff0001000100ff01, 0xff00010001000000, 0xff000100010001ff, 0xff0001000101ff00,
    0xff00010001010001, 0xff00010001010100, 0xff000101ffff0100, 0xff000101ff000001,
    0xff000101ff0100ff, 0xff000101ff010001, 0xff00010100ff00ff, 0xff00010100ff0001,
    0xff00010100ff0100, 0xff0001010000ffff, 0xff0001010000ff01, 0xff00010100000000,
    0xff000101000001ff, 0xff0001010001ff00, 0xff00010100010001, 0xff00010100010100,
    0xff00010101ff0000, 0xff0001010100ff00, 0xff00010101000001, 0xff00010101000101,
    0xff01ffffffffffff, 0xff01ffffffffff01, 0xff01ffffffff01ff, 0xff01ffffffff0101,
    0xff01ffffff000000, 0xff01ffffff01ffff, 0xff01ffffff01ff01, 0xff01ffffff010000,
    0xff01ffffff0101ff, 0xff01ffffff010101, 0xff01ffff00ff0000, 0xff01ffff0000ff00,
    0xff01ffff00000100, 0xff01ffff0001ff00, 0xff01ffff00010000, 0xff01ffff01ffffff,
    0xff01ffff01ffff01, 0xff01ffff01ff01ff, 0xff01ffff01ff0101, 0xff01ffff01000000,
    0xff01ffff0101ffff, 0xff01ffff0101ff01, 0xff01ffff01010000, 0xff01ffff010101ff,
    0xff01ffff01010101, 0xff01ff00ffff0000, 0xff01ff00ff00ff00, 0xff01ff00ff0000ff,
    0xff01ff00ff000100, 0xff01ff00ff010000, 0xff01ff0000ffff01, 0xff01ff0000ff00ff,
    0xff01ff0000ff0100, 0xff01ff0000000000, 0xff01ff00000001ff, 0xff01ff0000000101,
    0xff01ff000001ff00, 0xff01ff00000100ff, 0xff01ff0000010000, 0xff01ff0000010001,
    0xff01ff0001ff0000, 0xff01ff000100ffff, 0xff01ff0001000001, 0xff01ff0001000100,
    0xff01ff0001010000, 0xff01ff01ffffff00, 0xff01ff01ffff01ff, 0xff01ff01ffff0101,
    0xff01ff01ff00ff00, 0xff01ff01ff000000, 0xff01ff01ff01ffff, 0xff01ff01ff01ff01,
    0xff01ff01ff0101ff, 0xff01ff01ff010101, 0xff01ff0100ff0000, 0xff01ff010000ff00,
    0xff01ff0100000001, 0xff01ff0100000100, 0xff01ff0100010000, 0xff01ff0101ffff00,
    0xff01ff0101ff01ff, 0xff01ff0101ff0101, 0xff01ff010100ff00, 0xff01ff0101000000,
    0xff01ff010101ffff, 0xff01ff010101ff01, 0xff01ff01010101ff, 0xff01ff0101010101,
    0xff0100ffffff0000, 0xff0100ffff0000ff, 0xff0100ffff000001, 0xff0100ffff000100,
    0xff0100ffff010000, 0xff0100ff00ff00ff, 0xff0100ff00ff0000, 0xff0100ff00ff0001,
    0xff0100ff00ff0100, 0xff0100ff0000ff01, 0xff0100ff00000000, 0xff0100ff000001ff,
    0xff0100ff00000101, 0xff0100ff00010001, 0xff0100ff01ff0000, 0xff0100ff0100ff00,
    0xff0100ff010000ff, 0xff0100ff01000100, 0xff0100ff0101ff00, 0xff0100ff01010000,
    0xff010000ffff0100, 0xff010000ff000000, 0xff010000ff01ff00, 0xff010000ff010100,
    0xff01000000ffffff, 0xff01000000ff0000, 0xff01000000ff01ff, 0xff0100000000ff00,
    0xff010000000000ff, 0xff01000000000000, 0xff01000000000100, 0xff0100000001ff01,
    0xff01000000010000, 0xff010000000101ff, 0xff01000001ff0100, 0xff0100000100ffff,
    0xff010000010000ff, 0xff01000001000000, 0xff010000010001ff, 0xff01000001000101,
    0xff0100000101ff00, 0xff010000010100ff, 0xff01000001010001, 0xff01000001010100,
    0xff010001ffff0000, 0xff010001ff00ffff, 0xff010001ff00ff01, 0xff010001ff000100,
    0xff010001ff010000, 0xff01000100ffff00, 0xff01000100ff0100, 0xff01000100000000,
    0xff0100010001ffff, 0xff0100010001ff00, 0xff01000100010100, 0xff01000101ff00ff,
    0xff01000101ff0001, 0xff0100010100ffff, 0xff01000101000101, 0xff0101ffffffffff,
    0xff0101ffffffff01, 0xff0101ffffff01ff, 0xff0101ffffff0101, 0xff0101ffff000000,
    0xff0101ffff01ffff, 0xff0101ffff01ff01, 0xff0101ffff0101ff, 0xff0101ffff010101,
    0xff0101ff00ff0000, 0xff0101ff0000ff00, 0xff0101ff000000ff, 0xff0101ff00010000,
    0xff0101ff01ffffff, 0xff0101ff01ffff01, 0xff0101ff01ff01ff, 0xff0101ff01ff0101,
    0xff0101ff0101ffff, 0xff0101ff0101ff01, 0xff0101ff010101ff, 0xff0101ff01010101,
    0xff010100ffff0100, 0xff010100ff00ff00, 0xff010100ff0000ff, 0xff010100ff000100,
    0xff010100ff010000, 0xff01010000ff0001, 0xff01010000ff0100, 0xff0101000000ff01,
    0xff01010000000000, 0xff0101000001ff00, 0xff010100000100ff, 0xff01010000010001,
    0xff01010000010100, 0xff01010001ff0000, 0xff0101000100ffff, 0xff01010001000001,
    0xff01010001000100, 0xff010100010100ff, 0xff01010001010000, 0xff010101ffffffff,
    0xff010101ffffff01, 0xff010101ffff01ff, 0xff010101ffff0101, 0xff010101ff01ffff,
    0xff010101ff01ff01, 0xff010101ff0101ff, 0xff010101ff010101, 0xff01010100ff0000,
    0xff0101010000ff00, 0xff01010100000001, 0xff01010100000100, 0xff01010100010000,
    0xff01010101ffffff, 0xff01010101ffff01, 0xff01010101ff01ff, 0xff01010101ff0101,
    0xff01010101000000, 0xff0101010101ffff, 0xff0101010101ff01, 0xff010101010101ff,
    0xff01010101010101, 0x00ffffffffff0000, 0x00ffffffff00ff00, 0x00ffffffff000001,
    0x00ffffffff010000, 0x00ffffff00ff0100, 0x00ffffff0000ff01, 0x00ffffff00000000,
    0x00ffffff000001ff, 0x00ffffff00000101, 0x00ffffff0001ff00, 0x00ffffff000100ff,
    0x00ffffff00010001, 0x00ffffff010000ff, 0x00ffffff01000100, 0x00ffffff0101ff00,
    0x00ffffff01010001, 0x00ffff00ffffffff, 0x00ffff00ffffff00, 0x00ffff00ffff00ff,
    0x00ffff00ffff0001, 0x00ffff00ffff0100, 0x00ffff00ff00ff01, 0x00ffff00ff000000,
    0x00ffff00ff000001, 0x00ffff00ff0001ff, 0x00ffff00ff000101, 0x00ffff00ff01ff00,
    0x00ffff00ff010001, 0x00ffff00ff010100, 0x00ffff0000ff0000, 0x00ffff0000ff01ff,
    0x00ffff0000ff0101, 0x00ffff000000ff00, 0x00ffff00000000ff, 0x00ffff0000000000,
    0x00ffff0000000001, 0x00ffff0000000100, 0x00ffff0000000101, 0x00ffff0000010000,
    0x00ffff00000101ff, 0x00ffff0000010101, 0x00ffff0001ffff00, 0x00ffff0001ff00ff,
    0x00ffff0001ff0001, 0x00ffff000100ffff, 0x00ffff000100ff01, 0x00ffff0001000000,
    0x00ffff000101ffff, 0x00ffff000101ff00, 0x00ffff000101ff01, 0x00ffff01ffff0000,
    0x00ffff01ff00ff00, 0x00ffff01ff0000ff, 0x00ffff01ff000001, 0x00ffff01ff010000,
    0x00ffff0100ffff00, 0x00ffff010000ff01, 0x00ffff0100000000, 0x00ffff0100000101,
    0x00ffff01000100ff, 0x00ffff0100010100, 0x00ffff0101ff0100, 0x00ffff01010000ff,
    0x00ffff0101010000, 0x00ff00ffffffff00, 0x00ff00ffff000000, 0x00ff00ffff000100,
    0x00ff00ffff010100, 0x00ff00ff00ff0000, 0x00ff00ff00ff01ff, 0x00ff00ff00ff0101,
    0x00ff00ff0000ff00, 0x00ff00ff000000ff, 0x00ff00ff00000000, 0x00ff00ff00000001,
    0x00ff00ff0001ff00, 0x00ff00ff0001ff01, 0x00ff00ff00010000, 0x00ff00ff000101ff,
    0x00ff00ff00010101, 0x00ff00ff01ffff00, 0x00ff00ff01ff0001, 0x00ff00ff01ff0100,
    0x00ff00ff0100ffff, 0x00ff00ff0100ff01, 0x00ff00ff01000000, 0x00ff00ff0101ffff,
    0x00ff00ff0101ff00, 0x00ff00ff01010100, 0x00ff0000ffffff00, 0x00ff0000ffffff01,
    0x00ff0000ffff0000, 0x00ff0000ffff0101, 0x00ff0000ff00ff00, 0x00ff0000ff0000ff,
    0x00ff0000ff000000, 0x00ff0000ff000001, 0x00ff0000ff000100, 0x00ff0000ff01ffff,
    0x00ff0000ff010000, 0x00ff0000ff010101, 0x00ff000000ffff00, 0x00ff000000ff00ff,
    0x00ff000000ff0000, 0x00ff000000ff0001, 0x00ff000000ff0100, 0x00ff00000000ffff,
    0x00ff00000000ff00, 0x00ff0000000000ff, 0x00ff000000000000, 0x00ff000000000001,
    0x00ff0000000001ff, 0x00ff000000000100, 0x00ff00000001ff00, 0x00ff0000000100ff,
    0x00ff000000010000, 0x00ff000000010001, 0x00ff000000010100, 0x00ff000001ffff01,
    0x00ff000001ff00ff, 0x00ff000001ff0000, 0x00ff000001ff01ff, 0x00ff00000100ff00,
    0x00ff0000010000ff, 0x00ff000001000000, 0x00ff000001000001, 0x00ff000001000100,
    0x00ff000001000101, 0x00ff000001010000, 0x00ff0000010101ff, 0x00ff000001010101,
    0x00ff0001ffffff00, 0x00ff0001ffff0000, 0x00ff0001ffff0100, 0x00ff0001ff0000ff,
    0x00ff0001ff000000, 0x00ff0001ff0001ff, 0x00ff0001ff000101, 0x00ff0001ff01ff00,
    0x00ff0001ff0100ff, 0x00ff0001ff010100, 0x00ff000100ffffff, 0x00ff000100ffff01,
    0x00ff000100ff0000, 0x00ff000100ff01ff, 0x00ff00010000ffff, 0x00ff00010000ff00,
    0x00ff00010000ff01, 0x00ff000100000000, 0x00ff000100000001, 0x00ff000100000100,
    0x00ff00010001ff01, 0x00ff000100010000, 0x00ff0001000101ff, 0x00ff000101ffff00,
    0x00ff000101ff0000, 0x00ff000101ff0101, 0x00ff0001010000ff, 0x00ff000101000000,
    0x00ff00010101ff00, 0x00ff0001010100ff, 0x00ff000101010001, 0x00ff01ffffff0000,
    0x00ff01ffff00ff00, 0x00ff01ffff000000, 0x00ff01ffff000101, 0x00ff01ffff010000,
    0x00ff01ff00ffff01, 0x00ff01ff00ff0100, 0x00ff01ff0000ffff, 0x00ff01ff00000000,
    0x00ff01ff000001ff, 0x00ff01ff0001ff00, 0x00ff01ff000100ff, 0x00ff01ff00010001,
    0x00ff01ff00010100, 0x00ff01ff01ff0000, 0x00ff01ff0100ff00, 0x00ff01ff010000ff,
    0x00ff01ff01000001, 0x00ff01ff01000100, 0x00ff01ff01010000, 0x00ff0100ffffff00,
    0x00ff0100ffff0000, 0x00ff0100ffff0001, 0x00ff0100ffff0101, 0x00ff0100ff00ffff,
    0x00ff0100ff0000ff, 0x00ff0100ff000000, 0x00ff0100ff0001ff, 0x00ff0100ff01ff00,
    0x00ff0100ff0100ff, 0x00ff0100ff010001, 0x00ff010000ffffff, 0x00ff010000ff0000,
    0x00ff010000ff0101, 0x00ff01000000ff00, 0x00ff01000000ff01, 0x00ff0100000000ff,
    0x00ff010000000000, 0x00ff010000000001, 0x00ff010000000100, 0x00ff01000001ffff,
    0x00ff01000001ff01, 0x00ff010000010000, 0x00ff010000010001, 0x00ff010000010101,
    0x00ff010001ff0001, 0x00ff010001ff0100, 0x00ff01000100ff01, 0x00ff010001000000,
    0x00ff010001000001, 0x00ff0100010001ff, 0x00ff01000101ff00, 0x00ff0100010100ff,
    0x00ff010001010001, 0x00ff010001010100, 0x00ff0101ff000001, 0x00ff010100ff00ff,
    0x00ff010100ff0001, 0x00ff010100ff0100, 0x00ff010100000000, 0x00ff0101000001ff,
    0x00ff010100000101, 0x00ff0101000100ff, 0x00ff010100010100, 0x00ff0101010000ff,
    0x00ff010101010000, 0x0000ffffffffff00, 0x0000ffffffff00ff, 0x0000ffffffff0000,
    0x0000ffffffff0001, 0x0000ffffffff0100, 0x0000ffffff00ff01, 0x0000ffffff000000,
    0x0000ffffff000101, 0x0000ffffff01ff00, 0x0000ffffff0100ff, 0x0000ffffff010100,
    0x0000ffff00ffffff, 0x0000ffff00ff0000, 0x0000ffff00ff01ff, 0x0000ffff0000ff00,
    0x0000ffff000000ff, 0x0000ffff00000000, 0x0000ffff00000001, 0x0000ffff00000100,
    0x0000ffff00010000, 0x0000ffff000101ff, 0x0000ffff01ff0001, 0x0000ffff01ff0100,
    0x0000ffff01000000, 0x0000ffff010001ff, 0x0000ffff0101ffff, 0x0000ffff0101ff00,
    0x0000ffff01010001, 0x0000ffff01010100, 0x0000ff00ffff0000, 0x0000ff00ffff01ff,
    0x0000ff00ffff0100, 0x0000ff00ffff0101, 0x0000ff00ff00ff00, 0x0000ff00ff0000ff,
    0x0000ff00ff000000, 0x0000ff00ff000001, 0x0000ff00ff0001ff, 0x0000ff00ff000100,
    0x0000ff00ff01ffff, 0x0000ff00ff010000, 0x0000ff00ff010001, 0x0000ff00ff0101ff,
    0x0000ff00ff010101, 0x0000ff0000ffff00, 0x0000ff0000ff00ff, 0x0000ff0000ff0000,
    0x0000ff0000ff0001, 0x0000ff0000ff0100, 0x0000ff000000ffff, 0x0000ff000000ff00,
    0x0000ff000000ff01, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001,
    0x0000ff00000001ff, 0x0000ff0000000100, 0x0000ff0000000101, 0x0000ff000001ff00,
    0x0000ff00000100ff, 0x0000ff0000010000, 0x0000ff0000010001, 0x0000ff0000010100,
    0x0000ff0001ffff01, 0x0000ff0001ff0000, 0x0000ff000100ff00, 0x0000ff00010000ff,
    0x0000ff0001000000, 0x0000ff0001000001, 0x0000ff0001000100, 0x0000ff000101ffff,
    0x0000ff0001010000, 0x0000ff0001010101, 0x0000ff01ffffff00, 0x0000ff01ffff0001,
    0x0000ff01ff00ff01, 0x0000ff01ff000000, 0x0000ff01ff000101, 0x0000ff01ff01ff00,
    0x0000ff01ff0100ff, 0x0000ff0100ffff01, 0x0000ff0100ff0000, 0x0000ff0100ff0101,
    0x0000ff010000ff00, 0x0000ff01000000ff, 0x0000ff0100000000, 0x0000ff0100000001,
    0x0000ff0100000100, 0x0000ff010001ff01, 0x0000ff0100010000, 0x0000ff0101ff0000,
    0x0000ff010100ffff, 0x0000ff010100ff01, 0x0000ff0101000000, 0x0000ff0101000100,
    0x0000ff0101000101, 0x0000ff01010100ff, 0x000000ffffff00ff, 0x000000ffffff0000,
    0x000000ffff00ff00, 0x000000ffff0000ff, 0x000000ffff000000, 0x000000ffff000001,
    0x000000ffff0001ff, 0x000000ffff000100, 0x000000ffff01ff00, 0x000000ffff010000,
    0x000000ffff0101ff, 0x000000ffff010101, 0x000000ff00ffff00, 0x000000ff00ff00ff,
    0x000000ff00ff0000, 0x000000ff00ff0001, 0x000000ff00ff0100, 0x000000ff00ff0101,
    0x000000ff0000ffff, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000,
    0x000000ff00000001, 0x000000ff000001ff, 0x000000ff00000100, 0x000000ff00000101,
    0x000000ff0001ff00, 0x000000ff0001ff01, 0x000000ff000100ff, 0x000000ff00010000,
    0x000000ff00010001, 0x000000ff00010100, 0x000000ff01ffffff, 0x000000ff01ff01ff,
    0x000000ff01ff0101, 0x000000ff0100ff00, 0x000000ff010000ff, 0x000000ff01000000,
    0x000000ff01000001, 0x000000ff01000100, 0x000000ff0101ff00, 0x000000ff010100ff,
    0x000000ff01010000, 0x000000ff01010101, 0x00000000ffffff00, 0x00000000ffffff01,
    0x00000000ffff00ff, 0x00000000ffff0000, 0x00000000ffff0001, 0x00000000ffff0100,
    0x00000000ff00ffff, 0x00000000ff00ff00, 0x00000000ff00ff01, 0x00000000ff0000ff,
    0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff000101,
    0x00000000ff01ff00, 0x00000000ff0100ff, 0x00000000ff010000, 0x00000000ff010001,
    0x00000000ff010100, 0x0000000000ffffff, 0x0000000000ffff00, 0x0000000000ffff01,
    0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001, 0x0000000000ff01ff,
    0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01,
    0x00000000000000ff, 0x0000000000000000, 0x0000000000000001, 0x00000000000001ff,
    0x0000000000000100, 0x0000000000000101, 0x000000000001ffff, 0x000000000001ff00,
    0x00000000000100ff, 0x0000000000010000, 0x0000000000010001, 0x00000000000101ff,
    0x0000000000010100, 0x0000000000010101, 0x0000000001ffff00, 0x0000000001ff00ff,
    0x0000000001ff0000, 0x0000000001ff0100, 0x0000000001ff0101, 0x000000000100ffff,
    0x000000000100ff00, 0x00000000010000ff, 0x0000000001000000, 0x0000000001000001,
    0x00000000010001ff, 0x0000000001000100, 0x000000000101ff00, 0x00000000010100ff,
    0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x00000001ffffffff,
    0x00000001ffffff00, 0x00000001ffffff01, 0x00000001ffff00ff, 0x00000001ffff0001,
    0x00000001ffff01ff, 0x00000001ffff0100, 0x00000001ff00ff00, 0x00000001ff0000ff,
    0x00000001ff000000, 0x00000001ff0001ff, 0x00000001ff000100, 0x00000001ff01ffff,
    0x00000001ff01ff00, 0x00000001ff01ff01, 0x00000001ff0100ff, 0x00000001ff010000,
    0x00000001ff010001, 0x00000001ff0101ff, 0x00000001ff010100, 0x0000000100ffff00,
    0x0000000100ff0000, 0x0000000100ff0001, 0x0000000100ff01ff, 0x0000000100ff0100,
    0x0000000100ff0101, 0x000000010000ffff, 0x000000010000ff00, 0x000000010000ff01,
    0x00000001000000ff, 0x0000000100000000, 0x0000000100000001, 0x00000001000001ff,
    0x0000000100000100, 0x0000000100000101, 0x000000010001ff00, 0x00000001000100ff,
    0x0000000100010000, 0x0000000100010100, 0x0000000101ffff01, 0x0000000101ff0000,
    0x0000000101ff0001, 0x0000000101ff01ff, 0x0000000101ff0100, 0x0000000101ff0101,
    0x000000010100ff00, 0x0000000101000000, 0x0000000101000101, 0x000000010101ff01,
    0x0000000101010000, 0x0000000101010001, 0x00000001010101ff, 0x0000000101010100,
    0x000001ffffff00ff, 0x000001ffffff0000, 0x000001ffffff0001, 0x000001ffffff0100,
    0x000001ffff00ffff, 0x000001ffff000000, 0x000001ffff0001ff, 0x000001ffff01ff00,
    0x000001ffff010101, 0x000001ff00ff0000, 0x000001ff00ff01ff, 0x000001ff00ff0101,
    0x000001ff0000ff00, 0x000001ff000000ff, 0x000001ff00000000, 0x000001ff00000001,
    0x000001ff000001ff, 0x000001ff00000100, 0x000001ff0001ffff, 0x000001ff0001ff01,
    0x000001ff000100ff, 0x000001ff00010000, 0x000001ff01ffff01, 0x000001ff01ff0100,
    0x000001ff0100ffff, 0x000001ff0100ff01, 0x000001ff01000000, 0x000001ff010001ff,
    0x000001ff0101ff00, 0x000001ff01010100, 0x00000100ffffff00, 0x00000100ffffff01,
    0x00000100ffff0000, 0x00000100ffff0101, 0x00000100ff00ff00, 0x00000100ff0000ff,
    0x00000100ff000000, 0x00000100ff000001, 0x00000100ff000100, 0x00000100ff010000,
    0x0000010000ffff00, 0x0000010000ff00ff, 0x0000010000ff0000, 0x0000010000ff0001,
    0x0000010000ff0100, 0x000001000000ffff, 0x000001000000ff00, 0x000001000000ff01,
    0x00000100000000ff, 0x0000010000000000, 0x0000010000000001, 0x00000100000001ff,
    0x0000010000000100, 0x0000010000000101, 0x000001000001ff00, 0x00000100000100ff,
    0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010001ffff00,
    0x0000010001ff0000, 0x0000010001ff0100, 0x000001000100ff00, 0x00000100010000ff,
    0x0000010001000000, 0x0000010001000001, 0x00000100010001ff, 0x0000010001000100,
    0x0000010001010000, 0x00000101ffff00ff, 0x00000101ffff01ff, 0x00000101ff000000,
    0x00000101ff000101, 0x00000101ff01ffff, 0x00000101ff010000, 0x00000101ff010001,
    0x00000101ff010100, 0x0000010100ff0000, 0x0000010100ff01ff, 0x0000010100ff0100,
    0x000001010000ff00, 0x0000010100000000, 0x0000010100000001, 0x00000101000001ff,
    0x0000010100000100, 0x000001010001ff01, 0x0000010100010000, 0x00000101000101ff,
    0x0000010100010101, 0x0000010101ffff00, 0x0000010101ff0101, 0x000001010100ff01,
    0x0000010101000000, 0x0000010101000001, 0x00000101010001ff, 0x0000010101000101,
    0x000001010101ff00, 0x0001ffffffff0000, 0x0001ffffff0000ff, 0x0001ffffff000001,
    0x0001ffffff000100, 0x0001ffffff010000, 0x0001ffff00ff00ff, 0x0001ffff0000ffff,
    0x0001ffff00000000, 0x0001ffff00000001, 0x0001ffff000001ff, 0x0001ffff00000101,
    0x0001ffff0001ff00, 0x0001ffff000100ff, 0x0001ffff00010001, 0x0001ffff00010100,
    0x0001ffff01ffff00, 0x0001ffff01000001, 0x0001ffff01010000, 0x0001ff00ffffff00,
    0x0001ff00ffff00ff, 0x0001ff00ffff0001, 0x0001ff00ffff0100, 0x0001ff00ff00ff01,
    0x0001ff00ff000000, 0x0001ff00ff01ff00, 0x0001ff00ff01ff01, 0x0001ff00ff010001,
    0x0001ff00ff010100, 0x0001ff0000ff0000, 0x0001ff0000ff0100, 0x0001ff000000ff00,
    0x0001ff0000000000, 0x0001ff0000000001, 0x0001ff0000000100, 0x0001ff0000010000,
    0x0001ff0000010001, 0x0001ff0000010101, 0x0001ff0001ff00ff, 0x0001ff0001ff0101,
    0x0001ff000100ff01, 0x0001ff0001000000, 0x0001ff000101ff00, 0x0001ff0001010001,
    0x0001ff0001010100, 0x0001ff01ff00ff00, 0x0001ff01ff000001, 0x0001ff01ff000100,
    0x0001ff0100ffffff, 0x0001ff0100ffff00, 0x0001ff0100ff0001, 0x0001ff0100000000,
    0x0001ff0100000001, 0x0001ff01000001ff, 0x0001ff010001ffff, 0x0001ff0101ff0000,
    0x0001ff010100ff00, 0x0001ff0101000001, 0x0001ff0101010000, 0x000100ffff00ff00,
    0x000100ffff00ff01, 0x000100ffff000000, 0x000100ffff000001, 0x000100ffff000101,
    0x000100ffff01ff00, 0x000100ffff010001, 0x000100ffff010100, 0x000100ff00ffffff,
    0x000100ff00ffff01, 0x000100ff00ff0000, 0x000100ff00ff01ff, 0x000100ff00ff0101,
    0x000100ff0000ff00, 0x000100ff000000ff, 0x000100ff00000000, 0x000100ff00000001,
    0x000100ff00000100, 0x000100ff00000101, 0x000100ff0001ffff, 0x000100ff0001ff01,
    0x000100ff00010000, 0x000100ff01ff00ff, 0x000100ff01ff0000, 0x000100ff01ff0100,
    0x000100ff0100ffff, 0x000100ff0100ff01, 0x000100ff010000ff, 0x000100ff01000000,
    0x000100ff01000001, 0x000100ff010001ff, 0x000100ff01000101, 0x000100ff0101ff00,
    0x000100ff010100ff, 0x000100ff01010100, 0x00010000ffff0000, 0x00010000ffff01ff,
    0x00010000ffff0101, 0x00010000ff00ff00, 0x00010000ff000000, 0x00010000ff000001,
    0x00010000ff000100, 0x0001000000ff00ff, 0x0001000000ff0000, 0x0001000000ff0001,
    0x0001000000ff0100, 0x000100000000ffff, 0x000100000000ff00, 0x00010000000000ff,
    0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x000100000001ff00,
    0x00010000000100ff, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100,
    0x0001000001ff0001, 0x0001000001ff0100, 0x0001000001ff0101, 0x000100000100ff00,
    0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101,
    0x000100000101ff01, 0x0001000001010000, 0x0001000001010001, 0x00010000010101ff,
    0x00010001ffffff01, 0x00010001ffff0100, 0x00010001ff000000, 0x00010001ff01ffff,
    0x00010001ff010001, 0x00010001ff0101ff, 0x00010001ff010100, 0x0001000100ffffff,
    0x0001000100ff0000, 0x0001000100ff01ff, 0x0001000100ff0101, 0x000100010000ff00,
    0x00010001000000ff, 0x0001000100000000, 0x0001000100000001, 0x00010001000001ff,
    0x0001000100000101, 0x000100010001ffff, 0x0001000100010000, 0x00010001000101ff,
    0x0001000101ffffff, 0x0001000101ffff01, 0x0001000101ff0000, 0x0001000101ff0101,
    0x00010001010000ff, 0x0001000101000001, 0x00010001010001ff, 0x0001000101000100,
    0x000100010101ffff, 0x00010001010100ff, 0x0001000101010001, 0x0001000101010101,
    0x000101ffff000001, 0x000101ffff000100, 0x000101ffff010000, 0x000101ff00ffff00,
    0x000101ff0000ff01, 0x000101ff00000000, 0x000101ff00000101, 0x000101ff0001ff00,
    0x000101ff00010100, 0x000101ff01ff0000, 0x000101ff0100ff00, 0x000101ff010001ff,
    0x000101ff01010001, 0x00010100ffffff00, 0x00010100ffff00ff, 0x00010100ff00ffff,
    0x00010100ff000000, 0x00010100ff01ff00, 0x00010100ff0100ff, 0x00010100ff010001,
    0x00010100ff010100, 0x0001010000ffffff, 0x0001010000ffff00, 0x0001010000ff0000,
    0x0001010000ff0001, 0x0001010000ff01ff, 0x000101000000ff00, 0x00010100000000ff,
    0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x000101000001ffff,
    0x0001010000010000, 0x0001010000010101, 0x0001010001ffff01, 0x0001010001ff00ff,
    0x0001010001ff0101, 0x0001010001000000, 0x000101000101ff00, 0x00010100010100ff,
    0x0001010001010000, 0x0001010001010100, 0x00010101ff00ff00, 0x00010101ff000001,
    0x00010101ff0001ff, 0x0001010100ffff00, 0x0001010100ff00ff, 0x0001010100ff0100,
    0x000101010000ffff, 0x0001010100000000, 0x00010101000001ff, 0x0001010100000101,
    0x00010101000100ff, 0x0001010100010000, 0x0001010100010100, 0x0001010101ff0001,
    0x00010101010000ff, 0x00010101010001ff, 0x0001010101000101, 0x0001010101010001,
    0x01ffffffffffffff, 0x01ffffffffffff01, 0x01ffffffffff01ff, 0x01ffffffffff0101,
    0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff, 0x01ffffffff010101,
    0x01ffffff00ff0000, 0x01ffffff0000ffff, 0x01ffffff0000ff00, 0x01ffffff000000ff,
    0x01ffffff00000001, 0x01ffffff00000100, 0x01ffffff00010000, 0x01ffffff01ffffff,
    0x01ffffff01ffff01, 0x01ffffff01ff01ff, 0x01ffffff01ff0101, 0x01ffffff01000000,
    0x01ffffff0101ffff, 0x01ffffff0101ff01, 0x01ffffff010101ff, 0x01ffffff01010101,
    0x01ffff00ffff0000, 0x01ffff00ff00ff00, 0x01ffff00ff0000ff, 0x01ffff00ff000001,
    0x01ffff00ff000100, 0x01ffff00ff010000, 0x01ffff0000ffff00, 0x01ffff0000ff00ff,
    0x01ffff0000ff0100, 0x01ffff000000ffff, 0x01ffff000000ff01, 0x01ffff0000000000,
    0x01ffff0000000001, 0x01ffff00000001ff, 0x01ffff0000000100, 0x01ffff00000100ff,
    0x01ffff0000010001, 0x01ffff0000010100, 0x01ffff0001ff0000, 0x01ffff0001ff0100,
    0x01ffff00010000ff, 0x01ffff0001000001, 0x01ffff0001000100, 0x01ffff0001010000,
    0x01ffff01ffffffff, 0x01ffff01ffffff01, 0x01ffff01ffff01ff, 0x01ffff01ffff0101,
    0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff01ff01, 0x01ffff01ff0101ff,
    0x01ffff01ff010101, 0x01ffff010000ff00, 0x01ffff01000000ff, 0x01ffff0100000100,
    0x01ffff0100010000, 0x01ffff0101ffffff, 0x01ffff0101ffff01, 0x01ffff0101ff01ff,
    0x01ffff0101ff0101, 0x01ffff0101000000, 0x01ffff010101ffff, 0x01ffff010101ff01,
    0x01ffff01010101ff, 0x01ffff0101010101, 0x01ff00ffff0000ff, 0x01ff00ffff000100,
    0x01ff00ff00ffff00, 0x01ff00ff00ff00ff, 0x01ff00ff0000ff00, 0x01ff00ff00000000,
    0x01ff00ff00000101, 0x01ff00ff0001ff00, 0x01ff00ff000100ff, 0x01ff00ff00010100,
    0x01ff00ff010000ff, 0x01ff00ff01000100, 0x01ff0000ffffff00, 0x01ff0000ffff0100,
    0x01ff0000ff00ff01, 0x01ff0000ff000000, 0x01ff0000ff000101, 0x01ff0000ff010001,
    0x01ff0000ff010100, 0x01ff000000ffffff, 0x01ff000000ffff00, 0x01ff000000ff0000,
    0x01ff000000ff01ff, 0x01ff00000000ff00, 0x01ff0000000000ff, 0x01ff000000000000,
    0x01ff000000000001, 0x01ff000000000100, 0x01ff000000000101, 0x01ff000000010000,
    0x01ff000000010001, 0x01ff0000000101ff, 0x01ff000000010101, 0x01ff000001ffff00,
    0x01ff000001ff00ff, 0x01ff000001ff0001, 0x01ff000001ff0100, 0x01ff00000100ffff,
    0x01ff00000100ff01, 0x01ff000001000000, 0x01ff0000010001ff, 0x01ff000001010001,
    0x01ff0001ff00ff00, 0x01ff0001ff000001, 0x01ff0001ff000100, 0x01ff0001ff010000,
    0x01ff000100ffff00, 0x01ff000100ff00ff, 0x01ff000100ff0100, 0x01ff000100ff0101,
    0x01ff00010000ffff, 0x01ff000100000000, 0x01ff000100000100, 0x01ff000100000101,
    0x01ff00010001ff00, 0x01ff000100010001, 0x01ff000100010101, 0x01ff000101ff0000,
    0x01ff00010100ff00, 0x01ff000101000101, 0x01ff0001010100ff, 0x01ff01ffffffffff,
    0x01ff01ffffffff01, 0x01ff01ffffff01ff, 0x01ff01ffffff0101, 0x01ff01ffff000000,
    0x01ff01ffff01ffff, 0x01ff01ffff01ff01, 0x01ff01ffff0101ff, 0x01ff01ffff010101,
    0x01ff01ff00ffff00, 0x01ff01ff00ff0000, 0x01ff01ff0000ff00, 0x01ff01ff000000ff,
    0x01ff01ff00000100, 0x01ff01ff00010000, 0x01ff01ff00010100, 0x01ff01ff01ffffff,
    0x01ff01ff01ffff01, 0x01ff01ff01ff01ff, 0x01ff01ff01ff0101, 0x01ff01ff01000000,
    0x01ff01ff0101ffff, 0x01ff01ff0101ff01, 0x01ff01ff010101ff, 0x01ff01ff01010101,
    0x01ff0100ffff0000, 0x01ff0100ffff0001, 0x01ff0100ff00ff00, 0x01ff0100ff0000ff,
    0x01ff0100ff000001, 0x01ff0100ff010000, 0x01ff010000ffff00, 0x01ff010000ff00ff,
    0x01ff010000ff0001, 0x01ff010000ff0100, 0x01ff01000000ffff, 0x01ff01000000ff01,
    0x01ff010000000000, 0x01ff010000000101, 0x01ff01000001ff00, 0x01ff0100000100ff,
    0x01ff010001ff0000, 0x01ff010001000001, 0x01ff010001000100, 0x01ff010001010000,
    0x01ff0101ffffffff, 0x01ff0101ffffff01, 0x01ff0101ffff01ff, 0x01ff0101ffff0101,
    0x01ff0101ff000000, 0x01ff0101ff01ffff, 0x01ff0101ff01ff01, 0x01ff0101ff0101ff,
    0x01ff0101ff010101, 0x01ff010100ff0000, 0x01ff01010000ff00, 0x01ff0101000000ff,
    0x01ff010100000001, 0x01ff010101ffffff, 0x01ff010101ffff01, 0x01ff010101ff01ff,
    0x01ff010101ff0101, 0x01ff010101000000, 0x01ff01010101ffff, 0x01ff01010101ff01,
    0x01ff0101010101ff, 0x01ff010101010101, 0x0100ffffffff0000, 0x0100ffffff00ff00,
    0x0100ffffff000001, 0x0100ffffff0001ff, 0x0100ffffff000100, 0x0100ffffff010000,
    0x0100ffff00ffff00, 0x0100ffff00ff0001, 0x0100ffff00ff0100, 0x0100ffff00000000,
    0x0100ffff000001ff, 0x0100ffff00000101, 0x0100ffff00010100, 0x0100ffff00010101,
    0x0100ffff01ff0000, 0x0100ffff0100ff00, 0x0100ffff010000ff, 0x0100ffff01000001,
    0x0100ffff01000100, 0x0100ffff01010000, 0x0100ff00ffffff00, 0x0100ff00ffff00ff,
    0x0100ff00ffff0001, 0x0100ff00ffff0100, 0x0100ff00ff00ffff, 0x0100ff00ff000000,
    0x0100ff00ff0001ff, 0x0100ff00ff000101, 0x0100ff00ff01ff00, 0x0100ff00ff0100ff,
    0x0100ff00ff010001, 0x0100ff00ff010100, 0x0100ff0000ffffff, 0x0100ff0000ff0000,
    0x0100ff000000ffff, 0x0100ff000000ff00, 0x0100ff00000000ff, 0x0100ff0000000000,
    0x0100ff0000000001, 0x0100ff0000000100, 0x0100ff000001ff01, 0x0100ff0000010000,
    0x0100ff0001ff00ff, 0x0100ff0001ff0001, 0x0100ff000100ff01, 0x0100ff0001000000,
    0x0100ff00010001ff, 0x0100ff000101ff00, 0x0100ff00010100ff, 0x0100ff0001010001,
    0x0100ff0001010100, 0x0100ff01ffff0000, 0x0100ff01ff00ff00, 0x0100ff01ff0000ff,
    0x0100ff01ff000100, 0x0100ff01ff010000, 0x0100ff0100ff00ff, 0x0100ff0100ff0001,
    0x0100ff0100ff0100, 0x0100ff010000ffff, 0x0100ff010000ff01, 0x0100ff0100000000,
    0x0100ff01000001ff, 0x0100ff0100010001, 0x0100ff0100010100, 0x0100ff0101ff0000,
    0x0100ff01010000ff, 0x0100ff0101000001, 0x0100ff0101010100, 0x010000ffffffff00,
    0x010000ffffff00ff, 0x010000ffffff0001, 0x010000ffff00ffff, 0x010000ffff000000,
    0x010000ffff0001ff, 0x010000ffff010001, 0x010000ff00ffffff, 0x010000ff00ff0101,
    0x010000ff0000ff00, 0x010000ff000000ff, 0x010000ff00000000, 0x010000ff00000001,
    0x010000ff000001ff, 0x010000ff00000100, 0x010000ff0001ffff, 0x010000ff0001ff00,
    0x010000ff0001ff01, 0x010000ff00010000, 0x010000ff01ff00ff, 0x010000ff01ff0001,
    0x010000ff0100ff01, 0x010000ff010000ff, 0x010000ff01000000, 0x010000ff010001ff,
    0x010000ff0101ff00, 0x010000ff01010100, 0x01000000ffffffff, 0x01000000ffff0000,
    0x01000000ffff01ff, 0x01000000ffff0101, 0x01000000ff00ffff, 0x01000000ff00ff00,
    0x01000000ff0000ff, 0x01000000ff000000, 0x01000000ff000001, 0x01000000ff000100,
    0x01000000ff01ff00, 0x01000000ff010000, 0x01000000ff010100, 0x01000000ff010101,
    0x0100000000ffff00, 0x0100000000ff00ff, 0x0100000000ff0000, 0x0100000000ff0001,
    0x0100000000ff0100, 0x010000000000ffff, 0x010000000000ff00, 0x010000000000ff01,
    0x01000000000000ff, 0x0100000000000000, 0x0100000000000001, 0x01000000000001ff,
    0x0100000000000100, 0x0100000000000101, 0x010000000001ff00, 0x01000000000100ff,
    0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000001ffff00,
    0x0100000001ff0000, 0x0100000001ff01ff, 0x010000000100ff00, 0x010000000100ff01,
    0x01000000010000ff, 0x0100000001000000, 0x0100000001000001, 0x0100000001000100,
    0x0100000001000101, 0x010000000101ffff, 0x010000000101ff01, 0x0100000001010000,
    0x01000000010101ff, 0x0100000001010101, 0x01000001ffffff00, 0x01000001ffff00ff,
    0x01000001ff00ffff, 0x01000001ff000000, 0x01000001ff000100, 0x01000001ff01ffff,
    0x01000001ff010001, 0x01000001ff010100, 0x0100000100ff0000, 0x0100000100ff01ff,
    0x0100000100ff0100, 0x010000010000ff00, 0x010000010000ff01, 0x0100000100000000,
    0x0100000100000001, 0x0100000100000100, 0x0100000100010000, 0x01000001000101ff,
    0x0100000101ffff01, 0x0100000101ff00ff, 0x0100000101ff0100, 0x0100000101ff0101,
    0x010000010100ff01, 0x01000001010000ff, 0x0100000101000000, 0x01000001010100ff,
    0x0100000101010001, 0x0100000101010100, 0x010001ffffff0000, 0x010001ffff000001,
    0x010001ffff000100, 0x010001ffff010000, 0x010001ff00ffff00, 0x010001ff00ff0001,
    0x010001ff0000ffff, 0x010001ff0000ff01, 0x010001ff00000000, 0x010001ff00000001,
    0x010001ff00000101, 0x010001ff000100ff, 0x010001ff00010000, 0x010001ff01ff0000,
    0x010001ff0100ff00, 0x010001ff01000001, 0x010001ff01000100, 0x010001ff01010000,
    0x01000100ffff00ff, 0x01000100ffff0001, 0x01000100ffff0100, 0x01000100ff00ffff,
    0x01000100ff00ff01, 0x01000100ff000000, 0x01000100ff0001ff, 0x01000100ff000101,
    0x01000100ff01ffff, 0x01000100ff01ff00, 0x01000100ff0100ff, 0x01000100ff010001,
    0x0100010000ffffff, 0x0100010000ffff01, 0x0100010000ff0000, 0x0100010000ff01ff,
    0x0100010000ff0101, 0x010001000000ff00, 0x01000100000000ff, 0x0100010000000000,
    0x0100010000000001, 0x0100010000000100, 0x010001000001ff01, 0x0100010000010000,
    0x0100010000010001, 0x0100010000010101, 0x0100010001ffff00, 0x0100010001ff00ff,
    0x010001000100ffff, 0x010001000100ff01, 0x0100010001000000, 0x0100010001000101,
    0x010001000101ff00, 0x0100010001010001, 0x01000101ffff0000, 0x01000101ff000000,
    0x01000101ff010000, 0x0100010100ff00ff, 0x0100010100ff0001, 0x0100010100ff0100,
    0x010001010000ffff, 0x0100010100000000, 0x01000101000001ff, 0x010001010001ff00,
    0x0100010101ff0000, 0x010001010100ff00, 0x01000101010000ff, 0x0100010101000000,
    0x0100010101000001, 0x0101ffffffffffff, 0x0101ffffffffff01, 0x0101ffffffff01ff,
    0x0101ffffffff0101, 0x0101ffffff000000, 0x0101ffffff01ffff, 0x0101ffffff01ff01,
    0x0101ffffff0101ff, 0x0101ffffff010101, 0x0101ffff00ff0000, 0x0101ffff0000ff00,
    0x0101ffff000000ff, 0x0101ffff00000001, 0x0101ffff00000100, 0x0101ffff01ffffff,
    0x0101ffff01ffff01, 0x0101ffff01ff01ff, 0x0101ffff01ff0101, 0x0101ffff01000000,
    0x0101ffff0101ffff, 0x0101ffff0101ff01, 0x0101ffff010101ff, 0x0101ffff01010101,
    0x0101ff00ffff0000, 0x0101ff00ffff0100, 0x0101ff00ff00ff00, 0x0101ff00ff0000ff,
    0x0101ff00ff000001, 0x0101ff00ff000100, 0x0101ff00ff000101, 0x0101ff0000ff0001,
    0x0101ff0000ff0100, 0x0101ff000000ff00, 0x0101ff0000000000, 0x0101ff00000001ff,
    0x0101ff0000000101, 0x0101ff000001ff00, 0x0101ff00000100ff, 0x0101ff0001ff0000,
    0x0101ff000100ffff, 0x0101ff000100ff01, 0x0101ff0001000001, 0x0101ff0001000100,
    0x0101ff01ffffff01, 0x0101ff01ffff01ff, 0x0101ff01ffff0101, 0x0101ff01ff00ffff,
    0x0101ff01ff000100, 0x0101ff01ff01ff01, 0x0101ff01ff0101ff, 0x0101ff01ff010101,
    0x0101ff0100ff0000, 0x0101ff010000ff00, 0x0101ff0100000001, 0x0101ff0100000100,
    0x0101ff0100010000, 0x0101ff0101ffffff, 0x0101ff0101ffff01, 0x0101ff0101ff01ff,
    0x0101ff0101ff0101, 0x0101ff0101000000, 0x0101ff010101ffff, 0x0101ff010101ff01,
    0x0101ff01010101ff, 0x0101ff0101010101, 0x010100ffff000100, 0x010100ffff010000,
    0x010100ff00ffff00, 0x010100ff00ff00ff, 0x010100ff0000ffff, 0x010100ff000000ff,
    0x010100ff00000000, 0x010100ff000001ff, 0x010100ff00000101, 0x010100ff0001ff00,
    0x010100ff00010000, 0x010100ff00010001, 0x010100ff000101ff, 0x010100ff00010100,
    0x010100ff01ff0000, 0x01010000ffff0001, 0x01010000ffff0100, 0x01010000ff00ffff,
    0x01010000ff00ff01, 0x01010000ff000000, 0x01010000ff0001ff, 0x01010000ff010001,
    0x01010000ff010100, 0x0101000000ffff01, 0x0101000000ff0000, 0x010100000000ff00,
    0x01010000000000ff, 0x0101000000000000, 0x0101000000000001, 0x0101000000000100,
    0x0101000000010000, 0x0101000000010101, 0x0101000001ffff00, 0x0101000001ff00ff,
    0x0101000001ff0000, 0x0101000001ff0001, 0x0101000001ff0100, 0x010100000100ff01,
    0x0101000001000000, 0x01010000010001ff, 0x01010001ffff0000, 0x01010001ff00ff00,
    0x01010001ff000001, 0x01010001ff000101, 0x01010001ff01ff00, 0x01010001ff010000,
    0x0101000100ff00ff, 0x0101000100ff0001, 0x0101000100ff0101, 0x010100010000ff01,
    0x0101000100000000, 0x0101000100000001, 0x01010001000001ff, 0x010100010001ffff,
    0x010100010001ff01, 0x0101000101ff0001, 0x010100010100ffff, 0x0101000101000000,
    0x0101000101000001, 0x0101000101000100, 0x010100010101ff00, 0x01010001010100ff,
    0x0101000101010001, 0x010101ffffffffff, 0x010101ffffffff01, 0x010101ffffff01ff,
    0x010101ffffff0101, 0x010101ffff01ffff, 0x010101ffff01ff01, 0x010101ffff0101ff,
    0x010101ffff010101, 0x010101ff0000ff00, 0x010101ff000000ff, 0x010101ff00000001,
    0x010101ff00000100, 0x010101ff01ffffff, 0x010101ff01ffff01, 0x010101ff01ff01ff,
    0x010101ff01ff0101, 0x010101ff01000000, 0x010101ff0101ffff, 0x010101ff0101ff01,
    0x010101ff010101ff, 0x010101ff01010101, 0x01010100ffff0000, 0x01010100ff0000ff,
    0x01010100ff000100, 0x01010100ff01ff00, 0x01010100ff010000, 0x0101010000ffff00,
    0x010101000000ffff, 0x0101010000000000, 0x0101010000000101, 0x010101000001ff00,
    0x0101010000010001, 0x0101010000010100, 0x010101000100ffff, 0x0101010001000001,
    0x01010101ffffffff, 0x01010101ffffff01, 0x01010101ffff01ff, 0x01010101ffff0101,
    0x01010101ff01ffff, 0x01010101ff01ff01, 0x01010101ff0101ff, 0x01010101ff010101,
    0x010101010000ff00, 0x01010101000000ff, 0x0101010100000001, 0x0101010101ffffff,
    0x0101010101ffff01, 0x0101010101ff01ff, 0x0101010101ff0101, 0x0101010101000000,
    0x010101010101ffff, 0x010101010101ff01, 0x01010101010101ff, 0x0101010101010101,
GGML_TABLE_END()
#else
GGML_TABLE_BEGIN(uint32_t, iq1s_grid_gpu, NGRID_IQ1S)
    0x00000000, 0x00000002, 0x00000101, 0x00000200, 0x00000202, 0x00010001, 0x00010101, 0x00020000,
    0x00020002, 0x00020200, 0x00020202, 0x01000101, 0x01010001, 0x01010100, 0x01010102, 0x01020101,
    0x02000000, 0x02000002, 0x02000200, 0x02000202, 0x02010101, 0x02020000, 0x02020002, 0x02020200,
    0x02020202, 0x00000110, 0x00000111, 0x00010011, 0x00010110, 0x00010112, 0x00010211, 0x00010212,
    0x00020111, 0x01000011, 0x01000112, 0x01000211, 0x01010012, 0x01010111, 0x01010212, 0x01020011,
    0x01020110, 0x01020112, 0x01020210, 0x02000111, 0x02010011, 0x02010110, 0x02010112, 0x02020111,
    0x00000020, 0x00000022, 0x00000220, 0x00000222, 0x00010121, 0x00020020, 0x00020022, 0x00020220,
    0x00020222, 0x01000121, 0x01010021, 0x01010221, 0x01020120, 0x01020221, 0x02000020, 0x02000022,
    0x02000220, 0x02000222, 0x02010021, 0x02010121, 0x02010221, 0x02020020, 0x02020022, 0x02020220,
    0x02020222, 0x00011001, 0x00011100, 0x00011102, 0x00021101, 0x01001001, 0x01001201, 0x01011101,
    0x01011202, 0x01021100, 0x01021101, 0x02011001, 0x02011201, 0x02021101, 0x00001011, 0x00001110,
    0x00001111, 0x00001112, 0x00011111, 0x00011210, 0x00011212, 0x00021211, 0x01001010, 0x01001111,
    0x01001212, 0x01011010, 0x01011011, 0x01011110, 0x01011111, 0x01011112, 0x01011211, 0x01021010,
    0x01021012, 0x01021111, 0x01021210, 0x01021212, 0x02001011, 0x02011011, 0x02011111, 0x02011210,
    0x02011212, 0x02021011, 0x02021110, 0x02021111, 0x02021112, 0x02021211, 0x00011120, 0x00011221,
    0x01001021, 0x01001120, 0x01011020, 0x01011022, 0x01011121, 0x01011220, 0x01021020, 0x01021021,
    0x01021122, 0x01021221, 0x02001121, 0x02011021, 0x02011120, 0x02011221, 0x00002000, 0x00002002,
    0x00002200, 0x00002202, 0x00012101, 0x00022000, 0x00022002, 0x00022200, 0x00022202, 0x01002101,
    0x01012001, 0x01012102, 0x01022101, 0x02002000, 0x02002002, 0x02002200, 0x02002202, 0x02012101,
    0x02022000, 0x02022002, 0x02022200, 0x02022202, 0x00002111, 0x00012011, 0x00012110, 0x00012211,
    0x00022110, 0x00022111, 0x01002011, 0x01012010, 0x01012011, 0x01012111, 0x01022011, 0x01022110,
    0x01022211, 0x02012011, 0x02012110, 0x02012112, 0x02012211, 0x02022111, 0x00002020, 0x00002022,
    0x00002220, 0x00002222, 0x00012121, 0x00022020, 0x00022022, 0x00022220, 0x00022222, 0x01002121,
    0x01012021, 0x01012221, 0x01022021, 0x01022121, 0x02002020, 0x02002022, 0x02002121, 0x02002220,
    0x02002222, 0x02012121, 0x02022020, 0x02022022, 0x02022220, 0x02022222, 0x00110000, 0x00110001,
    0x00110100, 0x00110201, 0x00120100, 0x00120101, 0x01100001, 0x01100100, 0x01110000, 0x01110101,
    0x01110200, 0x01120001, 0x01120100, 0x01120101, 0x01120201, 0x02110001, 0x02110100, 0x02110102,
    0x02120001, 0x02120101, 0x00100011, 0x00100110, 0x00100112, 0x00100211, 0x00110010, 0x00110012,
    0x00110111, 0x00110210, 0x00120011, 0x00120110, 0x00120211, 0x01100111, 0x01100212, 0x01110010,
    0x01110011, 0x01110012, 0x01110110, 0x01110111, 0x01110112, 0x01110211, 0x01120010, 0x01120111,
    0x02100110, 0x02110012, 0x02110111, 0x02120011, 0x02120110, 0x00110021, 0x00110120, 0x00110122,
    0x00120121, 0x01100020, 0x01100122, 0x01100221, 0x01110022, 0x01110121, 0x01110220, 0x01110222,
    0x01120120, 0x01120122, 0x02100121, 0x02110021, 0x02110120, 0x02110122, 0x02120121, 0x00101001,
    0x00101102, 0x00101201, 0x00111100, 0x00111101, 0x00111200, 0x00111201, 0x00121001, 0x00121102,
    0x01101001, 0x01101101, 0x01101102, 0x01101200, 0x01101202, 0x01111001, 0x01111100, 0x01111101,
    0x01111102, 0x01111201, 0x01121002, 0x01121101, 0x01121200, 0x02101100, 0x02101201, 0x02111000,
    0x02111100, 0x02111101, 0x02111200, 0x02111201, 0x02111202, 0x02121001, 0x02121100, 0x02121101,
    0x02121201, 0x00101012, 0x00101111, 0x00101212, 0x00111011, 0x00111110, 0x00111111, 0x00111112,
    0x00111211, 0x00121010, 0x00121012, 0x00121111, 0x00121210, 0x00121212, 0x01101011, 0x01101110,
    0x01101111, 0x01101112, 0x01111011, 0x01111012, 0x01111110, 0x01111111, 0x01111112, 0x01111211,
    0x01111212, 0x01121011, 0x01121110, 0x01121111, 0x01121112, 0x01121211, 0x02101010, 0x02101012,
    0x02101110, 0x02101111, 0x02101210, 0x02101212, 0x02111010, 0x02111011, 0x02111110, 0x02111111,
    0x02111112, 0x02111211, 0x02111212, 0x02121010, 0x02121012, 0x02121111, 0x00101021, 0x00101120,
    0x00101121, 0x00101122, 0x00111121, 0x00111122, 0x00111220, 0x00111222, 0x00121021, 0x00121122,
    0x01101020, 0x01101022, 0x01101120, 0x01101121, 0x01101220, 0x01101222, 0x01111021, 0x01111121,
    0x01111122, 0x01111220, 0x01111221, 0x01121021, 0x01121120, 0x01121121, 0x01121220, 0x01121221,
    0x01121222, 0x02101122, 0x02101222, 0x02111022, 0x02111121, 0x02121120, 0x02121221, 0x00112001,
    0x00112102, 0x00122101, 0x01102001, 0x01102100, 0x01102102, 0x01102201, 0x01112000, 0x01112101,
    0x01112200, 0x01112202, 0x01122000, 0x01122001, 0x01122100, 0x01122102, 0x01122201, 0x02102101,
    0x02112001, 0x02112100, 0x02122101, 0x00112010, 0x00112012, 0x00112111, 0x00112212, 0x00122011,
    0x00122111, 0x01102012, 0x01102110, 0x01102111, 0x01102210, 0x01112011, 0x01112110, 0x01112111,
    0x01112112, 0x01112211, 0x01112212, 0x01122010, 0x01122111, 0x01122212, 0x02102211, 0x02112011,
    0x02112012, 0x02112111, 0x02112210, 0x02122011, 0x02122112, 0x02122211, 0x00102221, 0x00112122,
    0x00122120, 0x00122122, 0x01102120, 0x01102122, 0x01102221, 0x01112020, 0x01112022, 0x01112121,
    0x01112220, 0x01122021, 0x01122122, 0x01122221, 0x02102121, 0x02112021, 0x02112122, 0x02112222,
    0x00200000, 0x00200002, 0x00200200, 0x00200202, 0x00210101, 0x00220000, 0x00220002, 0x00220101,
    0x00220200, 0x00220202, 0x01200101, 0x01210001, 0x01210201, 0x01220001, 0x01220101, 0x02200000,
    0x02200002, 0x02200200, 0x02200202, 0x02210101, 0x02220000, 0x02220002, 0x02220101, 0x02220200,
    0x02220202, 0x00200111, 0x00210011, 0x00210110, 0x00210211, 0x00220111, 0x01200012, 0x01200110,
    0x01200211, 0x01210111, 0x01210210, 0x01210212, 0x01220011, 0x01220110, 0x01220111, 0x01220112,
    0x02200111, 0x02210010, 0x02210112, 0x02210211, 0x02220111, 0x00200021, 0x00200220, 0x00200222,
    0x00210021, 0x00210121, 0x00220020, 0x00220022, 0x00220220, 0x00220222, 0x01200121, 0x01210021,
    0x01210122, 0x01210221, 0x01220121, 0x02200021, 0x02200220, 0x02200222, 0x02210021, 0x02210121,
    0x02220020, 0x02220022, 0x02220220, 0x02220222, 0x00201101, 0x00211100, 0x00211102, 0x00211201,
    0x00221101, 0x01201100, 0x01201101, 0x01201102, 0x01201201, 0x01211002, 0x01211101, 0x01211200,
    0x01211202, 0x01221102, 0x02201101, 0x02211001, 0x02211100, 0x02211201, 0x02221001, 0x02221101,
    0x00201211, 0x00211111, 0x00221011, 0x00221211, 0x01201010, 0x01201111, 0x01201210, 0x01211011,
    0x01211110, 0x01211111, 0x01211211, 0x01221012, 0x01221111, 0x01221210, 0x02201211, 0x02211010,
    0x02211110, 0x02211111, 0x02211210, 0x02211212, 0x02221011, 0x02221110, 0x02221112, 0x02221211,
    0x00201121, 0x00211020, 0x00211022, 0x00211221, 0x00221121, 0x01201021, 0x01201221, 0x01211121,
    0x01221020, 0x01221021, 0x01221221, 0x02201120, 0x02201122, 0x02211020, 0x02211222, 0x00202000,
    0x00202002, 0x00202200, 0x00202202, 0x00212101, 0x00222000, 0x00222002, 0x00222200, 0x00222202,
    0x01202101, 0x01212001, 0x01212100, 0x01222101, 0x02202000, 0x02202002, 0x02202200, 0x02202202,
    0x02222000, 0x02222002, 0x02222200, 0x02222202, 0x00202211, 0x00212011, 0x00212110, 0x00212211,
    0x00222111, 0x01202112, 0x01202211, 0x01212012, 0x01212111, 0x01222011, 0x01222110, 0x01222112,
    0x01222211, 0x02202111, 0x02212010, 0x02212112, 0x02212211, 0x02222110, 0x02222111, 0x00202020,
    0x00202022, 0x00202220, 0x00202222, 0x00222020, 0x00222022, 0x00222220, 0x00222222, 0x01202121,
    0x01212021, 0x01212122, 0x01212221, 0x01222121, 0x02202020, 0x02202022, 0x02202220, 0x02202222,
    0x02212121, 0x02222020, 0x02222022, 0x02222220, 0x02222222, 0x10000101, 0x10010001, 0x10010102,
    0x10020101, 0x11000201, 0x11010002, 0x11010101, 0x11010200, 0x11010202, 0x11020001, 0x11020100,
    0x11020102, 0x12010100, 0x12010201, 0x12020001, 0x12020102, 0x10000010, 0x10000011, 0x10000110,
    0x10000112, 0x10000211, 0x10010012, 0x10010111, 0x10010112, 0x10010210, 0x10010212, 0x10020011,
    0x10020112, 0x10020211, 0x11000111, 0x11000210, 0x11000212, 0x11010011, 0x11010110, 0x11010111,
    0x11010112, 0x11010211, 0x11010212, 0x11020111, 0x11020210, 0x11020212, 0x12000011, 0x12000110,
    0x12000112, 0x12010010, 0x12010012, 0x12010111, 0x12020010, 0x12020011, 0x12020012, 0x10000121,
    0x10010021, 0x10010120, 0x10010122, 0x10020121, 0x11000021, 0x11010022, 0x11010121, 0x11010222,
    0x11020120, 0x11020221, 0x12000221, 0x12010120, 0x12020121, 0x10001001, 0x10011101, 0x10011201,
    0x10021201, 0x11001101, 0x11001200, 0x11001202, 0x11011001, 0x11011100, 0x11011101, 0x11011102,
    0x11021001, 0x11021002, 0x11021101, 0x11021200, 0x11021202, 0x12001001, 0x12001102, 0x12001201,
    0x12011000, 0x12011002, 0x12011101, 0x12021000, 0x12021001, 0x12021201, 0x10001011, 0x10001012,
    0x10001111, 0x10001212, 0x10011011, 0x10011110, 0x10011111, 0x10011112, 0x10011211, 0x10021010,
    0x10021111, 0x10021212, 0x11001011, 0x11001110, 0x11001111, 0x11001112, 0x11001211, 0x11011010,
    0x11011011, 0x11011110, 0x11011111, 0x11011112, 0x11011210, 0x11011211, 0x11021011, 0x11021110,
    0x11021111, 0x11021112, 0x11021211, 0x12001012, 0x12001110, 0x12001111, 0x12001210, 0x12011011,
    0x12011110, 0x12011111, 0x12011112, 0x12011211, 0x12011212, 0x12021111, 0x12021210, 0x12021212,
    0x10001021, 0x10001121, 0x10001221, 0x10011120, 0x10011121, 0x10011220, 0x10011222, 0x10021021,
    0x10021120, 0x10021221, 0x11001020, 0x11001022, 0x11001121, 0x11001220, 0x11011020, 0x11011021,
    0x11011022, 0x11011121, 0x11011122, 0x11011221, 0x11021022, 0x11021121, 0x11021220, 0x12001021,
    0x12001121, 0x12001222, 0x12011120, 0x12011121, 0x12021021, 0x12021120, 0x12021122, 0x10002101,
    0x10012001, 0x10012101, 0x10012202, 0x10022101, 0x11002002, 0x11002201, 0x11012000, 0x11012101,
    0x11012200, 0x11022001, 0x11022100, 0x11022102, 0x11022201, 0x12002101, 0x12012001, 0x12012100,
    0x12012102, 0x12012201, 0x12022101, 0x10002011, 0x10002111, 0x10002112, 0x10002212, 0x10012010,
    0x10012110, 0x10012111, 0x10012210, 0x10022011, 0x10022110, 0x10022112, 0x11002010, 0x11002111,
    0x11002212, 0x11012011, 0x11012012, 0x11012110, 0x11012111, 0x11012112, 0x11012211, 0x11022010,
    0x11022012, 0x11022111, 0x11022112, 0x11022212, 0x12002112, 0x12002211, 0x12012012, 0x12012111,
    0x12012112, 0x12012210, 0x12022011, 0x12022110, 0x12022112, 0x12022211, 0x10012122, 0x11002120,
    0x11002122, 0x11002221, 0x11012121, 0x11012220, 0x11012222, 0x11022120, 0x11022221, 0x12012120,
    0x12022121, 0x10100001, 0x10100100, 0x10100101, 0x10100102, 0x10100201, 0x10110002, 0x10110101,
    0x10110202, 0x10120001, 0x10120100, 0x10120201, 0x11100000, 0x11100101, 0x11100200, 0x11110001,
    0x11110100, 0x11110101, 0x11110102, 0x11110201, 0x11120101, 0x11120200, 0x12100102, 0x12100201,
    0x12110101, 0x12110200, 0x12120000, 0x12120001, 0x12120102, 0x12120201, 0x10100111, 0x10100210,
    0x10100211, 0x10100212, 0x10110011, 0x10110110, 0x10110111, 0x10110112, 0x10110210, 0x10110211,
    0x10120010, 0x10120111, 0x10120112, 0x10120210, 0x10120212, 0x11100011, 0x11100110, 0x11100111,
    0x11100112, 0x11100211, 0x11110010, 0x11110011, 0x11110012, 0x11110110, 0x11110111, 0x11110112,
    0x11110210, 0x11110211, 0x11110212, 0x11120011, 0x11120110, 0x11120111, 0x11120112, 0x11120211,
    0x12100012, 0x12100111, 0x12110011, 0x12110110, 0x12110111, 0x12110112, 0x12110211, 0x12120010,
    0x12120111, 0x12120212, 0x10100021, 0x10100122, 0x10110022, 0x10110121, 0x10110222, 0x10120021,
    0x10120120, 0x11100022, 0x11100121, 0x11100222, 0x11110021, 0x11110120, 0x11110121, 0x11110122,
    0x11110221, 0x11120022, 0x11120121, 0x12100121, 0x12110020, 0x12110022, 0x12110121, 0x12110221,
    0x12110222, 0x12120120, 0x10101100, 0x10101101, 0x10111001, 0x10111100, 0x10111101, 0x10111102,
    0x10111200, 0x10111201, 0x10121001, 0x10121101, 0x10121200, 0x10121202, 0x11101001, 0x11101100,
    0x11101101, 0x11101102, 0x11101201, 0x11101202, 0x11111000, 0x11111001, 0x11111100, 0x11111101,
    0x11111102, 0x11111200, 0x11111201, 0x11111202, 0x11121001, 0x11121002, 0x11121100, 0x11121101,
    0x11121102, 0x11121201, 0x12101000, 0x12101200, 0x12101202, 0x12111001, 0x12111100, 0x12111101,
    0x12111102, 0x12111201, 0x12121001, 0x12121100, 0x12121101, 0x12121202, 0x10101011, 0x10101012,
    0x10101110, 0x10101111, 0x10101112, 0x10101211, 0x10111010, 0x10111011, 0x10111012, 0x10111110,
    0x10111111, 0x10111112, 0x10111211, 0x10111212, 0x10121011, 0x10121110, 0x10121111, 0x10121112,
    0x10121211, 0x11101010, 0x11101011, 0x11101012, 0x11101110, 0x11101111, 0x11101112, 0x11101210,
    0x11101211, 0x11111010, 0x11111011, 0x11111012, 0x11111110, 0x11111111, 0x11111112, 0x11111210,
    0x11111211, 0x11111212, 0x11121010, 0x11121011, 0x11121110, 0x11121111, 0x11121112, 0x11121210,
    0x11121211, 0x11121212, 0x12101011, 0x12101110, 0x12101111, 0x12101211, 0x12101212, 0x12111010,
    0x12111011, 0x12111110, 0x12111111, 0x12111112, 0x12111210, 0x12111211, 0x12121011, 0x12121110,
    0x12121111, 0x12121112, 0x12121211, 0x10101020, 0x10101021, 0x10101022, 0x10101120, 0x10101122,
    0x10101220, 0x10101221, 0x10111021, 0x10111120, 0x10111121, 0x10111220, 0x10111221, 0x10121020,
    0x10121021, 0x10121022, 0x10121120, 0x10121121, 0x10121122, 0x10121220, 0x10121221, 0x11101021,
    0x11101121, 0x11101122, 0x11101220, 0x11101221, 0x11101222, 0x11111020, 0x11111021, 0x11111022,
    0x11111120, 0x11111121, 0x11111122, 0x11111220, 0x11111221, 0x11111222, 0x11121021, 0x11121120,
    0x11121121, 0x11121221, 0x12101022, 0x12101121, 0x12101122, 0x12101220, 0x12101221, 0x12101222,
    0x12111021, 0x12111121, 0x12111222, 0x12121022, 0x12121121, 0x12121122, 0x12121220, 0x12121221,
    0x10102100, 0x10102101, 0x10102102, 0x10102201, 0x10112000, 0x10112101, 0x10112200, 0x10122001,
    0x10122202, 0x11102101, 0x11102200, 0x11102202, 0x11112001, 0x11112100, 0x11112101, 0x11112102,
    0x11112200, 0x11112201, 0x11122000, 0x11122002, 0x11122100, 0x11122101, 0x12102002, 0x12102201,
    0x12112000, 0x12112002, 0x12112101, 0x12112200, 0x12122001, 0x12122201, 0x10102011, 0x10102012,
    0x10102111, 0x10102212, 0x10112011, 0x10112110, 0x10112111, 0x10112112, 0x10112211, 0x10122111,
    0x11102011, 0x11102110, 0x11102111, 0x11102112, 0x11102211, 0x11112010, 0x11112011, 0x11112012,
    0x11112110, 0x11112111, 0x11112112, 0x11112210, 0x11112211, 0x11112212, 0x11122011, 0x11122110,
    0x11122111, 0x11122112, 0x11122211, 0x12102011, 0x12102111, 0x12102211, 0x12112011, 0x12112110,
    0x12112111, 0x12112112, 0x12112210, 0x12112211, 0x12122111, 0x10102120, 0x10102220, 0x10112121,
    0x10112222, 0x10122020, 0x10122121, 0x10122122, 0x10122221, 0x11102121, 0x11102220, 0x11102221,
    0x11112021, 0x11112121, 0x11112122, 0x11112220, 0x11112221, 0x11122022, 0x11122121, 0x11122220,
    0x11122222, 0x12102021, 0x12102222, 0x12112022, 0x12112121, 0x12112122, 0x12112220, 0x12112222,
    0x12122021, 0x10200101, 0x10210100, 0x10210102, 0x10210201, 0x10220101, 0x11200100, 0x11210000,
    0x11210101, 0x11210102, 0x11210200, 0x11210202, 0x11220001, 0x11220100, 0x11220102, 0x11220201,
    0x12200001, 0x12210102, 0x12220101, 0x10200011, 0x10200110, 0x10200112, 0x10200211, 0x10210012,
    0x10210111, 0x10220011, 0x10220012, 0x10220112, 0x10220211, 0x11200111, 0x11200211, 0x11210011,
    0x11210111, 0x11210112, 0x11210211, 0x11220111, 0x11220112, 0x11220212, 0x12200110, 0x12200212,
    0x12210012, 0x12210111, 0x12220011, 0x12220112, 0x12220211, 0x10210021, 0x10210122, 0x10210221,
    0x11200020, 0x11200021, 0x11200122, 0x11210121, 0x11210122, 0x11210220, 0x11220020, 0x12200121,
    0x12210021, 0x12210122, 0x12220121, 0x10211001, 0x10211002, 0x10211101, 0x10211102, 0x10211202,
    0x10221001, 0x10221102, 0x10221201, 0x11201000, 0x11201002, 0x11201101, 0x11201200, 0x11201202,
    0x11211001, 0x11211100, 0x11211101, 0x11211102, 0x11211201, 0x11211202, 0x11221000, 0x11221002,
    0x11221101, 0x12201100, 0x12201101, 0x12201201, 0x12211000, 0x12211002, 0x12211100, 0x12211101,
    0x12211102, 0x12211200, 0x12211202, 0x12221001, 0x12221100, 0x12221201, 0x10201111, 0x10201210,
    0x10201212, 0x10211011, 0x10211111, 0x10211112, 0x10211211, 0x11201110, 0x11201111, 0x11201112,
    0x11201211, 0x11211010, 0x11211011, 0x11211110, 0x11211111, 0x11211112, 0x11211211, 0x11221011,
    0x11221110, 0x11221111, 0x11221112, 0x11221211, 0x12201112, 0x12201211, 0x12201212, 0x12211011,
    0x12211111, 0x12211112, 0x12211211, 0x12211212, 0x12221012, 0x12221111, 0x12221112, 0x12221210,
    0x10201022, 0x10201221, 0x10211121, 0x10221020, 0x10221122, 0x10221220, 0x10221221, 0x11201020,
    0x11201121, 0x11201220, 0x11201222, 0x11211021, 0x11211120, 0x11211121, 0x11211122, 0x11211220,
    0x11211222, 0x11221020, 0x11221121, 0x11221220, 0x12201020, 0x12201022, 0x12201121, 0x12201222,
    0x12211120, 0x12211122, 0x12211220, 0x12211221, 0x12221020, 0x12221120, 0x12221122, 0x12221222,
    0x10212102, 0x10212201, 0x10222101, 0x11202001, 0x11212002, 0x11212101, 0x11212202, 0x11222001,
    0x11222201, 0x12202101, 0x12212001, 0x12212200, 0x12222102, 0x10202011, 0x10202110, 0x10212010,
    0x10212111, 0x10222011, 0x10222110, 0x10222112, 0x10222211, 0x11202010, 0x11202011, 0x11202111,
    0x11202112, 0x11202210, 0x11212011, 0x11212110, 0x11212111, 0x11212112, 0x11212211, 0x11222010,
    0x11222111, 0x11222212, 0x12202012, 0x12202110, 0x12202212, 0x12212111, 0x12222011, 0x12222110,
    0x12222111, 0x12222211, 0x10212021, 0x10212122, 0x10212220, 0x11202021, 0x11202120, 0x11202221,
    0x11212020, 0x11212121, 0x11212220, 0x11212222, 0x11222120, 0x11222121, 0x11222221, 0x12202122,
    0x12212120, 0x12212220, 0x12212222, 0x12222122, 0x20000000, 0x20000002, 0x20000200, 0x20000202,
    0x20020000, 0x20020002, 0x20020200, 0x20020202, 0x21000101, 0x21010000, 0x21010001, 0x21010100,
    0x21010102, 0x21010201, 0x21020101, 0x22000000, 0x22000002, 0x22000200, 0x22000202, 0x22010101,
    0x22020000, 0x22020002, 0x22020200, 0x22020202, 0x20000111, 0x20010011, 0x20010110, 0x20010112,
    0x20010211, 0x20020111, 0x21000011, 0x21000110, 0x21000211, 0x21010010, 0x21010012, 0x21010111,
    0x21010112, 0x21010210, 0x21010211, 0x21020110, 0x21020112, 0x21020211, 0x22000111, 0x22000211,
    0x22010110, 0x22010112, 0x22010211, 0x22020111, 0x20000020, 0x20000022, 0x20000220, 0x20000222,
    0x20010121, 0x20020020, 0x20020022, 0x20020220, 0x20020222, 0x21010021, 0x21010120, 0x21010221,
    0x21020121, 0x22000020, 0x22000022, 0x22000220, 0x22000222, 0x22010121, 0x22020020, 0x22020022,
    0x22020220, 0x22020222, 0x20011100, 0x20011201, 0x21001001, 0x21001100, 0x21011001, 0x21011101,
    0x21011202, 0x21021001, 0x21021100, 0x21021201, 0x22011100, 0x22011201, 0x20001011, 0x20001211,
    0x20011012, 0x20011111, 0x20011212, 0x20021112, 0x20021211, 0x21001010, 0x21001011, 0x21001111,
    0x21001210, 0x21011011, 0x21011110, 0x21011111, 0x21011112, 0x21011211, 0x21011212, 0x21021111,
    0x21021112, 0x21021210, 0x21021212, 0x22001011, 0x22001110, 0x22001112, 0x22001211, 0x22011010,
    0x22011012, 0x22011111, 0x22011210, 0x22021112, 0x20011021, 0x20011122, 0x20011221, 0x20021121,
    0x21001021, 0x21001120, 0x21001221, 0x21001222, 0x21011020, 0x21011121, 0x21011221, 0x21011222,
    0x21021021, 0x21021122, 0x21021222, 0x22001121, 0x22011021, 0x22011222, 0x22021120, 0x20002000,
    0x20002002, 0x20002200, 0x20002202, 0x20012101, 0x20022000, 0x20022002, 0x20022200, 0x20022202,
    0x21002001, 0x21002101, 0x21012001, 0x21012100, 0x21012201, 0x21022101, 0x21022201, 0x22002000,
    0x22002002, 0x22002200, 0x22002202, 0x22012101, 0x22022000, 0x22022002, 0x22022200, 0x22022202,
    0x20002111, 0x20002112, 0x20012011, 0x20012110, 0x20012112, 0x20022111, 0x21002011, 0x21002110,
    0x21002112, 0x21002211, 0x21012010, 0x21012012, 0x21012111, 0x21012212, 0x21022011, 0x21022110,
    0x22002111, 0x22012112, 0x22012211, 0x22022111, 0x20002020, 0x20002022, 0x20002220, 0x20002222,
    0x20012121, 0x20022020, 0x20022022, 0x20022220, 0x20022222, 0x21002121, 0x21012021, 0x21012120,
    0x21012122, 0x22002020, 0x22002022, 0x22002220, 0x22002222, 0x22012121, 0x22022020, 0x22022022,
    0x22022220, 0x22022222, 0x20100101, 0x20110001, 0x20110102, 0x20110200, 0x20110201, 0x20120101,
    0x21100001, 0x21100102, 0x21100201, 0x21110101, 0x21110200, 0x21110202, 0x21120201, 0x21120202,
    0x22100101, 0x22110001, 0x22110100, 0x22110102, 0x22110201, 0x22120101, 0x20100011, 0x20100110,
    0x20100112, 0x20100211, 0x20110010, 0x20110111, 0x20110210, 0x20110212, 0x20120011, 0x20120110,
    0x20120112, 0x20120211, 0x21100010, 0x21100111, 0x21110010, 0x21110011, 0x21110110, 0x21110111,
    0x21110112, 0x21110211, 0x21120012, 0x21120111, 0x22100110, 0x22100112, 0x22110012, 0x22110111,
    0x22110210, 0x22120011, 0x22120110, 0x22120112, 0x22120211, 0x20100121, 0x20110021, 0x20110120,
    0x20110221, 0x20120121, 0x21100120, 0x21100122, 0x21100221, 0x21110020, 0x21110022, 0x21110121,
    0x21110220, 0x21120122, 0x21120221, 0x22100121, 0x22110120, 0x22110122, 0x22120221, 0x20101001,
    0x20101100, 0x20101102, 0x20111000, 0x20111101, 0x20111200, 0x20121102, 0x21101000, 0x21101202,
    0x21111001, 0x21111100, 0x21111101, 0x21111102, 0x21111200, 0x21111201, 0x21121000, 0x21121001,
    0x21121002, 0x21121101, 0x22101100, 0x22101102, 0x22111002, 0x22111100, 0x22111101, 0x22111200,
    0x22121001, 0x22121201, 0x20101010, 0x20101111, 0x20101210, 0x20101212, 0x20111010, 0x20111011,
    0x20111110, 0x20111111, 0x20111112, 0x20111211, 0x20121011, 0x20121111, 0x20121211, 0x20121212,
    0x21101011, 0x21101110, 0x21101111, 0x21101112, 0x21101211, 0x21111010, 0x21111011, 0x21111012,
    0x21111110, 0x21111111, 0x21111112, 0x21111210, 0x21111211, 0x21111212, 0x21121011, 0x21121110,
    0x21121111, 0x21121112, 0x21121211, 0x22101011, 0x22101111, 0x22101210, 0x22111011, 0x22111012,
    0x22111110, 0x22111111, 0x22111112, 0x22111211, 0x22111212, 0x22121010, 0x22121012, 0x22121111,
    0x22121210, 0x22121212, 0x20101021, 0x20101120, 0x20111020, 0x20111121, 0x20111221, 0x20121020,
    0x20121122, 0x20121221, 0x21101121, 0x21101220, 0x21101221, 0x21111021, 0x21111022, 0x21111121,
    0x21111122, 0x21111221, 0x21121121, 0x21121220, 0x22101022, 0x22101120, 0x22101221, 0x22101222,
    0x22111022, 0x22111120, 0x22111121, 0x22121120, 0x22121122, 0x22121221, 0x20102101, 0x20112102,
    0x20112201, 0x20122101, 0x21102001, 0x21102102, 0x21112000, 0x21112002, 0x21112101, 0x21112102,
    0x21112202, 0x21122100, 0x21122101, 0x22102101, 0x22112001, 0x22112102, 0x22112201, 0x22122101,
    0x20102110, 0x20102112, 0x20102211, 0x20112010, 0x20112012, 0x20112111, 0x20112210, 0x20112212,
    0x20122010, 0x20122011, 0x20122110, 0x20122112, 0x21102010, 0x21102012, 0x21102111, 0x21102210,
    0x21102212, 0x21112011, 0x21112110, 0x21112111, 0x21112112, 0x21112211, 0x21122012, 0x21122111,
    0x21122112, 0x21122212, 0x22102011, 0x22102110, 0x22112010, 0x22112012, 0x22112111, 0x22112212,
    0x22122011, 0x22122112, 0x20102121, 0x20112121, 0x20122121, 0x21102120, 0x21102122, 0x21102221,
    0x21112020, 0x21112121, 0x21112220, 0x21122021, 0x22102121, 0x22112021, 0x22112120, 0x22112121,
    0x22112122, 0x20200000, 0x20200002, 0x20200200, 0x20200202, 0x20210101, 0x20220000, 0x20220002,
    0x20220200, 0x20220202, 0x21200101, 0x21210001, 0x21210100, 0x21210102, 0x21210201, 0x22200000,
    0x22200002, 0x22200200, 0x22200202, 0x22210101, 0x22220000, 0x22220002, 0x22220200, 0x22220202,
    0x20200111, 0x20200211, 0x20210011, 0x20210110, 0x20210112, 0x20210211, 0x20210212, 0x21200112,
    0x21200211, 0x21210011, 0x21210111, 0x21210210, 0x21210212, 0x21220011, 0x21220110, 0x22200111,
    0x22210010, 0x22210012, 0x22210112, 0x22210211, 0x20200022, 0x20200220, 0x20200222, 0x20210020,
    0x20210221, 0x20220022, 0x20220220, 0x20220222, 0x21200121, 0x21210021, 0x21210122, 0x21210221,
    0x21220121, 0x22200020, 0x22200022, 0x22200220, 0x22200222, 0x22210121, 0x22220020, 0x22220022,
    0x22220220, 0x22220222, 0x20211201, 0x20221101, 0x21201001, 0x21201100, 0x21211000, 0x21211100,
    0x21211101, 0x21211200, 0x21211202, 0x21221001, 0x21221101, 0x21221102, 0x21221200, 0x21221201,
    0x22201101, 0x20201112, 0x20201211, 0x20211010, 0x20211012, 0x20211111, 0x20211210, 0x20221112,
    0x20221211, 0x21201012, 0x21201111, 0x21211011, 0x21211110, 0x21211111, 0x21211112, 0x21211211,
    0x21221111, 0x21221212, 0x22201011, 0x22201110, 0x22201111, 0x22201112, 0x22201211, 0x22211012,
    0x22211111, 0x22211210, 0x20201121, 0x20211021, 0x20211122, 0x20211222, 0x20221021, 0x20221121,
    0x21201120, 0x21201122, 0x21201222, 0x21211022, 0x21211121, 0x21211122, 0x21211220, 0x21221020,
    0x21221022, 0x22201122, 0x22211020, 0x22211121, 0x22211122, 0x22211221, 0x22221021, 0x22221120,
    0x22221122, 0x20202000, 0x20202002, 0x20202200, 0x20202202, 0x20222000, 0x20222002, 0x20222200,
    0x20222202, 0x21212001, 0x21212100, 0x21212102, 0x21212201, 0x22202000, 0x22202002, 0x22202200,
    0x22202202, 0x22212101, 0x22222000, 0x22222002, 0x22222200, 0x22222202, 0x20202111, 0x20212110,
    0x20212211, 0x20222011, 0x20222111, 0x21202011, 0x21212010, 0x21212111, 0x21212212, 0x21222011,
    0x21222112, 0x21222211, 0x22212010, 0x22212112, 0x20202020, 0x20202022, 0x20202220, 0x20202222,
    0x20222020, 0x20222022, 0x20222220, 0x20222222, 0x21212021, 0x21212120, 0x21212122, 0x22202020,
    0x22202022, 0x22202220, 0x22202222, 0x22212121, 0x22222020, 0x22222022, 0x22222220, 0x22222222,
GGML_TABLE_END()
#endif

#endif // GGML_COMMON_IMPL
#endif // GGML_COMMON_IMPL
