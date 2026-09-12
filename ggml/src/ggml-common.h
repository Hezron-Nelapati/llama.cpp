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
#define NEURON_VQ_SBB   4                               /* bits per sub-block index     */
#define NEURON_VQ_SBBY  (NEURON_VQ_NSB * NEURON_VQ_SBB / 8)   /* 4 bytes                */

#define NEURON_V_BLOCK(SFX)                                                       \
    typedef struct {                                                              \
        ggml_half d;                                 /* block scale            */ \
        uint8_t   sb[NEURON_VQ_SBBY];                /* 8 x 4-bit multipliers  */ \
        uint8_t   qs[(QK_NEURON / 2) * NEURON_VQ##SFX##_BITS / 8];                \
    } block_neuron_v##SFX;                                                        \
    static_assert(sizeof(block_neuron_v##SFX) == sizeof(ggml_half) +              \
                  NEURON_VQ_SBBY + (QK_NEURON / 2) * NEURON_VQ##SFX##_BITS / 8,   \
                  "wrong block_neuron_v" #SFX " size/padding");
// Pair p lives in sub-block s = p / (NEURON_VQ_SBV/2) = p/8, whose 4-bit multiplier index is
// nibble s of sb[]:  (sb[s >> 1] >> ((s & 1) * 4)) & 0xF.  Spelled out at each use rather
// than shared as a helper, because this header is also compiled as Metal, where a pointer
// parameter needs an address-space qualifier and a generic one will not compile.

NEURON_V_BLOCK(4)
NEURON_V_BLOCK(5)

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

// Sub-block scale multipliers, ratios of sub-block absmax to block absmax. Level 0 is
// pinned to exactly 1.0 because by construction some sub-block attains the block
// absmax -- that level is forced by the definition, not chosen. The other 15 are 1-D
// Lloyd levels on the empirical ratio distribution, fitted in LOG space (constant
// relative resolution is what a positive quantity of unknown scale wants). Mean
// relative error of the grid: 1.62%. See experiments/fit_subscale.py.
GGML_TABLE_BEGIN(float, kNeuronVQSB, 16)
    +1.0000000f, +0.9291289f, +0.8907700f, +0.8449818f, +0.8070589f, +0.7777284f, +0.7499992f, +0.7236124f, +0.6915383f, +0.6621733f, +0.6293483f, +0.5985167f, +0.5515646f, +0.4777620f, +0.3698162f, +0.1773243f,
GGML_TABLE_END()

// The neuron_v4 codebook: 256 points in the pair plane, interleaved x,y. Same fit as
// kNeuronVQ5 below -- 800k absmax-normalised pairs with their sign orbit, drawn from all
// 252 layer weight tensors of Qwen3-4B, ordered by radius. See experiments/fit_vq.py.
// Held out from Qwen3-0.6B, the model it is first measured on.
GGML_TABLE_BEGIN(float, kNeuronVQ4, 512)
    +0.0057626f, +0.0044268f, -0.0214538f, -0.0190144f, +0.0145248f, -0.0255991f, -0.0211066f, +0.0232543f,
    +0.0186425f, +0.0369755f, +0.0426181f, +0.0063373f, -0.0530573f, +0.0016880f, -0.0190835f, -0.0622703f,
    +0.0557096f, -0.0359766f, -0.0178597f, +0.0693282f, +0.0233867f, -0.0680010f, -0.0628667f, -0.0494510f,
    -0.0702716f, +0.0479250f, +0.0328734f, +0.0829003f, +0.0769726f, +0.0531597f, +0.0947416f, +0.0009524f,
    -0.0994444f, -0.0115470f, +0.0995805f, -0.0614066f, +0.0582026f, -0.1022784f, -0.0053520f, -0.1189543f,
    -0.0611548f, +0.1068663f, -0.0635636f, -0.1071797f, +0.0007464f, +0.1350103f, -0.1332805f, +0.0313209f,
    -0.1200738f, -0.0726600f, +0.0715326f, +0.1252549f, +0.1469608f, +0.0384544f, +0.1517835f, -0.0263306f,
    -0.1328241f, +0.0875041f, +0.1290981f, +0.1001902f, -0.1655110f, -0.0290498f, +0.0501820f, -0.1644160f,
    +0.1114988f, -0.1389140f, +0.1537405f, -0.0916426f, -0.0091044f, -0.1837607f, -0.1123576f, +0.1497755f,
    -0.1276096f, -0.1372469f, -0.0568442f, +0.1791151f, -0.0730860f, -0.1740720f, +0.0555272f, +0.1930844f,
    -0.2100234f, +0.0316810f, -0.0106354f, +0.2190746f, +0.2199201f, +0.0058991f, +0.1248830f, +0.1835917f,
    -0.1978126f, -0.1019501f, +0.2184351f, -0.0700175f, +0.2153190f, +0.0811556f, -0.2164776f, +0.0965412f,
    +0.1845541f, +0.1506334f, -0.1777745f, +0.1601631f, +0.1140815f, -0.2143037f, -0.2424235f, -0.0381868f,
    +0.0385223f, -0.2439324f, +0.1896924f, -0.1589199f, -0.1793126f, -0.1821845f, -0.0443611f, -0.2519398f,
    -0.1239647f, +0.2316648f, -0.1228528f, -0.2405226f, +0.0438624f, +0.2712659f, -0.0652020f, +0.2852256f,
    +0.1244673f, +0.2676184f, -0.2950341f, +0.0273899f, +0.2696899f, -0.1284152f, +0.2990767f, +0.0301195f,
    +0.2981266f, -0.0500861f, -0.2887913f, -0.1088452f, +0.2593091f, +0.1690071f, +0.1917662f, -0.2434739f,
    +0.1079942f, -0.2931257f, -0.2574351f, -0.1804210f, -0.2618698f, +0.1756927f, -0.3023084f, +0.0995006f,
    +0.2086073f, +0.2416400f, +0.0073709f, -0.3233559f, -0.2078782f, +0.2482826f, +0.3048138f, +0.1103409f,
    -0.2117305f, -0.2615575f, +0.0033065f, +0.3368750f, -0.3363468f, -0.0324012f, -0.0862643f, -0.3300035f,
    +0.2712704f, -0.2098846f, +0.0879281f, -0.3572219f, -0.1555888f, +0.3341491f, +0.0917703f, +0.3575921f,
    +0.3757751f, -0.0295002f, -0.1654003f, -0.3396166f, +0.1869157f, -0.3320164f, +0.3611255f, -0.1319291f,
    -0.3844607f, +0.0493543f, +0.1880191f, +0.3413859f, -0.0709258f, +0.3854822f, +0.3865949f, +0.0734640f,
    -0.3787864f, -0.1076584f, +0.3059691f, +0.2503570f, -0.3673348f, +0.1559186f, -0.3546399f, -0.1951927f,
    -0.3264266f, +0.2433908f, +0.2735484f, -0.3036141f, -0.0349199f, -0.4094127f, +0.3698448f, +0.1888191f,
    -0.3094948f, -0.2771509f, -0.2619871f, +0.3293481f, +0.3601756f, -0.2349430f, +0.0576236f, -0.4279039f,
    -0.2515938f, -0.3550692f, +0.0281018f, +0.4441960f, +0.2830516f, +0.3436011f, -0.4469417f, -0.0441303f,
    +0.1636098f, -0.4214845f, +0.4628515f, -0.0057115f, -0.1214605f, -0.4503644f, +0.4636011f, -0.1018802f,
    -0.1521759f, +0.4502239f, +0.1278307f, +0.4579072f, -0.4633945f, +0.1221440f, +0.2694167f, -0.4003848f,
    +0.4608220f, +0.1581202f, -0.4684304f, -0.1405503f, -0.4910217f, +0.0311214f, -0.3684607f, +0.3293264f,
    +0.3646441f, -0.3381592f, -0.0609650f, +0.4938903f, +0.3850394f, +0.3187920f, +0.4590299f, -0.2025661f,
    -0.4489105f, +0.2263456f, -0.2387448f, +0.4424514f, +0.2316587f, +0.4523860f, -0.2226101f, -0.4593225f,
    +0.5068619f, +0.0731669f, -0.4480833f, -0.2491689f, -0.0058650f, -0.5174976f, -0.3305465f, -0.4155231f,
    -0.4035956f, -0.3466004f, +0.1152056f, -0.5208129f, +0.4837835f, +0.2696416f, -0.3344944f, +0.4449748f,
    +0.4642464f, -0.3152956f, +0.2372436f, -0.5087142f, +0.5678191f, -0.0240586f, -0.5658138f, -0.0543144f,
    +0.3462698f, +0.4524627f, +0.0181633f, +0.5711977f, -0.5664366f, +0.1108579f, -0.1144841f, -0.5678639f,
    +0.3641351f, -0.4540052f, -0.4836108f, +0.3319443f, -0.5566355f, -0.1880289f, -0.1717742f, +0.5654368f,
    +0.5772349f, -0.1348214f, +0.5715373f, +0.1711939f, +0.1250473f, +0.5874835f, -0.5625407f, +0.2351704f,
    +0.4570713f, +0.4048346f, -0.2319877f, -0.5667601f, +0.2291959f, +0.5751781f, +0.5707119f, -0.2403689f,
    -0.5328978f, -0.3327345f, -0.4460737f, +0.4506406f, +0.0821738f, -0.6287856f, -0.0891635f, +0.6291638f,
    -0.3614949f, -0.5242452f, -0.6390609f, +0.0352972f, +0.6382751f, +0.0732587f, -0.0335344f, -0.6432490f,
    -0.3126225f, +0.5729958f, +0.4827550f, -0.4484060f, -0.6505882f, -0.1147472f, +0.2118112f, -0.6280847f,
    -0.4803496f, -0.4672040f, +0.3507559f, +0.5762386f, -0.6548862f, +0.1821485f, +0.6767614f, -0.0663713f,
    +0.3535556f, -0.5859551f, +0.6321014f, +0.2644607f, -0.1816615f, -0.6633403f, +0.5897742f, -0.3564068f,
    +0.0232533f, +0.6894779f, -0.6401248f, -0.2696701f, +0.5830980f, +0.3841939f, -0.2394880f, +0.6628857f,
    -0.6282018f, +0.3272039f, -0.3278056f, -0.6375607f, +0.4978487f, +0.5299426f, -0.4437687f, +0.5762683f,
    -0.5782290f, +0.4455145f, +0.6939179f, -0.2381807f, +0.2625571f, +0.7056395f, -0.7555294f, -0.0064624f,
    -0.6202692f, -0.4319443f, +0.7412288f, +0.1766165f, +0.4945303f, -0.5860231f, +0.1393900f, -0.7548975f,
    -0.1344397f, +0.7571619f, +0.1444689f, +0.7559181f, -0.0006062f, -0.7702098f, -0.7704998f, +0.1190163f,
    +0.7794210f, +0.0318084f, -0.7703736f, -0.1288347f, -0.1461680f, -0.7752874f, +0.4342841f, +0.6635930f,
    -0.4625991f, -0.6545221f, +0.2883561f, -0.7485584f, +0.7908400f, -0.1432885f, -0.7653355f, -0.2499415f,
    -0.7680839f, +0.2725690f, +0.6468763f, -0.4966822f, -0.4379107f, +0.7010582f, +0.0062618f, +0.8287978f,
    -0.5853284f, +0.5890075f, -0.3133012f, +0.7703444f, +0.7628065f, +0.3459839f, -0.3262824f, -0.7733732f,
    -0.5928006f, -0.6094415f, -0.7595689f, -0.3847171f, +0.7616984f, -0.3808340f, -0.7312986f, +0.4386828f,
    +0.4385181f, -0.7330917f, +0.6844617f, +0.5144737f, +0.4051889f, +0.7990699f, +0.6087682f, +0.6821080f,
    +0.6414750f, -0.6672287f, -0.7467205f, -0.5614088f, +0.0009756f, -0.9507070f, -0.9514757f, +0.0429310f,
    -0.1595821f, +0.9413828f, -0.9502523f, -0.1022445f, +0.9541913f, +0.0780118f, +0.1579985f, +0.9465021f,
    -0.7355973f, +0.6166306f, +0.1703659f, -0.9447811f, +0.9587864f, -0.0979200f, -0.1698965f, -0.9500563f,
    -0.5948787f, +0.7626244f, -0.9518770f, +0.1972618f, +0.9440097f, +0.2512488f, -0.9469680f, -0.2611949f,
    -0.0041065f, +0.9829591f, +0.9487926f, -0.2970740f, -0.3227750f, +0.9501563f, +0.3284940f, +0.9485107f,
    -0.9357410f, +0.3743182f, +0.3586914f, -0.9469596f, -0.3503438f, -0.9537799f, -0.5308077f, -0.8704176f,
    +0.9233665f, +0.4767434f, -0.9441964f, -0.4509374f, +0.9011447f, -0.5460813f, -0.5125322f, +0.9371253f,
    +0.5704882f, -0.9071597f, -0.7248828f, -0.8214039f, +0.5990103f, +0.9248255f, -0.9432954f, +0.5812419f,
    +0.8524562f, +0.7505783f, -0.9236146f, -0.7139006f, +0.8358065f, -0.8244682f, -0.8296225f, +0.8701367f,
GGML_TABLE_END()

// The neuron_v5 codebook: 1024 points in the pair plane, interleaved x,y.
// Fitted by Lloyd iteration on 800k absmax-normalised weight pairs drawn from all 252
// layer weight tensors of Qwen3-4B, carrying each pair's sign orbit because the weight
// distribution is symmetric about zero. Ordered by radius. See experiments/fit_vq5.py.
// Held out from Qwen3-0.6B, the model it is first measured on.
GGML_TABLE_BEGIN(float, kNeuronVQ5, 2048)
    -0.0018805f, -0.0072195f, +0.0092629f, +0.0080770f, -0.0082736f, +0.0093666f, +0.0134715f, -0.0110996f,
    -0.0206845f, -0.0059771f, -0.0031017f, -0.0259502f, +0.0003915f, +0.0279823f, +0.0296161f, +0.0058596f,
    -0.0252703f, +0.0199570f, +0.0235963f, +0.0292391f, -0.0263167f, -0.0294343f, +0.0183450f, -0.0359674f,
    +0.0363676f, -0.0180740f, -0.0439143f, -0.0052054f, -0.0133546f, +0.0495353f, -0.0072664f, -0.0526301f,
    -0.0379834f, +0.0414966f, +0.0569250f, -0.0018992f, -0.0543551f, +0.0176070f, +0.0499812f, +0.0285072f,
    +0.0142184f, +0.0559505f, +0.0417339f, -0.0452886f, -0.0531488f, -0.0332646f, -0.0380874f, -0.0573879f,
    +0.0165751f, -0.0699258f, +0.0461193f, +0.0572170f, +0.0682062f, -0.0329375f, -0.0757532f, -0.0085871f,
    -0.0074214f, +0.0806874f, -0.0657010f, +0.0482726f, -0.0383511f, +0.0723533f, +0.0797139f, +0.0190579f,
    -0.0206220f, -0.0808535f, +0.0469144f, -0.0760104f, -0.0871774f, +0.0237662f, +0.0897393f, -0.0131742f,
    +0.0267662f, +0.0876693f, -0.0822984f, -0.0431769f, +0.0839493f, +0.0512844f, -0.0660344f, -0.0739843f,
    +0.0766092f, -0.0656294f, +0.0065756f, -0.1070304f, +0.0687770f, +0.0831073f, -0.0696671f, +0.0837857f,
    -0.1114920f, +0.0012615f, +0.1111598f, +0.0152243f, -0.0353173f, +0.1069378f, -0.0021894f, +0.1149916f,
    +0.0362154f, -0.1096750f, -0.0987070f, +0.0605092f, -0.0236938f, -0.1135032f, -0.1122507f, -0.0303282f,
    +0.1078690f, -0.0480210f, -0.0551807f, -0.1083593f, +0.0556255f, +0.1115669f, +0.1246174f, -0.0192524f,
    -0.1082608f, -0.0676853f, -0.1241384f, +0.0360763f, +0.0714549f, -0.1080419f, +0.1222120f, +0.0495774f,
    +0.0291160f, +0.1320510f, +0.1070983f, +0.0856424f, -0.0906633f, -0.1055671f, +0.1103686f, -0.0854793f,
    -0.0686877f, +0.1220221f, -0.1051391f, +0.0992379f, +0.1442120f, +0.0161082f, +0.0036418f, -0.1479530f,
    -0.1483635f, -0.0203791f, -0.0343552f, -0.1483419f, -0.0358609f, +0.1487308f, -0.1528842f, +0.0157979f,
    +0.0398902f, -0.1485204f, -0.1358441f, +0.0744762f, +0.0001146f, +0.1555479f, +0.0917217f, +0.1272145f,
    +0.1512955f, -0.0516906f, +0.1592948f, -0.0179563f, -0.1507628f, -0.0575841f, +0.0632694f, +0.1519680f,
    +0.1090160f, -0.1242332f, -0.1355159f, -0.0973411f, -0.0725866f, -0.1512730f, +0.1602629f, +0.0500390f,
    +0.0764638f, -0.1499273f, +0.1486211f, +0.0827834f, +0.1480142f, -0.0853971f, +0.1261082f, +0.1212862f,
    -0.1099719f, -0.1366369f, -0.1658345f, +0.0571196f, -0.1092811f, +0.1387119f, -0.0763522f, +0.1597963f,
    +0.1773127f, +0.0175278f, -0.1465980f, +0.1112130f, +0.0368782f, +0.1817315f, -0.0152586f, -0.1875663f,
    +0.0210560f, -0.1879951f, -0.1885014f, -0.0196870f, +0.1487623f, -0.1243295f, -0.0413597f, +0.1897873f,
    -0.1935703f, +0.0225702f, +0.1949944f, -0.0158293f, -0.0008487f, +0.1963957f, +0.0586122f, -0.1885777f,
    -0.0523696f, -0.1908504f, +0.1128017f, -0.1634652f, +0.1628459f, +0.1151428f, +0.1941611f, -0.0480541f,
    -0.1473841f, -0.1370453f, -0.1940099f, -0.0594812f, +0.1985498f, +0.0502563f, +0.0933180f, +0.1823957f,
    +0.1893343f, -0.0785428f, +0.1251886f, +0.1634477f, +0.1889003f, +0.0837009f, -0.1824086f, -0.0974173f,
    -0.1499969f, +0.1499895f, -0.0835412f, +0.1959407f, -0.0911956f, -0.1931986f, -0.1221860f, -0.1754260f,
    +0.1824262f, -0.1117093f, -0.1859119f, +0.1059435f, -0.2042380f, +0.0664355f, +0.1494364f, -0.1582530f,
    +0.2177809f, +0.0137867f, -0.1209388f, +0.1830389f, +0.0967977f, -0.1971507f, +0.0646190f, +0.2132175f,
    +0.1654110f, +0.1513841f, -0.1819545f, -0.1387947f, -0.0011223f, -0.2347102f, -0.2350374f, -0.0145218f,
    +0.2034317f, +0.1186942f, +0.0330209f, -0.2342102f, -0.0210824f, +0.2360205f, +0.0230649f, +0.2364630f,
    -0.2373814f, +0.0276817f, +0.2373446f, -0.0280168f, -0.0373836f, -0.2371652f, +0.1897743f, -0.1485868f,
    -0.1894970f, +0.1500632f, -0.2369317f, -0.0574707f, +0.1521772f, -0.1906361f, -0.0629671f, +0.2359642f,
    +0.2393489f, +0.0502667f, -0.0755217f, -0.2326303f, -0.1631867f, -0.1824371f, +0.2353699f, -0.0682637f,
    +0.0676301f, -0.2356658f, +0.2191130f, -0.1111403f, +0.1423971f, +0.2056332f, +0.2352954f, +0.0884917f,
    -0.1647558f, +0.1907412f, -0.2441931f, +0.0720471f, +0.1356980f, -0.2164353f, -0.2336266f, -0.1051088f,
    -0.2328758f, +0.1108086f, +0.1079623f, +0.2344663f, +0.2590944f, +0.0098188f, -0.1072929f, +0.2364601f,
    -0.1277389f, -0.2278182f, +0.1052316f, -0.2407433f, +0.2106716f, +0.1587119f, -0.2160877f, -0.1514257f,
    +0.1857869f, +0.1902219f, +0.1857802f, -0.1902754f, +0.0655014f, +0.2600379f, +0.2527018f, -0.1057718f,
    +0.2427014f, +0.1322200f, +0.0222277f, +0.2758750f, -0.2769612f, +0.0050217f, -0.1516891f, +0.2333369f,
    -0.2762588f, -0.0366907f, -0.0290303f, +0.2775942f, -0.1028572f, -0.2596339f, -0.0038240f, -0.2797669f,
    -0.2370930f, +0.1487345f, +0.0337399f, -0.2782563f, -0.2763579f, +0.0494368f, +0.2388393f, -0.1477207f,
    +0.2187264f, -0.1765507f, -0.2065141f, -0.1931589f, +0.2797085f, +0.0447934f, -0.2156321f, +0.1837704f,
    +0.2833426f, -0.0262975f, -0.0412276f, -0.2818670f, +0.2808383f, -0.0694278f, -0.0805148f, +0.2780754f,
    -0.1685196f, -0.2354240f, +0.0718949f, -0.2816479f, -0.2811721f, -0.0788484f, +0.2797949f, +0.0862143f,
    +0.1755796f, -0.2348469f, +0.1768102f, +0.2348745f, -0.0763293f, -0.2848561f, -0.2783873f, +0.1053826f,
    +0.1055186f, +0.2787454f, +0.1447305f, -0.2625182f, -0.2543128f, -0.1603969f, -0.2060426f, +0.2192806f,
    -0.2766673f, -0.1187155f, +0.0066620f, +0.3028093f, +0.2388668f, +0.1873714f, +0.1511824f, +0.2651327f,
    +0.2174441f, -0.2150318f, -0.3064633f, -0.0115908f, +0.3062795f, +0.0166529f, -0.1325469f, +0.2770081f,
    -0.3059998f, +0.0306179f, +0.0501750f, +0.3042948f, +0.1104710f, -0.2880047f, +0.2888930f, -0.1116104f,
    -0.3061880f, -0.0506967f, -0.1371552f, -0.2805476f, -0.3036079f, +0.0763336f, +0.2929337f, +0.1164202f,
    +0.0312793f, -0.3155209f, -0.0284354f, +0.3168915f, +0.2164518f, +0.2335496f, -0.0134861f, -0.3186152f,
    -0.2133791f, -0.2376146f, +0.2809268f, -0.1520670f, -0.0627190f, +0.3137878f, -0.2833059f, +0.1525887f,
    +0.2840004f, +0.1517739f, +0.3170211f, -0.0568398f, +0.3169411f, +0.0664236f, +0.0867908f, +0.3121874f,
    +0.2608151f, -0.1924173f, -0.2613138f, +0.1924973f, -0.1811071f, +0.2695011f, -0.1104474f, +0.3071789f,
    +0.2143699f, -0.2467830f, -0.1067995f, -0.3097813f, +0.0718038f, -0.3205324f, -0.0571901f, -0.3249946f,
    +0.3299123f, -0.0144492f, -0.1722672f, -0.2833112f, -0.3132467f, -0.1092869f, -0.2530659f, -0.2149351f,
    -0.2974317f, -0.1512114f, +0.1811442f, -0.2804385f, +0.0158305f, +0.3353253f, -0.3361216f, +0.0104859f,
    -0.3127467f, +0.1238850f, -0.2157625f, +0.2594660f, -0.3373748f, -0.0300326f, +0.1334927f, +0.3119060f,
    -0.3313901f, -0.0734949f, +0.2780642f, +0.1949692f, -0.3355257f, +0.0554252f, +0.3283093f, -0.1009272f,
    -0.2872283f, -0.1929352f, -0.0918212f, +0.3347070f, +0.1503204f, -0.3143016f, +0.2096927f, +0.2792762f,
    -0.1553281f, +0.3136582f, -0.2143343f, -0.2768021f, -0.1412682f, -0.3224359f, -0.2612345f, +0.2365416f,
    +0.2612497f, +0.2372938f, +0.3531041f, +0.0299638f, +0.2639285f, -0.2369246f, +0.1065392f, -0.3392859f,
    -0.3419973f, +0.1031828f, +0.0586050f, +0.3531827f, +0.3407118f, +0.1107189f, +0.0108337f, -0.3582825f,
    +0.2199774f, -0.2832126f, +0.3244070f, -0.1546541f, +0.1831166f, +0.3110082f, -0.0308854f, +0.3598592f,
    -0.3111610f, +0.1883397f, -0.0359695f, -0.3623027f, +0.3598453f, -0.0583328f, +0.3044231f, -0.2006978f,
    -0.0936507f, -0.3533248f, +0.0605035f, -0.3609940f, +0.3138762f, +0.1891615f, +0.3608403f, +0.0712691f,
    -0.3677205f, +0.0370186f, -0.2613638f, -0.2618106f, -0.3420165f, -0.1416104f, -0.3700702f, -0.0097983f,
    +0.1065242f, +0.3547383f, +0.3381512f, +0.1517273f, +0.0154724f, +0.3704227f, -0.2034960f, +0.3123404f,
    -0.0750364f, +0.3651811f, -0.3687588f, -0.0559451f, -0.1230587f, +0.3529225f, -0.1817336f, -0.3273330f,
    +0.1972007f, -0.3184888f, -0.2167045f, -0.3073103f, +0.3762743f, -0.0088587f, -0.3676665f, +0.0806140f,
    -0.3618372f, -0.1051863f, -0.3443095f, +0.1560106f, -0.3324386f, -0.1812395f, -0.2455031f, +0.2916665f,
    +0.3633054f, -0.1192511f, +0.2646187f, +0.2771302f, -0.3032799f, +0.2361542f, +0.3030944f, +0.2372083f,
    -0.3057292f, -0.2351598f, +0.1540566f, +0.3548118f, +0.1394430f, -0.3610344f, +0.2633805f, -0.2844426f,
    -0.1470960f, -0.3615771f, -0.1542592f, +0.3630041f, -0.1863661f, +0.3500284f, +0.1765184f, -0.3595762f,
    +0.2299367f, +0.3282134f, +0.0283160f, -0.4002609f, -0.0669831f, -0.3974978f, -0.2877642f, +0.2835869f,
    -0.0189187f, -0.4037695f, +0.3480803f, -0.2055447f, -0.0053521f, +0.4043902f, -0.3421265f, +0.2160870f,
    +0.0843059f, +0.3960417f, -0.4050529f, +0.0149727f, -0.2652515f, -0.3066414f, -0.0494516f, +0.4026425f,
    +0.4052511f, +0.0273796f, +0.1067313f, -0.3921329f, +0.2429043f, -0.3264115f, +0.0418036f, +0.4047797f,
    -0.4062675f, -0.0279455f, -0.3864332f, +0.1288482f, +0.4053499f, -0.0406600f, +0.3710392f, -0.1690891f,
    -0.4040924f, +0.0548738f, +0.3233054f, -0.2496491f, +0.2653389f, +0.3106394f, +0.3991670f, -0.0904312f,
    -0.2193104f, -0.3459061f, +0.3918003f, +0.1229239f, +0.0711660f, -0.4051805f, +0.4038996f, +0.0782186f,
    +0.3589622f, +0.2010392f, -0.4052957f, -0.0719746f, +0.3377365f, +0.2367556f, +0.1973397f, +0.3623777f,
    -0.3524171f, -0.2174563f, -0.0953171f, +0.4034822f, -0.1183734f, -0.3974985f, -0.4047409f, +0.0938331f,
    +0.2135732f, -0.3597431f, -0.3059694f, -0.2859196f, -0.3878762f, -0.1594318f, +0.3028164f, -0.2924154f,
    -0.2494328f, +0.3400802f, -0.1928544f, -0.3758163f, -0.4058336f, -0.1183917f, -0.3745042f, +0.1983486f,
    +0.1263471f, +0.4046536f, +0.3919731f, +0.1619219f, -0.2175488f, +0.3647777f, +0.3178461f, +0.2829498f,
    -0.3343800f, +0.2635502f, -0.1446462f, +0.4025621f, +0.4044124f, -0.1437486f, -0.3978090f, +0.1673919f,
    +0.1520375f, -0.4046269f, -0.2591951f, -0.3461422f, +0.1701570f, +0.4029248f, -0.1670242f, -0.4076820f,
    -0.2973386f, +0.3252532f, -0.3540020f, -0.2641232f, -0.3966221f, -0.2047793f, -0.1903732f, +0.4038480f,
    +0.2763365f, -0.3510044f, +0.4483505f, -0.0081949f, +0.4039656f, +0.1974034f, +0.3672153f, +0.2604305f,
    +0.0082163f, -0.4502208f, -0.0237328f, +0.4498117f, +0.3806010f, -0.2412583f, +0.2511635f, +0.3743776f,
    -0.4507547f, +0.0090158f, -0.0380256f, -0.4499483f, +0.0239597f, +0.4513343f, +0.2017812f, -0.4046811f,
    -0.3722205f, +0.2574683f, -0.4511969f, -0.0395201f, +0.0548595f, -0.4498829f, +0.2443404f, -0.3817471f,
    +0.3055969f, -0.3347402f, +0.3147701f, +0.3262749f, -0.4503451f, +0.0568456f, +0.4514706f, +0.0473472f,
    +0.4503207f, -0.0625809f, +0.4059466f, -0.2053129f, -0.0717087f, +0.4492661f, +0.0719051f, +0.4494960f,
    +0.2852160f, +0.3552799f, -0.0831782f, -0.4491918f, +0.0989978f, -0.4487836f, -0.2975589f, -0.3515765f,
    -0.2384280f, -0.3944816f, -0.4518065f, -0.0913157f, +0.4494698f, +0.1031025f, -0.4498482f, +0.1066900f,
    -0.2383283f, +0.3975162f, +0.2263504f, +0.4051987f, +0.4499153f, -0.1149020f, -0.1191619f, +0.4495348f,
    +0.1192065f, +0.4506405f, -0.4065149f, +0.2287259f, +0.4022813f, +0.2374513f, -0.1300736f, -0.4487397f,
    +0.3532366f, -0.3067639f, -0.3535209f, +0.3080547f, +0.1392004f, -0.4505251f, -0.4494591f, -0.1441517f,
    -0.3480797f, -0.3198564f, +0.4482623f, +0.1510985f, -0.4491673f, +0.1533280f, +0.3617391f, +0.3096552f,
    -0.2847914f, +0.3826970f, +0.3840544f, -0.2830561f, +0.4504096f, -0.1677244f, -0.1678301f, +0.4504848f,
    +0.1677372f, +0.4507388f, -0.2065004f, -0.4364073f, -0.4033464f, -0.2707478f, +0.1851996f, -0.4506662f,
    -0.4474718f, +0.1969477f, -0.4492640f, -0.1941392f, -0.3313742f, +0.3602462f, +0.3390442f, -0.3542042f,
    +0.4514387f, +0.1930751f, +0.4038974f, +0.2884648f, +0.3052793f, -0.3915655f, +0.3425831f, +0.3601258f,
    +0.2138401f, +0.4503448f, +0.4504233f, -0.2138266f, -0.2846736f, -0.4094813f, -0.4029053f, +0.2940746f,
    -0.2147762f, +0.4503737f, +0.4995525f, +0.0122580f, +0.0124101f, -0.4999977f, -0.5000026f, +0.0155520f,
    +0.4992857f, -0.0348464f, +0.0245324f, +0.5000007f, -0.0237602f, +0.5001525f, -0.5000001f, -0.0437284f,
    -0.0557906f, -0.5000002f, +0.2706210f, -0.4244411f, -0.3255657f, -0.3840202f, +0.4999985f, +0.0626443f,
    +0.0697249f, +0.4999970f, -0.0725989f, +0.5000017f, -0.5000009f, +0.0775773f, +0.0799804f, -0.5000018f,
    +0.3113941f, +0.3996671f, +0.4999990f, -0.0840117f, +0.4497886f, +0.2370388f, -0.1725707f, -0.4782799f,
    +0.2366410f, -0.4517672f, -0.4999986f, -0.1010227f, -0.4507066f, +0.2393786f, +0.4999985f, +0.1155179f,
    -0.4486980f, -0.2492085f, +0.1174987f, +0.5000028f, -0.1198975f, -0.5000030f, +0.4422084f, -0.2647158f,
    -0.1251574f, +0.5000025f, +0.2650884f, +0.4430561f, -0.2526416f, -0.4502895f, -0.3952788f, -0.3330652f,
    +0.5000057f, -0.1334084f, -0.2642249f, +0.4453813f, -0.5000042f, +0.1380906f, +0.1501378f, -0.5000012f,
    +0.3982421f, +0.3426275f, -0.5000005f, -0.1625939f, +0.1672386f, +0.4999982f, -0.3871054f, +0.3590775f,
    +0.5000015f, +0.1728945f, -0.3646961f, -0.3851188f, +0.4498143f, +0.2833081f, +0.4236090f, -0.3212375f,
    -0.1817539f, +0.5000005f, +0.4999994f, -0.1860798f, -0.4498700f, +0.2884808f, +0.3643521f, +0.3948236f,
    -0.3102119f, +0.4390827f, -0.4999992f, +0.2041166f, +0.3928336f, -0.3722140f, +0.3536979f, -0.4102022f,
    +0.2140123f, +0.5000054f, +0.2202861f, -0.5006570f, -0.4470460f, -0.3154486f, -0.2244461f, -0.4992321f,
    -0.3585547f, +0.4148067f, -0.5000028f, -0.2267304f, +0.3141229f, +0.4503593f, +0.5000021f, +0.2299279f,
    +0.3160659f, -0.4506181f, +0.5509496f, +0.0054722f, +0.0142341f, -0.5515216f, +0.0057520f, +0.5518382f,
    -0.5512340f, -0.0297022f, -0.5519950f, +0.0190881f, -0.3267028f, -0.4458355f, -0.2362980f, +0.5000078f,
    -0.0451239f, -0.5514392f, -0.0517104f, +0.5508893f, +0.5014454f, -0.2356690f, +0.0600633f, +0.5510864f,
    +0.5519158f, +0.0540503f, +0.5534794f, -0.0414591f, +0.4496651f, +0.3262558f, -0.5519924f, +0.0689667f,
    +0.0742517f, -0.5513812f, -0.5522955f, -0.0759439f, +0.5509806f, -0.0928515f, -0.4445897f, +0.3406152f,
    -0.1083620f, +0.5506442f, +0.5528375f, +0.1040846f, -0.1103776f, -0.5520198f, -0.4073035f, -0.3890223f,
    -0.5520327f, -0.1206732f, +0.1167218f, +0.5529709f, -0.5520337f, +0.1218396f, -0.2813115f, -0.4913473f,
    +0.2658994f, +0.5000015f, +0.4067383f, +0.3962774f, +0.1304289f, -0.5530702f, +0.4944464f, -0.2862261f,
    -0.5000008f, +0.2766890f, +0.5529023f, -0.1481698f, +0.3584712f, +0.4481652f, +0.5520008f, +0.1605246f,
    -0.4966827f, -0.2906543f, +0.2893821f, -0.4978951f, -0.5521452f, -0.1669319f, -0.1738641f, -0.5502602f,
    +0.1741129f, +0.5513017f, +0.4999995f, +0.2905260f, -0.1790565f, +0.5505838f, -0.5521517f, +0.1811216f,
    +0.1868485f, -0.5511356f, -0.2930312f, +0.5043379f, +0.4480981f, -0.3806849f, +0.4865645f, -0.3304601f,
    -0.4506157f, -0.3790791f, +0.5519270f, -0.2052721f, -0.4329579f, +0.3997230f, -0.5511047f, -0.2134405f,
    +0.4516180f, +0.3830570f, -0.3189608f, -0.5000039f, -0.3896599f, -0.4476354f, +0.3202561f, +0.5006493f,
    +0.5522426f, +0.2250396f, +0.3999172f, -0.4435272f, -0.2297680f, -0.5519402f, +0.2297627f, +0.5522434f,
    -0.5528634f, +0.2362260f, -0.3590180f, +0.4829147f, +0.2455218f, -0.5517317f, +0.4038756f, +0.4498352f,
    -0.2480187f, +0.5518526f, +0.3507136f, -0.4953251f, -0.0413308f, +0.6064284f, +0.0424306f, -0.6073637f,
    -0.6088359f, -0.0131329f, +0.6089671f, +0.0105972f, -0.0206147f, -0.6091614f, -0.1346982f, +0.5958619f,
    +0.5013615f, +0.3498347f, -0.4973602f, +0.3557392f, -0.5517972f, -0.2647142f, -0.6108639f, +0.0400379f,
    +0.6062994f, -0.0850150f, -0.4989265f, -0.3552195f, +0.0652229f, +0.6093413f, +0.0089852f, +0.6127642f,
    -0.6100385f, -0.0691506f, -0.0859427f, -0.6079494f, +0.6103118f, +0.0679860f, -0.6100302f, +0.0927300f,
    +0.5530357f, -0.2744450f, +0.6170186f, -0.0412913f, -0.6064735f, -0.1261814f, -0.0903174f, +0.6150066f,
    +0.6098891f, +0.1213933f, +0.1090441f, -0.6126288f, -0.3709767f, -0.4999990f, -0.2889896f, -0.5516679f,
    +0.1285266f, +0.6100270f, +0.2890621f, +0.5525213f, -0.5527590f, +0.2893083f, -0.1514122f, -0.6054146f,
    -0.6085560f, +0.1493417f, +0.6124976f, -0.1340631f, +0.5527773f, +0.2960276f, -0.4476448f, -0.4449548f,
    +0.3059732f, -0.5525914f, +0.3871368f, +0.4999998f, +0.4508559f, +0.4441677f, -0.5509456f, -0.3139794f,
    +0.1799395f, -0.6088543f, +0.5026960f, -0.3898361f, -0.1891143f, +0.6074777f, -0.4277522f, +0.4732091f,
    +0.6121123f, -0.1852303f, -0.6064194f, -0.2032338f, +0.6142992f, +0.1816340f, +0.1930541f, +0.6116502f,
    -0.6128572f, +0.2084476f, -0.3381439f, +0.5530818f, -0.2263317f, -0.6076337f, +0.4738708f, -0.4481538f,
    +0.6093743f, -0.2345955f, +0.4059724f, -0.5114949f, +0.6064096f, +0.2430048f, -0.3499157f, -0.5524879f,
    +0.5055354f, +0.4157924f, -0.2385104f, +0.6102706f, -0.5504379f, +0.3560134f, +0.3551690f, +0.5512070f,
    +0.5526390f, -0.3535550f, -0.5000057f, -0.4254323f, -0.4271820f, -0.4999970f, -0.4954065f, +0.4331308f,
    -0.0080279f, -0.6582961f, +0.2505790f, -0.6137215f, +0.2573125f, +0.6110995f, -0.5547329f, -0.3632259f,
    -0.1488761f, +0.6471893f, -0.6082449f, -0.2694956f, -0.6448458f, -0.1640294f, +0.5527745f, +0.3765063f,
    -0.6693692f, +0.0131809f, -0.3032061f, +0.5969837f, +0.3698358f, -0.5589375f, +0.0576621f, -0.6679585f,
    -0.6150176f, +0.2672807f, +0.6686370f, +0.0538296f, +0.0206903f, +0.6712118f, +0.6713580f, -0.0193188f,
    -0.6703084f, -0.0490483f, -0.6692576f, +0.0753082f, -0.0428283f, +0.6724640f, +0.4489266f, +0.5035665f,
    -0.0644283f, -0.6718391f, +0.6689804f, -0.0938179f, +0.6106034f, -0.2903174f, +0.0834113f, +0.6711421f,
    -0.2939212f, -0.6092194f, -0.6690732f, -0.1098076f, -0.6657969f, +0.1367972f, -0.1809414f, -0.6572661f,
    -0.1258983f, -0.6704550f, +0.6706821f, +0.1311752f, -0.1042713f, +0.6766200f, +0.1095342f, -0.6772926f,
    +0.1591644f, -0.6678648f, +0.4496997f, -0.5189009f, -0.6068027f, +0.3216356f, -0.5519274f, -0.4102602f,
    +0.3223771f, +0.6081663f, +0.1519624f, +0.6721648f, +0.3248029f, -0.6086603f, +0.6727881f, -0.1591158f,
    -0.4184467f, +0.5527467f, +0.6095625f, +0.3317532f, +0.4999978f, +0.4821226f, -0.4260888f, -0.5507761f,
    +0.4243300f, +0.5522609f, +0.6401396f, +0.2751318f, -0.2904619f, +0.6334560f, -0.4957392f, -0.4933327f,
    -0.2320121f, -0.6601494f, -0.6635826f, -0.2237318f, -0.6135284f, -0.3378463f, -0.6724516f, +0.2006595f,
    -0.5550615f, +0.4303550f, -0.2059087f, +0.6717819f, +0.5541336f, -0.4335674f, +0.6138372f, -0.3461660f,
    +0.2242215f, +0.6681404f, +0.6691221f, -0.2278427f, +0.2102668f, -0.6752443f, +0.0092088f, -0.7077472f,
    +0.6718935f, +0.2228383f, -0.3515784f, -0.6196615f, -0.4895682f, +0.5228523f, +0.3880081f, +0.6056892f,
    -0.6748074f, +0.2597716f, -0.2623976f, +0.6743805f, -0.3860113f, +0.6124302f, -0.6123824f, +0.3865603f,
    +0.5250923f, -0.5014072f, -0.4018098f, -0.6051998f, +0.5782852f, +0.4404575f, -0.7063876f, -0.1730939f,
    -0.6677998f, -0.2908493f, -0.5520383f, -0.4785093f, -0.2905520f, -0.6703255f, -0.7305542f, +0.0281638f,
    +0.7279252f, +0.0816702f, +0.7316629f, -0.0539116f, +0.2979694f, +0.6705062f, +0.7341328f, +0.0156308f,
    +0.5511367f, +0.4869479f, -0.7344884f, -0.0387428f, -0.7302309f, -0.1044240f, -0.1659166f, +0.7195885f,
    +0.0365703f, +0.7381895f, -0.0352455f, +0.7384304f, +0.2817549f, -0.6835281f, -0.0677863f, -0.7372247f,
    -0.2191848f, -0.7079775f, +0.6776382f, -0.3009691f, +0.7174622f, +0.1876682f, +0.0820118f, -0.7372442f,
    -0.7371497f, +0.0862896f, +0.6248118f, -0.4007721f, -0.7292833f, +0.1388480f, +0.3519471f, -0.6544607f,
    -0.3451881f, +0.6581131f, +0.4241960f, -0.6111320f, -0.1049598f, +0.7371303f, +0.6730298f, +0.3185581f,
    +0.1065753f, +0.7369698f, -0.5511537f, +0.5033271f, -0.6745309f, +0.3200492f, +0.4941353f, -0.5625876f,
    +0.7388660f, -0.1252068f, -0.1524899f, -0.7338040f, -0.6244103f, -0.4193394f, +0.4939101f, +0.5679762f,
    +0.1774123f, +0.7315817f, +0.1551343f, -0.7368858f, -0.7108955f, -0.2500302f, +0.6320091f, +0.4128047f,
    +0.7291611f, -0.1961885f, -0.6689531f, -0.3521006f, +0.6080815f, -0.4524381f, +0.7467737f, +0.1399334f,
    -0.5242853f, -0.5521910f, -0.7403132f, +0.2045526f, +0.4459067f, +0.6261827f, -0.4707357f, -0.6083177f,
    +0.2366935f, +0.7321905f, +0.3786669f, +0.6701695f, +0.0146633f, -0.7696687f, -0.3523720f, -0.6866469f,
    +0.2319510f, -0.7388325f, -0.4706651f, +0.6155789f, -0.2287891f, +0.7406748f, -0.7566955f, -0.1684449f,
    -0.6339714f, +0.4497572f, -0.6763240f, +0.3922633f, +0.7407308f, +0.2522737f, +0.6840383f, +0.3818766f,
    -0.6155176f, -0.4875248f, -0.3011856f, +0.7252777f, +0.6898317f, -0.3768276f, +0.5567907f, +0.5605261f,
    -0.7425408f, +0.2742622f, +0.4093342f, -0.6777043f, +0.7506837f, -0.2641069f, +0.6221107f, +0.4963193f,
    -0.3041428f, -0.7364908f, +0.6064514f, -0.5169556f, -0.2297777f, -0.7634996f, -0.7906473f, -0.1079418f,
    -0.7978208f, +0.0288636f, -0.4075934f, -0.6870201f, +0.8013504f, -0.0085271f, +0.3005011f, +0.7429961f,
    -0.8011577f, -0.0277610f, +0.5584195f, -0.5780912f, -0.7631134f, -0.2548300f, -0.4099570f, +0.6931574f,
    -0.1278280f, -0.7951843f, -0.0331370f, -0.8052662f, -0.6122425f, +0.5250149f, -0.0075296f, +0.8069643f,
    -0.7417653f, -0.3183045f, -0.0914817f, +0.8029182f, +0.0711298f, +0.8070267f, +0.8060376f, +0.0856863f,
    +0.8061897f, -0.0941822f, +0.0780006f, -0.8080412f, +0.5014043f, -0.6393237f, -0.8080729f, +0.0889006f,
    -0.5599269f, +0.5922760f, +0.7465694f, +0.3296032f, -0.8028314f, +0.1523537f, +0.1477115f, +0.8042949f,
    -0.4781245f, -0.6644759f, +0.3673909f, -0.7315654f, +0.1615472f, -0.8043697f, -0.7409508f, +0.3531546f,
    -0.7209591f, -0.3924809f, +0.6746056f, -0.4712955f, +0.8020929f, -0.1878668f, -0.1787255f, +0.8045639f,
    +0.4969707f, +0.6600069f, +0.8097792f, +0.1794514f, +0.3754363f, +0.7400898f, +0.2306174f, +0.7986809f,
    -0.5703096f, -0.6049753f, -0.6846773f, -0.4760273f, -0.6252741f, -0.5517601f, +0.7694774f, -0.3342977f,
    -0.8237696f, -0.1617173f, -0.8095667f, -0.2241901f, -0.2077534f, -0.8145781f, -0.0819484f, -0.8376003f,
    +0.2565451f, -0.8025866f, +0.4409029f, +0.7203540f, -0.3656795f, +0.7619120f, -0.8151342f, +0.2260869f,
    +0.4751474f, -0.6998746f, -0.8423799f, -0.0781540f, +0.3225453f, -0.7838954f, -0.4848815f, +0.6981080f,
    -0.6691085f, +0.5250589f, -0.7356733f, +0.4358991f, -0.2914612f, +0.8056999f, +0.7337036f, +0.4443624f,
    +0.5692021f, +0.6441222f, +0.7434963f, -0.4314000f, -0.3032467f, -0.8050014f, +0.8161390f, +0.2754556f,
    +0.6283871f, +0.5903572f, +0.6554927f, -0.5663021f, +0.5821877f, -0.6446350f, -0.4735497f, -0.7287076f,
    +0.6953951f, +0.5246935f, -0.7946326f, -0.3612510f, -0.8158365f, +0.3106005f, +0.3425523f, +0.8032908f,
    -0.3846068f, -0.7871797f, -0.1548718f, -0.8643454f, -0.7500680f, -0.4595460f, -0.8290846f, -0.3055548f,
    -0.5744443f, -0.6714431f, +0.8064336f, +0.3657275f, +0.1979267f, +0.8663548f, -0.0146815f, -0.8916656f,
    +0.8924137f, +0.0018677f, +0.8561515f, -0.2573974f, -0.8932486f, +0.0448762f, -0.0084466f, +0.8948330f,
    -0.0791409f, +0.8930057f, -0.8057405f, +0.3942545f, +0.0655989f, +0.8949419f, +0.0533065f, -0.8958332f,
    +0.8948477f, +0.0807650f, -0.8995440f, -0.0297709f, -0.6347756f, +0.6388795f, +0.5593327f, +0.7061432f,
    +0.8973727f, -0.0808121f, -0.5641251f, +0.7028956f, -0.4362588f, +0.7887439f, +0.2877314f, +0.8543946f,
    +0.4700755f, -0.7694921f, +0.4009546f, -0.8097928f, +0.4958629f, +0.7558401f, -0.7155870f, -0.5543169f,
    +0.8139385f, -0.3978763f, -0.8968067f, +0.1301313f, +0.7384391f, -0.5301496f, +0.8947553f, -0.1611278f,
    -0.6561708f, -0.6293459f, +0.1335865f, -0.8995799f, +0.2043915f, -0.8872873f, -0.1560512f, +0.8981965f,
    -0.9053296f, -0.1104424f, +0.4306081f, +0.8058728f, +0.8995826f, +0.1665208f, +0.1436169f, +0.9068398f,
    -0.7538267f, +0.5265234f, -0.0936657f, -0.9147213f, -0.9033589f, -0.1825976f, -0.9010271f, +0.2136700f,
    -0.2398604f, +0.8946552f, -0.2545698f, -0.8945234f, -0.4732240f, -0.8031381f, -0.1832846f, -0.9157747f,
    +0.5776860f, -0.7340540f, +0.6479899f, -0.6738020f, +0.2793598f, -0.8923029f, -0.8993572f, -0.2617013f,
    -0.5713543f, -0.7439268f, +0.8282520f, +0.4442628f, -0.8053774f, -0.4880482f, +0.9003943f, +0.2772720f,
    +0.8046012f, -0.4928528f, -0.8465233f, -0.4239434f, -0.7177599f, +0.6191323f, -0.8223194f, +0.4744731f,
    +0.2492479f, +0.9192641f, -0.9021897f, +0.3080322f, -0.3389350f, -0.8931914f, -0.3483441f, +0.8918222f,
    +0.7254473f, -0.6251767f, +0.8971519f, -0.3378536f, +0.5760319f, +0.7664012f, +0.8065999f, +0.5185212f,
    -0.7402629f, -0.6108696f, +0.7015665f, +0.6569932f, -0.5230852f, +0.8074299f, +0.3544734f, -0.8975757f,
    +0.3675244f, +0.8947815f, +0.6429648f, +0.7286513f, -0.9059567f, -0.3605773f, +0.5451101f, -0.8122367f,
    -0.8043721f, -0.5634145f, +0.7787873f, +0.5995736f, -0.8955155f, +0.4082697f, +0.8997616f, +0.3997543f,
    -0.6842942f, -0.7204827f, -0.4343179f, -0.8954905f, +0.5385813f, +0.8382849f, +0.4347029f, -0.8981250f,
    -1.0000025f, -0.0020543f, +0.9999997f, -0.0074653f, +0.0201653f, -0.9999998f, -0.0355759f, +1.0000014f,
    +0.0400163f, +0.9999996f, +1.0000067f, +0.0677517f, +0.8962937f, -0.4491737f, -0.0715206f, -1.0000103f,
    +1.0000011f, -0.0872811f, -1.0000030f, +0.0931977f, -0.6956095f, +0.7249104f, -1.0000086f, -0.0971031f,
    -0.1110247f, +0.9999897f, +0.1125406f, -0.9999985f, +0.1155549f, +1.0000255f, -0.4571838f, +0.8985501f,
    -0.5566899f, -0.8425430f, +1.0000117f, +0.1437869f, +0.4599049f, +0.8998328f, +0.9830819f, -0.2401750f,
    -0.1622423f, -1.0000088f, +1.0000087f, -0.1666822f, -0.1820173f, +1.0000108f, -1.0000013f, +0.1888485f,
    +0.7039154f, -0.7356002f, -0.9038101f, -0.4695749f, +0.1953933f, +0.9999972f, -1.0000015f, -0.2010044f,
    -0.6325067f, +0.8008591f, -0.7659675f, -0.6747940f, +0.2072915f, -1.0000037f, +0.9987261f, +0.2182229f,
    +0.8477919f, -0.5826399f, -0.2486694f, -0.9999989f, -0.8100410f, +0.6377466f, -0.2598320f, +1.0000094f,
    -1.0000043f, +0.2758312f, +0.2943587f, +0.9990693f, -0.9999872f, -0.2998689f, -0.8947883f, +0.5381643f,
    +0.5305655f, -0.8997703f, +0.9999852f, +0.3039831f, +1.0000001f, -0.3114494f, +0.8983502f, +0.5417495f,
    +0.3173264f, -1.0000064f, -0.6777771f, -0.8010345f, -0.8454596f, -0.6295004f, -0.3351995f, -1.0000187f,
    -0.3459111f, +0.9999916f, +0.6421905f, -0.8428190f, -0.9018751f, -0.5600050f, -1.0000058f, -0.3702886f,
    -1.0000056f, +0.3704396f, +0.7261929f, +0.7810532f, +1.0000114f, +0.3906186f, +1.0000010f, -0.4005720f,
    -0.5948561f, +0.9021981f, +0.4098847f, +1.0000188f, +0.8278877f, -0.7028940f, -0.4266287f, -1.0000151f,
    +0.8356852f, +0.6960995f, +0.6357900f, +0.8868414f, -1.0000112f, -0.4399138f, +0.4419054f, -0.9999918f,
    -0.6159863f, -0.9070256f, -0.4516121f, +0.9999976f, -0.7876832f, -0.7692954f, -0.9936715f, +0.4799338f,
    -0.5089393f, -0.9798051f, +0.9974933f, +0.4919495f, +0.9828730f, -0.5223437f, +0.7559775f, -0.8246973f,
    -0.8567098f, +0.7278949f, -0.9999957f, -0.5238730f, -0.8894718f, -0.7016598f, +0.5521497f, +0.9900950f,
    -0.7632577f, +0.8408893f, -0.7301708f, -0.8887079f, +0.5841947f, -0.9916000f, -0.5768782f, -0.9999725f,
    -0.9847280f, +0.6155825f, -0.5958017f, +0.9999986f, +0.9834958f, +0.6307222f, +0.9777358f, -0.6448475f,
    -0.9961294f, -0.6351683f, +0.8234622f, +0.8502380f, -0.6610700f, -1.0000143f, +0.7070056f, -0.9791427f,
    +0.8316563f, -0.8881918f, -0.8612013f, -0.8646276f, +0.7181055f, +0.9905740f, +0.9693221f, +0.7733319f,
    -0.9870787f, -0.7726217f, +0.9744248f, -0.8005569f, -0.9836243f, +0.7965512f, -0.7910733f, -0.9914099f,
    -0.8234321f, +0.9702841f, +0.8736893f, -0.9841430f, +0.9226856f, +0.9534308f, -0.9594911f, -0.9470464f,
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
