#pragma once

#include "common.h"

void quantize_q1_0(device const float * src, device block_q1_0 & dst) {
    float sum_abs = 0.0f;
    for (int j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    for (int j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK1_0; j++) {
        if (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

void quantize_q2_0(device const float * src, device block_q2_0 & dst) {
    float amax = 0.0f;
    for (int j = 0; j < QK2_0; j++) {
        float a = fabs(src[j]);
        if (a > amax) amax = a;
    }
    const float d = amax;
    dst.d = d;

    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    for (int j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK2_0; j++) {
        int q = (int)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

void quantize_q4_0(device const float * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK4_0/2 + j]*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q4_1(device const float * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < QK4_1; j++) {
        const float v = src[j];
        if (min > v) min = v;
        if (max < v) max = v;
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK4_1/2 + j] - min)*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q5_0(device const float * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK5_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -16;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK5_0/2 + j]*id;

        const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q5_1(device const float * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    float max = src[0];
    float min = src[0];

    for (int j = 1; j < QK5_1; j++) {
        const float v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q8_0(device const float * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = src[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

void quantize_iq4_nl(device const float * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_NL; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / kvalues_iq4nl_f[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = src[0        + j]*id;
        const float x1 = src[QK4_NL/2 + j]*id;

        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        const float v0 = kvalues_iq4nl_f[xi0];
        const float v1 = kvalues_iq4nl_f[xi1];
        const float w0 = src[0        + j]*src[0        + j];
        const float w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

void quantize_tq2_0(device const float * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK_K; j++) {
        const float v = src[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    for (int j = 0; j < QK_K/4; j += 32) {
        for (int m = 0; m < 32; ++m) {
            uint8_t q = 0;
            for (int n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = (int)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}

// Neuron pair codec: encode on the way INTO the KV cache -------------------------------
//
// Same codec as the weights, run inline as a token is written. The anchors are the min,
// median and max of the block's log magnitudes; median is taken by counting rather than
// sorting, since a full sort of 64 values inside a kernel is not worth it and the anchor
// only needs to land near the middle to give the ladder its bend.
// KV encoder, one implementation for every layout. Mirrors the weight encoder's geometry
// -- 16-pair / (4m+3)-byte groups, joint checkerboard angles -- but not its search: this
// runs on every token written to the cache, so it takes the MEAN of the log magnitudes
// rather than a true median (counting ranks was O(n^2) and cost 2.4x in generation), and
// it does no refit.
// KV-cache encoder for the v family. Deliberately NOT the weight encoder: that one runs an
// exhaustive search over every representable sub-block multiplier plus a convex-hull query
// and a least-squares scale refit, which is affordable once per model and absurd once per
// token. Here each sub-block takes the multiplier nearest its own absmax ratio in one pass,
// and each pair takes its nearest codeword. Same block layout, so every decoder --
// dequantize_neuron_v* in the mul_mm tiles and in flash attention -- is unchanged.
//
// Packing goes through a THREAD-local buffer and writes each byte to device memory exactly
// once. Setting bits straight into dst.qs is a device read-modify-write per bit, and that,
// not the search, is what made the polar KV path 2.4x slower than an f16 cache.
#define NEURON_V_QUANT(SFX)                                                               \
void quantize_neuron_v##SFX(device const float * src, device block_neuron_v##SFX & dst) { \
    const int NP  = QK_NEURON / 2;                                                        \
    const int PSB = NEURON_VQ_SBV / 2;                                                    \
    const int B   = NEURON_VQ##SFX##_BITS;                                                \
    const int SBB = NEURON_VQ##SFX##_SBB;                                                 \
    const int NSBT = 1 << SBB;                                                            \
                                                                                          \
    float amax = 0.0f;                                                                    \
    for (int j = 0; j < QK_NEURON; ++j) { amax = fmax(amax, fabs(src[j])); }              \
    if (amax == 0.0f) {                                                                   \
        dst.d = (half) 0.0f;                                                              \
        for (int i = 0; i < NEURON_VQ_SBBY(SBB); ++i) { dst.sb[i] = 0; }                  \
        for (int i = 0; i < NP * B / 8; ++i) { dst.qs[i] = 0; }                           \
        return;                                                                           \
    }                                                                                     \
    dst.d = (half) amax;                                                                  \
    const float d = (float) dst.d;                                                        \
                                                                                          \
    uint8_t sbuf[NEURON_VQ_SBBY(12)];          /* widest: 8 sub-blocks x 6 bits */        \
    for (int i = 0; i < NEURON_VQ_SBBY(SBB); ++i) { sbuf[i] = 0; }                        \
    uint8_t qbuf[QK_NEURON / 2 * 12 / 8];      /* widest: 64 pairs x 12 bits */           \
    for (int i = 0; i < NP * B / 8; ++i) { qbuf[i] = 0; }                                 \
                                                                                          \
    for (int sblk = 0; sblk < NEURON_VQ_NSB; ++sblk) {                                    \
        /* sub-block absmax -> nearest representable multiplier, one pass, no search */   \
        float sa = 0.0f;                                                                  \
        for (int t = 0; t < NEURON_VQ_SBV; ++t) {                                         \
            sa = fmax(sa, fabs(src[sblk*NEURON_VQ_SBV + t]));                             \
        }                                                                                 \
        const float want = sa / d;                                                        \
        int   mj = 0;                                                                     \
        float mb = INFINITY;                                                              \
        for (int j = 0; j < NSBT; ++j) {                                                  \
            const float e = fabs(NEURON_VQ##SFX##_SBT[j] - want);                         \
            if (e < mb) { mb = e; mj = j; }                                               \
        }                                                                                 \
        const int sbit = sblk * SBB;                                                      \
        sbuf[sbit >> 3] |= (uint8_t)(mj << (sbit & 7));                                   \
        if (((sbit & 7) + SBB) > 8) {                                                     \
            sbuf[(sbit >> 3) + 1] |= (uint8_t)(mj >> (8 - (sbit & 7)));                   \
        }                                                                                 \
                                                                                          \
        const float g = d * NEURON_VQ##SFX##_SBT[mj];                                     \
        const float inv = g > 0.0f ? 1.0f / g : 0.0f;                                     \
        for (int t = 0; t < PSB; ++t) {                                                   \
            const int   p  = sblk*PSB + t;                                                \
            const float ax = src[2*p] * inv, ay = src[2*p + 1] * inv;                     \
            int   bc = 0;                                                                 \
            float bv = -INFINITY;                                                         \
            for (int c = 0; c < NEURON_VQ##SFX##_K; ++c) {                                \
                const float cx = kNeuronVQ##SFX[2*c], cy = kNeuronVQ##SFX[2*c + 1];       \
                const float v  = 2.0f*(ax*cx + ay*cy) - (cx*cx + cy*cy);                  \
                if (v > bv) { bv = v; bc = c; }                                           \
            }                                                                             \
            const int bit = p * B;                                                        \
            qbuf[bit >> 3] |= (uint8_t)(bc << (bit & 7));                                 \
            if (((bit & 7) + B) > 8)  { qbuf[(bit >> 3) + 1] |= (uint8_t)(bc >> (8 - (bit & 7))); }\
            if (((bit & 7) + B) > 16) { qbuf[(bit >> 3) + 2] |= (uint8_t)(bc >> (16 - (bit & 7))); }\
        }                                                                                 \
    }                                                                                     \
    for (int i = 0; i < NEURON_VQ_SBBY(SBB); ++i) { dst.sb[i] = sbuf[i]; }                \
    for (int i = 0; i < NP * B / 8; ++i) { dst.qs[i] = qbuf[i]; }                         \
}

/* Weight encoder: the same block layout, but the sub-block multiplier is SEARCHED for the
   one that minimises squared error, as the CPU encoder does, rather than taken as the
   nearest representable multiple of the sub-block absmax. beta3 measured that search to be
   worth real quality, so an encoder without it is not a drop-in for the CPU path.

   Deliberately NOT folded into quantize_neuron_v* above: that one encodes the KV cache and
   runs per token, where paying NSBT times the work to save a fraction of a bit would be a
   bad trade. Weights are encoded once. */
#define NEURON_V_QUANT_SS(SFX)                                                            \
void quantize_neuron_v##SFX##_ss(device const float * src, device block_neuron_v##SFX & dst) { \
    const int NP  = QK_NEURON / 2;                                                        \
    const int PSB = NEURON_VQ_SBV / 2;                                                    \
    const int B   = NEURON_VQ##SFX##_BITS;                                                \
    const int SBB = NEURON_VQ##SFX##_SBB;                                                 \
    const int NSBT = 1 << SBB;                                                            \
                                                                                          \
    float amax = 0.0f;                                                                    \
    for (int j = 0; j < QK_NEURON; ++j) { amax = fmax(amax, fabs(src[j])); }              \
    if (amax == 0.0f) {                                                                   \
        dst.d = (half) 0.0f;                                                              \
        for (int i = 0; i < NEURON_VQ_SBBY(SBB); ++i) { dst.sb[i] = 0; }                  \
        for (int i = 0; i < NP * B / 8; ++i) { dst.qs[i] = 0; }                           \
        return;                                                                           \
    }                                                                                     \
    dst.d = (half) amax;                                                                  \
    const float d = (float) dst.d;                                                        \
    uint8_t sbuf[NEURON_VQ_SBBY(12)];                                                     \
    for (int i = 0; i < NEURON_VQ_SBBY(SBB); ++i) { sbuf[i] = 0; }                        \
    uint8_t qbuf[QK_NEURON / 2 * 12 / 8];                                                 \
    for (int i = 0; i < NP * B / 8; ++i) { qbuf[i] = 0; }                                 \
                                                                                          \
    for (int sblk = 0; sblk < NEURON_VQ_NSB; ++sblk) {                                    \
        int   mj = 0;                                                                     \
        float mb = INFINITY;                                                              \
        for (int j = 0; j < NSBT; ++j) {                                                  \
            const float gj = d * NEURON_VQ##SFX##_SBT[j];                                 \
            if (!(gj > 0.0f)) { continue; }                                               \
            const float ij = 1.0f / gj;                                                   \
            float sse = 0.0f;                                                             \
            for (int t = 0; t < PSB; ++t) {                                               \
                const int   p  = sblk*PSB + t;                                            \
                const float ax = src[2*p] * ij, ay = src[2*p + 1] * ij;                   \
                float bv = -INFINITY;                                                     \
                for (int c = 0; c < NEURON_VQ##SFX##_K; ++c) {                            \
                    const float cx = kNeuronVQ##SFX[2*c], cy = kNeuronVQ##SFX[2*c + 1];   \
                    const float v  = 2.0f*(ax*cx + ay*cy) - (cx*cx + cy*cy);              \
                    if (v > bv) { bv = v; }                                               \
                }                                                                         \
                sse += (ax*ax + ay*ay - bv) * gj * gj;                                    \
            }                                                                             \
            if (sse < mb) { mb = sse; mj = j; }                                           \
        }                                                                                 \
        const int sbit = sblk * SBB;                                                      \
        sbuf[sbit >> 3] |= (uint8_t)(mj << (sbit & 7));                                   \
        if (((sbit & 7) + SBB) > 8) {                                                     \
            sbuf[(sbit >> 3) + 1] |= (uint8_t)(mj >> (8 - (sbit & 7)));                   \
        }                                                                                 \
        const float g = d * NEURON_VQ##SFX##_SBT[mj];                                     \
        const float inv = g > 0.0f ? 1.0f / g : 0.0f;                                     \
        for (int t = 0; t < PSB; ++t) {                                                   \
            const int   p  = sblk*PSB + t;                                                \
            const float ax = src[2*p] * inv, ay = src[2*p + 1] * inv;                     \
            int   bc = 0;                                                                 \
            float bv = -INFINITY;                                                         \
            for (int c = 0; c < NEURON_VQ##SFX##_K; ++c) {                                \
                const float cx = kNeuronVQ##SFX[2*c], cy = kNeuronVQ##SFX[2*c + 1];       \
                const float v  = 2.0f*(ax*cx + ay*cy) - (cx*cx + cy*cy);                  \
                if (v > bv) { bv = v; bc = c; }                                           \
            }                                                                             \
            const int bit = p * B;                                                        \
            qbuf[bit >> 3] |= (uint8_t)(bc << (bit & 7));                                 \
            if (((bit & 7) + B) > 8)  { qbuf[(bit >> 3) + 1] |= (uint8_t)(bc >> (8 - (bit & 7))); }\
            if (((bit & 7) + B) > 16) { qbuf[(bit >> 3) + 2] |= (uint8_t)(bc >> (16 - (bit & 7))); }\
        }                                                                                 \
    }                                                                                     \
    for (int i = 0; i < NEURON_VQ_SBBY(SBB); ++i) { dst.sb[i] = sbuf[i]; }                \
    for (int i = 0; i < NP * B / 8; ++i) { dst.qs[i] = qbuf[i]; }                         \
}

NEURON_V_QUANT_SS(4)
NEURON_V_QUANT_SS(5)
NEURON_V_QUANT_SS(6)

NEURON_V_QUANT(4)
NEURON_V_QUANT(5)
NEURON_V_QUANT(6)

#define NEURON_QUANT(LP)                                                                    \
void quantize_neuron_m##LP(device const float * src, device block_neuron_m##LP & dst) {   \
    const int nlv = 1 << (LP), last = nlv - 1, np = QK_NEURON / 2;                        \
    const int A = NEURON_ANG(LP), AW = NEURON_AW(LP);                                     \
    const int GBY = NEURON_GBY(LP), AOFF = NEURON_AOFF(LP), JSH = NEURON_JSH(LP);         \
    float lr[QK_NEURON / 2], th[QK_NEURON / 2], rr[QK_NEURON / 2];                        \
    float lo = INFINITY, hi = -INFINITY;                                                  \
    for (int p = 0; p < np; ++p) {                                                        \
        const float a = src[2*p], b = src[2*p + 1];                                       \
        const float r = fmax(sqrt(a*a + b*b), 1e-30f);                                    \
        rr[p] = r;                                                                        \
        lr[p] = log(r);                                                                   \
        th[p] = atan2(b, a);                                                              \
        lo = fmin(lo, lr[p]);                                                             \
        hi = fmax(hi, lr[p]);                                                             \
    }                                                                                     \
    float mid = 0.0f;                                                                     \
    for (int p = 0; p < np; ++p) { mid += lr[p]; }                                        \
    mid /= (float) np;                                                                    \
    mid = clamp(mid, lo + 1e-4f, hi - 1e-4f);                                             \
    dst.lo = lo; dst.mid = mid; dst.hi = hi;                                              \
    const float flo = (float) dst.lo, fmi = (float) dst.mid, fhi = (float) dst.hi;        \
    /* Pack into THREAD-local bytes, then write each byte to device memory exactly once.  \
       Setting bits straight into dst.qs is a device read-modify-write PER BIT, and that, \
       not the anchor search, is what made the KV path 2.4x slower than an f16 cache. */  \
    for (int g = 0; g < np / NEURON_GRP; ++g) {                                           \
        uint8_t buf[NEURON_GBY(LP)];                                                      \
        for (int i = 0; i < GBY; ++i) { buf[i] = 0; }                                     \
        const int p0 = g * NEURON_GRP;                                                    \
        for (int t = 0; t < NEURON_GRP; ++t) {                                            \
            const float l = lr[p0 + t];                                                   \
            const float u = l <= fmi ? (l - flo) / fmax(2.0f*(fmi - flo), 1e-30f)         \
                                     : 0.5f + (l - fmi) / fmax(2.0f*(fhi - fmi), 1e-30f); \
            const int mc = clamp((int) round(clamp(u, 0.0f, 1.0f) * (float) last), 0, last);\
            const int bit = t * (LP);                                                     \
            for (int i = 0; i < (LP); ++i) {                                              \
                if (mc & (1 << i)) { buf[(bit+i) >> 3] |= (uint8_t)(1u << ((bit+i) & 7)); }\
            }                                                                             \
        }                                                                                 \
        for (int s = 0; s < NEURON_GRP / 2; ++s) {                                        \
            const int pa = p0 + 2*s, pb = pa + 1;                                         \
            const float ua = (th[pa] + M_PI_F) / (2.0f * M_PI_F) * (float) A;             \
            const float ub = (th[pb] + M_PI_F) / (2.0f * M_PI_F) * (float) A;             \
            int ia = (int) round(ua), ib = (int) round(ub);                               \
            const float da = ua - (float) ia, db = ub - (float) ib;                       \
            ia = ((ia % A) + A) % A;                                                      \
            ib = ((ib % A) + A) % A;                                                      \
            if ((ia + ib) & 1) {                                                          \
                /* exact nearest point on the checkerboard: move whichever pair to its    \
                   second-nearest grid point costs less. Angular error shows up as        \
                   r*dtheta, so the squared cost changes by r^2 * (1 - 2|d|). */          \
                const float ca = rr[pa]*rr[pa]*(1.0f - 2.0f*fabs(da));                    \
                const float cb = rr[pb]*rr[pb]*(1.0f - 2.0f*fabs(db));                    \
                if (ca <= cb) ia = ((ia + (da >= 0.0f ? 1 : -1)) % A + A) % A;            \
                else          ib = ((ib + (db >= 0.0f ? 1 : -1)) % A + A) % A;            \
            }                                                                             \
            const uint v   = ((uint) ia << JSH) | ((uint) ib >> 1);                       \
            const int  bit = AOFF*8 + s*AW;                                               \
            for (int i = 0; i < AW; ++i) {                                                \
                if (v & (1u << i)) { buf[(bit+i) >> 3] |= (uint8_t)(1u << ((bit+i) & 7)); }\
            }                                                                             \
        }                                                                                 \
        for (int i = 0; i < GBY; ++i) { dst.qs[g*GBY + i] = buf[i]; }                     \
    }                                                                                     \
}

NEURON_QUANT(3)
NEURON_QUANT(4)
NEURON_QUANT(5)
NEURON_QUANT(6)
NEURON_QUANT(7)
