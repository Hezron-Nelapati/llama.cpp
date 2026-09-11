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
#define NEURON_QUANT(LP, BITS)                                                            \
void quantize_neuron_m##LP(device const float * src, device block_neuron_m##LP & dst) {   \
    const int M = (LP), K = (LP) + 1;                                                     \
    const int nlv = 1 << M, kn = 1 << K, last = nlv - 1, np = QK_NEURON / 2;              \
    float lr[QK_NEURON / 2], th[QK_NEURON / 2];                                           \
    float lo = INFINITY, hi = -INFINITY;                                                  \
    for (int p = 0; p < np; ++p) {                                                        \
        const float a = src[2*p], b = src[2*p + 1];                                       \
        const float r = fmax(sqrt(a*a + b*b), 1e-30f);                                    \
        lr[p] = log(r);                                                                   \
        th[p] = atan2(b, a);                                                              \
        lo = fmin(lo, lr[p]);                                                             \
        hi = fmax(hi, lr[p]);                                                             \
    }                                                                                     \
    /* MEAN of the log magnitudes, not the median. The weight encoder takes a true median
       because it runs once, offline; this runs on every token written to the cache, and
       counting ranks was O(n^2) -- 4096 comparisons per block -- which cost 2.4x in
       generation. The mean is O(n) and, for roughly log-normal magnitudes, lands close
       enough: the anchor only has to sit near the middle to give the ladder its bend. */  \
    float mid = 0.0f;                                                                     \
    for (int p = 0; p < np; ++p) { mid += lr[p]; }                                        \
    mid /= (float) np;                                                                    \
    mid = clamp(mid, lo + 1e-4f, hi - 1e-4f);                                             \
    dst.lo = lo; dst.mid = mid; dst.hi = hi;                                              \
    const float flo = (float) dst.lo, fmi = (float) dst.mid, fhi = (float) dst.hi;        \
    /* Pack into THREAD-local bytes, then write each byte to device memory exactly once.
       Setting bits straight into dst.qs is a device read-modify-write PER BIT -- 704 of
       them per block -- and that, not the anchor search, was what made the KV path 2.4x
       slower than an f16 cache. 8 pairs fill exactly (BITS) bytes, so a group needs no
       zeroing and no carry between groups. */                                            \
    for (int g = 0; g < np; g += 8) {                                                     \
        uint8_t buf[(BITS)];                                                              \
        for (int i = 0; i < (BITS); ++i) { buf[i] = 0; }                                  \
        for (int t = 0; t < 8; ++t) {                                                     \
            const int p = g + t;                                                          \
            int ac = (int) round((th[p] + M_PI_F) / (2.0f * M_PI_F) * kn);                \
            ac = ((ac % kn) + kn) % kn;                                                    \
            const float u = lr[p] <= fmi                                                   \
                ? (lr[p] - flo) / fmax(2.0f*(fmi - flo), 1e-30f)                           \
                : 0.5f + (lr[p] - fmi) / fmax(2.0f*(fhi - fmi), 1e-30f);                   \
            const int mc = clamp((int) round(clamp(u, 0.0f, 1.0f) * last), 0, last);       \
            const uint code = (uint)(ac * nlv + mc);                                       \
            for (int i = 0; i < (BITS); ++i) {                                             \
                const int bit = t * (BITS) + i;                                            \
                if (code & (1u << i)) { buf[bit >> 3] |= (uint8_t)(1u << (bit & 7)); }     \
            }                                                                              \
        }                                                                                  \
        for (int i = 0; i < (BITS); ++i) { dst.qs[(g / 8) * (BITS) + i] = buf[i]; }        \
    }                                                                                      \
}

NEURON_QUANT(3,  7)
NEURON_QUANT(4,  9)
NEURON_QUANT(5, 11)
NEURON_QUANT(6, 13)
NEURON_QUANT(7, 15)
