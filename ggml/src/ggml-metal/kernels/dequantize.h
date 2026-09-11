#pragma once

#include "common.h"

#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL
#if defined(GGML_METAL_EMBED_LIBRARY)
__embed_ggml-common.h__
#else
#include "ggml-common.h"
#endif

#define QK_NL 16 // shared by mul_mm and get_rows_q instantiations

// NOTE: this is not dequantizing - we are simply fitting the template
template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f32_t4(device const float4 * src, short il, thread type4 & reg) {
    reg = (type4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}

#if defined(GGML_METAL_HAS_BF16)
template <typename type4x4>
void dequantize_bf16(device const bfloat4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_bf16_t4(device const bfloat4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}
#endif

template <typename type4x4>
void dequantize_q1_0(device const block_q1_0 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * qs = xb->qs;
    const float d = xb->d;
    const float neg_d = -d;

    const int byte_offset = il * 2;  // il*16 bits = il*2 bytes
    const uint8_t b0 = qs[byte_offset];
    const uint8_t b1 = qs[byte_offset + 1];

    float4x4 reg_f;

    reg_f[0][0] = select(neg_d, d, bool(b0 & 0x01));
    reg_f[0][1] = select(neg_d, d, bool(b0 & 0x02));
    reg_f[0][2] = select(neg_d, d, bool(b0 & 0x04));
    reg_f[0][3] = select(neg_d, d, bool(b0 & 0x08));
    reg_f[1][0] = select(neg_d, d, bool(b0 & 0x10));
    reg_f[1][1] = select(neg_d, d, bool(b0 & 0x20));
    reg_f[1][2] = select(neg_d, d, bool(b0 & 0x40));
    reg_f[1][3] = select(neg_d, d, bool(b0 & 0x80));

    reg_f[2][0] = select(neg_d, d, bool(b1 & 0x01));
    reg_f[2][1] = select(neg_d, d, bool(b1 & 0x02));
    reg_f[2][2] = select(neg_d, d, bool(b1 & 0x04));
    reg_f[2][3] = select(neg_d, d, bool(b1 & 0x08));
    reg_f[3][0] = select(neg_d, d, bool(b1 & 0x10));
    reg_f[3][1] = select(neg_d, d, bool(b1 & 0x20));
    reg_f[3][2] = select(neg_d, d, bool(b1 & 0x40));
    reg_f[3][3] = select(neg_d, d, bool(b1 & 0x80));

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q1_0_t4(device const block_q1_0 * xb, short il, thread type4 & reg) {
    const float d = xb->d;
    const float neg_d = -d;
    const int base = il * 4;
    const uint8_t byte = xb->qs[base / 8];
    const int s = base % 8;

    float4 reg_f;
    reg_f[0] = select(neg_d, d, bool((byte >> (s    )) & 1));
    reg_f[1] = select(neg_d, d, bool((byte >> (s + 1)) & 1));
    reg_f[2] = select(neg_d, d, bool((byte >> (s + 2)) & 1));
    reg_f[3] = select(neg_d, d, bool((byte >> (s + 3)) & 1));

    reg = (type4) reg_f;
}

template <typename type4x4>
void dequantize_q2_0(device const block_q2_0 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * qs = xb->qs;
    const float d = xb->d;

    const int byte_offset = il * 4;  // il*16 elements = il*4 bytes (4 elements per byte)
    float4x4 reg_f;

    for (int i = 0; i < 4; i++) {
        const uint8_t b = qs[byte_offset + i];
        reg_f[i][0] = ((float)((b >> 0) & 3) - 1.0f) * d;
        reg_f[i][1] = ((float)((b >> 2) & 3) - 1.0f) * d;
        reg_f[i][2] = ((float)((b >> 4) & 3) - 1.0f) * d;
        reg_f[i][3] = ((float)((b >> 6) & 3) - 1.0f) * d;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q2_0_t4(device const block_q2_0 * xb, short il, thread type4 & reg) {
    const float d = xb->d;
    const uint8_t b = xb->qs[il];

    float4 reg_f;
    reg_f[0] = ((float)((b >> 0) & 3) - 1.0f) * d;
    reg_f[1] = ((float)((b >> 2) & 3) - 1.0f) * d;
    reg_f[2] = ((float)((b >> 4) & 3) - 1.0f) * d;
    reg_f[3] = ((float)((b >> 6) & 3) - 1.0f) * d;

    reg = (type4) reg_f;
}

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_0_t4(device const block_q4_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}



template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_1_t4(device const block_q4_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + m;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + m;
    }
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_0_t4(device const block_q5_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + md;
        reg[2*ii + 1] = d * x1 + md;
    }
}

template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_1_t4(device const block_q5_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + m;
        reg[2*ii + 1] = d * x1 + m;
    }
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    reg = (type4) (float4(qs[il]) * d);
}

template <typename type4x4>
void dequantize_mxfp4(device const block_mxfp4 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * q2 = (device const uint8_t *)xb->qs;

    const float d = e8m0_to_fp32(xb->e);
    const uint8_t shr = il >= 1 ? 4 : 0;

    for (int i = 0; i < 4; ++i) {
        reg[i][0] = d * kvalues_mxfp4_f[(q2[4*i + 0] >> shr) & 0x0F];
        reg[i][1] = d * kvalues_mxfp4_f[(q2[4*i + 1] >> shr) & 0x0F];
        reg[i][2] = d * kvalues_mxfp4_f[(q2[4*i + 2] >> shr) & 0x0F];
        reg[i][3] = d * kvalues_mxfp4_f[(q2[4*i + 3] >> shr) & 0x0F];
    }
}

template <typename type4>
void dequantize_mxfp4_t4(device const block_mxfp4 * xb, short il, thread type4 & reg) {
    device const uint8_t * q2 = (device const uint8_t *)xb->qs;

    const float d = e8m0_to_fp32(xb->e);
    const short il4 = il%4;

    const uint8_t shr = il >= 4 ? 4 : 0;

    reg[0] = d * kvalues_mxfp4_f[(q2[4*il4 + 0] >> shr) & 0x0F];
    reg[1] = d * kvalues_mxfp4_f[(q2[4*il4 + 1] >> shr) & 0x0F];
    reg[2] = d * kvalues_mxfp4_f[(q2[4*il4 + 2] >> shr) & 0x0F];
    reg[3] = d * kvalues_mxfp4_f[(q2[4*il4 + 3] >> shr) & 0x0F];
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * q3 = xb->qs + 8*ib32;
    device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_s(device const block_iq3_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 8*ib32;
    device const uint8_t * signs = xb->signs + 4*ib32 + 2*il;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]);
        reg[1][i] = dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]);
    }
    grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]);
        reg[3][i] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]);
    }
}

template <typename type4x4>
void dequantize_iq2_s(device const block_iq2_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * signs = qs + QK_K/8;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
    for (int i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]);
        reg[i/4+2][i%4] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]);
    }
}

template <typename type4x4>
void dequantize_iq1_s(device const block_iq1_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    const float d = xb->d;
    device const uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
    device const uint16_t * qh = xb->qh;
    const float dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
    const float ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
    const uint16_t h = qh[ib32] >> 6*il;
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml;
        reg[1][i] = dl * (grid1[i] >>  4) + ml;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml;
        reg[3][i] = dl * (grid2[i] >>  4) + ml;
    }
}

template <typename type4x4>
void dequantize_iq1_m(device const block_iq1_m * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * sc = (device const uint16_t *)xb->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = scale.f16;

    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * qh = xb->qh + 2*ib32 + il;

    const float dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
    const float ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    const float ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml1;
        reg[1][i] = dl * (grid1[i] >>  4) + ml1;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml2;
        reg[3][i] = dl * (grid2[i] >>  4) + ml2;
    }
}

template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template <typename type4>
void dequantize_iq4_nl_t4(device const block_iq4_nl * xb, short il, thread type4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    aux32 = ((q4[2*(il%4)] | (q4[2*(il%4)+1] << 16)) >> 4*(il/4)) & 0x0f0f0f0f;
    reg[0] = d * kvalues_iq4nl_f[q8[0]];
    reg[1] = d * kvalues_iq4nl_f[q8[1]];
    reg[2] = d * kvalues_iq4nl_f[q8[2]];
    reg[3] = d * kvalues_iq4nl_f[q8[3]];
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template <typename type4x4>
void dequantize_tq2_0(device const block_tq2_0 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * qs = xb->qs;
    const float d = xb->d;

    float4x4 reg_f;

    // 2 bits per element, 4 elements per byte, 128 elements per 32-byte group
    const short base = il * 16;
    for (int k = 0; k < 16; k++) {
        const int i = base + k;
        const int byte = ((i >> 7) & 1) * 32 + (i & 31);
        const int l = (i >> 5) & 3;
        reg_f[k/4][k%4] = d * (float)(((qs[byte] >> (2*l)) & 3) - 1);
    }

    reg = (type4x4) reg_f;
}



// Angle table in the CONSTANT address space. theta depends only on the code, so these are
// compile-time values -- but cos() is not a constant expression in Metal, so the literals
// are generated. This is what lets the TILE dequantiser drop its cos/sin: unlike mul_mv it
// takes no threadgroup argument, and constant-space reads are hardware cached.
//
// Built at the widest grid (m8 -> A = 2^10 = 1024); a narrower A strides it by
// 1 << (10 - log2(A)), the same trick neuron_ang_at() uses on the CPU side.
constant half2 kNeuronAng[NEURON_ANGTAB] = {
    half2(-1.00000000f,-0.00000000f), half2(-0.99998118f,-0.00613588f), half2(-0.99992470f,-0.01227154f), half2(-0.99983058f,-0.01840673f),
    half2(-0.99969882f,-0.02454123f), half2(-0.99952942f,-0.03067480f), half2(-0.99932238f,-0.03680722f), half2(-0.99907773f,-0.04293826f),
    half2(-0.99879546f,-0.04906767f), half2(-0.99847558f,-0.05519524f), half2(-0.99811811f,-0.06132074f), half2(-0.99772307f,-0.06744392f),
    half2(-0.99729046f,-0.07356456f), half2(-0.99682030f,-0.07968244f), half2(-0.99631261f,-0.08579731f), half2(-0.99576741f,-0.09190896f),
    half2(-0.99518473f,-0.09801714f), half2(-0.99456457f,-0.10412163f), half2(-0.99390697f,-0.11022221f), half2(-0.99321195f,-0.11631863f),
    half2(-0.99247953f,-0.12241068f), half2(-0.99170975f,-0.12849811f), half2(-0.99090264f,-0.13458071f), half2(-0.99005821f,-0.14065824f),
    half2(-0.98917651f,-0.14673047f), half2(-0.98825757f,-0.15279719f), half2(-0.98730142f,-0.15885814f), half2(-0.98630810f,-0.16491312f),
    half2(-0.98527764f,-0.17096189f), half2(-0.98421009f,-0.17700422f), half2(-0.98310549f,-0.18303989f), half2(-0.98196387f,-0.18906866f),
    half2(-0.98078528f,-0.19509032f), half2(-0.97956977f,-0.20110463f), half2(-0.97831737f,-0.20711138f), half2(-0.97702814f,-0.21311032f),
    half2(-0.97570213f,-0.21910124f), half2(-0.97433938f,-0.22508391f), half2(-0.97293995f,-0.23105811f), half2(-0.97150389f,-0.23702361f),
    half2(-0.97003125f,-0.24298018f), half2(-0.96852209f,-0.24892761f), half2(-0.96697647f,-0.25486566f), half2(-0.96539444f,-0.26079412f),
    half2(-0.96377607f,-0.26671276f), half2(-0.96212140f,-0.27262136f), half2(-0.96043052f,-0.27851969f), half2(-0.95870347f,-0.28440754f),
    half2(-0.95694034f,-0.29028468f), half2(-0.95514117f,-0.29615089f), half2(-0.95330604f,-0.30200595f), half2(-0.95143502f,-0.30784964f),
    half2(-0.94952818f,-0.31368174f), half2(-0.94758559f,-0.31950203f), half2(-0.94560733f,-0.32531029f), half2(-0.94359346f,-0.33110631f),
    half2(-0.94154407f,-0.33688985f), half2(-0.93945922f,-0.34266072f), half2(-0.93733901f,-0.34841868f), half2(-0.93518351f,-0.35416353f),
    half2(-0.93299280f,-0.35989504f), half2(-0.93076696f,-0.36561300f), half2(-0.92850608f,-0.37131719f), half2(-0.92621024f,-0.37700741f),
    half2(-0.92387953f,-0.38268343f), half2(-0.92151404f,-0.38834505f), half2(-0.91911385f,-0.39399204f), half2(-0.91667906f,-0.39962420f),
    half2(-0.91420976f,-0.40524131f), half2(-0.91170603f,-0.41084317f), half2(-0.90916798f,-0.41642956f), half2(-0.90659570f,-0.42200027f),
    half2(-0.90398929f,-0.42755509f), half2(-0.90134885f,-0.43309382f), half2(-0.89867447f,-0.43861624f), half2(-0.89596625f,-0.44412214f),
    half2(-0.89322430f,-0.44961133f), half2(-0.89044872f,-0.45508359f), half2(-0.88763962f,-0.46053871f), half2(-0.88479710f,-0.46597650f),
    half2(-0.88192126f,-0.47139674f), half2(-0.87901223f,-0.47679923f), half2(-0.87607009f,-0.48218377f), half2(-0.87309498f,-0.48755016f),
    half2(-0.87008699f,-0.49289819f), half2(-0.86704625f,-0.49822767f), half2(-0.86397286f,-0.50353838f), half2(-0.86086694f,-0.50883014f),
    half2(-0.85772861f,-0.51410274f), half2(-0.85455799f,-0.51935599f), half2(-0.85135519f,-0.52458968f), half2(-0.84812034f,-0.52980362f),
    half2(-0.84485357f,-0.53499762f), half2(-0.84155498f,-0.54017147f), half2(-0.83822471f,-0.54532499f), half2(-0.83486287f,-0.55045797f),
    half2(-0.83146961f,-0.55557023f), half2(-0.82804505f,-0.56066158f), half2(-0.82458930f,-0.56573181f), half2(-0.82110251f,-0.57078075f),
    half2(-0.81758481f,-0.57580819f), half2(-0.81403633f,-0.58081396f), half2(-0.81045720f,-0.58579786f), half2(-0.80684755f,-0.59075970f),
    half2(-0.80320753f,-0.59569930f), half2(-0.79953727f,-0.60061648f), half2(-0.79583690f,-0.60551104f), half2(-0.79210658f,-0.61038281f),
    half2(-0.78834643f,-0.61523159f), half2(-0.78455660f,-0.62005721f), half2(-0.78073723f,-0.62485949f), half2(-0.77688847f,-0.62963824f),
    half2(-0.77301045f,-0.63439328f), half2(-0.76910334f,-0.63912444f), half2(-0.76516727f,-0.64383154f), half2(-0.76120239f,-0.64851440f),
    half2(-0.75720885f,-0.65317284f), half2(-0.75318680f,-0.65780669f), half2(-0.74913639f,-0.66241578f), half2(-0.74505779f,-0.66699992f),
    half2(-0.74095113f,-0.67155895f), half2(-0.73681657f,-0.67609270f), half2(-0.73265427f,-0.68060100f), half2(-0.72846439f,-0.68508367f),
    half2(-0.72424708f,-0.68954054f), half2(-0.72000251f,-0.69397146f), half2(-0.71573083f,-0.69837625f), half2(-0.71143220f,-0.70275474f),
    half2(-0.70710678f,-0.70710678f), half2(-0.70275474f,-0.71143220f), half2(-0.69837625f,-0.71573083f), half2(-0.69397146f,-0.72000251f),
    half2(-0.68954054f,-0.72424708f), half2(-0.68508367f,-0.72846439f), half2(-0.68060100f,-0.73265427f), half2(-0.67609270f,-0.73681657f),
    half2(-0.67155895f,-0.74095113f), half2(-0.66699992f,-0.74505779f), half2(-0.66241578f,-0.74913639f), half2(-0.65780669f,-0.75318680f),
    half2(-0.65317284f,-0.75720885f), half2(-0.64851440f,-0.76120239f), half2(-0.64383154f,-0.76516727f), half2(-0.63912444f,-0.76910334f),
    half2(-0.63439328f,-0.77301045f), half2(-0.62963824f,-0.77688847f), half2(-0.62485949f,-0.78073723f), half2(-0.62005721f,-0.78455660f),
    half2(-0.61523159f,-0.78834643f), half2(-0.61038281f,-0.79210658f), half2(-0.60551104f,-0.79583690f), half2(-0.60061648f,-0.79953727f),
    half2(-0.59569930f,-0.80320753f), half2(-0.59075970f,-0.80684755f), half2(-0.58579786f,-0.81045720f), half2(-0.58081396f,-0.81403633f),
    half2(-0.57580819f,-0.81758481f), half2(-0.57078075f,-0.82110251f), half2(-0.56573181f,-0.82458930f), half2(-0.56066158f,-0.82804505f),
    half2(-0.55557023f,-0.83146961f), half2(-0.55045797f,-0.83486287f), half2(-0.54532499f,-0.83822471f), half2(-0.54017147f,-0.84155498f),
    half2(-0.53499762f,-0.84485357f), half2(-0.52980362f,-0.84812034f), half2(-0.52458968f,-0.85135519f), half2(-0.51935599f,-0.85455799f),
    half2(-0.51410274f,-0.85772861f), half2(-0.50883014f,-0.86086694f), half2(-0.50353838f,-0.86397286f), half2(-0.49822767f,-0.86704625f),
    half2(-0.49289819f,-0.87008699f), half2(-0.48755016f,-0.87309498f), half2(-0.48218377f,-0.87607009f), half2(-0.47679923f,-0.87901223f),
    half2(-0.47139674f,-0.88192126f), half2(-0.46597650f,-0.88479710f), half2(-0.46053871f,-0.88763962f), half2(-0.45508359f,-0.89044872f),
    half2(-0.44961133f,-0.89322430f), half2(-0.44412214f,-0.89596625f), half2(-0.43861624f,-0.89867447f), half2(-0.43309382f,-0.90134885f),
    half2(-0.42755509f,-0.90398929f), half2(-0.42200027f,-0.90659570f), half2(-0.41642956f,-0.90916798f), half2(-0.41084317f,-0.91170603f),
    half2(-0.40524131f,-0.91420976f), half2(-0.39962420f,-0.91667906f), half2(-0.39399204f,-0.91911385f), half2(-0.38834505f,-0.92151404f),
    half2(-0.38268343f,-0.92387953f), half2(-0.37700741f,-0.92621024f), half2(-0.37131719f,-0.92850608f), half2(-0.36561300f,-0.93076696f),
    half2(-0.35989504f,-0.93299280f), half2(-0.35416353f,-0.93518351f), half2(-0.34841868f,-0.93733901f), half2(-0.34266072f,-0.93945922f),
    half2(-0.33688985f,-0.94154407f), half2(-0.33110631f,-0.94359346f), half2(-0.32531029f,-0.94560733f), half2(-0.31950203f,-0.94758559f),
    half2(-0.31368174f,-0.94952818f), half2(-0.30784964f,-0.95143502f), half2(-0.30200595f,-0.95330604f), half2(-0.29615089f,-0.95514117f),
    half2(-0.29028468f,-0.95694034f), half2(-0.28440754f,-0.95870347f), half2(-0.27851969f,-0.96043052f), half2(-0.27262136f,-0.96212140f),
    half2(-0.26671276f,-0.96377607f), half2(-0.26079412f,-0.96539444f), half2(-0.25486566f,-0.96697647f), half2(-0.24892761f,-0.96852209f),
    half2(-0.24298018f,-0.97003125f), half2(-0.23702361f,-0.97150389f), half2(-0.23105811f,-0.97293995f), half2(-0.22508391f,-0.97433938f),
    half2(-0.21910124f,-0.97570213f), half2(-0.21311032f,-0.97702814f), half2(-0.20711138f,-0.97831737f), half2(-0.20110463f,-0.97956977f),
    half2(-0.19509032f,-0.98078528f), half2(-0.18906866f,-0.98196387f), half2(-0.18303989f,-0.98310549f), half2(-0.17700422f,-0.98421009f),
    half2(-0.17096189f,-0.98527764f), half2(-0.16491312f,-0.98630810f), half2(-0.15885814f,-0.98730142f), half2(-0.15279719f,-0.98825757f),
    half2(-0.14673047f,-0.98917651f), half2(-0.14065824f,-0.99005821f), half2(-0.13458071f,-0.99090264f), half2(-0.12849811f,-0.99170975f),
    half2(-0.12241068f,-0.99247953f), half2(-0.11631863f,-0.99321195f), half2(-0.11022221f,-0.99390697f), half2(-0.10412163f,-0.99456457f),
    half2(-0.09801714f,-0.99518473f), half2(-0.09190896f,-0.99576741f), half2(-0.08579731f,-0.99631261f), half2(-0.07968244f,-0.99682030f),
    half2(-0.07356456f,-0.99729046f), half2(-0.06744392f,-0.99772307f), half2(-0.06132074f,-0.99811811f), half2(-0.05519524f,-0.99847558f),
    half2(-0.04906767f,-0.99879546f), half2(-0.04293826f,-0.99907773f), half2(-0.03680722f,-0.99932238f), half2(-0.03067480f,-0.99952942f),
    half2(-0.02454123f,-0.99969882f), half2(-0.01840673f,-0.99983058f), half2(-0.01227154f,-0.99992470f), half2(-0.00613588f,-0.99998118f),
    half2(+0.00000000f,-1.00000000f), half2(+0.00613588f,-0.99998118f), half2(+0.01227154f,-0.99992470f), half2(+0.01840673f,-0.99983058f),
    half2(+0.02454123f,-0.99969882f), half2(+0.03067480f,-0.99952942f), half2(+0.03680722f,-0.99932238f), half2(+0.04293826f,-0.99907773f),
    half2(+0.04906767f,-0.99879546f), half2(+0.05519524f,-0.99847558f), half2(+0.06132074f,-0.99811811f), half2(+0.06744392f,-0.99772307f),
    half2(+0.07356456f,-0.99729046f), half2(+0.07968244f,-0.99682030f), half2(+0.08579731f,-0.99631261f), half2(+0.09190896f,-0.99576741f),
    half2(+0.09801714f,-0.99518473f), half2(+0.10412163f,-0.99456457f), half2(+0.11022221f,-0.99390697f), half2(+0.11631863f,-0.99321195f),
    half2(+0.12241068f,-0.99247953f), half2(+0.12849811f,-0.99170975f), half2(+0.13458071f,-0.99090264f), half2(+0.14065824f,-0.99005821f),
    half2(+0.14673047f,-0.98917651f), half2(+0.15279719f,-0.98825757f), half2(+0.15885814f,-0.98730142f), half2(+0.16491312f,-0.98630810f),
    half2(+0.17096189f,-0.98527764f), half2(+0.17700422f,-0.98421009f), half2(+0.18303989f,-0.98310549f), half2(+0.18906866f,-0.98196387f),
    half2(+0.19509032f,-0.98078528f), half2(+0.20110463f,-0.97956977f), half2(+0.20711138f,-0.97831737f), half2(+0.21311032f,-0.97702814f),
    half2(+0.21910124f,-0.97570213f), half2(+0.22508391f,-0.97433938f), half2(+0.23105811f,-0.97293995f), half2(+0.23702361f,-0.97150389f),
    half2(+0.24298018f,-0.97003125f), half2(+0.24892761f,-0.96852209f), half2(+0.25486566f,-0.96697647f), half2(+0.26079412f,-0.96539444f),
    half2(+0.26671276f,-0.96377607f), half2(+0.27262136f,-0.96212140f), half2(+0.27851969f,-0.96043052f), half2(+0.28440754f,-0.95870347f),
    half2(+0.29028468f,-0.95694034f), half2(+0.29615089f,-0.95514117f), half2(+0.30200595f,-0.95330604f), half2(+0.30784964f,-0.95143502f),
    half2(+0.31368174f,-0.94952818f), half2(+0.31950203f,-0.94758559f), half2(+0.32531029f,-0.94560733f), half2(+0.33110631f,-0.94359346f),
    half2(+0.33688985f,-0.94154407f), half2(+0.34266072f,-0.93945922f), half2(+0.34841868f,-0.93733901f), half2(+0.35416353f,-0.93518351f),
    half2(+0.35989504f,-0.93299280f), half2(+0.36561300f,-0.93076696f), half2(+0.37131719f,-0.92850608f), half2(+0.37700741f,-0.92621024f),
    half2(+0.38268343f,-0.92387953f), half2(+0.38834505f,-0.92151404f), half2(+0.39399204f,-0.91911385f), half2(+0.39962420f,-0.91667906f),
    half2(+0.40524131f,-0.91420976f), half2(+0.41084317f,-0.91170603f), half2(+0.41642956f,-0.90916798f), half2(+0.42200027f,-0.90659570f),
    half2(+0.42755509f,-0.90398929f), half2(+0.43309382f,-0.90134885f), half2(+0.43861624f,-0.89867447f), half2(+0.44412214f,-0.89596625f),
    half2(+0.44961133f,-0.89322430f), half2(+0.45508359f,-0.89044872f), half2(+0.46053871f,-0.88763962f), half2(+0.46597650f,-0.88479710f),
    half2(+0.47139674f,-0.88192126f), half2(+0.47679923f,-0.87901223f), half2(+0.48218377f,-0.87607009f), half2(+0.48755016f,-0.87309498f),
    half2(+0.49289819f,-0.87008699f), half2(+0.49822767f,-0.86704625f), half2(+0.50353838f,-0.86397286f), half2(+0.50883014f,-0.86086694f),
    half2(+0.51410274f,-0.85772861f), half2(+0.51935599f,-0.85455799f), half2(+0.52458968f,-0.85135519f), half2(+0.52980362f,-0.84812034f),
    half2(+0.53499762f,-0.84485357f), half2(+0.54017147f,-0.84155498f), half2(+0.54532499f,-0.83822471f), half2(+0.55045797f,-0.83486287f),
    half2(+0.55557023f,-0.83146961f), half2(+0.56066158f,-0.82804505f), half2(+0.56573181f,-0.82458930f), half2(+0.57078075f,-0.82110251f),
    half2(+0.57580819f,-0.81758481f), half2(+0.58081396f,-0.81403633f), half2(+0.58579786f,-0.81045720f), half2(+0.59075970f,-0.80684755f),
    half2(+0.59569930f,-0.80320753f), half2(+0.60061648f,-0.79953727f), half2(+0.60551104f,-0.79583690f), half2(+0.61038281f,-0.79210658f),
    half2(+0.61523159f,-0.78834643f), half2(+0.62005721f,-0.78455660f), half2(+0.62485949f,-0.78073723f), half2(+0.62963824f,-0.77688847f),
    half2(+0.63439328f,-0.77301045f), half2(+0.63912444f,-0.76910334f), half2(+0.64383154f,-0.76516727f), half2(+0.64851440f,-0.76120239f),
    half2(+0.65317284f,-0.75720885f), half2(+0.65780669f,-0.75318680f), half2(+0.66241578f,-0.74913639f), half2(+0.66699992f,-0.74505779f),
    half2(+0.67155895f,-0.74095113f), half2(+0.67609270f,-0.73681657f), half2(+0.68060100f,-0.73265427f), half2(+0.68508367f,-0.72846439f),
    half2(+0.68954054f,-0.72424708f), half2(+0.69397146f,-0.72000251f), half2(+0.69837625f,-0.71573083f), half2(+0.70275474f,-0.71143220f),
    half2(+0.70710678f,-0.70710678f), half2(+0.71143220f,-0.70275474f), half2(+0.71573083f,-0.69837625f), half2(+0.72000251f,-0.69397146f),
    half2(+0.72424708f,-0.68954054f), half2(+0.72846439f,-0.68508367f), half2(+0.73265427f,-0.68060100f), half2(+0.73681657f,-0.67609270f),
    half2(+0.74095113f,-0.67155895f), half2(+0.74505779f,-0.66699992f), half2(+0.74913639f,-0.66241578f), half2(+0.75318680f,-0.65780669f),
    half2(+0.75720885f,-0.65317284f), half2(+0.76120239f,-0.64851440f), half2(+0.76516727f,-0.64383154f), half2(+0.76910334f,-0.63912444f),
    half2(+0.77301045f,-0.63439328f), half2(+0.77688847f,-0.62963824f), half2(+0.78073723f,-0.62485949f), half2(+0.78455660f,-0.62005721f),
    half2(+0.78834643f,-0.61523159f), half2(+0.79210658f,-0.61038281f), half2(+0.79583690f,-0.60551104f), half2(+0.79953727f,-0.60061648f),
    half2(+0.80320753f,-0.59569930f), half2(+0.80684755f,-0.59075970f), half2(+0.81045720f,-0.58579786f), half2(+0.81403633f,-0.58081396f),
    half2(+0.81758481f,-0.57580819f), half2(+0.82110251f,-0.57078075f), half2(+0.82458930f,-0.56573181f), half2(+0.82804505f,-0.56066158f),
    half2(+0.83146961f,-0.55557023f), half2(+0.83486287f,-0.55045797f), half2(+0.83822471f,-0.54532499f), half2(+0.84155498f,-0.54017147f),
    half2(+0.84485357f,-0.53499762f), half2(+0.84812034f,-0.52980362f), half2(+0.85135519f,-0.52458968f), half2(+0.85455799f,-0.51935599f),
    half2(+0.85772861f,-0.51410274f), half2(+0.86086694f,-0.50883014f), half2(+0.86397286f,-0.50353838f), half2(+0.86704625f,-0.49822767f),
    half2(+0.87008699f,-0.49289819f), half2(+0.87309498f,-0.48755016f), half2(+0.87607009f,-0.48218377f), half2(+0.87901223f,-0.47679923f),
    half2(+0.88192126f,-0.47139674f), half2(+0.88479710f,-0.46597650f), half2(+0.88763962f,-0.46053871f), half2(+0.89044872f,-0.45508359f),
    half2(+0.89322430f,-0.44961133f), half2(+0.89596625f,-0.44412214f), half2(+0.89867447f,-0.43861624f), half2(+0.90134885f,-0.43309382f),
    half2(+0.90398929f,-0.42755509f), half2(+0.90659570f,-0.42200027f), half2(+0.90916798f,-0.41642956f), half2(+0.91170603f,-0.41084317f),
    half2(+0.91420976f,-0.40524131f), half2(+0.91667906f,-0.39962420f), half2(+0.91911385f,-0.39399204f), half2(+0.92151404f,-0.38834505f),
    half2(+0.92387953f,-0.38268343f), half2(+0.92621024f,-0.37700741f), half2(+0.92850608f,-0.37131719f), half2(+0.93076696f,-0.36561300f),
    half2(+0.93299280f,-0.35989504f), half2(+0.93518351f,-0.35416353f), half2(+0.93733901f,-0.34841868f), half2(+0.93945922f,-0.34266072f),
    half2(+0.94154407f,-0.33688985f), half2(+0.94359346f,-0.33110631f), half2(+0.94560733f,-0.32531029f), half2(+0.94758559f,-0.31950203f),
    half2(+0.94952818f,-0.31368174f), half2(+0.95143502f,-0.30784964f), half2(+0.95330604f,-0.30200595f), half2(+0.95514117f,-0.29615089f),
    half2(+0.95694034f,-0.29028468f), half2(+0.95870347f,-0.28440754f), half2(+0.96043052f,-0.27851969f), half2(+0.96212140f,-0.27262136f),
    half2(+0.96377607f,-0.26671276f), half2(+0.96539444f,-0.26079412f), half2(+0.96697647f,-0.25486566f), half2(+0.96852209f,-0.24892761f),
    half2(+0.97003125f,-0.24298018f), half2(+0.97150389f,-0.23702361f), half2(+0.97293995f,-0.23105811f), half2(+0.97433938f,-0.22508391f),
    half2(+0.97570213f,-0.21910124f), half2(+0.97702814f,-0.21311032f), half2(+0.97831737f,-0.20711138f), half2(+0.97956977f,-0.20110463f),
    half2(+0.98078528f,-0.19509032f), half2(+0.98196387f,-0.18906866f), half2(+0.98310549f,-0.18303989f), half2(+0.98421009f,-0.17700422f),
    half2(+0.98527764f,-0.17096189f), half2(+0.98630810f,-0.16491312f), half2(+0.98730142f,-0.15885814f), half2(+0.98825757f,-0.15279719f),
    half2(+0.98917651f,-0.14673047f), half2(+0.99005821f,-0.14065824f), half2(+0.99090264f,-0.13458071f), half2(+0.99170975f,-0.12849811f),
    half2(+0.99247953f,-0.12241068f), half2(+0.99321195f,-0.11631863f), half2(+0.99390697f,-0.11022221f), half2(+0.99456457f,-0.10412163f),
    half2(+0.99518473f,-0.09801714f), half2(+0.99576741f,-0.09190896f), half2(+0.99631261f,-0.08579731f), half2(+0.99682030f,-0.07968244f),
    half2(+0.99729046f,-0.07356456f), half2(+0.99772307f,-0.06744392f), half2(+0.99811811f,-0.06132074f), half2(+0.99847558f,-0.05519524f),
    half2(+0.99879546f,-0.04906767f), half2(+0.99907773f,-0.04293826f), half2(+0.99932238f,-0.03680722f), half2(+0.99952942f,-0.03067480f),
    half2(+0.99969882f,-0.02454123f), half2(+0.99983058f,-0.01840673f), half2(+0.99992470f,-0.01227154f), half2(+0.99998118f,-0.00613588f),
    half2(+1.00000000f,+0.00000000f), half2(+0.99998118f,+0.00613588f), half2(+0.99992470f,+0.01227154f), half2(+0.99983058f,+0.01840673f),
    half2(+0.99969882f,+0.02454123f), half2(+0.99952942f,+0.03067480f), half2(+0.99932238f,+0.03680722f), half2(+0.99907773f,+0.04293826f),
    half2(+0.99879546f,+0.04906767f), half2(+0.99847558f,+0.05519524f), half2(+0.99811811f,+0.06132074f), half2(+0.99772307f,+0.06744392f),
    half2(+0.99729046f,+0.07356456f), half2(+0.99682030f,+0.07968244f), half2(+0.99631261f,+0.08579731f), half2(+0.99576741f,+0.09190896f),
    half2(+0.99518473f,+0.09801714f), half2(+0.99456457f,+0.10412163f), half2(+0.99390697f,+0.11022221f), half2(+0.99321195f,+0.11631863f),
    half2(+0.99247953f,+0.12241068f), half2(+0.99170975f,+0.12849811f), half2(+0.99090264f,+0.13458071f), half2(+0.99005821f,+0.14065824f),
    half2(+0.98917651f,+0.14673047f), half2(+0.98825757f,+0.15279719f), half2(+0.98730142f,+0.15885814f), half2(+0.98630810f,+0.16491312f),
    half2(+0.98527764f,+0.17096189f), half2(+0.98421009f,+0.17700422f), half2(+0.98310549f,+0.18303989f), half2(+0.98196387f,+0.18906866f),
    half2(+0.98078528f,+0.19509032f), half2(+0.97956977f,+0.20110463f), half2(+0.97831737f,+0.20711138f), half2(+0.97702814f,+0.21311032f),
    half2(+0.97570213f,+0.21910124f), half2(+0.97433938f,+0.22508391f), half2(+0.97293995f,+0.23105811f), half2(+0.97150389f,+0.23702361f),
    half2(+0.97003125f,+0.24298018f), half2(+0.96852209f,+0.24892761f), half2(+0.96697647f,+0.25486566f), half2(+0.96539444f,+0.26079412f),
    half2(+0.96377607f,+0.26671276f), half2(+0.96212140f,+0.27262136f), half2(+0.96043052f,+0.27851969f), half2(+0.95870347f,+0.28440754f),
    half2(+0.95694034f,+0.29028468f), half2(+0.95514117f,+0.29615089f), half2(+0.95330604f,+0.30200595f), half2(+0.95143502f,+0.30784964f),
    half2(+0.94952818f,+0.31368174f), half2(+0.94758559f,+0.31950203f), half2(+0.94560733f,+0.32531029f), half2(+0.94359346f,+0.33110631f),
    half2(+0.94154407f,+0.33688985f), half2(+0.93945922f,+0.34266072f), half2(+0.93733901f,+0.34841868f), half2(+0.93518351f,+0.35416353f),
    half2(+0.93299280f,+0.35989504f), half2(+0.93076696f,+0.36561300f), half2(+0.92850608f,+0.37131719f), half2(+0.92621024f,+0.37700741f),
    half2(+0.92387953f,+0.38268343f), half2(+0.92151404f,+0.38834505f), half2(+0.91911385f,+0.39399204f), half2(+0.91667906f,+0.39962420f),
    half2(+0.91420976f,+0.40524131f), half2(+0.91170603f,+0.41084317f), half2(+0.90916798f,+0.41642956f), half2(+0.90659570f,+0.42200027f),
    half2(+0.90398929f,+0.42755509f), half2(+0.90134885f,+0.43309382f), half2(+0.89867447f,+0.43861624f), half2(+0.89596625f,+0.44412214f),
    half2(+0.89322430f,+0.44961133f), half2(+0.89044872f,+0.45508359f), half2(+0.88763962f,+0.46053871f), half2(+0.88479710f,+0.46597650f),
    half2(+0.88192126f,+0.47139674f), half2(+0.87901223f,+0.47679923f), half2(+0.87607009f,+0.48218377f), half2(+0.87309498f,+0.48755016f),
    half2(+0.87008699f,+0.49289819f), half2(+0.86704625f,+0.49822767f), half2(+0.86397286f,+0.50353838f), half2(+0.86086694f,+0.50883014f),
    half2(+0.85772861f,+0.51410274f), half2(+0.85455799f,+0.51935599f), half2(+0.85135519f,+0.52458968f), half2(+0.84812034f,+0.52980362f),
    half2(+0.84485357f,+0.53499762f), half2(+0.84155498f,+0.54017147f), half2(+0.83822471f,+0.54532499f), half2(+0.83486287f,+0.55045797f),
    half2(+0.83146961f,+0.55557023f), half2(+0.82804505f,+0.56066158f), half2(+0.82458930f,+0.56573181f), half2(+0.82110251f,+0.57078075f),
    half2(+0.81758481f,+0.57580819f), half2(+0.81403633f,+0.58081396f), half2(+0.81045720f,+0.58579786f), half2(+0.80684755f,+0.59075970f),
    half2(+0.80320753f,+0.59569930f), half2(+0.79953727f,+0.60061648f), half2(+0.79583690f,+0.60551104f), half2(+0.79210658f,+0.61038281f),
    half2(+0.78834643f,+0.61523159f), half2(+0.78455660f,+0.62005721f), half2(+0.78073723f,+0.62485949f), half2(+0.77688847f,+0.62963824f),
    half2(+0.77301045f,+0.63439328f), half2(+0.76910334f,+0.63912444f), half2(+0.76516727f,+0.64383154f), half2(+0.76120239f,+0.64851440f),
    half2(+0.75720885f,+0.65317284f), half2(+0.75318680f,+0.65780669f), half2(+0.74913639f,+0.66241578f), half2(+0.74505779f,+0.66699992f),
    half2(+0.74095113f,+0.67155895f), half2(+0.73681657f,+0.67609270f), half2(+0.73265427f,+0.68060100f), half2(+0.72846439f,+0.68508367f),
    half2(+0.72424708f,+0.68954054f), half2(+0.72000251f,+0.69397146f), half2(+0.71573083f,+0.69837625f), half2(+0.71143220f,+0.70275474f),
    half2(+0.70710678f,+0.70710678f), half2(+0.70275474f,+0.71143220f), half2(+0.69837625f,+0.71573083f), half2(+0.69397146f,+0.72000251f),
    half2(+0.68954054f,+0.72424708f), half2(+0.68508367f,+0.72846439f), half2(+0.68060100f,+0.73265427f), half2(+0.67609270f,+0.73681657f),
    half2(+0.67155895f,+0.74095113f), half2(+0.66699992f,+0.74505779f), half2(+0.66241578f,+0.74913639f), half2(+0.65780669f,+0.75318680f),
    half2(+0.65317284f,+0.75720885f), half2(+0.64851440f,+0.76120239f), half2(+0.64383154f,+0.76516727f), half2(+0.63912444f,+0.76910334f),
    half2(+0.63439328f,+0.77301045f), half2(+0.62963824f,+0.77688847f), half2(+0.62485949f,+0.78073723f), half2(+0.62005721f,+0.78455660f),
    half2(+0.61523159f,+0.78834643f), half2(+0.61038281f,+0.79210658f), half2(+0.60551104f,+0.79583690f), half2(+0.60061648f,+0.79953727f),
    half2(+0.59569930f,+0.80320753f), half2(+0.59075970f,+0.80684755f), half2(+0.58579786f,+0.81045720f), half2(+0.58081396f,+0.81403633f),
    half2(+0.57580819f,+0.81758481f), half2(+0.57078075f,+0.82110251f), half2(+0.56573181f,+0.82458930f), half2(+0.56066158f,+0.82804505f),
    half2(+0.55557023f,+0.83146961f), half2(+0.55045797f,+0.83486287f), half2(+0.54532499f,+0.83822471f), half2(+0.54017147f,+0.84155498f),
    half2(+0.53499762f,+0.84485357f), half2(+0.52980362f,+0.84812034f), half2(+0.52458968f,+0.85135519f), half2(+0.51935599f,+0.85455799f),
    half2(+0.51410274f,+0.85772861f), half2(+0.50883014f,+0.86086694f), half2(+0.50353838f,+0.86397286f), half2(+0.49822767f,+0.86704625f),
    half2(+0.49289819f,+0.87008699f), half2(+0.48755016f,+0.87309498f), half2(+0.48218377f,+0.87607009f), half2(+0.47679923f,+0.87901223f),
    half2(+0.47139674f,+0.88192126f), half2(+0.46597650f,+0.88479710f), half2(+0.46053871f,+0.88763962f), half2(+0.45508359f,+0.89044872f),
    half2(+0.44961133f,+0.89322430f), half2(+0.44412214f,+0.89596625f), half2(+0.43861624f,+0.89867447f), half2(+0.43309382f,+0.90134885f),
    half2(+0.42755509f,+0.90398929f), half2(+0.42200027f,+0.90659570f), half2(+0.41642956f,+0.90916798f), half2(+0.41084317f,+0.91170603f),
    half2(+0.40524131f,+0.91420976f), half2(+0.39962420f,+0.91667906f), half2(+0.39399204f,+0.91911385f), half2(+0.38834505f,+0.92151404f),
    half2(+0.38268343f,+0.92387953f), half2(+0.37700741f,+0.92621024f), half2(+0.37131719f,+0.92850608f), half2(+0.36561300f,+0.93076696f),
    half2(+0.35989504f,+0.93299280f), half2(+0.35416353f,+0.93518351f), half2(+0.34841868f,+0.93733901f), half2(+0.34266072f,+0.93945922f),
    half2(+0.33688985f,+0.94154407f), half2(+0.33110631f,+0.94359346f), half2(+0.32531029f,+0.94560733f), half2(+0.31950203f,+0.94758559f),
    half2(+0.31368174f,+0.94952818f), half2(+0.30784964f,+0.95143502f), half2(+0.30200595f,+0.95330604f), half2(+0.29615089f,+0.95514117f),
    half2(+0.29028468f,+0.95694034f), half2(+0.28440754f,+0.95870347f), half2(+0.27851969f,+0.96043052f), half2(+0.27262136f,+0.96212140f),
    half2(+0.26671276f,+0.96377607f), half2(+0.26079412f,+0.96539444f), half2(+0.25486566f,+0.96697647f), half2(+0.24892761f,+0.96852209f),
    half2(+0.24298018f,+0.97003125f), half2(+0.23702361f,+0.97150389f), half2(+0.23105811f,+0.97293995f), half2(+0.22508391f,+0.97433938f),
    half2(+0.21910124f,+0.97570213f), half2(+0.21311032f,+0.97702814f), half2(+0.20711138f,+0.97831737f), half2(+0.20110463f,+0.97956977f),
    half2(+0.19509032f,+0.98078528f), half2(+0.18906866f,+0.98196387f), half2(+0.18303989f,+0.98310549f), half2(+0.17700422f,+0.98421009f),
    half2(+0.17096189f,+0.98527764f), half2(+0.16491312f,+0.98630810f), half2(+0.15885814f,+0.98730142f), half2(+0.15279719f,+0.98825757f),
    half2(+0.14673047f,+0.98917651f), half2(+0.14065824f,+0.99005821f), half2(+0.13458071f,+0.99090264f), half2(+0.12849811f,+0.99170975f),
    half2(+0.12241068f,+0.99247953f), half2(+0.11631863f,+0.99321195f), half2(+0.11022221f,+0.99390697f), half2(+0.10412163f,+0.99456457f),
    half2(+0.09801714f,+0.99518473f), half2(+0.09190896f,+0.99576741f), half2(+0.08579731f,+0.99631261f), half2(+0.07968244f,+0.99682030f),
    half2(+0.07356456f,+0.99729046f), half2(+0.06744392f,+0.99772307f), half2(+0.06132074f,+0.99811811f), half2(+0.05519524f,+0.99847558f),
    half2(+0.04906767f,+0.99879546f), half2(+0.04293826f,+0.99907773f), half2(+0.03680722f,+0.99932238f), half2(+0.03067480f,+0.99952942f),
    half2(+0.02454123f,+0.99969882f), half2(+0.01840673f,+0.99983058f), half2(+0.01227154f,+0.99992470f), half2(+0.00613588f,+0.99998118f),
    half2(+0.00000000f,+1.00000000f), half2(-0.00613588f,+0.99998118f), half2(-0.01227154f,+0.99992470f), half2(-0.01840673f,+0.99983058f),
    half2(-0.02454123f,+0.99969882f), half2(-0.03067480f,+0.99952942f), half2(-0.03680722f,+0.99932238f), half2(-0.04293826f,+0.99907773f),
    half2(-0.04906767f,+0.99879546f), half2(-0.05519524f,+0.99847558f), half2(-0.06132074f,+0.99811811f), half2(-0.06744392f,+0.99772307f),
    half2(-0.07356456f,+0.99729046f), half2(-0.07968244f,+0.99682030f), half2(-0.08579731f,+0.99631261f), half2(-0.09190896f,+0.99576741f),
    half2(-0.09801714f,+0.99518473f), half2(-0.10412163f,+0.99456457f), half2(-0.11022221f,+0.99390697f), half2(-0.11631863f,+0.99321195f),
    half2(-0.12241068f,+0.99247953f), half2(-0.12849811f,+0.99170975f), half2(-0.13458071f,+0.99090264f), half2(-0.14065824f,+0.99005821f),
    half2(-0.14673047f,+0.98917651f), half2(-0.15279719f,+0.98825757f), half2(-0.15885814f,+0.98730142f), half2(-0.16491312f,+0.98630810f),
    half2(-0.17096189f,+0.98527764f), half2(-0.17700422f,+0.98421009f), half2(-0.18303989f,+0.98310549f), half2(-0.18906866f,+0.98196387f),
    half2(-0.19509032f,+0.98078528f), half2(-0.20110463f,+0.97956977f), half2(-0.20711138f,+0.97831737f), half2(-0.21311032f,+0.97702814f),
    half2(-0.21910124f,+0.97570213f), half2(-0.22508391f,+0.97433938f), half2(-0.23105811f,+0.97293995f), half2(-0.23702361f,+0.97150389f),
    half2(-0.24298018f,+0.97003125f), half2(-0.24892761f,+0.96852209f), half2(-0.25486566f,+0.96697647f), half2(-0.26079412f,+0.96539444f),
    half2(-0.26671276f,+0.96377607f), half2(-0.27262136f,+0.96212140f), half2(-0.27851969f,+0.96043052f), half2(-0.28440754f,+0.95870347f),
    half2(-0.29028468f,+0.95694034f), half2(-0.29615089f,+0.95514117f), half2(-0.30200595f,+0.95330604f), half2(-0.30784964f,+0.95143502f),
    half2(-0.31368174f,+0.94952818f), half2(-0.31950203f,+0.94758559f), half2(-0.32531029f,+0.94560733f), half2(-0.33110631f,+0.94359346f),
    half2(-0.33688985f,+0.94154407f), half2(-0.34266072f,+0.93945922f), half2(-0.34841868f,+0.93733901f), half2(-0.35416353f,+0.93518351f),
    half2(-0.35989504f,+0.93299280f), half2(-0.36561300f,+0.93076696f), half2(-0.37131719f,+0.92850608f), half2(-0.37700741f,+0.92621024f),
    half2(-0.38268343f,+0.92387953f), half2(-0.38834505f,+0.92151404f), half2(-0.39399204f,+0.91911385f), half2(-0.39962420f,+0.91667906f),
    half2(-0.40524131f,+0.91420976f), half2(-0.41084317f,+0.91170603f), half2(-0.41642956f,+0.90916798f), half2(-0.42200027f,+0.90659570f),
    half2(-0.42755509f,+0.90398929f), half2(-0.43309382f,+0.90134885f), half2(-0.43861624f,+0.89867447f), half2(-0.44412214f,+0.89596625f),
    half2(-0.44961133f,+0.89322430f), half2(-0.45508359f,+0.89044872f), half2(-0.46053871f,+0.88763962f), half2(-0.46597650f,+0.88479710f),
    half2(-0.47139674f,+0.88192126f), half2(-0.47679923f,+0.87901223f), half2(-0.48218377f,+0.87607009f), half2(-0.48755016f,+0.87309498f),
    half2(-0.49289819f,+0.87008699f), half2(-0.49822767f,+0.86704625f), half2(-0.50353838f,+0.86397286f), half2(-0.50883014f,+0.86086694f),
    half2(-0.51410274f,+0.85772861f), half2(-0.51935599f,+0.85455799f), half2(-0.52458968f,+0.85135519f), half2(-0.52980362f,+0.84812034f),
    half2(-0.53499762f,+0.84485357f), half2(-0.54017147f,+0.84155498f), half2(-0.54532499f,+0.83822471f), half2(-0.55045797f,+0.83486287f),
    half2(-0.55557023f,+0.83146961f), half2(-0.56066158f,+0.82804505f), half2(-0.56573181f,+0.82458930f), half2(-0.57078075f,+0.82110251f),
    half2(-0.57580819f,+0.81758481f), half2(-0.58081396f,+0.81403633f), half2(-0.58579786f,+0.81045720f), half2(-0.59075970f,+0.80684755f),
    half2(-0.59569930f,+0.80320753f), half2(-0.60061648f,+0.79953727f), half2(-0.60551104f,+0.79583690f), half2(-0.61038281f,+0.79210658f),
    half2(-0.61523159f,+0.78834643f), half2(-0.62005721f,+0.78455660f), half2(-0.62485949f,+0.78073723f), half2(-0.62963824f,+0.77688847f),
    half2(-0.63439328f,+0.77301045f), half2(-0.63912444f,+0.76910334f), half2(-0.64383154f,+0.76516727f), half2(-0.64851440f,+0.76120239f),
    half2(-0.65317284f,+0.75720885f), half2(-0.65780669f,+0.75318680f), half2(-0.66241578f,+0.74913639f), half2(-0.66699992f,+0.74505779f),
    half2(-0.67155895f,+0.74095113f), half2(-0.67609270f,+0.73681657f), half2(-0.68060100f,+0.73265427f), half2(-0.68508367f,+0.72846439f),
    half2(-0.68954054f,+0.72424708f), half2(-0.69397146f,+0.72000251f), half2(-0.69837625f,+0.71573083f), half2(-0.70275474f,+0.71143220f),
    half2(-0.70710678f,+0.70710678f), half2(-0.71143220f,+0.70275474f), half2(-0.71573083f,+0.69837625f), half2(-0.72000251f,+0.69397146f),
    half2(-0.72424708f,+0.68954054f), half2(-0.72846439f,+0.68508367f), half2(-0.73265427f,+0.68060100f), half2(-0.73681657f,+0.67609270f),
    half2(-0.74095113f,+0.67155895f), half2(-0.74505779f,+0.66699992f), half2(-0.74913639f,+0.66241578f), half2(-0.75318680f,+0.65780669f),
    half2(-0.75720885f,+0.65317284f), half2(-0.76120239f,+0.64851440f), half2(-0.76516727f,+0.64383154f), half2(-0.76910334f,+0.63912444f),
    half2(-0.77301045f,+0.63439328f), half2(-0.77688847f,+0.62963824f), half2(-0.78073723f,+0.62485949f), half2(-0.78455660f,+0.62005721f),
    half2(-0.78834643f,+0.61523159f), half2(-0.79210658f,+0.61038281f), half2(-0.79583690f,+0.60551104f), half2(-0.79953727f,+0.60061648f),
    half2(-0.80320753f,+0.59569930f), half2(-0.80684755f,+0.59075970f), half2(-0.81045720f,+0.58579786f), half2(-0.81403633f,+0.58081396f),
    half2(-0.81758481f,+0.57580819f), half2(-0.82110251f,+0.57078075f), half2(-0.82458930f,+0.56573181f), half2(-0.82804505f,+0.56066158f),
    half2(-0.83146961f,+0.55557023f), half2(-0.83486287f,+0.55045797f), half2(-0.83822471f,+0.54532499f), half2(-0.84155498f,+0.54017147f),
    half2(-0.84485357f,+0.53499762f), half2(-0.84812034f,+0.52980362f), half2(-0.85135519f,+0.52458968f), half2(-0.85455799f,+0.51935599f),
    half2(-0.85772861f,+0.51410274f), half2(-0.86086694f,+0.50883014f), half2(-0.86397286f,+0.50353838f), half2(-0.86704625f,+0.49822767f),
    half2(-0.87008699f,+0.49289819f), half2(-0.87309498f,+0.48755016f), half2(-0.87607009f,+0.48218377f), half2(-0.87901223f,+0.47679923f),
    half2(-0.88192126f,+0.47139674f), half2(-0.88479710f,+0.46597650f), half2(-0.88763962f,+0.46053871f), half2(-0.89044872f,+0.45508359f),
    half2(-0.89322430f,+0.44961133f), half2(-0.89596625f,+0.44412214f), half2(-0.89867447f,+0.43861624f), half2(-0.90134885f,+0.43309382f),
    half2(-0.90398929f,+0.42755509f), half2(-0.90659570f,+0.42200027f), half2(-0.90916798f,+0.41642956f), half2(-0.91170603f,+0.41084317f),
    half2(-0.91420976f,+0.40524131f), half2(-0.91667906f,+0.39962420f), half2(-0.91911385f,+0.39399204f), half2(-0.92151404f,+0.38834505f),
    half2(-0.92387953f,+0.38268343f), half2(-0.92621024f,+0.37700741f), half2(-0.92850608f,+0.37131719f), half2(-0.93076696f,+0.36561300f),
    half2(-0.93299280f,+0.35989504f), half2(-0.93518351f,+0.35416353f), half2(-0.93733901f,+0.34841868f), half2(-0.93945922f,+0.34266072f),
    half2(-0.94154407f,+0.33688985f), half2(-0.94359346f,+0.33110631f), half2(-0.94560733f,+0.32531029f), half2(-0.94758559f,+0.31950203f),
    half2(-0.94952818f,+0.31368174f), half2(-0.95143502f,+0.30784964f), half2(-0.95330604f,+0.30200595f), half2(-0.95514117f,+0.29615089f),
    half2(-0.95694034f,+0.29028468f), half2(-0.95870347f,+0.28440754f), half2(-0.96043052f,+0.27851969f), half2(-0.96212140f,+0.27262136f),
    half2(-0.96377607f,+0.26671276f), half2(-0.96539444f,+0.26079412f), half2(-0.96697647f,+0.25486566f), half2(-0.96852209f,+0.24892761f),
    half2(-0.97003125f,+0.24298018f), half2(-0.97150389f,+0.23702361f), half2(-0.97293995f,+0.23105811f), half2(-0.97433938f,+0.22508391f),
    half2(-0.97570213f,+0.21910124f), half2(-0.97702814f,+0.21311032f), half2(-0.97831737f,+0.20711138f), half2(-0.97956977f,+0.20110463f),
    half2(-0.98078528f,+0.19509032f), half2(-0.98196387f,+0.18906866f), half2(-0.98310549f,+0.18303989f), half2(-0.98421009f,+0.17700422f),
    half2(-0.98527764f,+0.17096189f), half2(-0.98630810f,+0.16491312f), half2(-0.98730142f,+0.15885814f), half2(-0.98825757f,+0.15279719f),
    half2(-0.98917651f,+0.14673047f), half2(-0.99005821f,+0.14065824f), half2(-0.99090264f,+0.13458071f), half2(-0.99170975f,+0.12849811f),
    half2(-0.99247953f,+0.12241068f), half2(-0.99321195f,+0.11631863f), half2(-0.99390697f,+0.11022221f), half2(-0.99456457f,+0.10412163f),
    half2(-0.99518473f,+0.09801714f), half2(-0.99576741f,+0.09190896f), half2(-0.99631261f,+0.08579731f), half2(-0.99682030f,+0.07968244f),
    half2(-0.99729046f,+0.07356456f), half2(-0.99772307f,+0.06744392f), half2(-0.99811811f,+0.06132074f), half2(-0.99847558f,+0.05519524f),
    half2(-0.99879546f,+0.04906767f), half2(-0.99907773f,+0.04293826f), half2(-0.99932238f,+0.03680722f), half2(-0.99952942f,+0.03067480f),
    half2(-0.99969882f,+0.02454123f), half2(-0.99983058f,+0.01840673f), half2(-0.99992470f,+0.01227154f), half2(-0.99998118f,+0.00613588f),
};

// One tile dequantiser for every layout. `il` names a 16-value slice -- half a 16-pair
// group -- so il>>1 picks the group and il&1 picks which eight pairs inside it.
//
// No trig and no general exp: the angle comes from the constant table above, and the
// ladder is linear in log magnitude with an integer code, so exp(ladder) is exp2(base +
// code*step) with base and step hoisted per block.
#define NEURON_DEQ(LP)                                                                    \
template <typename type4x4>                                                               \
void dequantize_neuron_m##LP(device const block_neuron_m##LP *xb, short il,               \
                             thread type4x4 & reg) {                                      \
    const int last = (1 << (LP)) - 1;                                                     \
    const int AK = (LP) + 2, AW = NEURON_AW(LP), JSH = NEURON_JSH(LP);                    \
    const uint JM = (1u << JSH) - 1u;                                                     \
    const float lo  = (float) xb->lo;                                                     \
    const float mid = (float) xb->mid;                                                    \
    const float hi  = (float) xb->hi;                                                     \
    const float L   = M_LOG2E_F;                                                          \
    const float b0 = lo * L,              s0 = 2.0f*(mid - lo)*L/(float) last;            \
    const float b1 = (2.0f*mid - hi) * L, s1 = 2.0f*(hi - mid)*L/(float) last;            \
    device const uint8_t * grp = xb->qs + (il >> 1) * NEURON_GBY(LP);                     \
    device const uint8_t * ang = grp + NEURON_AOFF(LP);                                   \
    const short t0 = (il & 1) * 8;                                                        \
    /* Walk pair-COUPLES, not pairs. One joint code carries the angles of both, so a      \
       per-pair loop fetches and unpacks the same field twice -- half the angle reads in  \
       this path were redundant. `il` is a 16-value slice, so four couples fill it and each\
       writes one whole reg row. */                                                       \
    FOR_UNROLL (short s = 0; s < 4; ++s) {                                                \
        const short pa = t0 + 2*s, pb = pa + 1;                                           \
        const int   ba = (pa >> 1) * AW;                                                  \
        uint av = (uint) ang[ba >> 3];                                                    \
        if (((ba & 7) + AW) >  8) av |= (uint) ang[(ba >> 3) + 1] <<  8;                  \
        if (((ba & 7) + AW) > 16) av |= (uint) ang[(ba >> 3) + 2] << 16;                  \
        if (((ba & 7) + AW) > 24) av |= (uint) ang[(ba >> 3) + 3] << 24;                  \
        const uint v  = (av >> (ba & 7)) & ((1u << AW) - 1u);                             \
        const int  ia = (int)(v >> JSH);                                                  \
        const int  ib = (int)(2u*(v & JM)) + (ia & 1);                                    \
        const int  bma = pa * (LP), bmb = pb * (LP);                                      \
        uint mva = (uint) grp[bma >> 3];                                                  \
        if (((bma & 7) + (LP)) > 8) mva |= (uint) grp[(bma >> 3) + 1] << 8;               \
        uint mvb = (uint) grp[bmb >> 3];                                                  \
        if (((bmb & 7) + (LP)) > 8) mvb |= (uint) grp[(bmb >> 3) + 1] << 8;               \
        const int  ma = (int)((mva >> (bma & 7)) & (uint) last);                          \
        const int  mb = (int)((mvb >> (bmb & 7)) & (uint) last);                          \
        const bool ua = (2*ma > last), ub = (2*mb > last);                                \
        const float ra = exp2(fma((float) ma, ua ? s1 : s0, ua ? b1 : b0));               \
        const float rb = exp2(fma((float) mb, ub ? s1 : s0, ub ? b1 : b0));               \
        const half2 ca = kNeuronAng[ia << (10 - AK)];                                     \
        const half2 cb = kNeuronAng[ib << (10 - AK)];                                     \
        reg[s][0] = ra * (float) ca.x;                                                    \
        reg[s][1] = ra * (float) ca.y;                                                    \
        reg[s][2] = rb * (float) cb.x;                                                    \
        reg[s][3] = rb * (float) cb.y;                                                    \
    }                                                                                     \
}

NEURON_DEQ(1)
NEURON_DEQ(2)
NEURON_DEQ(3)
NEURON_DEQ(4)
NEURON_DEQ(5)
NEURON_DEQ(6)
NEURON_DEQ(7)
NEURON_DEQ(8)
