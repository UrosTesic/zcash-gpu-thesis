// Implementation of primitives found in pairing/lib.rs


// The wrapper to split a 64 bit integer into two 32 bit numbers (passed as arguments)
//
inline void split_ulong(const ulong src, ulong* high, ulong* low) {
    *high = src >> 32;
    *low = src & 0xFFFFFFFF;
}

// The wrapper to combine two 32 bit integers into a 64 bit one
//
inline ulong combine_ulong(const ulong high, const ulong low) {
    return high << 32 | low;
}

// Calculate a - b - borrow, returning the difference, and modifying borrow
// passed as the argument
//
inline ulong sbb(const ulong a, const ulong b, ulong* borrow) {
    ulong a_high, a_low;
    ulong b_high, b_low;
    ulong bor, r0, r1;

    split_ulong(a, &a_high, &a_low);
    split_ulong(b, &b_high, &b_low);
    split_ulong((1ul << 32) + a_low - b_low - *borrow, &bor, &r0);
    split_ulong((1ul << 32) + a_high - b_high - (bor == 0), &bor, &r1);

    *borrow = (bor == 0);
    return combine_ulong(r1, r0);
}

// Calculate a + b + carry, returning the sum, and modifying the varry value
//
inline ulong adc(const ulong a, const ulong b, ulong* carry) {
    ulong a_high, a_low;
    ulong b_high, b_low;
    ulong carry_high, carry_low;
    ulong r0, r1;
    ulong temp;

    split_ulong(a, &a_high, &a_low);
    split_ulong(b, &b_high, &b_low);
    split_ulong(*carry, &carry_high, &carry_low);

    split_ulong(a_low + b_low + carry_low, &temp, &r0);
    split_ulong(temp + a_high + b_high + carry_high, &temp, &r1);

    *carry = temp;

    return combine_ulong(r1, r0);
}

// Calculate a + (b * c) + carry, returning the least significant bytes
// and setting the carry to the most significant bytes
//
inline ulong mac_with_carry(const ulong a, const ulong b, const ulong c, ulong* carry) {
        /*
                                [  b_hi  |  b_lo  ]
                                [  c_hi  |  c_lo  ] *
        -------------------------------------------
                                [  b_lo  *  c_lo  ] <-- w
                       [  b_hi  *  c_lo  ]          <-- x
                       [  b_lo  *  c_hi  ]          <-- y
             [   b_hi  *  c_lo  ]                   <-- z
                                [  a_hi  |  a_lo  ]
                                [  C_hi  |  C_lo  ]
        */

        ulong a_high, a_low;
        ulong b_high, b_low;
        ulong c_high, c_low;
        ulong carry_high, carry_low;  
        ulong w_high, w_low;
        ulong x_high, x_low;
        ulong y_high, y_low;
        ulong z_high, z_low;
        ulong temp;
        ulong r0, r1, r2, r3;

        split_ulong(a, &a_high, &a_low);
        split_ulong(b, &b_high, &b_low);
        split_ulong(c, &c_high, &c_low);
        split_ulong(*carry, &carry_high, &carry_low);

        split_ulong(b_low * c_low, &w_high, &w_low);
        split_ulong(b_high * c_low, &x_high, &x_low);
        split_ulong(b_low * c_high, &y_high, &y_low);
        split_ulong(b_high * c_high, &z_high, &z_low);

        split_ulong(w_low + a_low + carry_low, &temp, &r0);
        split_ulong(temp + w_high + x_low + y_low + a_high + carry_high,
                    &temp, &r1);
        split_ulong(temp + x_high + y_high + z_low, &temp, &r2);
        split_ulong(temp + z_high, &temp, &r3);

        *carry = combine_ulong(r3, r2);
        return combine_ulong(r1, r0); 
}


__kernel void test_split_ulong(__global const ulong* src, __global ulong* res_high,
                               __global ulong* res_low, uint length) {
    const uint id = get_global_id(0);

    if (id >= length) return;

    ulong priv_res_high, priv_res_low;
    split_ulong(src[id], &priv_res_high, &priv_res_low);

    res_high[id] = priv_res_high;
    res_low[id] = priv_res_low;
}

__kernel void test_combine_ulong(__global const ulong* a, __global const ulong* b,
                                 __global ulong* c, uint length) {
    const uint id = get_global_id(0);

    if (id >= length) return;

    c[id] = combine_ulong(a[id], b[id]);
}

__kernel void test_sbb(__global const ulong* a, __global const ulong* b,
                       __global ulong* borrow, __global ulong* result,
                       uint length) {
    const uint id = get_global_id(0);

    if (id >= length) return;

    ulong priv_borrow = borrow[id];

    result[id] = sbb(a[id], b[id], &priv_borrow);
    borrow[id] = priv_borrow;
}

__kernel void test_adc(__global const ulong* a, __global const ulong* b,
                       __global ulong* carry, __global ulong* result,
                       uint length) {
    const uint id = get_global_id(0);

    if (id >= length) return;

    ulong priv_carry = carry[id];

    result[id] = adc(a[id], b[id], &priv_carry);
    carry[id] = priv_carry;
}

__kernel void test_mac_with_carry(__global const ulong* a, __global const ulong* b,
                                  __global const ulong* c, __global ulong* carry,
                                  __global ulong* result, uint length) {
    const uint id = get_global_id(0);

    if (id >= length) return;

    ulong priv_carry = carry[id];
    result[id] = mac_with_carry(a[id], b[id], c[id], &priv_carry);
    carry[id] = priv_carry;
}

// *** FqRepr **

#define FQ_WIDTH 6

#define FQ_MODULUS  {\
                        {\
                        0xb9feffffffffaaab,\
                        0x1eabfffeb153ffff,\
                        0x6730d2a0f6b0f624,\
                        0x64774b84f38512bf,\
                        0x4b1ba7b6434bacd7,\
                        0x1a0111ea397fe69a\
                        }\
                    }

#define FQ_ONE  {\
                    {\
                        {\
                        0x760900000002fffd,\
                        0xebf4000bc40c0002,\
                        0x5f48985753c758ba,\
                        0x77ce585370525745,\
                        0x5c071a97a256ec6d,\
                        0x15f65ec3fa80e493\
                        }\
                    }\
                }

#define FQ_ZERO {\
                    {\
                        {\
                        0x0ul,\
                        0x0ul,\
                        0x0ul,\
                        0x0ul,\
                        0x0ul,\
                        0x0ul,\
                        }\
                    }\
                }

#define FQ_MODULUS_BITS 381u

#define FQ_REPR_SHAVE_BITS 3u

#define FQ_INV 0x89f3fffcfffcfffdul

#define FQ_S 1u

struct FqRepr {
    ulong data[FQ_WIDTH];
};

typedef struct FqRepr FqRepr;

inline void fqrepr_assign(FqRepr* left, const FqRepr* right) {
    #pragma unroll
    for (int i = 0; i < FQ_WIDTH; i++) {
        left->data[i] = right->data[i];
    }
}

inline void fqrepr_assign_global(FqRepr* left, __global const FqRepr* right) {
    #pragma unroll
    for (int i = 0; i < FQ_WIDTH; i++) {
        left->data[i] = right->data[i];
    }
}

inline void fqrepr_assign_to_global(__global FqRepr* left, const FqRepr* right) {
    #pragma unroll
    for (int i = 0; i < FQ_WIDTH; i++) {
        left->data[i] = right->data[i];
    }
}

inline void fqrepr_assign_local(FqRepr* left, __local const FqRepr* right) {
    #pragma unroll
    for (int i = 0; i < FQ_WIDTH; i++) {
        left->data[i] = right->data[i];
    }
}

inline void fqrepr_assign_to_local(__local FqRepr* left, const FqRepr* right) {
    #pragma unroll
    for (int i = 0; i < FQ_WIDTH; i++) {
        left->data[i] = right->data[i];
    }
}

inline bool fqrepr_is_odd(const FqRepr* self) {
    return (self->data[0] & 0x01) == 1;
}

inline bool fqrepr_is_even(const FqRepr* self) {
    return !fqrepr_is_odd(self);
}

inline bool fqrepr_is_zero(const FqRepr* self) {
    return (self->data[0] == 0)
        && (self->data[1] == 0)
        && (self->data[2] == 0)
        && (self->data[3] == 0)
        && (self->data[4] == 0)
        && (self->data[5] == 0);
}

inline int fqrepr_cmp(const FqRepr* a, const FqRepr* b) {
    for (int i = FQ_WIDTH - 1; i >= 0; i--) {
        if (a->data[i] < b->data[i]) return -1;
        else if (a->data[i] > b->data[i]) return 1;
    }
    return 0;
}

inline bool fqrepr_eq(const FqRepr* a, const FqRepr* b) {
    return fqrepr_cmp(a,b) == 0;
}

inline bool fqrepr_lt(const FqRepr* a, const FqRepr* b) {
    return fqrepr_cmp(a,b) == -1;
}

inline void fqrepr_sub_noborrow(FqRepr* self, const FqRepr* other) {
    ulong borrow = 0;

    for (int i = 0; i < FQ_WIDTH; i++) {
        self->data[i] = sbb(self->data[i], other->data[i], &borrow);
    }
}

inline void fqrepr_add_nocarry(FqRepr* self, const FqRepr* other) {
    ulong carry = 0;

    for (int i = 0; i < FQ_WIDTH; i++) {
        self->data[i] = adc(self->data[i], other->data[i], &carry);
    }
}

inline void fqrepr_div2(FqRepr* self) {
    ulong t = 0;

    for (int i = FQ_WIDTH - 1; i >= 0; i--) {
        ulong t2 = self->data[i] << 63;
        self->data[i] >>= 1;
        self->data[i] |= t;
        t = t2;
    }
}

inline void fqrepr_mul2(FqRepr* self) {
    ulong last = 0;

    for (int i = 0; i < FQ_WIDTH; i++) {
        ulong tmp = self->data[i] >> 63;
        self->data[i] <<= 1;
        self->data[i] |= last;
        last = tmp;
    }
}

__kernel void test_fqrepr_is_odd(__global const FqRepr* a, __global char* result,
                                 uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FqRepr temp;
    fqrepr_assign_global(&temp, &a[id]);
    result[id] = fqrepr_is_odd(&temp);
}

__kernel void test_fqrepr_is_even(__global const FqRepr* a, __global char* result,
                                  uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FqRepr temp;
    fqrepr_assign_global(&temp, &a[id]);
    result[id] = fqrepr_is_even(&temp);
}

__kernel void test_fqrepr_is_zero(__global const FqRepr* a, __global char* result,
                                  uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FqRepr temp;
    fqrepr_assign_global(&temp, &a[id]);
    result[id] = fqrepr_is_zero(&temp);
}

__kernel void test_fqrepr_cmp(__global const FqRepr* a, __global const FqRepr* b, 
                       __global char* result, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FqRepr temp_a, temp_b;
    fqrepr_assign_global(&temp_a, &a[id]);
    fqrepr_assign_global(&temp_b, &b[id]);
    result[id] = fqrepr_cmp(&temp_a, &temp_b);
}

__kernel void test_fqrepr_lt(__global const FqRepr* a, __global const FqRepr* b, 
                             __global char* result, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FqRepr temp_a, temp_b;
    fqrepr_assign_global(&temp_a, &a[id]);
    fqrepr_assign_global(&temp_b, &b[id]);
    result[id] = fqrepr_lt(&temp_a, &temp_b);
}

__kernel void test_fqrepr_eq(__global const FqRepr* a, __global const FqRepr* b, 
                       __global char* result, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FqRepr temp_a, temp_b;
    fqrepr_assign_global(&temp_a, &a[id]);
    fqrepr_assign_global(&temp_b, &b[id]);
    result[id] = fqrepr_eq(&temp_a, &temp_b);
}

__kernel void test_fqrepr_sub_noborrow(__global FqRepr* a, __global const FqRepr* b,
                                       uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FqRepr t1, t2;
    fqrepr_assign_global(&t1, &a[id]);
    fqrepr_assign_global(&t2, &b[id]);

    fqrepr_sub_noborrow(&t1, &t2);

    fqrepr_assign_to_global(&a[id], &t1);
}

__kernel void test_fqrepr_add_nocarry(__global FqRepr* a, __global const FqRepr* b,
                                uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FqRepr t1, t2;
    fqrepr_assign_global(&t1, &a[id]);
    fqrepr_assign_global(&t2, &b[id]);

    fqrepr_add_nocarry(&t1, &t2);

    fqrepr_assign_to_global(&a[id], &t1);
}

__kernel void test_fqrepr_div2(__global FqRepr* a, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FqRepr temp;
    fqrepr_assign_global(&temp, &a[id]);

    fqrepr_div2(&temp);

    fqrepr_assign_to_global(&a[id], &temp);
}

__kernel void test_fqrepr_mul2(__global FqRepr* a, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FqRepr temp;
    fqrepr_assign_global(&temp, &a[id]);

    fqrepr_mul2(&temp);
    fqrepr_assign_to_global(&a[id], &temp);
}

// *** Fq ***
//
struct Fq {
    FqRepr repr;
};

typedef struct Fq Fq;

inline bool fq_is_valid(const Fq* self) {
    const FqRepr temp = FQ_MODULUS;
    
    return fqrepr_lt(&self->repr, &temp);
}

inline void fq_assign(Fq* left, const Fq* right) {
    fqrepr_assign(&left->repr, &right->repr);
}

inline void fq_assign_global(Fq* left, __global const Fq* right) {
    fqrepr_assign_global(&left->repr, &right->repr);
}

inline void fq_assign_to_global(__global Fq* left, const Fq* right) {
    fqrepr_assign_to_global(&left->repr, &right->repr);
}

inline void fq_assign_local(Fq* left, __local const Fq* right) {
    fqrepr_assign_local(&left->repr, &right->repr);
}

inline void fq_assign_to_local(__local Fq* left, const Fq* right) {
    fqrepr_assign_to_local(&left->repr, &right->repr);
}

inline bool fq_eq(const Fq* left, const Fq* right) {
    return fqrepr_eq(&left->repr, &right->repr);
}

inline void fq_reduce(Fq* self) {
    const FqRepr temp = FQ_MODULUS;

    if (!fq_is_valid(self)) {
        fqrepr_sub_noborrow(&self->repr, &temp);
    }
}

inline void fq_mont_reduce(Fq* self,
                        const ulong r0,
                        ulong r1,
                        ulong r2,
                        ulong r3,
                        ulong r4,
                        ulong r5,
                        ulong r6,
                        ulong r7,
                        ulong r8,
                        ulong r9,
                        ulong r10,
                        ulong r11) {
    const FqRepr MOD_TEMP = FQ_MODULUS;

    ulong k = r0 * FQ_INV;
    ulong carry = 0;
    mac_with_carry(r0, k, MOD_TEMP.data[0], &carry);
    r1 = mac_with_carry(r1, k, MOD_TEMP.data[1], &carry);
    r2 = mac_with_carry(r2, k, MOD_TEMP.data[2], &carry);
    r3 = mac_with_carry(r3, k, MOD_TEMP.data[3], &carry);
    r4 = mac_with_carry(r4, k, MOD_TEMP.data[4], &carry);
    r5 = mac_with_carry(r5, k, MOD_TEMP.data[5], &carry);
    r6 = adc(r6, 0, &carry);

    ulong carry2 = carry;
    k = r1 * FQ_INV;
    carry = 0;
    mac_with_carry(r1, k, MOD_TEMP.data[0], &carry);
    r2 = mac_with_carry(r2, k, MOD_TEMP.data[1], &carry);
    r3 = mac_with_carry(r3, k, MOD_TEMP.data[2], &carry);
    r4 = mac_with_carry(r4, k, MOD_TEMP.data[3], &carry);
    r5 = mac_with_carry(r5, k, MOD_TEMP.data[4], &carry);
    r6 = mac_with_carry(r6, k, MOD_TEMP.data[5], &carry);
    r7 = adc(r7, carry2, &carry);

    carry2 = carry;
    k = r2 * FQ_INV;
    carry = 0;
    mac_with_carry(r2, k, MOD_TEMP.data[0], &carry);
    r3 = mac_with_carry(r3, k, MOD_TEMP.data[1], &carry);
    r4 = mac_with_carry(r4, k, MOD_TEMP.data[2], &carry);
    r5 = mac_with_carry(r5, k, MOD_TEMP.data[3], &carry);
    r6 = mac_with_carry(r6, k, MOD_TEMP.data[4], &carry);
    r7 = mac_with_carry(r7, k, MOD_TEMP.data[5], &carry);
    r8 = adc(r8, carry2, &carry);

    carry2 = carry;
    k = r3 * FQ_INV;
    carry = 0;
    mac_with_carry(r3, k, MOD_TEMP.data[0], &carry);
    r4 = mac_with_carry(r4, k, MOD_TEMP.data[1], &carry);
    r5 = mac_with_carry(r5, k, MOD_TEMP.data[2], &carry);
    r6 = mac_with_carry(r6, k, MOD_TEMP.data[3], &carry);
    r7 = mac_with_carry(r7, k, MOD_TEMP.data[4], &carry);
    r8 = mac_with_carry(r8, k, MOD_TEMP.data[5], &carry);
    r9 = adc(r9, carry2, &carry);

    carry2 = carry;
    k = r4 * FQ_INV;
    carry = 0;
    mac_with_carry(r4, k, MOD_TEMP.data[0], &carry);
    r5 = mac_with_carry(r5, k, MOD_TEMP.data[1], &carry);
    r6 = mac_with_carry(r6, k, MOD_TEMP.data[2], &carry);
    r7 = mac_with_carry(r7, k, MOD_TEMP.data[3], &carry);
    r8 = mac_with_carry(r8, k, MOD_TEMP.data[4], &carry);
    r9 = mac_with_carry(r9, k, MOD_TEMP.data[5], &carry);
    r10 = adc(r10, carry2, &carry);

    carry2 = carry;
    k = r5 * FQ_INV;
    carry = 0;
    mac_with_carry(r5, k, MOD_TEMP.data[0], &carry);
    r6 = mac_with_carry(r6, k, MOD_TEMP.data[1], &carry);
    r7 = mac_with_carry(r7, k, MOD_TEMP.data[2], &carry);
    r8 = mac_with_carry(r8, k, MOD_TEMP.data[3], &carry);
    r9 = mac_with_carry(r9, k, MOD_TEMP.data[4], &carry);
    r10 = mac_with_carry(r10, k, MOD_TEMP.data[5], &carry);
    r11 = adc(r11, carry2, &carry);

    self->repr.data[0] = r6;
    self->repr.data[1] = r7;
    self->repr.data[2] = r8;
    self->repr.data[3] = r9;
    self->repr.data[4] = r10;
    self->repr.data[5] = r11;
    fq_reduce(self);
}

inline void fq_mul_assign(Fq* self, const Fq* other) {
    ulong carry, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;
    
    carry = 0;
    r0 = mac_with_carry(0, self->repr.data[0], other->repr.data[0], &carry);
    r1 = mac_with_carry(0, self->repr.data[0], other->repr.data[1], &carry);
    r2 = mac_with_carry(0, self->repr.data[0], other->repr.data[2], &carry);
    r3 = mac_with_carry(0, self->repr.data[0], other->repr.data[3], &carry);
    r4 = mac_with_carry(0, self->repr.data[0], other->repr.data[4], &carry);
    r5 = mac_with_carry(0, self->repr.data[0], other->repr.data[5], &carry);
    r6 = carry;

    carry = 0;
    r1 = mac_with_carry(r1, self->repr.data[1], other->repr.data[0], &carry);
    r2 = mac_with_carry(r2, self->repr.data[1], other->repr.data[1], &carry);
    r3 = mac_with_carry(r3, self->repr.data[1], other->repr.data[2], &carry);
    r4 = mac_with_carry(r4, self->repr.data[1], other->repr.data[3], &carry);
    r5 = mac_with_carry(r5, self->repr.data[1], other->repr.data[4], &carry);
    r6 = mac_with_carry(r6, self->repr.data[1], other->repr.data[5], &carry);
    r7 = carry;

    carry = 0;
    r2 = mac_with_carry(r2, self->repr.data[2], other->repr.data[0], &carry);
    r3 = mac_with_carry(r3, self->repr.data[2], other->repr.data[1], &carry);
    r4 = mac_with_carry(r4, self->repr.data[2], other->repr.data[2], &carry);
    r5 = mac_with_carry(r5, self->repr.data[2], other->repr.data[3], &carry);
    r6 = mac_with_carry(r6, self->repr.data[2], other->repr.data[4], &carry);
    r7 = mac_with_carry(r7, self->repr.data[2], other->repr.data[5], &carry);
    r8 = carry;

    carry = 0;
    r3 = mac_with_carry(r3, self->repr.data[3], other->repr.data[0], &carry);
    r4 = mac_with_carry(r4, self->repr.data[3], other->repr.data[1], &carry);
    r5 = mac_with_carry(r5, self->repr.data[3], other->repr.data[2], &carry);
    r6 = mac_with_carry(r6, self->repr.data[3], other->repr.data[3], &carry);
    r7 = mac_with_carry(r7, self->repr.data[3], other->repr.data[4], &carry);
    r8 = mac_with_carry(r8, self->repr.data[3], other->repr.data[5], &carry);
    r9 = carry;

    carry = 0;
    r4 = mac_with_carry(r4, self->repr.data[4], other->repr.data[0], &carry);
    r5 = mac_with_carry(r5, self->repr.data[4], other->repr.data[1], &carry);
    r6 = mac_with_carry(r6, self->repr.data[4], other->repr.data[2], &carry);
    r7 = mac_with_carry(r7, self->repr.data[4], other->repr.data[3], &carry);
    r8 = mac_with_carry(r8, self->repr.data[4], other->repr.data[4], &carry);
    r9 = mac_with_carry(r9, self->repr.data[4], other->repr.data[5], &carry);
    r10 = carry;

    carry = 0;
    r5 = mac_with_carry(r5, self->repr.data[5], other->repr.data[0], &carry);
    r6 = mac_with_carry(r6, self->repr.data[5], other->repr.data[1], &carry);
    r7 = mac_with_carry(r7, self->repr.data[5], other->repr.data[2], &carry);
    r8 = mac_with_carry(r8, self->repr.data[5], other->repr.data[3], &carry);
    r9 = mac_with_carry(r9, self->repr.data[5], other->repr.data[4], &carry);
    r10 = mac_with_carry(r10, self->repr.data[5], other->repr.data[5], &carry);
    r11 = carry;

    fq_mont_reduce(self, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11);
}

inline void fq_square(Fq* self) {
    ulong carry, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;
    
    carry = 0;
    r1 = mac_with_carry(0, self->repr.data[0], self->repr.data[1], &carry);
    r2 = mac_with_carry(0, self->repr.data[0], self->repr.data[2], &carry);
    r3 = mac_with_carry(0, self->repr.data[0], self->repr.data[3], &carry);
    r4 = mac_with_carry(0, self->repr.data[0], self->repr.data[4], &carry);
    r5 = mac_with_carry(0, self->repr.data[0], self->repr.data[5], &carry);
    r6 = carry;

    carry = 0;
    r3 = mac_with_carry(r3, self->repr.data[1], self->repr.data[2], &carry);
    r4 = mac_with_carry(r4, self->repr.data[1], self->repr.data[3], &carry);
    r5 = mac_with_carry(r5, self->repr.data[1], self->repr.data[4], &carry);
    r6 = mac_with_carry(r6, self->repr.data[1], self->repr.data[5], &carry);
    r7 = carry;

    carry = 0;
    r5 = mac_with_carry(r5, self->repr.data[2], self->repr.data[3], &carry);
    r6 = mac_with_carry(r6, self->repr.data[2], self->repr.data[4], &carry);
    r7 = mac_with_carry(r7, self->repr.data[2], self->repr.data[5], &carry);
    r8 = carry;

    carry = 0;
    r7 = mac_with_carry(r7, self->repr.data[3], self->repr.data[4], &carry);
    r8 = mac_with_carry(r8, self->repr.data[3], self->repr.data[5], &carry);
    r9 = carry;

    carry = 0;
    r9 = mac_with_carry(r9, self->repr.data[4], self->repr.data[5], &carry);
    r10 = carry;

    r11 = r10 >> 63;
    r10 = (r10 << 1) | (r9 >> 63);
    r9 = (r9 << 1) | (r8 >> 63);
    r8 = (r8 << 1) | (r7 >> 63);
    r7 = (r7 << 1) | (r6 >> 63);
    r6 = (r6 << 1) | (r5 >> 63);
    r5 = (r5 << 1) | (r4 >> 63);
    r4 = (r4 << 1) | (r3 >> 63);
    r3 = (r3 << 1) | (r2 >> 63);
    r2 = (r2 << 1) | (r1 >> 63);
    r1 = r1 << 1;

    carry = 0;
    r0 = mac_with_carry(0, self->repr.data[0], self->repr.data[0], &carry);
    r1 = adc(r1, 0, &carry);
    r2 = mac_with_carry(r2, self->repr.data[1], self->repr.data[1], &carry);
    r3 = adc(r3, 0, &carry);
    r4 = mac_with_carry(r4, self->repr.data[2], self->repr.data[2], &carry);
    r5 = adc(r5, 0, &carry);
    r6 = mac_with_carry(r6, self->repr.data[3], self->repr.data[3], &carry);
    r7 = adc(r7, 0, &carry);
    r8 = mac_with_carry(r8, self->repr.data[4], self->repr.data[4], &carry);
    r9 = adc(r9, 0, &carry);
    r10 = mac_with_carry(r10, self->repr.data[5], self->repr.data[5], &carry);
    r11 = adc(r11, 0, &carry);
    fq_mont_reduce(self, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11);
}

inline void fq_add_assign(Fq* self, const Fq* other) {
    fqrepr_add_nocarry(&self->repr, &other->repr);
    fq_reduce(self);
}

inline void fq_double(Fq* self) {
    fqrepr_mul2(&self->repr);
    fq_reduce(self);
}

inline void fq_sub_assign(Fq* self, const Fq* other) {
    if (fqrepr_lt(&self->repr, &other->repr)) {
        const FqRepr MOD_TEMP = FQ_MODULUS;
        fqrepr_add_nocarry(&self->repr, &MOD_TEMP);
    }

    fqrepr_sub_noborrow(&self->repr, &other->repr);
}

inline void fq_negate(Fq* self) {
    if (!fqrepr_is_zero(&self->repr)) {
        FqRepr MOD_TEMP = FQ_MODULUS;
        fqrepr_sub_noborrow(&MOD_TEMP, &self->repr);
        fqrepr_assign(&self->repr, &MOD_TEMP);
    }
}

inline bool fq_is_zero(const Fq* self) {
    return fqrepr_is_zero(&self->repr);
}

__kernel void test_fq_is_valid(__global const Fq* a, __global char* result,
                            uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fq temp;
    fq_assign_global(&temp, &a[id]);
    result[id] = fq_is_valid(&temp);
}

__kernel void test_mul_assign(__global Fq* a, __global const Fq* b, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fq left;
    Fq right;

    fqrepr_assign_global(&left.repr, &a[id].repr);
    fqrepr_assign_global(&right.repr, &b[id].repr);

    fq_mul_assign(&left, &right);

    fqrepr_assign_to_global(&a[id].repr, &left.repr);
}

__kernel void test_fq_square(__global Fq* a, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fq temp;

    fqrepr_assign_global(&temp.repr, &a[id].repr);

    fq_square(&temp);

    fqrepr_assign_to_global(&a[id].repr, &temp.repr);
}

__kernel void test_fq_add_assign(__global Fq* a, __global const Fq* b,
                                uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fq t1, t2;
    fq_assign_global(&t1, &a[id]);
    fq_assign_global(&t2, &b[id]);

    fq_add_assign(&t1, &t2);

    fq_assign_to_global(&a[id], &t1);
}

__kernel void test_fq_double(__global Fq* a, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fq t1;
    fq_assign_global(&t1, &a[id]);

    fq_double(&t1);

    fq_assign_to_global(&a[id], &t1);
}

__kernel void test_sub_assign(__global Fq* a, __global const Fq* b,
                                uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fq t1, t2;
    fq_assign_global(&t1, &a[id]);
    fq_assign_global(&t2, &b[id]);

    fq_sub_assign(&t1, &t2);

    fq_assign_to_global(&a[id], &t1);
}

__kernel void test_fq_negate(__global Fq* a, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fq t1;
    fq_assign_global(&t1, &a[id]);

    fq_negate(&t1);

    fq_assign_to_global(&a[id], &t1);
}

// Implementation of projective arithmetic
//

#define PROJECTIVE_ZERO {\
                        FQ_ZERO,\
                        FQ_ONE,\
                        FQ_ZERO\
                        }

struct Projective {
    Fq x;
    Fq y;
    Fq z;
};

typedef struct Projective Projective;

inline void projective_assign(Projective* left, const Projective* right) {
    fq_assign(&left->x, &right->x);
    fq_assign(&left->y, &right->y);
    fq_assign(&left->z, &right->z);
}

inline void projective_assign_global(Projective* left, __global const Projective* right) {
    fq_assign_global(&left->x, &right->x);
    fq_assign_global(&left->y, &right->y);
    fq_assign_global(&left->z, &right->z);
}

inline void projective_assign_to_global(__global Projective* left, const Projective* right) {
    fq_assign_to_global(&left->x, &right->x);
    fq_assign_to_global(&left->y, &right->y);
    fq_assign_to_global(&left->z, &right->z);
}

inline void projective_assign_local(Projective* left, __local const Projective* right) {
    fq_assign_local(&left->x, &right->x);
    fq_assign_local(&left->y, &right->y);
    fq_assign_local(&left->z, &right->z);
}

inline void projective_assign_to_local(__local Projective* left, const Projective* right) {
    fq_assign_to_local(&left->x, &right->x);
    fq_assign_to_local(&left->y, &right->y);
    fq_assign_to_local(&left->z, &right->z);
}

inline bool projective_is_zero(const Projective* self) {
    return fq_is_zero(&self->z);
}

inline void projective_negate(Projective* self) {
    fq_negate(&self->y);
}

void projective_double(Projective* self) {
    if (projective_is_zero(self)) {
        return;
    }

    Fq a;
    fq_assign(&a, &self->x);
    fq_square(&a);

    Fq b;
    fq_assign(&b, &self->y);
    fq_square(&b);

    Fq c;
    fq_assign(&c, &b);
    fq_square(&c);

    Fq d;
    fq_assign(&d, &self->x);
    fq_add_assign(&d, &b);
    fq_square(&d);
    fq_sub_assign(&d, &a);
    fq_sub_assign(&d, &c);
    fq_double(&d);

    Fq e;
    fq_assign(&e, &a);
    fq_double(&e);
    fq_add_assign(&e, &a);

    Fq f;
    fq_assign(&f, &e);
    fq_square(&f);

    fq_mul_assign(&self->z, &self->y);
    fq_double(&self->z);

    fq_assign(&self->x, &f);
    fq_sub_assign(&self->x, &d);
    fq_sub_assign(&self->x, &d);

    fq_assign(&self->y, &d);
    fq_sub_assign(&self->y, &self->x);
    fq_mul_assign(&self->y, &e);
    fq_double(&c);
    fq_double(&c);
    fq_double(&c);
    fq_sub_assign(&self->y, &c);
}


void projective_add_assign(Projective* self, const Projective* other) {
    if (projective_is_zero(self)) {
        projective_assign(self, other);
        return;
    }

    if (projective_is_zero(other)) {
        return;
    }

    Fq z1z1;
    fq_assign(&z1z1, &self->z);
    fq_square(&z1z1);

    Fq z2z2;
    fq_assign(&z2z2, &other->z);
    fq_square(&z2z2);

    Fq u1;
    fq_assign(&u1, &self->x);
    fq_mul_assign(&u1, &z2z2);

    Fq u2;
    fq_assign(&u2, &other->x);
    fq_mul_assign(&u2, &z1z1);

    Fq s1;
    fq_assign(&s1, &self->y);
    fq_mul_assign(&s1, &other->z);
    fq_mul_assign(&s1, &z2z2);

    Fq s2;
    fq_assign(&s2, &other->y);
    fq_mul_assign(&s2, &self->z);
    fq_mul_assign(&s2, &z1z1);

    if (fq_eq(&u1, &u2) && fq_eq(&s1, &s2)) {
        projective_double(self);
    } else {
        Fq h;
        fq_assign(&h, &u2);
        fq_sub_assign(&h, &u1);

        Fq i;
        fq_assign(&i, &h);
        fq_double(&i);
        fq_square(&i);

        Fq j;
        fq_assign(&j, &h);
        fq_mul_assign(&j, &i);

        Fq r;
        fq_assign(&r, &s2);
        fq_sub_assign(&r, &s1);
        fq_double(&r);

        Fq v;
        fq_assign(&v, &u1);
        fq_mul_assign(&v, &i);

        fq_assign(&self->x, &r);
        fq_square(&self->x);
        fq_sub_assign(&self->x, &j);
        fq_sub_assign(&self->x, &v);
        fq_sub_assign(&self->x, &v);

        fq_assign(&self->y, &v);
        fq_sub_assign(&self->y, &self->x);
        fq_mul_assign(&self->y, &r);
        fq_mul_assign(&s1, &j);
        fq_double(&s1);
        fq_sub_assign(&self->y, &s1);

        fq_add_assign(&self->z, &other->z);
        fq_square(&self->z);
        fq_sub_assign(&self->z, &z1z1);
        fq_sub_assign(&self->z, &z2z2);
        fq_mul_assign(&self->z, &h);
    }
}

__kernel void test_projective_add_assign(__global Projective* a, __global const Projective* b,
                                    uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Projective t1, t2;
    projective_assign_global(&t1, &a[id]);
    projective_assign_global(&t2, &b[id]);
    projective_add_assign(&t1, &t2);
    projective_assign_to_global(&a[id], &t1);
}

__kernel void test_projective_double(__global Projective* a, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Projective t;
    projective_assign_global(&t, &a[id]);
    projective_double(&t);
    projective_assign_to_global(&a[id], &t);
}

struct Affine {
    Fq x;
    Fq y;
    ulong infinity;
};

typedef struct Affine Affine;

inline void affine_assign(Affine* left, const Affine* right) {
    fq_assign(&left->x, &right->x);
    fq_assign(&left->y, &right->y);
    left->infinity = right->infinity;
}

inline void affine_assign_global(Affine* left, __global const Affine* right) {
    fq_assign_global(&left->x, &right->x);
    fq_assign_global(&left->y, &right->y);
    left->infinity = right->infinity;
}

inline bool affine_is_zero(const Affine* self) {
    return self->infinity != 0;
}

void projective_add_assign_mixed(Projective* self, const Affine* other) {
    if (affine_is_zero(other)) {
        return;
    }

    if (projective_is_zero(self)) {
        fq_assign(&self->x, &other->x);
        fq_assign(&self->y, &other->y);

        const Fq ONE = FQ_ONE;
        fq_assign(&self->z, &ONE);
        return;
    }

    Fq z1z1;
    fq_assign(&z1z1, &self->z);
    fq_square(&z1z1);

    Fq u2;
    fq_assign(&u2, &other->x);
    fq_mul_assign(&u2, &z1z1);

    Fq s2;
    fq_assign(&s2, &other->y);
    fq_mul_assign(&s2, &self->z);
    fq_mul_assign(&s2, &z1z1);

    if (fq_eq(&self->x, &u2) && fq_eq(&self->y, &s2)) {
        projective_double(self);
        return;
    } else {
        Fq h;
        fq_assign(&h, &u2);
        fq_sub_assign(&h, &self->x);

        Fq hh;
        fq_assign(&hh, &h);
        fq_square(&hh);

        Fq i;
        fq_assign(&i, &hh);
        fq_double(&i);
        fq_double(&i);

        Fq j;
        fq_assign(&j, &h);
        fq_mul_assign(&j, &i);

        Fq r;
        fq_assign(&r, &s2);
        fq_sub_assign(&r, &self->y);
        fq_double(&r);

        Fq v;
        fq_assign(&v, &self->x);
        fq_mul_assign(&v, &i);

        fq_assign(&self->x, &r);
        fq_square(&self->x);
        fq_sub_assign(&self->x, &j);
        fq_sub_assign(&self->x, &v);
        fq_sub_assign(&self->x, &v);

        fq_mul_assign(&j, &self->y);
        fq_double(&j);
        fq_assign(&self->y, &v);
        fq_sub_assign(&self->y, &self->x);
        fq_mul_assign(&self->y, &r);
        fq_sub_assign(&self->y, &j);

        fq_add_assign(&self->z, &h);
        fq_square(&self->z);
        fq_sub_assign(&self->z, &z1z1);
        fq_sub_assign(&self->z, &hh);
        return;
    }
}

__kernel void test_projective_add_assign_mixed(__global Projective* a, __global const Affine* b,
                                    uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Projective proj;
    Affine aff;
    projective_assign_global(&proj, &a[id]);
    affine_assign_global(&aff, &b[id]);

    projective_add_assign_mixed(&proj, &aff);

    projective_assign_to_global(&a[id], &proj);
}

// *** FrRepr ***

#define FR_WIDTH 4

#define FR_MODULUS {\
                    {\
                    0xffffffff00000001,\
                    0x53bda402fffe5bfe,\
                    0x3339d80809a1d805,\
                    0x73eda753299d7d48\
                    }\
                }

#define FR_MODULUS_BITS 255u

#define FR_REPR_SHAVE_BITS 1u

#define FR_INV 0xfffffffefffffffful;

#define FR_S 32u

struct FrRepr {
    ulong data[FR_WIDTH];
};

typedef struct FrRepr FrRepr;

inline void frrepr_assign(FrRepr* left, const FrRepr* right) {
    #pragma unroll
    for (int i = 0; i < FR_WIDTH; i++) {
        left->data[i] = right->data[i];
    }
}

inline void frrepr_assign_global(FrRepr* left, __global const FrRepr* right) {
    #pragma unroll
    for (int i = 0; i < FR_WIDTH; i++) {
        left->data[i] = right->data[i];
    }
}

inline void frrepr_assign_to_global(__global FrRepr* left, const FrRepr* right) {
    #pragma unroll
    for (int i = 0; i < FR_WIDTH; i++) {
        left->data[i] = right->data[i];
    }
}

inline bool frrepr_is_odd(const FrRepr* self) {
    return (self->data[0] & 0x01) == 1;
}

inline bool frrepr_is_even(const FrRepr* self) {
    return !frrepr_is_odd(self);
}

inline bool frrepr_is_zero(const FrRepr* self) {
    return (self->data[0] == 0)
        && (self->data[1] == 0)
        && (self->data[2] == 0)
        && (self->data[3] == 0);
}

inline int frrepr_cmp(const FrRepr* a, const FrRepr* b) {
    for (int i = FR_WIDTH - 1; i >= 0; i--) {
        if (a->data[i] < b->data[i]) return -1;
        else if (a->data[i] > b->data[i]) return 1;
    }
    return 0;
}

inline bool frrepr_eq(const FrRepr* a, const FrRepr* b) {
    return frrepr_cmp(a,b) == 0;
}

inline bool frrepr_lt(const FrRepr* a, const FrRepr* b) {
    return frrepr_cmp(a,b) == -1;
}

inline void frrepr_sub_noborrow(FrRepr* self, const FrRepr* other) {
    ulong borrow = 0;

    for (int i = 0; i < FR_WIDTH; i++) {
        self->data[i] = sbb(self->data[i], other->data[i], &borrow);
    }
}

inline void frrepr_add_nocarry(FrRepr* self, const FrRepr* other) {
    ulong carry = 0;

    for (int i = 0; i < FR_WIDTH; i++) {
        self->data[i] = adc(self->data[i], other->data[i], &carry);
    }
}

inline void frrepr_div2(FrRepr* self) {
    ulong t = 0;

    for (int i = FR_WIDTH - 1; i >= 0; i--) {
        ulong t2 = self->data[i] << 63;
        self->data[i] >>= 1;
        self->data[i] |= t;
        t = t2;
    }
}

inline void frrepr_mul2(FrRepr* self) {
    ulong last = 0;

    for (int i = 0; i < FR_WIDTH; i++) {
        ulong tmp = self->data[i] >> 63;
        self->data[i] <<= 1;
        self->data[i] |= last;
        last = tmp;
    }
}

__kernel void test_frrepr_is_odd(__global const FrRepr* a, __global char* result,
                                 uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp;
    frrepr_assign_global(&temp, &a[id]);
    result[id] = frrepr_is_odd(&temp);
}

__kernel void test_frrepr_is_even(__global const FrRepr* a, __global char* result,
                                  uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp;
    frrepr_assign_global(&temp, &a[id]);
    result[id] = frrepr_is_even(&temp);
}

__kernel void test_frrepr_is_zero(__global const FrRepr* a, __global char* result,
                                  uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp;
    frrepr_assign_global(&temp, &a[id]);
    result[id] = frrepr_is_zero(&temp);
}

__kernel void test_frrepr_cmp(__global const FrRepr* a, __global const FrRepr* b, 
                       __global char* result, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp_a, temp_b;
    frrepr_assign_global(&temp_a, &a[id]);
    frrepr_assign_global(&temp_b, &b[id]);
    result[id] = frrepr_cmp(&temp_a, &temp_b);
}

__kernel void test_frrepr_lt(__global const FrRepr* a, __global const FrRepr* b, 
                             __global char* result, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp_a, temp_b;
    frrepr_assign_global(&temp_a, &a[id]);
    frrepr_assign_global(&temp_b, &b[id]);
    result[id] = frrepr_lt(&temp_a, &temp_b);
}

__kernel void test_frrepr_eq(__global const FrRepr* a, __global const FrRepr* b, 
                       __global char* result, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp_a, temp_b;
    frrepr_assign_global(&temp_a, &a[id]);
    frrepr_assign_global(&temp_b, &b[id]);
    result[id] = frrepr_eq(&temp_a, &temp_b);
}

__kernel void test_frrepr_sub_noborrow(__global FrRepr* a, __global const FrRepr* b,
                                       uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr t1, t2;
    frrepr_assign_global(&t1, &a[id]);
    frrepr_assign_global(&t2, &b[id]);

    frrepr_sub_noborrow(&t1, &t2);

    frrepr_assign_to_global(&a[id], &t1);
}

__kernel void test_frrepr_add_nocarry(__global FrRepr* a, __global const FrRepr* b,
                                uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr t1, t2;
    frrepr_assign_global(&t1, &a[id]);
    frrepr_assign_global(&t2, &b[id]);

    frrepr_add_nocarry(&t1, &t2);

    frrepr_assign_to_global(&a[id], &t1);
}

__kernel void test_frrepr_div2(__global FrRepr* a, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp;
    frrepr_assign_global(&temp, &a[id]);

    frrepr_div2(&temp);

    frrepr_assign_to_global(&a[id], &temp);
}

__kernel void test_frrepr_mul2(__global FrRepr* a, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp;
    frrepr_assign_global(&temp, &a[id]);

    frrepr_mul2(&temp);
    frrepr_assign_to_global(&a[id], &temp);
}

// *** Fr ***
//
struct Fr {
    FrRepr repr;
};

typedef struct Fr Fr;

inline bool fr_is_valid(const Fr* self) {
    const FrRepr temp = FR_MODULUS;
    
    return frrepr_lt(&self->repr, &temp);
}

inline void fr_assign(Fr* left, const Fr* right) {
    frrepr_assign(&left->repr, &right->repr);
}

inline void fr_assign_global(Fr* left, __global const Fr* right) {
    frrepr_assign_global(&left->repr, &right->repr);
}

inline void fr_assign_to_global(__global Fr* left, const Fr* right) {
    frrepr_assign_to_global(&left->repr, &right->repr);
}

inline bool fr_eq(const Fr* left, const Fr* right) {
    return frrepr_eq(&left->repr, &right->repr);
}

inline void fr_reduce(Fr* self) {
    const FrRepr temp = FR_MODULUS;

    if (!fr_is_valid(self)) {
        frrepr_sub_noborrow(&self->repr, &temp);
    }
}

inline void fr_mont_reduce(Fr* self,
                        const ulong r0,
                        ulong r1,
                        ulong r2,
                        ulong r3,
                        ulong r4,
                        ulong r5,
                        ulong r6,
                        ulong r7) {
    const FrRepr MOD_TEMP = FR_MODULUS;

    ulong k = r0 * FR_INV;
    ulong carry = 0;
    mac_with_carry(r0, k, MOD_TEMP.data[0], &carry);
    r1 = mac_with_carry(r1, k, MOD_TEMP.data[1], &carry);
    r2 = mac_with_carry(r2, k, MOD_TEMP.data[2], &carry);
    r3 = mac_with_carry(r3, k, MOD_TEMP.data[3], &carry);
    r4 = adc(r4, 0, &carry);

    ulong carry2 = carry;
    k = r1 * FR_INV;
    carry = 0;
    mac_with_carry(r1, k, MOD_TEMP.data[0], &carry);
    r2 = mac_with_carry(r2, k, MOD_TEMP.data[1], &carry);
    r3 = mac_with_carry(r3, k, MOD_TEMP.data[2], &carry);
    r4 = mac_with_carry(r4, k, MOD_TEMP.data[3], &carry);
    r5 = adc(r5, carry2, &carry);

    carry2 = carry;
    k = r2 * FR_INV;
    carry = 0;
    mac_with_carry(r2, k, MOD_TEMP.data[0], &carry);
    r3 = mac_with_carry(r3, k, MOD_TEMP.data[1], &carry);
    r4 = mac_with_carry(r4, k, MOD_TEMP.data[2], &carry);
    r5 = mac_with_carry(r5, k, MOD_TEMP.data[3], &carry);
    r6 = adc(r6, carry2, &carry);

    carry2 = carry;
    k = r3 * FR_INV;
    carry = 0;
    mac_with_carry(r3, k, MOD_TEMP.data[0], &carry);
    r4 = mac_with_carry(r4, k, MOD_TEMP.data[1], &carry);
    r5 = mac_with_carry(r5, k, MOD_TEMP.data[2], &carry);
    r6 = mac_with_carry(r6, k, MOD_TEMP.data[3], &carry);
    r7 = adc(r7, carry2, &carry);

    self->repr.data[0] = r4;
    self->repr.data[1] = r5;
    self->repr.data[2] = r6;
    self->repr.data[3] = r7;
    fr_reduce(self);
}

inline void fr_mul_assign(Fr* self, const Fr* other) {
    ulong carry, r0, r1, r2, r3, r4, r5, r6, r7;
    
    carry = 0;
    r0 = mac_with_carry(0, self->repr.data[0], other->repr.data[0], &carry);
    r1 = mac_with_carry(0, self->repr.data[0], other->repr.data[1], &carry);
    r2 = mac_with_carry(0, self->repr.data[0], other->repr.data[2], &carry);
    r3 = mac_with_carry(0, self->repr.data[0], other->repr.data[3], &carry);
    r4 = carry;

    carry = 0;
    r1 = mac_with_carry(r1, self->repr.data[1], other->repr.data[0], &carry);
    r2 = mac_with_carry(r2, self->repr.data[1], other->repr.data[1], &carry);
    r3 = mac_with_carry(r3, self->repr.data[1], other->repr.data[2], &carry);
    r4 = mac_with_carry(r4, self->repr.data[1], other->repr.data[3], &carry);
    r5 = carry;

    carry = 0;
    r2 = mac_with_carry(r2, self->repr.data[2], other->repr.data[0], &carry);
    r3 = mac_with_carry(r3, self->repr.data[2], other->repr.data[1], &carry);
    r4 = mac_with_carry(r4, self->repr.data[2], other->repr.data[2], &carry);
    r5 = mac_with_carry(r5, self->repr.data[2], other->repr.data[3], &carry);
    r6 = carry;

    carry = 0;
    r3 = mac_with_carry(r3, self->repr.data[3], other->repr.data[0], &carry);
    r4 = mac_with_carry(r4, self->repr.data[3], other->repr.data[1], &carry);
    r5 = mac_with_carry(r5, self->repr.data[3], other->repr.data[2], &carry);
    r6 = mac_with_carry(r6, self->repr.data[3], other->repr.data[3], &carry);
    r7 = carry;

    fr_mont_reduce(self, r0, r1, r2, r3, r4, r5, r6, r7);
}

inline void fr_square(Fr* self) {
    ulong carry, r0, r1, r2, r3, r4, r5, r6, r7;
    
    carry = 0;
    r1 = mac_with_carry(0, self->repr.data[0], self->repr.data[1], &carry);
    r2 = mac_with_carry(0, self->repr.data[0], self->repr.data[2], &carry);
    r3 = mac_with_carry(0, self->repr.data[0], self->repr.data[3], &carry);
    r4 = carry;

    carry = 0;
    r3 = mac_with_carry(r3, self->repr.data[1], self->repr.data[2], &carry);
    r4 = mac_with_carry(r4, self->repr.data[1], self->repr.data[3], &carry);
    r5 = carry;

    carry = 0;
    r5 = mac_with_carry(r5, self->repr.data[2], self->repr.data[3], &carry);
    r6 = carry;

    r7 = r6 >> 63;
    r6 = (r6 << 1) | (r5 >> 63);
    r5 = (r5 << 1) | (r4 >> 63);
    r4 = (r4 << 1) | (r3 >> 63);
    r3 = (r3 << 1) | (r2 >> 63);
    r2 = (r2 << 1) | (r1 >> 63);
    r1 = r1 << 1;

    carry = 0;
    r0 = mac_with_carry(0, self->repr.data[0], self->repr.data[0], &carry);
    r1 = adc(r1, 0, &carry);
    r2 = mac_with_carry(r2, self->repr.data[1], self->repr.data[1], &carry);
    r3 = adc(r3, 0, &carry);
    r4 = mac_with_carry(r4, self->repr.data[2], self->repr.data[2], &carry);
    r5 = adc(r5, 0, &carry);
    r6 = mac_with_carry(r6, self->repr.data[3], self->repr.data[3], &carry);
    r7 = adc(r7, 0, &carry);
    fr_mont_reduce(self, r0, r1, r2, r3, r4, r5, r6, r7);
}

inline void fr_add_assign(Fr* self, const Fr* other) {
    frrepr_add_nocarry(&self->repr, &other->repr);
    fr_reduce(self);
}

inline void fr_double(Fr* self) {
    frrepr_mul2(&self->repr);
    fr_reduce(self);
}

inline void fr_sub_assign(Fr* self, const Fr* other) {
    if (frrepr_lt(&self->repr, &other->repr)) {
        const FrRepr MOD_TEMP = FR_MODULUS;
        frrepr_add_nocarry(&self->repr, &MOD_TEMP);
    }

    frrepr_sub_noborrow(&self->repr, &other->repr);
}

inline void fr_negate(Fr* self) {
    if (!frrepr_is_zero(&self->repr)) {
        FrRepr MOD_TEMP = FR_MODULUS;
        frrepr_sub_noborrow(&MOD_TEMP, &self->repr);
        frrepr_assign(&self->repr, &MOD_TEMP);
    }
}

inline bool fr_is_zero(const Fr* self) {
    return frrepr_is_zero(&self->repr);
}

__kernel void test_fr_is_valid(__global const Fr* a, __global char* result,
                            uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fr temp;
    fr_assign_global(&temp, &a[id]);
    result[id] = fr_is_valid(&temp);
}

__kernel void test_fr_mul_assign(__global Fr* a, __global const Fr* b, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fr left;
    Fr right;

    fr_assign_global(&left, &a[id]);
    fr_assign_global(&right, &b[id]);

    fr_mul_assign(&left, &right);

    fr_assign_to_global(&a[id], &left);
}

__kernel void test_fr_square(__global Fr* a, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fr temp;

    fr_assign_global(&temp, &a[id]);

    fr_square(&temp);

    fr_assign_to_global(&a[id], &temp);
}

__kernel void test_fr_add_assign(__global Fr* a, __global const Fr* b,
                                uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fr t1, t2;
    fr_assign_global(&t1, &a[id]);
    fr_assign_global(&t2, &b[id]);

    fr_add_assign(&t1, &t2);

    fr_assign_to_global(&a[id], &t1);
}

__kernel void test_fr_double(__global Fr* a, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fr t1;
    fr_assign_global(&t1, &a[id]);

    fr_double(&t1);

    fr_assign_to_global(&a[id], &t1);
}

__kernel void test_fr_sub_assign(__global Fr* a, __global const Fr* b,
                                uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fr t1, t2;
    fr_assign_global(&t1, &a[id]);
    fr_assign_global(&t2, &b[id]);

    fr_sub_assign(&t1, &t2);

    fr_assign_to_global(&a[id], &t1);
}

__kernel void test_fr_negate(__global Fr* a, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Fr t1;
    fr_assign_global(&t1, &a[id]);

    fr_negate(&t1);

    fr_assign_to_global(&a[id], &t1);
}

void inline affine_mul_binary(const Affine* base, const FrRepr* exp, Projective* out) {
    Projective result = PROJECTIVE_ZERO;

    bool found_one = false;

    for (int i = FR_WIDTH*64-1-FR_REPR_SHAVE_BITS; i >= 0; i--) {
        if (found_one) {
            projective_double(&result);
        }

        int part = i / 64;
        int pos = i % 64;

        bool bit = exp->data[part] & 1ul << pos;

        if (bit) {
            found_one = true;
            projective_add_assign_mixed(&result, base);
        }
    }

    projective_assign(out, &result);
}

#define WINDOW_SIZE 4

inline void affine_generate_table(const Affine* base, Projective* table) {
    Projective current = PROJECTIVE_ZERO;
    
    #pragma unroll
    for (int i = 1; i < 1 << WINDOW_SIZE; i++) {
        projective_add_assign_mixed(&current, base);
        projective_assign(&table[i-1], &current);
    }
}

// Optimized for window width of 4. The real minimum is around 4.6.
inline void affine_mul_window(const Affine* base, const FrRepr* exp, Projective* out) {
    Projective result = PROJECTIVE_ZERO;
    Projective table[(1 << WINDOW_SIZE) - 1];
    ulong window = (1u << WINDOW_SIZE) - 1;

    affine_generate_table(base, table);

    bool found_one = false;

    for (int i = (FR_WIDTH*64-1) / WINDOW_SIZE; i >= 0; i--) {
        if (found_one) {
            for (int j = 0; j < WINDOW_SIZE; j++) {
                projective_double(&result);
            } 
        }

        int part = i / 16;
        int pos = (i % 16) * WINDOW_SIZE;
        int value = (exp->data[part] & window << pos) >> pos;

        if (value) {
            found_one = true;
            projective_add_assign(&result, &table[value-1]);
        }
    }

    projective_assign(out, &result);
}

inline void projective_reduce_local(__local Projective* redBuf, Projective* out) {
    uint local_idx = get_local_id(0);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = get_local_size(0)/2; i > 0; i /= 2) {
        
        if (local_idx < i) {
            Projective first, second;
            projective_assign_local(&first, &redBuf[local_idx]);
            projective_assign_local(&second, &redBuf[local_idx+i]);
            projective_add_assign(&first, &second);

            projective_assign_to_local(&redBuf[local_idx], &first);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_idx == 0) {
        projective_assign_local(out, &redBuf[0]);
    }
}

inline void projective_reduce_local_smart(__local Projective* redBuf, Projective* out) {
    const uint local_idx = get_local_id(0);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = get_local_size(0)/2, num_shifts = 1; i > 0; i /= 2, num_shifts *= 2) {
        
        if (local_idx < i) {
            Projective first, second;
            first = redBuf[2*local_idx];
            second = redBuf[2*local_idx+1];

            for (int j = 0; j < num_shifts; j++) {
                projective_double(&second);
                projective_double(&second);
                projective_double(&second);
                projective_double(&second);
            }

            projective_add_assign(&first, &second);

            redBuf[local_idx] = first;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_idx == 0) {
        *out = redBuf[0];
    }
}

__kernel void projective_reduce_step(__global Projective* points, uint len,
                                     __local Projective* redBuf) {
    const uint idx = get_global_id(0);
    const uint local_idx = get_local_id(0);
    
    Projective point = PROJECTIVE_ZERO;

    if (idx < len) {
        projective_assign_global(&point, &points[idx]);
    }
    projective_assign_to_local(&redBuf[local_idx], &point);

    projective_reduce_local(redBuf, &point);
    if (local_idx == 0) {
        projective_assign_to_global(&points[idx/get_local_size(0)], &point);
    }
}

__kernel void affine_mulexp_binary(__global const Affine* bases, __global const FrRepr* exps, uint len,
                                   __local Projective* redBuf, __global Projective* out){
    uint idx = get_global_id(0);
    uint local_idx = get_local_id(0);

    Projective point = PROJECTIVE_ZERO;
    if (idx < len) {
        Affine base;
        FrRepr exp;
        affine_assign_global(&base, &bases[idx]);
        frrepr_assign_global(&exp, &exps[idx]);
        affine_mul_binary(&base, &exp, &point);
    }

    projective_assign_to_local(&redBuf[local_idx], &point);

    projective_reduce_local(redBuf, &point);

    if (local_idx == 0) {
        projective_assign_to_global(&out[idx/get_local_size(0)], &point);
    }
}

struct Task {
    Affine base;
    FrRepr exp;
};

typedef struct Task Task;
// Use group size of 64!!!
#define BUCKET_BITS 4

__kernel void affine_mulexp_smart(__global const Affine* points, __global const FrRepr* exps,
                                  uint len, uint chunk_size, __global Projective* out) {
    Projective buckets[15] = {
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO,
        PROJECTIVE_ZERO
    };
    const ulong mask = 0xf;
    __local Projective chunk_values[64];

    Affine current_point;
    FrRepr current_exp;

    const uint idx = get_global_id(0);
    const uint idx_local = get_local_id(0);
    const uint idx_group = get_group_id(0);

    const uint start_idx = idx_group * chunk_size;
    uint end_idx = (idx_group + 1) * chunk_size;


    const uint part = idx_local / 16;
    const uint shift = (idx_local % 16) * 4;

    if (end_idx > len) end_idx = len;

    if (start_idx >= len) return;

    for (uint i = start_idx; i < end_idx; i++) {
        current_point = points[i];
        current_exp = exps[i];
        uint exp_idx = (current_exp.data[part] >> shift) & mask;

        if (exp_idx > 0) {
            projective_add_assign_mixed(&buckets[exp_idx - 1], &current_point);
        }
    }
    
    Projective sum = PROJECTIVE_ZERO;
    Projective partial_sum = PROJECTIVE_ZERO;

    for (int i = 15; i > 0; i--) {
        projective_add_assign(&partial_sum, &buckets[i-1]);
        projective_add_assign(&sum, &partial_sum);
    }

    chunk_values[idx_local] = sum;

    projective_reduce_local_smart(chunk_values, &sum);
    if (idx_local == 0) {
        out[idx_group] = sum;
    }
}

__kernel void test_affine_mul_binary(__global const Affine* bases,
                                    __global const FrRepr* exps,
                                    __global Projective* results, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Affine base;
    FrRepr exp;
    Projective result;

    affine_assign_global(&base, &bases[id]);
    frrepr_assign_global(&exp, &exps[id]);

    affine_mul_binary(&base, &exp, &result);

    projective_assign_to_global(&results[id], &result);
}

__kernel void test_affine_mul_window(__global const Affine* bases,
                                    __global const FrRepr* exps,
                                    __global Projective* results, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    Affine base;
    FrRepr exp;
    Projective result;

    affine_assign_global(&base, &bases[id]);
    frrepr_assign_global(&exp, &exps[id]);
    
    affine_mul_window(&base, &exp, &result);

    projective_assign_to_global(&results[id], &result);
}

__kernel void projective_reduce_step_global(__global Projective* points, uint len) {
    const uint idx = get_global_id(0);
    
    Projective point1 = PROJECTIVE_ZERO;
    Projective point2 = PROJECTIVE_ZERO;

    if (idx < len/2) {
        point1 = points[2*idx];
    } else {
        return;
    }

    if (2*idx+1 < len) {
        point2 = points[2*idx+1];
    }
    
    projective_add_assign(&point1, &point2);

    points[idx] = point1;
}