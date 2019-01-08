// Implementation of primitives found in pairing/lib.rs


// The wrapper to split a 64 bit integer into two 32 bit numbers (passed as arguments)
//
inline void split_ulong(ulong src, ulong* high, ulong* low) {
    *high = src >> 32;
    *low = src & 0xFFFFFFFF;
}

// The wrapper to combine two 32 bit integers into a 64 bit one
//
inline ulong combine_ulong(ulong high, ulong low) {
    return high << 32 | low;
}

// Calculate a - b - borrow, returning the difference, and modifying borrow
// passed as the argument
//
inline ulong sbb(ulong a, ulong b, ulong* borrow) {
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
inline ulong adc(ulong a, ulong b, ulong* carry) {
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
inline ulong mac_with_carry(ulong a, ulong b, ulong c, ulong* carry) {
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


__kernel void test_split_ulong(__global ulong* src, __global ulong* res_high,
                               __global ulong* res_low, uint length) {
    const uint id = get_global_id(0);

    if (id >= length) return;

    ulong priv_res_high, priv_res_low;
    split_ulong(src[id], &priv_res_high, &priv_res_low);

    res_high[id] = priv_res_high;
    res_low[id] = priv_res_low;
}

__kernel void test_combine_ulong(__global ulong* a, __global ulong* b,
                                 __global ulong* c, uint length) {
    const uint id = get_global_id(0);

    if (id >= length) return;

    c[id] = combine_ulong(a[id], b[id]);
}

__kernel void test_sbb(__global ulong* a, __global ulong* b,
                       __global ulong* borrow, __global ulong* result,
                       uint length) {
    const uint id = get_global_id(0);

    if (id >= length) return;

    ulong priv_borrow = borrow[id];

    result[id] = sbb(a[id], b[id], &priv_borrow);
    borrow[id] = priv_borrow;
}

__kernel void test_adc(__global ulong* a, __global ulong* b,
                       __global ulong* carry, __global ulong* result,
                       uint length) {
    const uint id = get_global_id(0);

    if (id >= length) return;

    ulong priv_carry = carry[id];

    result[id] = adc(a[id], b[id], &priv_carry);
    carry[id] = priv_carry;
}

__kernel void test_mac_with_carry(__global ulong* a, __global ulong* b,
                                  __global ulong* c, __global ulong* carry,
                                  __global ulong* result, uint length) {
    const uint id = get_global_id(0);

    if (id >= length) return;

    ulong priv_carry = carry[id];
    result[id] = mac_with_carry(a[id], b[id], c[id], &priv_carry);
    carry[id] = priv_carry;
}

// Implementation of finite field arithmetic found in fr.rs
struct FrRepr {
    ulong data[4];
};

typedef struct FrRepr FrRepr;

inline void assign(FrRepr* left, FrRepr* right) {
    for (int i = 0; i < 4; i++) {
        left->data[i] = right->data[i];
    }
}

inline void assign_global(FrRepr* left, __global FrRepr* right) {
    for (int i = 0; i < 4; i++) {
        left->data[i] = right->data[i];
    }
}

/* FrRepr MODULUS = {
    {
        0xffffffff00000001,
        0x53bda402fffe5bfe,
        0x3339d80809a1d805,
        0x73eda753299d7d48
    }
}; */

#define MODULUS {\
                    {\
                    0xffffffff00000001,\
                    0x53bda402fffe5bfe,\
                    0x3339d80809a1d805,\
                    0x73eda753299d7d48\
                    }\
                }

#define MODULUS_BITS 255u;

#define REPR_SHAVE_BITS 1u;

__constant FrRepr R = {
    {
        0x1fffffffe,
        0x5884b7fa00034802,
        0x998c4fefecbc4ff5,
        0x1824b159acc5056f 
    }
};

__constant FrRepr R2 = {
    {
        0xc999e990f3f29c6d,
        0x2b6cedcb87925c23,
        0x5d314967254398f,
        0x748d9d99f59ff11
    }
};

#define INV 0xfffffffefffffffful;

__constant FrRepr GENERATOR = {
    {
        0xefffffff1,
        0x17e363d300189c0f,
        0xff9c57876f8457b0,
        0x351332208fc5a8c4  
    }
};

#define S 32u;

__constant FrRepr ROOT_OF_UNITY = {
    {
        0xb9b58d8c5f0e466a,
        0x5b1b4c801819d7ec,
        0xaf53ae352a31e64,
        0x5bf3adda19e9b27b
    }
};

inline bool is_odd(FrRepr* self) {
    return (self->data[0] & 0x01) == 1;
}

__kernel void test_is_odd(__global FrRepr* a, __global char* result,
                          uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp;
    assign_global(&temp, &(a[id]));
    result[id] = is_odd(&temp);
}

inline bool is_even(FrRepr* self) {
    return !is_odd(self);
}

__kernel void test_is_even(__global FrRepr* a, __global char* result,
                          uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp;
    assign_global(&temp, &(a[id]));
    result[id] = is_even(&temp);
}

inline bool is_zero(FrRepr* self) {
    return (self->data[0] == 0)
        && (self->data[1] == 0)
        && (self->data[2] == 0)
        && (self->data[3] == 0);
}

__kernel void test_is_zero(__global FrRepr* a, __global char* result,
                          uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp;
    assign_global(&temp, &(a[id]));
    result[id] = is_zero(&temp);
}

inline int cmp(FrRepr* a, FrRepr* b) {
    for (int i = 3; i >= 0; i--) {
        if (a->data[i] < b->data[i]) return -1;
        else if (a->data[i] > b->data[i]) return 1;
    }
    return 0;
}

__kernel void test_cmp(__global FrRepr* a, __global FrRepr* b, 
                       __global char* result, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp_a, temp_b;
    assign_global(&temp_a, &(a[id]));
    assign_global(&temp_b, &(b[id]));
    result[id] = cmp(&temp_a, &temp_b);
}

inline bool lt(FrRepr* a, FrRepr* b) {
    return cmp(a,b) == -1;
}

__kernel void test_lt(__global FrRepr* a, __global FrRepr* b, 
                       __global char* result, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp_a, temp_b;
    assign_global(&temp_a, &(a[id]));
    assign_global(&temp_b, &(b[id]));
    result[id] = lt(&temp_a, &temp_b);
}

inline bool eq(FrRepr* a, FrRepr* b) {
    return cmp(a,b) == 0;
}

__kernel void test_eq(__global FrRepr* a, __global FrRepr* b, 
                       __global char* result, uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp_a, temp_b;
    assign_global(&temp_a, &(a[id]));
    assign_global(&temp_b, &(b[id]));
    result[id] = eq(&temp_a, &temp_b);
}

inline bool is_valid(FrRepr* self) {
    FrRepr temp = MODULUS;
    
    return lt(self, &temp);
}

__kernel void test_is_valid(__global FrRepr* a, __global char* result,
                            uint length) {
    uint id = get_global_id(0);
    if (id >= length) return;

    FrRepr temp;
    assign_global(&temp, &(a[id]));
    result[id] = is_valid(&temp);
}
/*
inline void sub_noborrow(FrRepr* self, FrRepr* other) {
    ulong borrow = 0;

    for (int i = 0; i < 4; i++) {
        self->data[i] = sbb(self->data[0], other->data[0], &borrow);
    }
}

__kernel void test_sub_noborrow(__global FrRepr* a, __global FrRepr*,
                                __global ulong* borrow, uint length) {
    
}

inline void add_nocarry(FrRepr* self, FrRepr* other) {
    ulong carry = 0;

    for (int i = 0; i < 4; i++) {
        self->data[i] = adc(self->data[0], self->data[1], &carry);
    }
}

inline void reduce(FrRepr* self) {
    if (!is_valid(self)) {
        sub_noborrow(self, &MODULUS);
    }
}

inline void mont_reduce(FrRepr* self,
                        const ulong r0,
                        ulong r1,
                        ulong r2,
                        ulong r3,
                        ulong r4,
                        ulong r5,
                        ulong r6,
                        ulong r7) {
    ulong k = r0 * INV;
    ulong carry = 0;
    mac_with_carry(r0, k, MODULUS.data[0], &carry);
    r1 = mac_with_carry(r1, k, MODULUS.data[1], &carry);
    r2 = mac_with_carry(r2, k, MODULUS.data[2], &carry);
    r3 = mac_with_carry(r3, k, MODULUS.data[3], &carry);
    r4 = adc(r4, 0, &carry);

    ulong carry2 = carry;
    k = r1 * INV;
    carry = 0;
    mac_with_carry(r1, k, MODULUS.data[0], &carry);
    r2 = mac_with_carry(r2, k, MODULUS.data[1], &carry);
    r3 = mac_with_carry(r3, k, MODULUS.data[2], &carry);
    r4 = mac_with_carry(r4, k, MODULUS.data[3], &carry);
    r5 = adc(r5, carry2, &carry);

    carry2 = carry;
    k = r2 * INV;
    carry = 0;
    mac_with_carry(r2, k, MODULUS.data[0], &carry);
    r3 = mac_with_carry(r3, k, MODULUS.data[1], &carry);
    r4 = mac_with_carry(r4, k, MODULUS.data[2], &carry);
    r5 = mac_with_carry(r5, k, MODULUS.data[3], &carry);
    r6 = adc(r6, carry2, &carry);

    carry2 = carry;
    k = r3 * INV;
    carry = 0;
    mac_with_carry(r3, k, MODULUS.data[0], &carry);
    r4 = mac_with_carry(r4, k, MODULUS.data[1], &carry);
    r5 = mac_with_carry(r5, k, MODULUS.data[2], &carry);
    r6 = mac_with_carry(r6, k, MODULUS.data[3], &carry);
    r7 = adc(r7, carry2, &carry);

    self->data[0] = r4;
    self->data[1] = r5;
    self->data[2] = r6;
    self->data[3] = r7;
    reduce(self);
}

inline void mul_assign(FrRepr* self, FrRepr* other) {
    ulong carry, r0, r1, r2, r3, r4, r5, r6, r7;
    
    carry = 0;
    r0 = mac_with_carry(0, self->data[0], other->data[0], &carry);
    r1 = mac_with_carry(0, self->data[0], other->data[1], &carry);
    r2 = mac_with_carry(0, self->data[0], other->data[2], &carry);
    r3 = mac_with_carry(0, self->data[0], other->data[3], &carry);
    r4 = carry;

    carry = 0;
    r1 = mac_with_carry(r1, self->data[1], other->data[0], &carry);
    r2 = mac_with_carry(r2, self->data[1], other->data[1], &carry);
    r3 = mac_with_carry(r3, self->data[1], other->data[2], &carry);
    r4 = mac_with_carry(r4, self->data[1], other->data[3], &carry);
    r5 = carry;

    carry = 0;
    r2 = mac_with_carry(r2, self->data[2], other->data[0], &carry);
    r3 = mac_with_carry(r3, self->data[2], other->data[1], &carry);
    r4 = mac_with_carry(r4, self->data[2], other->data[2], &carry);
    r5 = mac_with_carry(r5, self->data[2], other->data[3], &carry);
    r6 = carry;

    carry = 0;
    r3 = mac_with_carry(r3, self->data[3], other->data[0], &carry);
    r4 = mac_with_carry(r4, self->data[3], other->data[1], &carry);
    r5 = mac_with_carry(r5, self->data[3], other->data[2], &carry);
    r6 = mac_with_carry(r6, self->data[3], other->data[3], &carry);
    r7 = carry;

    mont_reduce(self, r0, r1, r2, r3, r4, r5, r6, r7);
}

inline void square(FrRepr* self) {
    ulong carry, r0, r1, r2, r3, r4, r5, r6, r7;
    
    carry = 0;
    r1 = mac_with_carry(0, self->data[0], self->data[1], &carry);
    r2 = mac_with_carry(0, self->data[0], self->data[2], &carry);
    r3 = mac_with_carry(0, self->data[0], self->data[3], &carry);
    r4 = carry;

    carry = 0;
    r3 = mac_with_carry(r3, self->data[1], self->data[2], &carry);
    r4 = mac_with_carry(r4, self->data[1], self->data[3], &carry);
    r5 = carry;

    r5 = mac_with_carry(r5, self->data[2], self->data[3], &carry);
    r6 = carry;

    r7 = r6 >> 63;
    r6 = (r6 << 1) | (r5 >> 63);
    r5 = (r5 << 1) | (r4 >> 63);
    r4 = (r4 << 1) | (r3 >> 63);
    r3 = (r3 << 1) | (r2 >> 63);
    r2 = (r2 << 1) | (r1 >> 63);
    r1 = r1 << 1;

    carry = 0;
    r0 = mac_with_carry(0, self->data[0], self->data[0], &carry);
    r1 = adc(r1, 0, &carry);
    r2 = mac_with_carry(r2, self->data[1], self->data[1], &carry);
    r3 = adc(r3, 0, &carry);
    r4 = mac_with_carry(r4, self->data[2], self->data[2], &carry);
    r5 = adc(r5, 0, &carry);
    r6 = mac_with_carry(r6, self->data[3], self->data[3], &carry);
    r7 = adc(r7, 0, &carry);

    mont_reduce(self, r0, r1, r2, r3, r4, r5, r6, r7);
}

inline void div2(FrRepr* self) {
    ulong t = 0;

    for (int i = 3; i >= 0; i--) {
        ulong t2 = self->data[i] << 63;
        self->data[i] >>= 1;
        self->data[i] |= t;
        t = t2;
    }
}

inline void mul2(FrRepr* self) {
    ulong last = 0;

    for (int i = 0; i < 4; i++) {
        ulong tmp = self->data[i] >> 63;
        self->data[i] <<= 1;
        self->data[i] |= last;
        last = tmp;
    }
}

inline void add_assign(FrRepr* self, FrRepr* other) {
    add_nocarry(self, other);
    reduce(self);
}

inline void perform_double(FrRepr* self) {
    mul2(self);
    reduce(self);
}

inline void sub_assign(FrRepr* self, FrRepr* other) {
    if lt(self, other) {
        add_nocarry(self, other);
    }

    sub_noborrow(self, other);
}

inline void negate(FrRepr* self) {
    if (!is_zero(self)) {
        FrRepr tmp;
        assign(&tmp, &MODULUS);
        sub_noborrow(&tmp, self);
        assign(self, &tmp);
    }
}

// Implementation of projective arithmetic
//
struct Projective {
    FrRepr x;
    FrRepr y;
    FrRepr z;
};

inline void bool is_zero(Projective* self) {
    return is_zero(self->z);
}

void doubling(Projective* self) {
    if (is_zero(self)) {
        return;
    }

    FrRepr a;
    assign(&a, &(self->x));
    square(&a);

    FrRepr b;
    assign(&b, &(self->y));

    FrRepr c;
    assign(&c, &b);
    square(&c);

    FrRepr d;
    assign(&d, &(self->x));
    add_assign(&d, &b);
    square(&d);
    sub_assign(&d, &a);
    sub_assign(&d, &c);
    doubling(&d);

    FrRepr e;
    assign(&e, &a);
    doubling(&e);
    add_assign(&a);

    FrRepr f;
    assign(&f, &e);
    square(&f);

    mul_assign(&(self->z), &(self->y));
    doubling(&(self->z));

    assign(&(self->x), &f);
    sub_assign(&(self->x), &d);
    sub_assign(&(self->x), &d);

    assign(&(self->y), &d);
    sub_assign(&(self->y), &(self->x));
    mul_assign(&(self->y), &e);
    doubling(&c);
    doubling(&c);
    doubling(&c);
    sub_assign(&(self->y), &c);
}

void assign(Projective* left, Projective* right) {
    assign(left->x, right->x);
    assign(left->y, right->y);
    assign(seft->z, right->z);
}

void add_assign(Projective* self, Projective* other) {
    if (is_zero(self)) {
        assign(self, other);
        return;
    }

    if (is_zero(other)) {
        return;
    }

    FrRepr z1z1;
    assign(&z1z1, &(self->z));
    square(&z1z1);

    FrRepr z2z2;
    assign(&z2z2, &(other->z));
    square(&z2z2);

    FrRepr u1;
    assign(&u1, &(self->x));
    mul_assign(&u1, &z2z2);

    FrRepr u2;
    assign(&u2, &(other->x));
    mul_assign(&u2, &z1z1);

    FrRepr s1;
    assign(&s1, &(self->y));
    mul_assign(&s1, &(other->z));
    mul_assign(&s1, &z2z2);

    FrRepr s2;
    assign(&s2, &(other->y));
    mul_assign(&s2, &(self->z));
    mul_assign(&s2, &(z1z1));

    if (eq(&u1, &u2) && eq(&s1, &s2)) {
        doubling(self);
    } else {
        FrRepr h;
        assign(&h, &u2);
        sub_assign(&h, &u1);

        FrRepr i;
        assign(&i, &h);
        doubling(&i);
        square(&i);

        FrRepr j;
        assign(&j, &h);
        mul_assign(&j, &i);

        FrRepr r;
        assign(&r, &s2);
        sub_assign(&r, &s1);
        doubling(&r);

        FrRepr v;
        assign(&v, &u1);
        mul_assign(&v, &i);

        assign(&(self->x), &r);
        square(&(self->x));
        sub_assign(&(self->x), &j);
        sub_assign(&(self->x), &v);
        sub_assign(&(self->x), &v);

        assign(&(self->y), &v);
        sub_assign(&(self->y), &(self->x));
        mul_assign(&(self->y), &r);
        mul_assign(&s1, &j);
        doubling(&s1);
        sub_assign(&(self->y), &s1);

        add_assign(&(self->z), &(other->z));
        square(&(self->z));
        sub_assign(&(self->z), &z1z1);
        sub_assign(&(self->z), &z2z2);
        mul_assign(&(self->z), &h);
    }
}
*/