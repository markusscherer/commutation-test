#ifndef BRUTE_FORCE_EVALUATION_POLICY_HPP_MSDA
#define BRUTE_FORCE_EVALUATION_POLICY_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

#include "../simd_tools.hpp"
#include "../simd_evaluation_components.hpp"
#include "../array_function.hpp"

template <uint64_t D, uint64_t A, class ElementType>
struct brute_force_evaluation_policy {};

template <> struct brute_force_evaluation_policy<4, 1, uint32_t> {
    typedef simd_evaluation_constants<4, 4> constants;
    typedef simd_evaluation_registers<4, 1> registers;

    inline static void init_registers(registers& r,
                                      array_function<4, 1, uint32_t> f,
                                      const constants& c) {
        __m128i partial_function = _mm_set1_epi32(f.storage[0]);

        // "blow up" function from packed 2-bit integers to packed 8-bit integers

#ifdef __AVX2__
        partial_function = _mm_srlv_epi32(partial_function, c.shift128);
#else
        __m128i tmp = _mm_srli_epi32(partial_function, 2);
        partial_function = _mm_blend_epi16(partial_function, tmp, 0xFC);
        tmp = _mm_srli_epi32(tmp, 2);
        partial_function = _mm_blend_epi16(partial_function, tmp, 0xF0);
        tmp = _mm_srli_epi32(tmp, 2);
        partial_function = _mm_blend_epi16(partial_function, tmp, 0xC0);
#endif

        partial_function = _mm_and_si128(partial_function, c.epi8_2lsb_mask_128);
        partial_function = _mm_shuffle_epi8(partial_function, c.shuf128);

        r.f0 = partial_function;
    }

    inline static void eval(__m128i& res, uint64_t matcount, __m128i& matl,
                            registers& r, const constants) {
        res = _mm_shuffle_epi8(r.f0, matl);
    }
};

template <> struct brute_force_evaluation_policy<4, 2, uint32_t> {
    typedef simd_evaluation_constants<4, 4> constants;
    typedef simd_evaluation_registers<4, 2> registers;

    inline static void init_registers(registers& r,
                                      array_function<4, 2, uint32_t> f,
                                      const constants& c) {
        __m128i partial_function = _mm_set1_epi32(f.storage[0]);

        // "blow up" function from packed 2-bit integers to packed 8-bit integers

#ifdef __AVX2__
        partial_function = _mm_srlv_epi32(partial_function, c.shift128);
#else
        __m128i tmp = _mm_srli_epi32(partial_function, 2);
        partial_function = _mm_blend_epi16(partial_function, tmp, 0xFC);
        tmp = _mm_srli_epi32(tmp, 2);
        partial_function = _mm_blend_epi16(partial_function, tmp, 0xF0);
        tmp = _mm_srli_epi32(tmp, 2);
        partial_function = _mm_blend_epi16(partial_function, tmp, 0xC0);
#endif

        partial_function = _mm_and_si128(partial_function, c.epi8_2lsb_mask_128);
        partial_function = _mm_shuffle_epi8(partial_function, c.shuf128);

        r.f0 = partial_function;
    }

    inline static void eval(__m128i& res, uint64_t matcount, __m128i& matl,
                            registers& r, const constants&) {
        res = _mm_shuffle_epi8(r.f0, matl);
    }
};

template <class ElementType>
struct brute_force_evaluation_policy<4, 3, ElementType> {
    typedef simd_evaluation_constants<4, 4> constants;
    typedef simd_evaluation_registers<4, 3> registers;

    inline static void init_registers(registers& r,
                                      array_function<4, 3, ElementType> f) {
        array_to_si128(f.storage, r.f0);
    }

    inline static void eval(__m128i& res, uint64_t matcount, __m128i& matl,
                            __m128i& math, registers& r, const constants& c) {
        partial_eval<0>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<1>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<2>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<3>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
    }
};

template <class ElementType>
struct brute_force_evaluation_policy<4, 4, ElementType> {
    typedef simd_evaluation_constants<4, 4> constants;
    typedef simd_evaluation_registers<4, 4> registers;

    inline static void init_registers(registers& r,
                                      array_function<4, 4, ElementType> f) {
        array_to_si128(f.storage, r.f0, r.f1, r.f2, r.f3);
    }

    inline static void eval(__m128i& res, uint64_t matcount, __m128i& matl,
                            __m128i& math, registers& r, const constants& c) {
        partial_eval<0>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<1>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<2>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<3>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);

        partial_eval<4>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<5>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<6>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<7>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);

        partial_eval<8>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<9>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<10>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);
        partial_eval<11>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);

        partial_eval<12>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);
        partial_eval<13>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);
        partial_eval<14>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);
        partial_eval<15>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);
    }
};

#endif
