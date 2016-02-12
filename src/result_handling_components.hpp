#ifndef RESULT_HANDLING_COMPONENTS_HPP_MSDA
#define RESULT_HANDLING_COMPONENTS_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

template <uint64_t NumMatrices> __m128i init_const3333();

template <> __m128i init_const3333<4>() {
    return _mm_set1_epi32(0x03);
}

template <> __m128i init_const3333<8>() {
    return _mm_set1_epi16(0x03);
}

template <uint64_t NumMatrices> __m128i init_constFFFF();

template <> __m128i init_constFFFF<4>() {
    return _mm_set1_epi32(0x0F);
}

template <> __m128i init_constFFFF<8>() {
    return _mm_set1_epi16(0x0F);
}

inline static void result_to_matrix(const __m128i& res, __m128i& matl,
                                    __m128i& math, const __m128i& shuf128,
                                    const __m128i& const2020,
                                    const __m128i& const_matl_mask,
                                    const __m128i& const_math_mask) {
    // unite neighbouring 2-bit-integers to 4-integers
    // afterwards every even 8-bit-field field of math contains a valid
    // 4-bit integer
    math = _mm_shuffle_epi8(res, shuf128);
#ifdef __AVX2__
    math = _mm_sllv_epi32(math, const2020);
#else
    __m128i tmp = _mm_slli_epi32(math, 2);
    math = _mm_blend_epi16(math, tmp, 0xCC);
#endif
    math = _mm_shuffle_epi8(math, shuf128);
    math = _mm_srli_epi32(math, 8);
    math = _mm_or_si128(math, res);

    // write fields 0,4,8 and 12 to matl
    matl = _mm_and_si128(math, const_matl_mask);
    // write fields 2,6,10 and 14 to math (at 0,4,8 and 12)
    math = _mm_srli_epi32(math, 16);
    math = _mm_and_si128(math, const_math_mask);
}

inline static void result_to_matrix(const __m128i& res, __m128i& matl,
                                    const __m128i& shuf128,
                                    const __m128i& const2020,
                                    const __m128i& const_matl_mask) {
    // unite neighbouring 2-bit-integers to 4-integers
    // afterwards every even 8-bit-field field of math contains a valid
    // 4-bit integer
    matl = _mm_shuffle_epi8(res, shuf128);
#ifdef __AVX2__
    matl = _mm_sllv_epi32(matl, const2020);
#else
    __m128i tmp = _mm_slli_epi32(matl, 2);
    matl = _mm_blend_epi16(matl, tmp, 0xCC);
#endif
    matl = _mm_shuffle_epi8(matl, shuf128);
    matl = _mm_srli_epi32(matl, 8);
    matl = _mm_or_si128(matl, res);

    // write fields 0,4,8 and 12 to matl
    matl = _mm_and_si128(matl, const_matl_mask);
}

inline static void dense_result_to_matrix(const __m128i& res, __m128i& matl,
                                          const __m128i& const_matl_mask) {
    matl = _mm_srli_epi32(res, 6);
    matl = _mm_or_si128(matl, res);
    matl = _mm_and_si128(matl, const_matl_mask);
}

#endif
