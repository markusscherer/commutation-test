#ifndef SIMD_EVALUATION_COMPONENTS_HPP_MSDA
#define SIMD_EVALUATION_COMPONENTS_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

template <uint64_t, uint64_t> struct simd_evaluation_registers {};

template <> struct simd_evaluation_registers<4, 1> {
    __m128i f0;
};

template <> struct simd_evaluation_registers<4, 2> {
    __m128i f0;
};

template <> struct simd_evaluation_registers<4, 3> {
    __m128i f0;
};

template <> struct simd_evaluation_registers<4, 4> {
    __m128i f0;
    __m128i f1;
    __m128i f2;
    __m128i f3;
};

template <uint64_t, uint64_t> struct simd_evaluation_constants {};

template <> struct simd_evaluation_constants<4, 4> {
    const __m128i shuf128;
    const __m128i shift128;
    const __m128i epi8_2lsb_mask_128;

    simd_evaluation_constants()
        : shuf128(
              _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0)),
          shift128(_mm_set_epi32(6, 4, 2, 0)),
          epi8_2lsb_mask_128(_mm_set1_epi32(0x03030303)) {
    }
};

template <uint8_t R>
inline void partial_eval(const __m128i& function, __m128i& res,
                         const __m128i& epi8_2lsb_mask, const __m128i& shuf128,
                         const __m128i& shift128, const __m128i& matl,
                         const __m128i& math) {

    const uint8_t function_select = 0x55 * (R & 0x03);
    // select part of function table according to to most significant parameter(s)
    // (R)
    __m128i partial_function = _mm_shuffle_epi32(function, function_select);

    // "blow up" function from packed 2-bit integers to packed 8-bit integers

#ifdef __AVX2__
    partial_function = _mm_srlv_epi32(partial_function, shift128);
#else
    __m128i tmp = _mm_srli_epi32(partial_function, 2);
    partial_function = _mm_blend_epi16(partial_function, tmp, 0xFC);
    tmp = _mm_srli_epi32(tmp, 2);
    partial_function = _mm_blend_epi16(partial_function, tmp, 0xF0);
    tmp = _mm_srli_epi32(tmp, 2);
    partial_function = _mm_blend_epi16(partial_function, tmp, 0xC0);
#endif

    partial_function = _mm_and_si128(partial_function, epi8_2lsb_mask);
    partial_function = _mm_shuffle_epi8(partial_function, shuf128);

    // apply matrix of least significant parameters to partial_function
    __m128i partial_res = _mm_shuffle_epi8(partial_function, matl);

    // write R in every field of mask
    __m128i mask = _mm_set1_epi8(R);

    // set all bits in mask, that equal most significant parameter(s)
    mask = _mm_cmpeq_epi8(math, mask);

    // blend all matching results in partial result with final result
    res = _mm_blendv_epi8(res, partial_res, mask);
}
template <uint8_t R>
inline void partial_eval_with_selector(const __m128i& function, __m128i& res,
                                       const __m128i& epi8_2lsb_mask,
                                       const __m128i& shuf128,
                                       const __m128i& shift128,
                                       const __m128i& matl, const __m128i& math,
                                       uint8_t selector) {

    const uint8_t function_select = 0x55 * (R & 0x03);
    // select part of function table according to to most significant parameter(s)
    // (R)
    __m128i partial_function = _mm_shuffle_epi32(function, function_select);

    // "blow up" function from packed 2-bit integers to packed 8-bit integers
#ifdef __AVX2__
    partial_function = _mm_srlv_epi32(partial_function, shift128);
#else
    __m128i tmp = _mm_srli_epi32(partial_function, 2);
    partial_function = _mm_blend_epi16(partial_function, tmp, 0xFC);
    tmp = _mm_srli_epi32(tmp, 2);
    partial_function = _mm_blend_epi16(partial_function, tmp, 0xF0);
    tmp = _mm_srli_epi32(tmp, 2);
    partial_function = _mm_blend_epi16(partial_function, tmp, 0xC0);
#endif
    partial_function = _mm_and_si128(partial_function, epi8_2lsb_mask);
    partial_function = _mm_shuffle_epi8(partial_function, shuf128);

    // apply matrix of least significant parameters to partial_function
    __m128i partial_res = _mm_shuffle_epi8(partial_function, matl);

    // write selector in every field of mask
    __m128i mask = _mm_set1_epi8(selector);

    // set all bits in mask, that equal most significant parameter(s)
    mask = _mm_cmpeq_epi8(math, mask);

    // blend all matching results in partial result with final result
    res = _mm_blendv_epi8(res, partial_res, mask);
}

#endif
