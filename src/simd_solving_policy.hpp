#ifndef SIMD_SOLVING_POLICY_HPP_MSDA
#define SIMD_SOLVING_POLICY_HPP_MSDA

#include <type_traits>
#include <cstdint>
#include <cmath>

#include <iostream>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

#include "array_function.hpp"
#include "simd_tools.hpp"

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
    partial_function = _mm_srlv_epi32(partial_function, shift128);
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

inline void complete_eval(const __m128i& f0, const __m128i& f1,
                          const __m128i& f2, const __m128i& f3, __m128i& resf,
                          const __m128i& matl, const __m128i& math) {
    __m128i shuf128 =
        _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
    __m128i shift128 = _mm_set_epi32(6, 4, 2, 0);
    __m128i epi8_2lsb_mask_128 = _mm_set1_epi32(0x03030303);

    partial_eval<0>(f0, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<1>(f0, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<2>(f0, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<3>(f0, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);

    partial_eval<4>(f1, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<5>(f1, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<6>(f1, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<7>(f1, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);

    partial_eval<8>(f2, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<9>(f2, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<10>(f2, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<11>(f2, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);

    partial_eval<12>(f3, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<13>(f3, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<14>(f3, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
    partial_eval<15>(f3, resf, epi8_2lsb_mask_128, shuf128, shift128, matl, math);
}

inline void result_to_matrix(const __m128i& res, __m128i& matl, __m128i& math) {
    __m128i shuf128 =
        _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);

    math = _mm_shuffle_epi8(res, shuf128);
    math = _mm_sllv_epi32(math, _mm_set1_epi64x(0x0000000200000000));
    math = _mm_shuffle_epi8(math, shuf128);

    math = _mm_srli_epi32(math, 8);
    math = _mm_or_si128(math, res);

    matl = _mm_and_si128(math, _mm_set1_epi32(0x0F));
    math = _mm_srli_epi32(math, 16);
    math = _mm_and_si128(math, _mm_set1_epi32(0x0F));
}

inline void blow_up_to_matrix(__m128i& mat, const uint16_t matcount,
                              const __m128i& epi8_4lsb_mask_128,
                              const __m128i& shuf128, const __m128i& shift128) {
    mat = _mm_set1_epi32(matcount);

    mat = _mm_srlv_epi32(mat, shift128);
    mat = _mm_srlv_epi32(mat, shift128);
    mat = _mm_and_si128(mat, epi8_4lsb_mask_128);
    mat = _mm_shuffle_epi8(mat, shuf128);
    mat = _mm_broadcastd_epi32(mat);
}

inline void blow_up_to_transposed_matrix(__m128i& mat, const uint16_t matcount,
                                         const __m128i& shuf128,
                                         const __m128i& shift128) {
    mat = _mm_set1_epi16(matcount);
    __m128i tmp = _mm_setzero_si128();
    tmp = _mm_insert_epi64(tmp, 0x0000000400000004, 1);
    mat = _mm_srlv_epi32(mat, _mm_add_epi32(shift128, tmp));
    tmp = _mm_srlv_epi32(mat, _mm_set1_epi32(2));
    mat = _mm_and_si128(mat, _mm_set1_epi32(0x00000003));
    tmp = _mm_and_si128(tmp, _mm_set1_epi32(0x0000000C));
    mat = _mm_or_si128(mat, tmp);
    mat = _mm_shuffle_epi8(mat, shuf128);
    mat = _mm_broadcastd_epi32(mat);
}

template <uint64_t D, uint64_t A1, uint64_t A2>
struct blowup_matrix_generation_policy {
    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math) {
        __m128i shuf128 =
            _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
        __m128i shift128 = _mm_set_epi32(6, 4, 2, 0);
        __m128i epi8_4lsb_mask_128 = _mm_set1_epi32(0x0F0F0F0F);
        __m128i const0123 = _mm_set_epi32(3, 2, 1, 0);

        blow_up_to_matrix(matl, matcount, epi8_4lsb_mask_128, shuf128, shift128);
        matl = _mm_add_epi32(matl, const0123);

        if (matcount % 65536 == 0) {
            blow_up_to_matrix(math, matcount >> 16, epi8_4lsb_mask_128, shuf128,
                              shift128);
        }
    }
};

template <uint64_t D, uint64_t A1, uint64_t A2>
struct blowup_transposed_matrix_generation_policy {
    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math) {
        __m128i shuf128 =
            _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
        __m128i shift128 = _mm_set_epi32(6, 4, 2, 0);
        __m128i const0123 = _mm_set_epi32(3, 2, 1, 0);
        const uint32_t lcount =
            ((matcount & 0x00FF0000) >> 8) | (matcount & 0x000000FF);
        blow_up_to_transposed_matrix(matl, lcount, shuf128, shift128);
        matl = _mm_add_epi32(matl, const0123);

        if (matcount % 256 == 0) {
            const uint32_t hcount =
                ((matcount & 0xFF000000) | ((matcount & 0x0000FF00) << 8)) >> 16;
            blow_up_to_transposed_matrix(math, hcount, shuf128, shift128);
        }
    }
};

template <uint64_t D, uint64_t A1, uint64_t A2,
          template <uint64_t, uint64_t, typename> class FunctionType,
          class ElementType, class MatrixGenerationPolicy =
          blowup_matrix_generation_policy<D, A1, A2>,
          class TransposedMatrixGenerationPolicy =
          blowup_transposed_matrix_generation_policy<D, A1, A2>>
struct simd_solving_policy {
    typedef ElementType element_type;
    static const uint64_t domain_size = D;
    static const uint64_t cell_count = A1 * A2;
    static const uint64_t matrix_count = cpow(domain_size, cell_count);

    static bool commutes(FunctionType<D, A1, ElementType> f1,
                         FunctionType<D, A2, ElementType> f2);
};

template <template <uint64_t, uint64_t, typename> class FunctionType,
          class ElementType, class MatrixGenerationPolicy,
          class TransposedMatrixGenerationPolicy>
struct simd_solving_policy<4, 4, 4, FunctionType, ElementType,
           MatrixGenerationPolicy,
           TransposedMatrixGenerationPolicy> {
    typedef ElementType element_type;
    static const uint64_t domain_size = 4;
    static const uint64_t cell_count = 4 * 4;
    static const uint64_t matrix_count = cpow(domain_size, cell_count);

    static bool commutes(FunctionType<4, 4, ElementType> f,
                         FunctionType<4, 4, ElementType> g) {

        __m128i regf0;
        __m128i regf1;
        __m128i regf2;
        __m128i regf3;

        __m128i regg0;
        __m128i regg1;
        __m128i regg2;
        __m128i regg3;

        __m128i resf;
        __m128i resg;

        __m128i math = _mm_setzero_si128();
        __m128i matl = _mm_setzero_si128();

        __m128i tmath = _mm_setzero_si128();
        __m128i tmatl = _mm_setzero_si128();

        __m128i accmatfh = _mm_setzero_si128();
        __m128i accmatfl = _mm_setzero_si128();
        __m128i accmatgh = _mm_setzero_si128();
        __m128i accmatgl = _mm_setzero_si128();

        array_to_si128(f.storage, regf0, regf1, regf2, regf3);
        array_to_si128(g.storage, regg0, regg1, regg2, regg3);

        for (uint64_t i = 0; i < matrix_count; i += 4) {
            MatrixGenerationPolicy::next_matrix(i, matl, math);
            TransposedMatrixGenerationPolicy::next_matrix(i, tmatl, tmath);

            complete_eval(regf0, regf1, regf2, regf3, resf, matl, math);
            complete_eval(regg0, regg1, regg2, regg3, resg, tmatl, tmath);

            __m128i matfh = _mm_setzero_si128();
            __m128i matfl = _mm_setzero_si128();
            result_to_matrix(resf, matfl, matfh);

            accmatfh = _mm_bslli_si128(accmatfh, 1);
            accmatfh = _mm_or_si128(accmatfh, matfh);

            accmatfl = _mm_bslli_si128(accmatfl, 1);
            accmatfl = _mm_or_si128(accmatfl, matfl);
            __m128i matgh = _mm_setzero_si128();
            __m128i matgl = _mm_setzero_si128();
            result_to_matrix(resg, matgl, matgh);

            accmatgh = _mm_bslli_si128(accmatgh, 1);
            accmatgh = _mm_or_si128(accmatgh, matgh);

            accmatgl = _mm_bslli_si128(accmatgl, 1);
            accmatgl = _mm_or_si128(accmatgl, matgl);

            if (i % 16 == 0) {
                complete_eval(regf0, regf1, regf2, regf3, resf, accmatgl, accmatgh);
                complete_eval(regg0, regg1, regg2, regg3, resg, accmatfl, accmatfh);

                int funceq = _mm_movemask_epi8(_mm_cmpeq_epi8(resg, resf));

                if (funceq != 0xFFFF) {
                    return false;
                }

                accmatfh = _mm_setzero_si128();
                accmatfl = _mm_setzero_si128();
                accmatgh = _mm_setzero_si128();
                accmatgl = _mm_setzero_si128();
            }
        }

        return true;
    }
};
#endif
