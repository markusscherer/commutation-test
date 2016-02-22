#ifndef BLOWUP_TRANSPOSED_MATRIX_GENERATION_POLICY_HPP_MSDA
#define BLOWUP_TRANSPOSED_MATRIX_GENERATION_POLICY_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

template <uint64_t D, uint64_t A1, uint64_t A2>
struct blowup_transposed_matrix_generation_policy {};

template <> struct blowup_transposed_matrix_generation_policy<4, 4, 4> {
    static const uint64_t matrices_per_step = 4;
    struct constants {
        const __m128i shuf128;
        const __m128i shift128;
        const __m128i const0123;

        constants()
            : shuf128(_mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8,
                                   4, 0)),
              shift128(_mm_set_epi32(6, 4, 2, 0)),
              const0123(_mm_set_epi32(3, 2, 1, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl, __m128i& math) {
        matl = _mm_setzero_si128();
        math = _mm_setzero_si128();
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math, const constants& c) {
        const uint32_t lcount =
            ((matcount & 0x00FF0000) >> 8) | (matcount & 0x000000FF);
        blowup_to_transposed_matrix(matl, lcount, c.shuf128, c.shift128);
        matl = _mm_add_epi32(matl, c.const0123);

        if (matcount % 256 == 0) {
            const uint32_t hcount =
                ((matcount & 0xFF000000) | ((matcount & 0x0000FF00) << 8)) >> 16;
            blowup_to_transposed_matrix(math, hcount, c.shuf128, c.shift128);
        }
    }

private:
    inline static void blowup_to_transposed_matrix(__m128i& mat,
                                                   const uint16_t matcount,
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
};

#endif
