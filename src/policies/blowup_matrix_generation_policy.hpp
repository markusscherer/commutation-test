#ifndef BLOWUP_MATRIX_GENERATION_POLICY_HPP_MSDA
#define BLOWUP_MATRIX_GENERATION_POLICY_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

template <uint64_t D, uint64_t A1, uint64_t A2>
struct blowup_matrix_generation_policy {};

template <> struct blowup_matrix_generation_policy<4, 4, 4> {
    struct constants {
        const __m128i shuf128;
        const __m128i shift128;
        const __m128i epi8_4lsb_mask_128;
        const __m128i const0123;

        constants()
            : shuf128(_mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8,
                                   4, 0)),
              shift128(_mm_set_epi32(6, 4, 2, 0)),
              epi8_4lsb_mask_128(_mm_set1_epi32(0x0F0F0F0F)),
              const0123(_mm_set_epi32(3, 2, 1, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl, __m128i& math) {
        matl = _mm_setzero_si128();
        math = _mm_setzero_si128();
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math, const constants& c) {
        blowup_to_matrix(matl, matcount, c.epi8_4lsb_mask_128, c.shuf128,
                         c.shift128);
        matl = _mm_add_epi32(matl, c.const0123);

        if (matcount % 65536 == 0) {
            blowup_to_matrix(math, matcount >> 16, c.epi8_4lsb_mask_128, c.shuf128,
                             c.shift128);
        }
    }

private:
    inline static void blowup_to_matrix(__m128i& mat, const uint16_t matcount,
                                        const __m128i& epi8_4lsb_mask_128,
                                        const __m128i& shuf128,
                                        const __m128i& shift128) {
        mat = _mm_set1_epi32(matcount);

        mat = _mm_srlv_epi32(mat, shift128);
        mat = _mm_srlv_epi32(mat, shift128);
        mat = _mm_and_si128(mat, epi8_4lsb_mask_128);
        mat = _mm_shuffle_epi8(mat, shuf128);
        mat = _mm_broadcastd_epi32(mat);
    }
};
#endif
