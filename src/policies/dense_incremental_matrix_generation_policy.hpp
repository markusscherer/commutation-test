#ifndef DENSE_INCREMENTAL_MATRIX_GENERATION_POLICY_HPP_MSDA
#define DENSE_INCREMENTAL_MATRIX_GENERATION_POLICY_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

template <uint64_t D, uint64_t A1, uint64_t A2>
struct dense_incremental_matrix_generation_policy {};

template <> struct dense_incremental_matrix_generation_policy<4, 1, 2> {
    static const uint64_t matrices_per_step = 8;
    struct constants {
        const __m128i const0010;
        const __m128i const01_40;

        constants()
            : const0010(
                  _mm_set_epi8(0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8)),
              const01_40(_mm_set_epi8(0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1,
                                      -4, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {
        matl = _mm_add_epi8(matl, c.const0010);
    }
};

template <> struct dense_incremental_matrix_generation_policy<4, 2, 2> {
    static const uint64_t matrices_per_step = 8;
    struct constants {
        const __m128i const0004;
        const __m128i const0010;

        constants()
            : const0004(
                  _mm_set_epi8(0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8)),
              const0010(_mm_set_epi8(1, -16, 1, -16, 1, -16, 1, -16, 1, -16, 1, -16,
                                     1, -16, 1, -16)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {
        matl = _mm_add_epi8(matl, c.const0004);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const0010);
        }
    }
};
#endif
