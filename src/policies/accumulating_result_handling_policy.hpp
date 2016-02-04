#ifndef ACCUMULATING_RESULT_HANDLING_POLICY_HPP_MSDA
#define ACCUMULATING_RESULT_HANDLING_POLICY_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

#include "brute_force_evaluation_policy.hpp"

template <uint64_t D, uint64_t A1, uint64_t A2>
struct accumulating_result_handling_policy {};

template <> struct accumulating_result_handling_policy<4, 4, 4> {
    struct registers {
        __m128i accmatfh = _mm_setzero_si128();
        __m128i accmatfl = _mm_setzero_si128();
        __m128i accmatgh = _mm_setzero_si128();
        __m128i accmatgl = _mm_setzero_si128();
    };

    struct constants {
        const __m128i shuf128;
        const __m128i const2020;
        const __m128i constFFFF;

        constants()
            : shuf128(_mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8,
                                   4, 0)),
              const2020(_mm_set1_epi64x(0x0000000200000000)),
              constFFFF(_mm_set1_epi32(0x0F)) {
        }
    };

    template <class R1, class R2, class C1, class C2>
    inline static bool handle_results(__m128i resf, __m128i resg,
                                      uint64_t matcount, R1& ep1_reg, R2& ep2_reg,
                                      C1& ep1_constants, C2& ep2_constants,
                                      registers& r, const constants& c) {
        __m128i matfh = _mm_setzero_si128();
        __m128i matfl = _mm_setzero_si128();
        result_to_matrix(resf, matfl, matfh, c.shuf128, c.const2020, c.constFFFF);

        r.accmatfh = _mm_bslli_si128(r.accmatfh, 1);
        r.accmatfh = _mm_or_si128(r.accmatfh, matfh);

        r.accmatfl = _mm_bslli_si128(r.accmatfl, 1);
        r.accmatfl = _mm_or_si128(r.accmatfl, matfl);
        __m128i matgh = _mm_setzero_si128();
        __m128i matgl = _mm_setzero_si128();
        result_to_matrix(resg, matgl, matgh, c.shuf128, c.const2020, c.constFFFF);

        r.accmatgh = _mm_bslli_si128(r.accmatgh, 1);
        r.accmatgh = _mm_or_si128(r.accmatgh, matgh);

        r.accmatgl = _mm_bslli_si128(r.accmatgl, 1);
        r.accmatgl = _mm_or_si128(r.accmatgl, matgl);

        if (matcount % 16 == 12) {
            brute_force_evaluation_policy<4, 4>::eval(
                resf, matcount, r.accmatgl, r.accmatgh, ep1_reg, ep1_constants);
            brute_force_evaluation_policy<4, 4>::eval(
                resg, matcount, r.accmatfl, r.accmatfh, ep2_reg, ep2_constants);

            int funceq = _mm_movemask_epi8(_mm_cmpeq_epi8(resg, resf));

            if (funceq != 0xFFFF) {
                return false;
            }

            r.accmatfh = _mm_setzero_si128();
            r.accmatfl = _mm_setzero_si128();
            r.accmatgh = _mm_setzero_si128();
            r.accmatgl = _mm_setzero_si128();
        }

        return true;
    }

private:
    inline static void result_to_matrix(const __m128i& res, __m128i& matl,
                                        __m128i& math, const __m128i& shuf128,
                                        const __m128i& const2020,
                                        const __m128i& constFFFF) {
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
        matl = _mm_and_si128(math, constFFFF);
        // write fields 2,6,10 and 14 to math (at 0,4,8 and 12)
        math = _mm_srli_epi32(math, 16);
        math = _mm_and_si128(math, constFFFF);
    }
};

#endif
