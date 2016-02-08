#ifndef ACCUMULATING_RESULT_HANDLING_POLICY_HPP_MSDA
#define ACCUMULATING_RESULT_HANDLING_POLICY_HPP_MSDA

#include <cstdint>
#include <type_traits>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

#include "../code_generators.hpp"

#include "brute_force_evaluation_policy.hpp"

namespace impl {

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
    matl = _mm_sllv_epi32(math, const2020);
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

template <uint64_t D, uint64_t A1, uint64_t A2, class ElementType>
struct accumulating_result_handling_policy_two_low_matrices {
    struct registers {
        __m128i accmatfl = _mm_setzero_si128();
        __m128i accmatgl = _mm_setzero_si128();
    };

    struct constants {
        const __m128i shuf128;
        const __m128i const2020;
        const __m128i const3333;
        const __m128i constFFFF;

        constants()
            : shuf128(_mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8,
                                   4, 0)),
              const2020(_mm_set1_epi64x(0x0000000200000000)),
              const3333(_mm_set1_epi32(0x03)), constFFFF(_mm_set1_epi32(0x0F)) {
        }
    };

    template <class R1, class R2, class C1, class C2>
    inline static bool handle_results(__m128i resf, __m128i resg,
                                      uint64_t matcount, R1& ep1_reg, R2& ep2_reg,
                                      C1& ep1_constants, C2& ep2_constants,
                                      registers& r, const constants& c) {
        __m128i matfl = _mm_setzero_si128();
        result_to_matrix(resf, matfl, c.shuf128, c.const2020,
                         A2 == 1 ? c.const3333 : c.constFFFF);

        r.accmatfl = _mm_bslli_si128(r.accmatfl, 1);
        r.accmatfl = _mm_or_si128(r.accmatfl, matfl);

        __m128i matgl = _mm_setzero_si128();
        result_to_matrix(resg, matgl, c.shuf128, c.const2020,
                         A1 == 1 ? c.const3333 : c.constFFFF);

        r.accmatgl = _mm_bslli_si128(r.accmatgl, 1);
        r.accmatgl = _mm_or_si128(r.accmatgl, matgl);

        if (matcount % 16 == 12) {
            brute_force_evaluation_policy<D, A1, ElementType>::eval(
                resf, matcount, r.accmatgl, ep1_reg, ep1_constants);
            brute_force_evaluation_policy<D, A2, ElementType>::eval(
                resg, matcount, r.accmatfl, ep2_reg, ep2_constants);

            int funceq = _mm_movemask_epi8(_mm_cmpeq_epi8(resg, resf));

            if (funceq != 0xFFFF) {
                return false;
            }

            r.accmatfl = _mm_setzero_si128();
            r.accmatgl = _mm_setzero_si128();
        }

        return true;
    }
};

template <uint64_t D, uint64_t A1, uint64_t A2, class ElementType>
struct accumulating_result_handling_policy_mixed_matrices {
    struct registers {
        __m128i accmatfh = _mm_setzero_si128();
        __m128i accmatfl = _mm_setzero_si128();
        __m128i accmatgl = _mm_setzero_si128();
    };

    struct constants {
        const __m128i shuf128;
        const __m128i const2020;
        const __m128i const3333;
        const __m128i constFFFF;

        constants()
            : shuf128(_mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8,
                                   4, 0)),
              const2020(_mm_set1_epi64x(0x0000000200000000)),
              const3333(_mm_set1_epi32(0x03)), constFFFF(_mm_set1_epi32(0x0F)) {
        }
    };

    template <class R1, class R2, class C1, class C2>
    inline static bool handle_results(__m128i resf, __m128i resg,
                                      uint64_t matcount, R1& ep1_reg, R2& ep2_reg,
                                      C1& ep1_constants, C2& ep2_constants,
                                      registers& r, const constants& c) {
        __m128i matfh = _mm_setzero_si128();
        __m128i matfl = _mm_setzero_si128();
        result_to_matrix(resf, matfl, matfh, c.shuf128, c.const2020, c.constFFFF,
                         A2 == 3 ? c.const3333 : c.constFFFF);

        r.accmatfh = _mm_bslli_si128(r.accmatfh, 1);
        r.accmatfh = _mm_or_si128(r.accmatfh, matfh);

        r.accmatfl = _mm_bslli_si128(r.accmatfl, 1);
        r.accmatfl = _mm_or_si128(r.accmatfl, matfl);
        __m128i matgl = _mm_setzero_si128();
        result_to_matrix(resg, matgl, c.shuf128, c.const2020,
                         A1 == 1 ? c.const3333 : c.constFFFF);

        r.accmatgl = _mm_bslli_si128(r.accmatgl, 1);
        r.accmatgl = _mm_or_si128(r.accmatgl, matgl);

        if (matcount % 16 == 12) {
            brute_force_evaluation_policy<D, A1, ElementType>::eval(
                resf, matcount, r.accmatgl, ep1_reg, ep1_constants);
            brute_force_evaluation_policy<D, A2, ElementType>::eval(
                resg, matcount, r.accmatfl, r.accmatfh, ep2_reg, ep2_constants);

            int funceq = _mm_movemask_epi8(_mm_cmpeq_epi8(resg, resf));

            if (funceq != 0xFFFF) {
                return false;
            }

            r.accmatfh = _mm_setzero_si128();
            r.accmatfl = _mm_setzero_si128();
            r.accmatgl = _mm_setzero_si128();
        }

        return true;
    }
};

template <uint64_t D, uint64_t A1, uint64_t A2, class ElementType>
struct accumulating_result_handling_policy_two_high_matrices {
    struct registers {
        __m128i accmatfh = _mm_setzero_si128();
        __m128i accmatfl = _mm_setzero_si128();
        __m128i accmatgh = _mm_setzero_si128();
        __m128i accmatgl = _mm_setzero_si128();
    };

    struct constants {
        const __m128i shuf128;
        const __m128i const2020;
        const __m128i const3333;
        const __m128i constFFFF;

        constants()
            : shuf128(_mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8,
                                   4, 0)),
              const2020(_mm_set1_epi64x(0x0000000200000000)),
              const3333(_mm_set1_epi32(0x03)), constFFFF(_mm_set1_epi32(0x0F)) {
        }
    };

    template <class R1, class R2, class C1, class C2>
    inline static bool handle_results(__m128i resf, __m128i resg,
                                      uint64_t matcount, R1& ep1_reg, R2& ep2_reg,
                                      C1& ep1_constants, C2& ep2_constants,
                                      registers& r, const constants& c) {
        __m128i matfh = _mm_setzero_si128();
        __m128i matfl = _mm_setzero_si128();
        result_to_matrix(resf, matfl, matfh, c.shuf128, c.const2020, c.constFFFF,
                         A2 == 3 ? c.const3333 : c.constFFFF);

        r.accmatfh = _mm_bslli_si128(r.accmatfh, 1);
        r.accmatfh = _mm_or_si128(r.accmatfh, matfh);

        r.accmatfl = _mm_bslli_si128(r.accmatfl, 1);
        r.accmatfl = _mm_or_si128(r.accmatfl, matfl);
        __m128i matgh = _mm_setzero_si128();
        __m128i matgl = _mm_setzero_si128();
        result_to_matrix(resg, matgl, matgh, c.shuf128, c.const2020, c.constFFFF,
                         A1 == 3 ? c.const3333 : c.constFFFF);
        r.accmatgh = _mm_bslli_si128(r.accmatgh, 1);
        r.accmatgh = _mm_or_si128(r.accmatgh, matgh);

        r.accmatgl = _mm_bslli_si128(r.accmatgl, 1);
        r.accmatgl = _mm_or_si128(r.accmatgl, matgl);

        if (matcount % 16 == 12) {
            brute_force_evaluation_policy<D, A1, ElementType>::eval(
                resf, matcount, r.accmatgl, r.accmatgh, ep1_reg, ep1_constants);
            brute_force_evaluation_policy<D, A2, ElementType>::eval(
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
};
}

template <uint64_t D, uint64_t A1, uint64_t A2, class ElementType>
struct accumulating_result_handling_policy {
private:
    typedef impl::accumulating_result_handling_policy_two_low_matrices<
    D, A1, A2, ElementType> TwoLowImplementation;

    typedef impl::accumulating_result_handling_policy_mixed_matrices<
    D, A1, A2, ElementType> MixedImplementation;

    typedef impl::accumulating_result_handling_policy_two_high_matrices<
    D, A1, A2, ElementType> TwoHighImplementation;

public:
    typedef typename variadic_conditional<
    vc_tuple<D == 4 && A1 >= 3 && A2 >= 3, TwoHighImplementation>,
             vc_tuple<D == 4 && A1 <= 2 && A2 <= 2, TwoLowImplementation>,
             vc_tuple<D == 4, MixedImplementation>>::type Implementation;

    typedef typename Implementation::registers registers;
    typedef typename Implementation::constants constants;

    template <class R1, class R2, class C1, class C2>
    inline static bool handle_results(__m128i resf, __m128i resg,
                                      uint64_t matcount, R1& ep1_reg, R2& ep2_reg,
                                      C1& ep1_constants, C2& ep2_constants,
                                      registers& r, const constants& c) {
        return Implementation::handle_results(resf, resg, matcount, ep1_reg,
                                              ep2_reg, ep1_constants, ep2_constants,
                                              r, c);
    }
};

#endif
