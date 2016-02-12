#ifndef SELECTIVE_ACCUMULATING_RESULT_HANDLING_POLICY_HPP_MSDA
#define SELECTIVE_ACCUMULATING_RESULT_HANDLING_POLICY_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

#include "brute_force_evaluation_policy.hpp"
#include "../simd_evaluation_components.hpp"

template <uint64_t D, uint64_t A1, uint64_t A2>
struct selective_accumulating_result_handling_policy {};

template <> struct selective_accumulating_result_handling_policy<4, 4, 4> {
    const static uint64_t matrices_per_step = 4;
    struct registers {
        __m128i accmatfh = _mm_setzero_si128();
        __m128i accmatfl = _mm_setzero_si128();
        __m128i accmatgh = _mm_setzero_si128();
        __m128i accmatgl = _mm_setzero_si128();
        __m128i curf = _mm_setzero_si128();
        __m128i curg = _mm_setzero_si128();
        uint16_t selector1 = 0;
        uint16_t selector2 = 0;
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
        shift_lower_matrix(resf, r.accmatfl, c.constFFFF);
        shift_lower_matrix(resg, r.accmatgl, c.constFFFF);

        if (matcount % 256 < 16) {
            if (matcount % 256 == 0) {
                new_upper_matrix(r.accmatfh, r.curg, resf, r.selector2, ep2_reg);
            }

            shift_upper_matrix(resf, r.accmatfh, c.constFFFF);

            if (matcount % 65536 < 16) {
                if (matcount % 65536 == 0) {
                    new_upper_matrix(r.accmatgh, r.curf, resg, r.selector1, ep1_reg);
                }

                shift_upper_matrix(resg, r.accmatgh, c.constFFFF);
            }
        }

        if (matcount % 16 == 12) {
            eval(r.curf, resf, ep1_constants.epi8_2lsb_mask_128,
                 ep1_constants.shuf128, ep1_constants.shift128, r.accmatgl,
                 r.accmatgh, r.selector1);

            eval(r.curg, resg, ep2_constants.epi8_2lsb_mask_128,
                 ep2_constants.shuf128, ep2_constants.shift128, r.accmatfl,
                 r.accmatfh, r.selector2);

            int funceq = _mm_movemask_epi8(_mm_cmpeq_epi8(resg, resf));

            if (funceq != 0xFFFF) {
                return false;
            }

            r.accmatfl = _mm_setzero_si128();
            r.accmatgl = _mm_setzero_si128();
        }

        return true;
    }

private:
    template <typename R>
    inline static void new_upper_matrix(__m128i& accmath, __m128i& cur,
                                        const __m128i& res, uint16_t& selector,
                                        const R& ep_reg) {
        accmath = _mm_setzero_si128();
        selector = _mm_extract_epi16(res, 7);
        selector = (selector >> 6) | (selector & 0x03);

        switch (selector >> 2) {
            case 0:
                cur = ep_reg.f0;
                break;

            case 1:
                cur = ep_reg.f1;
                break;

            case 2:
                cur = ep_reg.f2;
                break;

            case 3:
                cur = ep_reg.f3;
                break;

            default:
                // This happens, if the result of a computation is not in {0,1,2,3}, which
                // should be impossible
                exit(13);
        }
    }

    inline static void shift_upper_matrix(const __m128i& res, __m128i& accmath,
                                          const __m128i& constFFFF) {
        __m128i math = _mm_setzero_si128();
        result_to_upper_matrix(res, math, constFFFF);
        accmath = _mm_bslli_si128(accmath, 1);
        accmath = _mm_or_si128(accmath, math);
    }

    inline static void shift_lower_matrix(const __m128i& res, __m128i& accmatl,
                                          const __m128i& constFFFF) {
        __m128i matl = _mm_setzero_si128();
        result_to_lower_matrix(res, matl, constFFFF);
        accmatl = _mm_bslli_si128(accmatl, 1);
        accmatl = _mm_or_si128(accmatl, matl);
    }

    inline static void result_to_lower_matrix(const __m128i& res, __m128i& matl,
                                              const __m128i& constFFFF) {
        matl = _mm_srli_epi32(res, 6);
        matl = _mm_or_si128(matl, res);
        matl = _mm_and_si128(matl, constFFFF);
    }

    inline static void result_to_upper_matrix(const __m128i& res, __m128i& math,
                                              const __m128i& constFFFF) {
        math = _mm_srli_epi32(res, 16);
        __m128i tmp = _mm_srli_epi32(res, 22);
        math = _mm_or_si128(math, tmp);
        math = _mm_and_si128(math, constFFFF);
    }

    inline static void eval(const __m128i& function, __m128i& res,
                            const __m128i& epi8_2lsb_mask, const __m128i& shuf128,
                            const __m128i& shift128, const __m128i& matl,
                            const __m128i& math, uint16_t selector) {
        switch (selector & 0x03) {
            case 0:
                partial_eval_with_selector<0>(function, res, epi8_2lsb_mask, shuf128,
                                              shift128, matl, math, selector);
                break;

            case 1:
                partial_eval_with_selector<1>(function, res, epi8_2lsb_mask, shuf128,
                                              shift128, matl, math, selector);
                break;

            case 2:
                partial_eval_with_selector<2>(function, res, epi8_2lsb_mask, shuf128,
                                              shift128, matl, math, selector);
                break;

            case 3:
                partial_eval_with_selector<3>(function, res, epi8_2lsb_mask, shuf128,
                                              shift128, matl, math, selector);
                break;
        }
    }
};

#endif
