#ifndef SELECTIVE_EVALUATION_POLICY_HPP_MSDA
#define SELECTIVE_EVALUATION_POLICY_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

#include "../simd_tools.hpp"
#include "../simd_evaluation_components.hpp"
#include "../array_function.hpp"

template <uint64_t P, uint64_t V> struct mustcalc {
    static const uint32_t selmask = 0x03030303 << ((P - 2) * 2);
    static const uint32_t eqmask = (V | (V << 8) | (V << 16) | (V << 24))
                                   << ((P - 2) * 2);

    static inline bool calc(uint32_t matcount) {
        __m64 a = _mm_cvtsi32_si64(matcount);
        __m64 b = _mm_cvtsi32_si64(selmask);
        a = _mm_and_si64(a, b);
        b = _mm_cvtsi32_si64(eqmask);
        b = _mm_cmpeq_pi8(a, b);
        return _mm_cvtsi64_si32(b);
    }
};

template <uint64_t P, uint64_t V> struct mustcalc_transposed {
    static const uint32_t selmask = 0x0C0C0303 << ((P - 2) * 4);

    static const uint32_t eqmask =
        (V | (V << 8) | ((V << 2) << 16) | ((V << 2) << 24)) << ((P - 2) * 4);

    static inline bool calc(uint32_t matcount) {
        __m64 a = _mm_set1_pi16(matcount);
        __m64 b = _mm_cvtsi32_si64(selmask);

        a = _mm_and_si64(a, b);
        b = _mm_cvtsi32_si64(eqmask);
        b = _mm_cmpeq_pi8(a, b);
        return _mm_cvtsi64_si32(b);
    }
};

template <uint64_t D, uint64_t A, class ElementType>
struct selective_evaluation_policy {};

template <class ElementType>
struct selective_evaluation_policy<4, 4, ElementType> {
    typedef simd_evaluation_constants<4, 4> constants;
    typedef simd_evaluation_registers<4, 4> registers;

    inline static void init_registers(registers& r,
                                      array_function<4, 4, ElementType> f) {
        array_to_si128(f.storage, r.f0, r.f1, r.f2, r.f3);
    }

    inline static void eval(__m128i& res, uint64_t matcount, __m128i& matl,
                            __m128i& math, registers& r, const constants& c) {
        matcount = (matcount & 0xFFFF0000) | (matcount >> 20);

        if (mustcalc<3, 0>::calc(matcount)) {
            if (mustcalc<2, 0>::calc(matcount)) {
                partial_eval<0>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc<2, 1>::calc(matcount)) {
                partial_eval<1>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc<2, 2>::calc(matcount)) {
                partial_eval<2>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc<2, 3>::calc(matcount)) {
                partial_eval<3>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }
        }

        if (mustcalc<3, 1>::calc(matcount)) {
            if (mustcalc<2, 0>::calc(matcount)) {
                partial_eval<4>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc<2, 1>::calc(matcount)) {
                partial_eval<5>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc<2, 2>::calc(matcount)) {
                partial_eval<6>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc<2, 3>::calc(matcount)) {
                partial_eval<7>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }
        }

        if (mustcalc<3, 2>::calc(matcount)) {
            if (mustcalc<2, 0>::calc(matcount)) {
                partial_eval<8>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc<2, 1>::calc(matcount)) {
                partial_eval<9>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc<2, 2>::calc(matcount)) {
                partial_eval<10>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }

            if (mustcalc<2, 3>::calc(matcount)) {
                partial_eval<11>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }
        }

        if (mustcalc<3, 3>::calc(matcount)) {
            if (mustcalc<2, 0>::calc(matcount)) {
                partial_eval<12>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }

            if (mustcalc<2, 1>::calc(matcount)) {
                partial_eval<13>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }

            if (mustcalc<2, 2>::calc(matcount)) {
                partial_eval<14>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }

            if (mustcalc<2, 3>::calc(matcount)) {
                partial_eval<15>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }
        }
    }
};

template <uint64_t D, uint64_t A, class ElementType>
struct selective_transposed_evaluation_policy {};

template <class ElementType>
struct selective_transposed_evaluation_policy<4, 4, ElementType> {
    typedef simd_evaluation_constants<4, 4> constants;
    typedef simd_evaluation_registers<4, 4> registers;

    inline static void init_registers(registers& r,
                                      array_function<4, 4, ElementType> f) {
        array_to_si128(f.storage, r.f0, r.f1, r.f2, r.f3);
    }

    inline static void eval(__m128i& res, uint64_t matcount, __m128i& matl,
                            __m128i& math, registers& r, const constants& c) {
        matcount = ((matcount >> 8) & 0x00FF) | ((matcount >> 16) & 0xFF00);

        if (mustcalc_transposed<3, 0>::calc(matcount)) {
            if (mustcalc_transposed<2, 0>::calc(matcount)) {
                partial_eval<0>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc_transposed<2, 1>::calc(matcount)) {
                partial_eval<1>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc_transposed<2, 2>::calc(matcount)) {
                partial_eval<2>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc_transposed<2, 3>::calc(matcount)) {
                partial_eval<3>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }
        }

        if (mustcalc_transposed<3, 1>::calc(matcount)) {
            if (mustcalc_transposed<2, 0>::calc(matcount)) {
                partial_eval<4>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc_transposed<2, 1>::calc(matcount)) {
                partial_eval<5>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc_transposed<2, 2>::calc(matcount)) {
                partial_eval<6>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc_transposed<2, 3>::calc(matcount)) {
                partial_eval<7>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }
        }

        if (mustcalc_transposed<3, 2>::calc(matcount)) {
            if (mustcalc_transposed<2, 0>::calc(matcount)) {
                partial_eval<8>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc_transposed<2, 1>::calc(matcount)) {
                partial_eval<9>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                matl, math);
            }

            if (mustcalc_transposed<2, 2>::calc(matcount)) {
                partial_eval<10>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }

            if (mustcalc_transposed<2, 3>::calc(matcount)) {
                partial_eval<11>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }
        }

        if (mustcalc_transposed<3, 3>::calc(matcount)) {
            if (mustcalc_transposed<2, 0>::calc(matcount)) {
                partial_eval<12>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }

            if (mustcalc_transposed<2, 1>::calc(matcount)) {
                partial_eval<13>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }

            if (mustcalc_transposed<2, 2>::calc(matcount)) {
                partial_eval<14>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }

            if (mustcalc_transposed<2, 3>::calc(matcount)) {
                partial_eval<15>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                                 matl, math);
            }
        }
    }
};

#endif
