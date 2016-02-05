#ifndef BRUTE_FORCE_EVALUATION_POLICY_HPP_MSDA
#define BRUTE_FORCE_EVALUATION_POLICY_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

#include "../simd_tools.hpp"
#include "../simd_evaluation_components.hpp"
#include "../array_function.hpp"

template <uint64_t D, uint64_t A> struct brute_force_evaluation_policy {};

template <> struct brute_force_evaluation_policy<4, 3> {
    typedef simd_evaluation_constants<4, 4> constants;
    typedef simd_evaluation_registers<4, 3> registers;

    inline static void init_registers(registers& r, array_function<4, 3> f) {
        array_to_si128(f.storage, r.f0);
    }

    inline static void eval(__m128i& res, uint64_t matcount, __m128i& matl,
                            __m128i& math, registers& r, const constants& c) {
        partial_eval<0>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<1>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<2>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<3>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
    }
};

template <> struct brute_force_evaluation_policy<4, 4> {
    typedef simd_evaluation_constants<4, 4> constants;
    typedef simd_evaluation_registers<4, 4> registers;

    inline static void init_registers(registers& r, array_function<4, 4> f) {
        array_to_si128(f.storage, r.f0, r.f1, r.f2, r.f3);
    }

    inline static void eval(__m128i& res, uint64_t matcount, __m128i& matl,
                            __m128i& math, registers& r, const constants& c) {
        partial_eval<0>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<1>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<2>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<3>(r.f0, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);

        partial_eval<4>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<5>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<6>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<7>(r.f1, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);

        partial_eval<8>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<9>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                        matl, math);
        partial_eval<10>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);
        partial_eval<11>(r.f2, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);

        partial_eval<12>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);
        partial_eval<13>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);
        partial_eval<14>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);
        partial_eval<15>(r.f3, res, c.epi8_2lsb_mask_128, c.shuf128, c.shift128,
                         matl, math);
    }
};

#endif
