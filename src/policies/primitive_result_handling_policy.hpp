#ifndef PRIMITIVE_RESULT_HANDLING_POLICY_HPP_MSDA
#define PRIMITIVE_RESULT_HANDLING_POLICY_HPP_MSDA

#include <cstdint>
#include <type_traits>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

#include "../code_generators.hpp"
#include "../result_handling_components.hpp"

#include "brute_force_evaluation_policy.hpp"

namespace impl {

template <uint64_t D, uint64_t A1, uint64_t A2, class ElementType,
          uint64_t MatricesPerStep>
struct primitive_result_handling_policy_two_low_matrices {
    static const uint64_t matrices_per_step = MatricesPerStep;
    struct registers {};

    struct constants {
        const __m128i shuf128;
        const __m128i const2020;
        const __m128i const3333;
        const __m128i constFFFF;

        constants()
            : shuf128(_mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8,
                                   4, 0)),
              const2020(_mm_set1_epi64x(0x0000000200000000)),
              const3333(init_const3333<MatricesPerStep>()),
              constFFFF(init_constFFFF<MatricesPerStep>()) {
        }
    };

    static_assert(MatricesPerStep == 4 || MatricesPerStep == 8,
                  "MatricesPerStep must be 4 or 8.");

    template <class R1, class R2, class C1, class C2>
    inline static bool handle_results(__m128i resf, __m128i resg,
                                      uint64_t matcount, R1& ep1_reg, R2& ep2_reg,
                                      C1& ep1_constants, C2& ep2_constants,
                                      registers& r, const constants& c) {
        __m128i matfl = _mm_setzero_si128();
        __m128i matgl = _mm_setzero_si128();

        if (MatricesPerStep == 8) {
            dense_result_to_matrix(resf, matfl, A2 == 1 ? c.const3333 : c.constFFFF);
            dense_result_to_matrix(resg, matgl, A1 == 1 ? c.const3333 : c.constFFFF);
        } else {
            result_to_matrix(resf, matfl, c.shuf128, c.const2020,
                             A2 == 1 ? c.const3333 : c.constFFFF);
            result_to_matrix(resg, matgl, c.shuf128, c.const2020,
                             A1 == 1 ? c.const3333 : c.constFFFF);
        }

        brute_force_evaluation_policy<D, A1, ElementType>::eval(
            resf, matcount, matgl, ep1_reg, ep1_constants);
        brute_force_evaluation_policy<D, A2, ElementType>::eval(
            resg, matcount, matfl, ep2_reg, ep2_constants);

        uint32_t funceq = _mm_movemask_epi8(_mm_cmpeq_epi8(resg, resf));

        if ((funceq & mask) != mask) {
            return false;
        } else {
            return true;
        }
    }

private:
    static const uint32_t mask = MatricesPerStep == 8 ? 0x5555 : 0x1111;
};

template <uint64_t D, uint64_t A1, uint64_t A2, class ElementType>
struct primitive_result_handling_policy_mixed_matrices {
    static const uint64_t matrices_per_step = 4;
    struct registers {};

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

        __m128i matgl = _mm_setzero_si128();
        result_to_matrix(resg, matgl, c.shuf128, c.const2020,
                         A1 == 1 ? c.const3333 : c.constFFFF);

        brute_force_evaluation_policy<D, A1, ElementType>::eval(
            resf, matcount, matgl, ep1_reg, ep1_constants);
        brute_force_evaluation_policy<D, A2, ElementType>::eval(
            resg, matcount, matfl, matfh, ep2_reg, ep2_constants);

        int funceq = _mm_movemask_epi8(_mm_cmpeq_epi8(resg, resf));

        if ((funceq & mask) != mask) {
            return false;
        } else {
            return true;
        }
    }

private:
    static const uint32_t mask = 0x1111;
};

template <uint64_t D, uint64_t A1, uint64_t A2, class ElementType>
struct primitive_result_handling_policy_two_high_matrices {
    static const uint64_t matrices_per_step = 4;
    struct registers {};

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

        __m128i matgh = _mm_setzero_si128();
        __m128i matgl = _mm_setzero_si128();
        result_to_matrix(resg, matgl, matgh, c.shuf128, c.const2020, c.constFFFF,
                         A1 == 3 ? c.const3333 : c.constFFFF);
        brute_force_evaluation_policy<D, A1, ElementType>::eval(
            resf, matcount, matgl, matgh, ep1_reg, ep1_constants);
        brute_force_evaluation_policy<D, A2, ElementType>::eval(
            resg, matcount, matfl, matfh, ep2_reg, ep2_constants);

        int funceq = _mm_movemask_epi8(_mm_cmpeq_epi8(resg, resf));

        if ((funceq & mask) != mask) {
            return false;
        } else {
            return true;
        }
    }

private:
    static const uint32_t mask = 0x1111;
};
}

template <uint64_t D, uint64_t A1, uint64_t A2, class ElementType,
          uint64_t MatricesPerStep>
struct primitive_result_handling_policy {
private:
    typedef impl::primitive_result_handling_policy_two_low_matrices<
    D, A1, A2, ElementType, MatricesPerStep> TwoLowImplementation;

    typedef impl::primitive_result_handling_policy_mixed_matrices<
    D, A1, A2, ElementType> MixedImplementation;

    typedef impl::primitive_result_handling_policy_two_high_matrices<
    D, A1, A2, ElementType> TwoHighImplementation;

public:
    typedef typename variadic_conditional<
    vc_tuple<D == 4 && A1 >= 3 && A2 >= 3, TwoHighImplementation>,
             vc_tuple<D == 4 && A1 <= 2 && A2 <= 2, TwoLowImplementation>,
             vc_tuple<D == 4, MixedImplementation>>::type Implementation;

    typedef typename Implementation::registers registers;
    typedef typename Implementation::constants constants;

    const static uint64_t matrices_per_step = Implementation::matrices_per_step;

    static_assert(matrices_per_step == MatricesPerStep,
                  "MatricesPerStep must match with Implementation.");

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
