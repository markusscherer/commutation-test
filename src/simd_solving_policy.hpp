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

//#include "policies/blowup_matrix_generation_policy.hpp"
//#include "policies/incremental_matrix_generation_policy.hpp"
//#include "policies/blowup_transposed_matrix_generation_policy.hpp"
//#include "policies/incremental_transposed_matrix_generation_policy.hpp"
//#include "policies/brute_force_evaluation_policy.hpp"
//#include "policies/selective_evaluation_policy.hpp"
//#include "policies/accumulating_result_handling_policy.hpp"
//#include "policies/selective_accumulating_result_handling_policy.hpp"

namespace impl {

template <uint64_t D, uint64_t A1, uint64_t A2,
          template <uint64_t, uint64_t, typename> class FunctionType,
          class ElementType, class MatrixGenerationPolicy,
          class TransposedMatrixGenerationPolicy, class EvaluationPolicy1,
          class EvaluationPolicy2, class ResultsHandlingPolicy,
          uint64_t MatricesPerStep>
struct simd_solving_policy_two_low_matrices {
    typedef ElementType element_type;
    static const uint64_t domain_size = D;
    static const uint64_t cell_count = A1 * A2;
    static const uint64_t matrix_count = cpow(domain_size, cell_count);
    static const uint64_t matrices_per_step = MatricesPerStep;

    inline static bool commutes(const FunctionType<D, A1, ElementType>& f,
                                const FunctionType<D, A2, ElementType>& g) {

        const typename MatrixGenerationPolicy::constants mgp_constants;
        const typename TransposedMatrixGenerationPolicy::constants tmgp_constants;
        const typename EvaluationPolicy1::constants ep1_constants;
        const typename EvaluationPolicy2::constants ep2_constants;
        const typename ResultsHandlingPolicy::constants rhp_constants;

        typename EvaluationPolicy1::registers ep1_reg;
        typename EvaluationPolicy2::registers ep2_reg;
        typename ResultsHandlingPolicy::registers rhp_reg;

        EvaluationPolicy1::init_registers(ep1_reg, f, ep1_constants);
        EvaluationPolicy2::init_registers(ep2_reg, g, ep2_constants);

        __m128i resf;
        __m128i resg;

        __m128i matl;

        __m128i tmatl;

        MatrixGenerationPolicy::init_matrix(matl);
        TransposedMatrixGenerationPolicy::init_matrix(tmatl);

        for (uint64_t i = 0; i < matrix_count; i += matrices_per_step) {
            MatrixGenerationPolicy::next_matrix(i, matl, mgp_constants);
            TransposedMatrixGenerationPolicy::next_matrix(i, tmatl, tmgp_constants);

            EvaluationPolicy1::eval(resf, i, matl, ep1_reg, ep1_constants);
            EvaluationPolicy2::eval(resg, i, tmatl, ep2_reg, ep2_constants);

            if (!ResultsHandlingPolicy::handle_results(
                        resf, resg, i, ep1_reg, ep2_reg, ep1_constants, ep2_constants,
                        rhp_reg, rhp_constants)) {
                return false;
            }
        }

        return true;
    }
};

template <uint64_t D, uint64_t A1, uint64_t A2,
          template <uint64_t, uint64_t, typename> class FunctionType,
          class ElementType, class MatrixGenerationPolicy,
          class TransposedMatrixGenerationPolicy, class EvaluationPolicy1,
          class EvaluationPolicy2, class ResultsHandlingPolicy,
          uint64_t MatricesPerStep>
struct simd_solving_policy_mixed_matrices {
    typedef ElementType element_type;
    static const uint64_t domain_size = D;
    static const uint64_t cell_count = A1 * A2;
    static const uint64_t matrix_count = cpow(domain_size, cell_count);
    static const uint64_t matrices_per_step = MatricesPerStep;

    inline static bool commutes(const FunctionType<D, A1, ElementType>& f,
                                const FunctionType<D, A2, ElementType>& g) {

        const typename MatrixGenerationPolicy::constants mgp_constants;
        const typename TransposedMatrixGenerationPolicy::constants tmgp_constants;
        const typename EvaluationPolicy1::constants ep1_constants;
        const typename EvaluationPolicy2::constants ep2_constants;
        const typename ResultsHandlingPolicy::constants rhp_constants;

        typename EvaluationPolicy1::registers ep1_reg;
        typename EvaluationPolicy2::registers ep2_reg;
        typename ResultsHandlingPolicy::registers rhp_reg;

        EvaluationPolicy1::init_registers(ep1_reg, f, ep1_constants);
        EvaluationPolicy2::init_registers(ep2_reg, g);

        __m128i resf;
        __m128i resg;

        __m128i matl;

        __m128i tmath;
        __m128i tmatl;

        MatrixGenerationPolicy::init_matrix(matl);
        TransposedMatrixGenerationPolicy::init_matrix(tmatl, tmath);

        for (uint64_t i = 0; i < matrix_count; i += matrices_per_step) {
            MatrixGenerationPolicy::next_matrix(i, matl, mgp_constants);
            TransposedMatrixGenerationPolicy::next_matrix(i, tmatl, tmath,
                                                          tmgp_constants);

            EvaluationPolicy1::eval(resf, i, matl, ep1_reg, ep1_constants);
            EvaluationPolicy2::eval(resg, i, tmatl, tmath, ep2_reg, ep2_constants);

            if (!ResultsHandlingPolicy::handle_results(
                        resf, resg, i, ep1_reg, ep2_reg, ep1_constants, ep2_constants,
                        rhp_reg, rhp_constants)) {
                return false;
            }
        }

        return true;
    }
};

template <uint64_t D, uint64_t A1, uint64_t A2,
          template <uint64_t, uint64_t, typename> class FunctionType,
          class ElementType, class MatrixGenerationPolicy,
          class TransposedMatrixGenerationPolicy, class EvaluationPolicy1,
          class EvaluationPolicy2, class ResultsHandlingPolicy,
          uint64_t MatricesPerStep>
struct simd_solving_policy_two_high_matrices {
    typedef ElementType element_type;
    static const uint64_t domain_size = D;
    static const uint64_t cell_count = A1 * A2;
    static const uint64_t matrix_count = cpow(domain_size, cell_count);
    static const uint64_t matrices_per_step = MatricesPerStep;

    inline static bool commutes(const FunctionType<D, A1, ElementType>& f,
                                const FunctionType<D, A2, ElementType>& g) {

        const typename MatrixGenerationPolicy::constants mgp_constants;
        const typename TransposedMatrixGenerationPolicy::constants tmgp_constants;
        const typename EvaluationPolicy1::constants ep1_constants;
        const typename EvaluationPolicy2::constants ep2_constants;
        const typename ResultsHandlingPolicy::constants rhp_constants;

        typename EvaluationPolicy1::registers ep1_reg;
        typename EvaluationPolicy2::registers ep2_reg;
        typename ResultsHandlingPolicy::registers rhp_reg;

        EvaluationPolicy1::init_registers(ep1_reg, f);
        EvaluationPolicy2::init_registers(ep2_reg, g);

        __m128i resf;
        __m128i resg;

        __m128i math;
        __m128i matl;

        __m128i tmath;
        __m128i tmatl;

        MatrixGenerationPolicy::init_matrix(matl, math);
        TransposedMatrixGenerationPolicy::init_matrix(tmatl, tmath);

        for (uint64_t i = 0; i < matrix_count; i += matrices_per_step) {
            MatrixGenerationPolicy::next_matrix(i, matl, math, mgp_constants);
            TransposedMatrixGenerationPolicy::next_matrix(i, tmatl, tmath,
                                                          tmgp_constants);

            EvaluationPolicy1::eval(resf, i, matl, math, ep1_reg, ep1_constants);
            EvaluationPolicy2::eval(resg, i, tmatl, tmath, ep2_reg, ep2_constants);

            if (!ResultsHandlingPolicy::handle_results(
                        resf, resg, i, ep1_reg, ep2_reg, ep1_constants, ep2_constants,
                        rhp_reg, rhp_constants)) {
                return false;
            }
        }

        return true;
    }
};
}

template <uint64_t D, uint64_t A1, uint64_t A2,
          template <uint64_t, uint64_t, typename> class FunctionType,
          class ElementType, class MatrixGenerationPolicy,
          class TransposedMatrixGenerationPolicy, class EvaluationPolicy1,
          class EvaluationPolicy2, class ResultsHandlingPolicy,
          uint64_t MatricesPerStep>
struct simd_solving_policy {
private:
    typedef impl::simd_solving_policy_two_low_matrices<
    D, A1, A2, FunctionType, ElementType, MatrixGenerationPolicy,
    TransposedMatrixGenerationPolicy, EvaluationPolicy1, EvaluationPolicy2,
    ResultsHandlingPolicy, MatricesPerStep> TwoLowImplementation;

    typedef impl::simd_solving_policy_mixed_matrices<
    D, A1, A2, FunctionType, ElementType, MatrixGenerationPolicy,
    TransposedMatrixGenerationPolicy, EvaluationPolicy1, EvaluationPolicy2,
    ResultsHandlingPolicy, MatricesPerStep> MixedImplementation;

    typedef impl::simd_solving_policy_two_high_matrices<
    D, A1, A2, FunctionType, ElementType, MatrixGenerationPolicy,
    TransposedMatrixGenerationPolicy, EvaluationPolicy1, EvaluationPolicy2,
    ResultsHandlingPolicy, MatricesPerStep> TwoHighImplementation;

    static_assert(MatrixGenerationPolicy::matrices_per_step ==
                  TransposedMatrixGenerationPolicy::matrices_per_step,
                  "MatricesPerStep of matrix generation policies must match.");
    static_assert(MatrixGenerationPolicy::matrices_per_step ==
                  ResultsHandlingPolicy::matrices_per_step,
                  "MatricesPerStep of matrix generation policies and result "
                  "handling policy must match.");
    static_assert(
        MatricesPerStep == ResultsHandlingPolicy::matrices_per_step,
        "MatricesPerStep of solver and result handling policy must match.");

public:
    typedef ElementType element_type;
    static const uint64_t domain_size = D;
    static const uint64_t cell_count = A1 * A2;
    static const uint64_t matrix_count = cpow(domain_size, cell_count);

    typedef typename variadic_conditional<
    vc_tuple<D == 4 && A1 >= 3 && A2 >= 3, TwoHighImplementation>,
             vc_tuple<D == 4 && A1 <= 2 && A2 <= 2, TwoLowImplementation>,
             vc_tuple<D == 4, MixedImplementation>>::type Implementation;

    static bool commutes(const FunctionType<D, A1, ElementType>& f1,
                         const FunctionType<D, A2, ElementType>& f2) {
        return Implementation::commutes(f1, f2);
    };
};
#endif
