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
#include "simd_tools.hpp"

#include "policies/blowup_matrix_generation_policy.hpp"
#include "policies/incremental_matrix_generation_policy.hpp"
#include "policies/blowup_transposed_matrix_generation_policy.hpp"
#include "policies/incremental_transposed_matrix_generation_policy.hpp"
#include "policies/brute_force_evaluation_policy.hpp"
#include "policies/selective_evaluation_policy.hpp"
#include "policies/accumulating_result_handling_policy.hpp"
#include "policies/selective_accumulating_result_handling_policy.hpp"

template <uint64_t D, uint64_t A1, uint64_t A2,
          template <uint64_t, uint64_t, typename> class FunctionType,
          class ElementType, class MatrixGenerationPolicy,
          class TransposedMatrixGenerationPolicy, class EvaluationPolicy1,
          class EvaluationPolicy2, class ResultsHandlingPolicy>
struct simd_solving_policy {
    typedef ElementType element_type;
    static const uint64_t domain_size = D;
    static const uint64_t cell_count = A1 * A2;
    static const uint64_t matrix_count = cpow(domain_size, cell_count);

    static bool commutes(FunctionType<D, A1, ElementType> f1,
                         FunctionType<D, A2, ElementType> f2);
};

template <template <uint64_t, uint64_t, typename> class FunctionType,
          class ElementType, class MatrixGenerationPolicy,
          class TransposedMatrixGenerationPolicy, class EvaluationPolicy1,
          class EvaluationPolicy2, class ResultsHandlingPolicy>
struct simd_solving_policy<4, 3, 3, FunctionType, ElementType,
           MatrixGenerationPolicy,
           TransposedMatrixGenerationPolicy, EvaluationPolicy1,
           EvaluationPolicy2, ResultsHandlingPolicy> {
    typedef ElementType element_type;
    static const uint64_t domain_size = 4;
    static const uint64_t cell_count = 3 * 3;
    static const uint64_t matrix_count = cpow(domain_size, cell_count);

    static bool commutes(FunctionType<4, 3, ElementType> f,
                         FunctionType<4, 3, ElementType> g) {

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

        for (uint64_t i = 0; i < matrix_count; i += 4) {
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

template <template <uint64_t, uint64_t, typename> class FunctionType,
          class ElementType, class MatrixGenerationPolicy,
          class TransposedMatrixGenerationPolicy, class EvaluationPolicy1,
          class EvaluationPolicy2, class ResultsHandlingPolicy>
struct simd_solving_policy<4, 3, 4, FunctionType, ElementType,
           MatrixGenerationPolicy,
           TransposedMatrixGenerationPolicy, EvaluationPolicy1,
           EvaluationPolicy2, ResultsHandlingPolicy> {
    typedef ElementType element_type;
    static const uint64_t domain_size = 4;
    static const uint64_t cell_count = 3 * 4;
    static const uint64_t matrix_count = cpow(domain_size, cell_count);

    static bool commutes(FunctionType<4, 3, ElementType> f,
                         FunctionType<4, 4, ElementType> g) {

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

        for (uint64_t i = 0; i < matrix_count; i += 4) {
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

template <template <uint64_t, uint64_t, typename> class FunctionType,
          class ElementType, class MatrixGenerationPolicy,
          class TransposedMatrixGenerationPolicy, class EvaluationPolicy1,
          class EvaluationPolicy2, class ResultsHandlingPolicy>
struct simd_solving_policy<4, 4, 4, FunctionType, ElementType,
           MatrixGenerationPolicy,
           TransposedMatrixGenerationPolicy, EvaluationPolicy1,
           EvaluationPolicy2, ResultsHandlingPolicy> {
    typedef ElementType element_type;
    static const uint64_t domain_size = 4;
    static const uint64_t cell_count = 4 * 4;
    static const uint64_t matrix_count = cpow(domain_size, cell_count);

    static bool commutes(FunctionType<4, 4, ElementType> f,
                         FunctionType<4, 4, ElementType> g) {

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

        for (uint64_t i = 0; i < matrix_count; i += 4) {
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
#endif
