#include <cstdint>
#include <type_traits>

#include "code_generators.hpp"

#include "primitive_solving_policy.hpp"
#include "simd_solving_policy.hpp"
#include "simple_unary_solving_policy.hpp"
#include "policies/incremental_matrix_generation_policy.hpp"
#include "policies/blowup_matrix_generation_policy.hpp"
#include "policies/selective_accumulating_result_handling_policy.hpp"
#include "policies/most_significant_selective_evaluation_policy.hpp"
#include "policies/brute_force_evaluation_policy.hpp"
#include "policies/incremental_transposed_matrix_generation_policy.hpp"
#include "policies/accumulating_result_handling_policy.hpp"
#include "policies/primitive_result_handling_policy.hpp"
#include "policies/blowup_transposed_matrix_generation_policy.hpp"
#include "policies/dense_incremental_transposed_matrix_generation_policy.hpp"
#include "policies/dense_incremental_matrix_generation_policy.hpp"

template <uint64_t D, uint64_t A1, uint64_t A2> struct solver {
    typedef uint8_t ElementType;

    typedef simd_solving_policy<
    D, A1, A2, array_function, ElementType,
    incremental_matrix_generation_policy<D, A1, A2>,
    incremental_transposed_matrix_generation_policy<D, A1, A2>,
    brute_force_evaluation_policy<D, A1, ElementType>,
    brute_force_evaluation_policy<D, A2, ElementType>,
    primitive_result_handling_policy<D, A1, A2, ElementType, 4>,
    4> DefaultImplementation;

    typedef simd_solving_policy<
    D, A1, A2, array_function, ElementType,
    incremental_matrix_generation_policy<D, A1, A2>,
    incremental_transposed_matrix_generation_policy<D, A1, A2>,
    most_significant_selective_evaluation_policy<D, A1, ElementType>,
    most_significant_selective_transposed_evaluation_policy<D, A2,
    ElementType>,
    selective_accumulating_result_handling_policy<D, A1, A2>,
    4> BigImplementation;

    static_assert(A1 <= A2, "Please always instantiate with A1 <= A2!");

    typedef typename variadic_conditional<
    vc_tuple<D == 4 && A1 == 1 && A2 == 1,
             simple_unary_solving_policy<4, ElementType>>,
             vc_tuple<D == 4 && A1 == 4 && A2 == 4, BigImplementation>,
             vc_tuple<D == 4, DefaultImplementation>>::type Implementation;

    static inline bool commutes(const array_function<D, A1, ElementType>& f1,
                                const array_function<D, A2, ElementType>& f2) {
        return Implementation::commutes(f1, f2);
    }
};

#define SOLVER_DEFINED

#include "default_config/comtest.config.hpp"
