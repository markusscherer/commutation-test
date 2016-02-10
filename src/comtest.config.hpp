#include <cstdint>
#include <type_traits>

#include "code_generators.hpp"

#include "primitive_solving_policy.hpp"
#include "simd_solving_policy.hpp"
#include "simple_unary_solving_policy.hpp"
#include "policies/incremental_matrix_generation_policy.hpp"
#include "policies/blowup_matrix_generation_policy.hpp"
#include "policies/selective_accumulating_result_handling_policy.hpp"
#include "policies/selective_evaluation_policy.hpp"
#include "policies/brute_force_evaluation_policy.hpp"
#include "policies/incremental_transposed_matrix_generation_policy.hpp"
#include "policies/accumulating_result_handling_policy.hpp"
#include "policies/blowup_transposed_matrix_generation_policy.hpp"

template <uint64_t D, uint64_t A1, uint64_t A2> struct solver {
    typedef uint8_t ElementType;

    typedef simd_solving_policy<
    D, A1, A2, array_function, ElementType,
    incremental_matrix_generation_policy<D, A1, A2>,
    incremental_transposed_matrix_generation_policy<D, A1, A2>,
    brute_force_evaluation_policy<D, A1, ElementType>,
    brute_force_evaluation_policy<D, A2, ElementType>,
    accumulating_result_handling_policy<D, A1, A2, ElementType>>
    DefaultImplementation;

#ifdef __clang__
    typedef simd_solving_policy<
    D, A1, A2, array_function, ElementType,
    incremental_matrix_generation_policy<D, A1, A2>,
    incremental_transposed_matrix_generation_policy<D, A1, A2>,
    brute_force_evaluation_policy<D, A1, ElementType>,
    brute_force_evaluation_policy<D, A2, ElementType>,
    accumulating_result_handling_policy<D, A1, A2, ElementType>>
    BigArityImplementation;
#else
    typedef simd_solving_policy<
    D, A1, A2, array_function, ElementType,
    incremental_matrix_generation_policy<D, A1, A2>,
    incremental_transposed_matrix_generation_policy<D, A1, A2>,
    selective_evaluation_policy<D, A1, ElementType>,
    selective_transposed_evaluation_policy<D, A2, ElementType>,
    selective_accumulating_result_handling_policy<D, A1, A2>>
    BigArityImplementation;
#endif

    typedef typename variadic_conditional<
    vc_tuple<D == 4 && A1 == 4 && A2 == 4, BigArityImplementation>,
             vc_tuple<D == 4 && A1 == 1 && A2 == 1,
             simple_unary_solving_policy<4, ElementType>>,
             vc_tuple<D == 4, DefaultImplementation>>::type Implementation;

    static inline bool commutes(array_function<D, A1, ElementType>& f1,
                                array_function<D, A2, ElementType>& f2) {
        return Implementation::commutes(f1, f2);
    }
};
