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
#include "policies/primitive_result_handling_policy.hpp"
#include "policies/blowup_transposed_matrix_generation_policy.hpp"
#include "policies/dense_incremental_transposed_matrix_generation_policy.hpp"
#include "policies/dense_incremental_matrix_generation_policy.hpp"

#include "multi_threaded_commutation_tester.hpp"

typedef multi_threaded_commutation_tester<
match_map, simple_multi_threaded_tester_policy> tester;

#define TESTER_DEFINED

#include "default_config/comtest.config.hpp"
