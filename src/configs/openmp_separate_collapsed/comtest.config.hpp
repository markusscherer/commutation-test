#include "openmp_commutation_tester.hpp"
#include "matches_types.hpp"

typedef openmp_commutation_tester<
match_map, openmp_separate_collapsed_tester_policy> tester;

#define TESTER_DEFINED

#include "default_config/comtest.config.hpp"
