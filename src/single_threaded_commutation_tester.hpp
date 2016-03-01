#pragma once

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <cstdint>
#include <utility>

#include <omp.h>

#include "array_function.hpp"
#include "matches_types.hpp"
#include "range.hpp"

template <uint64_t, uint64_t, uint64_t> struct solver;

template <uint64_t D, uint64_t A1, uint64_t A2, class MatchesType>
struct single_threaded_tester_policy {
    typedef MatchesType matches_type;

    static inline matches_type
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2,
                     const range& r) {
        matches_type matches;

        for (uint64_t i = 0; i < vec1.size(); ++i) {
            for (uint64_t j = A1 != A2 ? 0 : i; j < vec2.size(); ++j) {
                if (solver<D, A1, A2>::commutes(vec1[i], vec2[j])) {
                    matches.add(i, j, A1, A2);
                }
            }
        }

        return matches;
    }
};

template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
struct single_threaded_commutation_tester {
    typedef MatchesType matches_type;

    static void init(const std::map<std::string, std::string>&) {
    }

    template <uint64_t D, uint64_t A1, uint64_t A2>
    static inline MatchesType
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2) {
        range ar;
        ar.startA = 0;
        ar.startB = 0;
        ar.endA = vec1.size();
        ar.endB = vec2.size();
        return TesterPolicy<D, A1, A2, MatchesType>::commutation_test(vec1, vec2,
                                                                      ar);
    }

    template <uint64_t D, uint64_t A1, uint64_t A2>
    static inline MatchesType
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2,
                     const range& r) {
        return TesterPolicy<D, A1, A2, MatchesType>::commutation_test(vec1, vec2,
                                                                      r);
    }
};
