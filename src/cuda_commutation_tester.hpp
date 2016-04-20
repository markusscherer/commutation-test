#pragma once

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <cstdint>
#include <utility>
#include <chrono>

#include "array_function.hpp"
#include "matches_types.hpp"
#include "range.hpp"

template <uint64_t, uint64_t, uint64_t> struct solver;

uint64_t init_time = 0;
uint64_t solve_time = 0;

template <uint64_t D, uint64_t A1, uint64_t A2, class MatchesType>
struct cuda_tester_policy {
    typedef MatchesType matches_type;

    static inline matches_type
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2,
                     const range& r, uint64_t group_size, uint64_t thread_count) {
        matches_type matches;

        auto start = std::chrono::high_resolution_clock::now();
        solver<D, A1, A2>::init(thread_count, group_size);
        auto end = std::chrono::high_resolution_clock::now();
        init_time +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

        for (uint64_t i = 0; i < vec1.size(); ++i) {
            auto f1 = sparse_array_function<D, A1>(vec1[i]);

            for (uint64_t j = A1 != A2 ? 0 : i; j < vec2.size(); ++j) {
                auto f2 = sparse_array_function<D, A2>(vec2[j]);
                start = std::chrono::high_resolution_clock::now();
                bool b = solver<D, A1, A2>::commutes(f1, f2);
                end = std::chrono::high_resolution_clock::now();
                solve_time +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                    .count();

                if (b) {
                    matches.add(i, j, A1, A2);
                }
            }
        }

        solver<D, A1, A2>::deinit();

        return matches;
    }
};

template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
struct cuda_commutation_tester {
    typedef MatchesType matches_type;
    static uint64_t thread_count;
    static uint64_t group_size;

    static void init(const std::map<std::string, std::string>& options) {
        auto val = options.find("group_size");

        if (val != options.end()) {
            group_size = strtol(val->second.c_str(), 0, 10);
        } else {
            group_size = (2ul << 29);
        }

        val = options.find("thread_count");

        if (val != options.end()) {
            thread_count = strtol(val->second.c_str(), 0, 10);
        } else {
            thread_count = 512;
        }

        gpuErrchk(cudaFree(0));
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
        return TesterPolicy<D, A1, A2, MatchesType>::commutation_test(
                   vec1, vec2, ar, group_size, thread_count);
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

template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
uint64_t cuda_commutation_tester<MatchesType, TesterPolicy>::thread_count;

template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
uint64_t cuda_commutation_tester<MatchesType, TesterPolicy>::group_size;
