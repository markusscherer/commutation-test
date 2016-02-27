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
struct openmp_critical_tester_policy {
    typedef MatchesType matches_type;

    static inline matches_type
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2,
                     const range& r) {
        matches_type matches;

        #pragma omp parallel for schedule(runtime)

        for (uint64_t i = 0; i < vec1.size(); ++i) {
            for (uint64_t j = A1 != A2 ? 0 : i; j < vec2.size(); ++j) {
                if (solver<D, A1, A2>::commutes(vec1[i], vec2[j])) {
                    #pragma omp critical
                    { matches.add(i, j, A1, A2); }
                }
            }
        }

        return matches;
    }
};

template <uint64_t D, uint64_t A1, uint64_t A2, class MatchesType>
struct openmp_separate_tester_policy {
    typedef MatchesType matches_type;

    static inline matches_type
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2,
                     const range& r) {
        const size_t max_threads = omp_get_max_threads();
        std::vector<matches_type> thread_matches(max_threads);

        #pragma omp parallel for schedule(runtime)

        for (uint64_t i = 0; i < vec1.size(); ++i) {
            for (uint64_t j = A1 != A2 ? 0 : i; j < vec2.size(); ++j) {
                if (solver<D, A1, A2>::commutes(vec1[i], vec2[j])) {
                    thread_matches[omp_get_thread_num()].add(i, j, A1, A2);
                }
            }
        }

        return std::accumulate(thread_matches.begin(), thread_matches.end(),
                               matches_type(), join_matches);
    }
};

template <uint64_t D, uint64_t A1, uint64_t A2, class MatchesType>
struct openmp_reduction_tester_policy {
    typedef MatchesType matches_type;

    static inline matches_type
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2,
                     const range& r) {
        #pragma omp declare reduction(match_join : matches_type : omp_out =            \
        join_matches(omp_out, omp_in))

        matches_type matches;

        #pragma omp parallel for schedule(runtime) reduction(match_join : matches)

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

template <uint64_t D, uint64_t A1, uint64_t A2, class MatchesType>
struct openmp_critical_collapsed_tester_policy {
    typedef MatchesType matches_type;

    static inline matches_type
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2,
                     const range& r) {
        matches_type matches;

        #pragma omp parallel for schedule(runtime) collapse(2)

        for (uint64_t i = 0; i < vec1.size(); ++i) {
            for (uint64_t j = 0; j < vec2.size(); ++j) {
                if (A1 == A2 && j < i) {
                    continue;
                }

                if (solver<D, A1, A2>::commutes(vec1[i], vec2[j])) {
                    #pragma omp critical
                    { matches.add(i, j, A1, A2); }
                }
            }
        }

        return matches;
    }
};

template <uint64_t D, uint64_t A1, uint64_t A2, class MatchesType>
struct openmp_separate_collapsed_tester_policy {
    typedef MatchesType matches_type;

    static inline matches_type
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2,
                     const range& r) {
        const size_t max_threads = omp_get_max_threads();
        std::vector<matches_type> thread_matches(max_threads);

        #pragma omp parallel for schedule(runtime) collapse(2)

        for (uint64_t i = 0; i < vec1.size(); ++i) {
            for (uint64_t j = 0; j < vec2.size(); ++j) {
                if (A1 == A2 && j < i) {
                    continue;
                }

                if (solver<D, A1, A2>::commutes(vec1[i], vec2[j])) {
                    thread_matches[omp_get_thread_num()].add(i, j, A1, A2);
                }
            }
        }

        matches_type ret;

        for (auto const& m : thread_matches) {
            ret = join_matches(ret, m);
        }

        return std::accumulate(thread_matches.begin(), thread_matches.end(),
                               matches_type(), join_matches);
    }
};

template <uint64_t D, uint64_t A1, uint64_t A2, class MatchesType>
struct openmp_reduction_collapsed_tester_policy {
    typedef MatchesType matches_type;

    static inline matches_type
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2,
                     const range& r) {
        #pragma omp declare reduction(match_join : matches_type : omp_out =            \
        join_matches(omp_out, omp_in))

        matches_type matches;

        #pragma omp parallel for schedule(runtime) reduction(match_join : matches)     \
        collapse(2)

        for (uint64_t i = 0; i < vec1.size(); ++i) {
            for (uint64_t j = 0; j < vec2.size(); ++j) {
                if (A1 == A2 && j < i) {
                    continue;
                }

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
struct openmp_commutation_tester {
    typedef MatchesType matches_type;

    static void init(const std::map<std::string, std::string>& options) {
        auto val = options.find("scheduler");
        omp_sched_t scheduler = omp_sched_static;
        uint64_t chunk_size = 0;
        uint64_t thread_count = 0;

        if (val != options.end()) {
            if (val->second == "auto") {
                scheduler = omp_sched_auto;
            } else if (val->second == "static") {
                scheduler = omp_sched_static;
            } else if (val->second == "dynamic") {
                scheduler = omp_sched_dynamic;
            } else if (val->second == "guided") {
                scheduler = omp_sched_guided;
            } else {
                std::cerr << "unknown scheduler type " << val->second << " ignored"
                          << std::endl;
            }
        }

        val = options.find("chunk_size");

        if (val != options.end()) {
            chunk_size = strtol(val->second.c_str(), 0, 10);
        }

        val = options.find("thread_count");

        if (val != options.end()) {
            thread_count = strtol(val->second.c_str(), 0, 10);
        }

        omp_set_num_threads(thread_count);
        omp_set_schedule(scheduler, chunk_size);
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
