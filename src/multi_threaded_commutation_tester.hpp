#pragma once

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <cstdint>
#include <utility>
#include <thread>
#include <mutex>
#include <limits>

#include "array_function.hpp"
#include "matches_types.hpp"
#include "range.hpp"

template <uint64_t, uint64_t, uint64_t> struct solver;
template <uint64_t D, uint64_t A1, uint64_t A2, class MatchesType>
struct commutation_thread;
template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
struct multi_threaded_commutation_tester;

template <uint64_t D, uint64_t A1, uint64_t A2, class MatchesType>
struct simple_multi_threaded_tester_policy {
    inline static MatchesType
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2,
                     uint64_t thread_count) {
        std::vector<std::thread> threads;
        threads.reserve(thread_count);
        std::vector<MatchesType> thread_matches;
        thread_matches.resize(thread_count);

        for (uint64_t i = 0; i < thread_count; ++i) {
            threads.push_back(std::thread(
                                  commutation_thread<D, A1, A2, MatchesType>(), std::cref(vec1),
                                  std::cref(vec2), std::ref(thread_matches[i])));
        }

        MatchesType ret;

        for (uint64_t i = 0; i < thread_count; ++i) {
            threads[i].join();
            ret = join_matches(ret, thread_matches[i]);
        }

        return ret;
    }
};

template <uint64_t D, uint64_t A1, uint64_t A2, class MatchesType>
struct commutation_thread {
    void operator()(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                    const std::vector<array_function<D, A2, uint8_t>>& vec2,
                    MatchesType& thread_matches) {
        bool run = true;
        uint64_t start = 0;
        uint64_t end = 0;

        typedef multi_threaded_commutation_tester<
        MatchesType, simple_multi_threaded_tester_policy> thread_manager;

        while (true) {
            run = thread_manager::get_new_bounds(start, end);

            if (!run) {
                break;
            }

            for (uint64_t i = start; i < end; ++i) {
                for (uint64_t j = A1 != A2 ? 0 : i; j < vec2.size(); ++j) {
                    if (solver<D, A1, A2>::commutes(vec1[i], vec2[j])) {
                        thread_matches.add(i, j, A1, A2);
                    }
                }
            }
        }
    }
};

template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
struct multi_threaded_commutation_tester {
    typedef MatchesType matches_type;
    static uint64_t chunk_size;
    static uint64_t thread_count;

    static std::mutex global_mutex;
    static uint64_t size1;
    static uint64_t size2;
    static uint64_t counter;

    static void init(const std::map<std::string, std::string>& options) {
        auto val = options.find("scheduler");

        val = options.find("chunk_size");

        if (val != options.end()) {
            chunk_size = strtol(val->second.c_str(), 0, 10);
        }

        val = options.find("thread_count");

        if (val != options.end()) {
            thread_count = strtol(val->second.c_str(), 0, 10);
        } else {
            thread_count = std::thread::hardware_concurrency();
        }
    }

    inline static bool get_new_bounds(uint64_t& start, uint64_t& end) {
        std::lock_guard<std::mutex> lock(global_mutex);

        if (counter >= size1) {
            return false;
        }

        start = counter;
        counter += chunk_size;
        end = std::min(size1, counter);

        return true;
    }

    template <uint64_t D, uint64_t A1, uint64_t A2>
    static inline MatchesType
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2) {
        /* range ar; */
        /* ar.startA = 0; */
        /* ar.startB = 0; */
        /* ar.endA = vec1.size(); */
        /* ar.endB = vec2.size(); */
        counter = 0;
        size1 = vec1.size();
        size2 = vec2.size();
        /* std::cout << A1 << "  " << A2 << " " << size1 << " " << size2 <<
         * std::endl; */
        return TesterPolicy<D, A1, A2, MatchesType>::commutation_test(vec1, vec2,
                                                                      thread_count);
    }

    template <uint64_t D, uint64_t A1, uint64_t A2>
    static inline MatchesType
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2,
                     const range& r) {
        exit(2);
        /* return TesterPolicy<D, A1, A2, MatchesType>::commutation_test(vec1, vec2,
         */
        /* r); */
    }
};

template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
uint64_t multi_threaded_commutation_tester<MatchesType, TesterPolicy>::counter =
    0;
template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
uint64_t multi_threaded_commutation_tester<MatchesType, TesterPolicy>::size1 =
    0;
template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
uint64_t multi_threaded_commutation_tester<MatchesType, TesterPolicy>::size2 =
    0;
template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
uint64_t multi_threaded_commutation_tester<MatchesType,
         TesterPolicy>::chunk_size = 1;
template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
uint64_t multi_threaded_commutation_tester<MatchesType,
         TesterPolicy>::thread_count = 1;

template <class MatchesType,
          template <uint64_t, uint64_t, uint64_t, class> class TesterPolicy>
std::mutex
multi_threaded_commutation_tester<MatchesType, TesterPolicy>::global_mutex;
