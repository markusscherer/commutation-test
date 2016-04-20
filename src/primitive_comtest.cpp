#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <chrono>

#include "constants.hpp"
#include "range.hpp"
#include "array_function.hpp"
#include "matches_types.hpp"

#include "configs/primitive_config/comtest.config.hpp"

const uint64_t MAX_ARITY = 4;

template <class Tester>
typename Tester::matches_type
test_all(const std::vector<array_function<2, 1, uint8_t>>& v1,
         const std::vector<array_function<2, 2, uint8_t>>& v2,
         const std::vector<array_function<2, 3, uint8_t>>& v3,
         const std::vector<array_function<2, 4, uint8_t>>& v4);

template <class Tester>
typename Tester::matches_type
test_ranges(const std::vector<array_function<2, 1, uint8_t>>& v1,
            const std::vector<array_function<2, 2, uint8_t>>& v2,
            const std::vector<array_function<2, 3, uint8_t>>& v3,
            const std::vector<array_function<2, 4, uint8_t>>& v4,
            const std::map<std::pair<uint64_t, uint64_t>, range>& ranges);

int main(int argc, char **argv) {
    const uint64_t D = 2;

    if (argc < 2) {
        return 1;
    }

    std::stringstream argparse;

    for (int i = 1; i < argc; ++i) {
        argparse << argv[i] << " ";
    }

    std::string s;
    std::string s2;
    std::map<std::string, std::string> options;
    std::string infilenames[MAX_ARITY];
    std::map<std::pair<uint64_t, uint64_t>, range> ranges;

    while (!argparse.eof()) {
        argparse >> s;

        if (argparse.bad() || argparse.eof()) {
            break;
        }

        if (s == "--functions-1" || s == "-1") {
            if (infilenames[0].empty()) {
                argparse >> infilenames[0];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "--functions-2" || s == "-2") {
            if (infilenames[1].empty()) {
                argparse >> infilenames[1];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "--functions-3" || s == "-3") {
            if (infilenames[2].empty()) {
                argparse >> infilenames[2];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "--functions-4" || s == "-4") {
            if (infilenames[3].empty()) {
                argparse >> infilenames[3];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "--range" || s == "-r") {
            std::cerr << "--range not supported!" << std::endl;
            return 1;
            argparse >> s;
            parse_range(ranges, s);
        } else if (s == "--option" || s == "-o") {
            argparse >> s;
            argparse >> s2;
            options[s] = s2;
        }
    }

    tester::init(options);

    std::vector<array_function<D, 1, uint8_t>> v1;
    std::vector<array_function<D, 2, uint8_t>> v2;
    std::vector<array_function<D, 3, uint8_t>> v3;
    std::vector<array_function<D, 4, uint8_t>> v4;

    if (!infilenames[0].empty()) {
        v1 = read_functions<2, 1>(infilenames[0]);
    }

    if (!infilenames[1].empty()) {
        v2 = read_functions<2, 2>(infilenames[1]);
    }

    if (!infilenames[2].empty()) {
        v3 = read_functions<2, 3>(infilenames[2]);
    }

    if (!infilenames[3].empty()) {
        v4 = read_functions<2, 4>(infilenames[3]);
    }

    typename tester::matches_type matches;

    auto start = std::chrono::high_resolution_clock::now();

    if (ranges.empty()) {
        matches = test_all<tester>(v1, v2, v3, v4);
    } else {
        matches = test_ranges<tester>(v1, v2, v3, v4, ranges);
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cerr << "total time: " << duration.count() << std::endl;
    std::cerr << "scheduler: " << options["scheduler"] << std::endl;
    std::cerr << "threads: " << options["thread_count"] << std::endl;
    std::cerr << "chunk size: " << options["chunk_size"] << std::endl;

    std::cout << matches.storage.size() << std::endl;

    for (auto& k : matches.storage) {
        std::cout << k.first.first << "/" << k.first.second;
        std::cout << ":";

        for (auto& c : k.second) {
            std::cout << c.first << "/" << c.second << " ";
        }

        std::cout << std::endl;
    }

    return 0;
}

template <class Tester>
typename Tester::matches_type
test_all(const std::vector<array_function<2, 1, uint8_t>>& v1,
         const std::vector<array_function<2, 2, uint8_t>>& v2,
         const std::vector<array_function<2, 3, uint8_t>>& v3,
         const std::vector<array_function<2, 4, uint8_t>>& v4) {
    typename Tester::matches_type ret;

    if (!v1.empty()) {
        auto tmp = tester::commutation_test(v1, v1);
        ret = join_matches(ret, tmp);

        if (!v2.empty()) {
            tmp = tester::commutation_test(v1, v2);
            ret = join_matches(ret, tmp);
        }

        if (!v3.empty()) {
            tmp = tester::commutation_test(v1, v3);
            ret = join_matches(ret, tmp);
        }

        if (!v4.empty()) {
            tmp = tester::commutation_test(v1, v4);
            ret = join_matches(ret, tmp);
        }
    }

    if (!v2.empty()) {
        auto tmp = tester::commutation_test(v2, v2);
        ret = join_matches(ret, tmp);

        if (!v3.empty()) {
            tmp = tester::commutation_test(v2, v3);
            ret = join_matches(ret, tmp);
        }

        if (!v4.empty()) {
            tmp = tester::commutation_test(v2, v4);
            ret = join_matches(ret, tmp);
        }
    }

    if (!v3.empty()) {
        auto tmp = tester::commutation_test(v3, v3);
        ret = join_matches(ret, tmp);

        if (!v4.empty()) {
            tmp = tester::commutation_test(v3, v4);
            ret = join_matches(ret, tmp);
        }
    }

    if (!v4.empty()) {
        auto tmp = tester::commutation_test(v4, v4);
        ret = join_matches(ret, tmp);
    }

    return ret;
}

template <class Tester>
typename Tester::matches_type
test_ranges(const std::vector<array_function<2, 1, uint8_t>>& v1,
            const std::vector<array_function<2, 2, uint8_t>>& v2,
            const std::vector<array_function<2, 3, uint8_t>>& v3,
            const std::vector<array_function<2, 4, uint8_t>>& v4,
            const std::map<std::pair<uint64_t, uint64_t>, range>& ranges) {
    std::cerr << "Ranges are buggy!!!" << std::endl;
    exit(2);

    typename Tester::matches_type ret;

    for (const auto& r : ranges) {
        const uint64_t a1 = r.first.first;
        const uint64_t a2 = r.first.second;
        typename Tester::matches_type tmp;
        std::cerr << "Checking range (at " << r.first.first << " " << r.first.second
                  << ") " << r.second.startA << "-" << r.second.endA << "  "
                  << r.second.startB << "-" << r.second.endB << std::endl;

        if (a1 == 1) {
            if (a2 == 1) {
                tmp = tester::commutation_test(v1, v1, r.second);
            } else if (a2 == 2) {
                tmp = tester::commutation_test(v1, v2, r.second);
            } else if (a2 == 3) {
                tmp = tester::commutation_test(v1, v3, r.second);
            } else if (a2 == 4) {
                tmp = tester::commutation_test(v1, v4, r.second);
            }
        } else if (a1 == 2) {
            if (a2 == 2) {
                tmp = tester::commutation_test(v2, v2, r.second);
            } else if (a2 == 3) {
                tmp = tester::commutation_test(v2, v3, r.second);
            } else if (a2 == 4) {
                tmp = tester::commutation_test(v2, v4, r.second);
            }
        } else if (a1 == 3) {
            if (a2 == 3) {
                tmp = tester::commutation_test(v3, v3, r.second);
            } else if (a2 == 4) {
                tmp = tester::commutation_test(v3, v4, r.second);
            }
        } else if (a1 == 4 && a2 == 4) {
            tmp = tester::commutation_test(v4, v4, r.second);
        }

        ret = join_matches(ret, tmp);
    }

    return ret;
}
