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

#include "comtest.config.hpp"

const uint64_t MAX_ARITY = 4;

int main(int argc, char **argv) {
    const uint64_t D = 4;

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
            std::cerr << "--functions-1 not supported!" << std::endl;
            return 1;
            if (infilenames[0].empty()) {
                argparse >> infilenames[0];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "--functions-2" || s == "-2") {
            std::cerr << "--functions-2 not supported!" << std::endl;
            return 1;
            if (infilenames[1].empty()) {
                argparse >> infilenames[1];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "--functions-3" || s == "-3") {
            std::cerr << "--functions-3 not supported!" << std::endl;
            return 1;
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

    std::vector<array_function<D, 4, uint8_t>> v4;

    if (!infilenames[3].empty()) {
        v4 = read_functions<4, 4>(infilenames[3]);
    }

    typename tester::matches_type matches;

    auto start = std::chrono::high_resolution_clock::now();

    matches = tester::commutation_test(v4, v4);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cerr << "total time: " << duration.count() << std::endl;
    std::cerr << "init time: " << init_time << std::endl;
    std::cerr << "solve time: " << solve_time << std::endl;
    std::cerr << "apply time: " << apply_time << std::endl;
    std::cerr << "reduce time: " << reduce_time << std::endl;
    std::cerr << "threads: " << options["thread_count"] << std::endl;
    std::cerr << "group size: " << options["group_size"] << std::endl;

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
