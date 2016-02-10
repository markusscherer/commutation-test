#include <array>
#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <string>
#include <sstream>

#include "array_function.hpp"
#include "constants.hpp"
#include "bitset_function.hpp"
#include "misc_tools.hpp"

#include "comtest.config.hpp"

std::map<std::string, std::set<std::string>> matches;

template <uint64_t D, uint64_t A>
std::vector<array_function<D, A, uint8_t>>
read_functions(std::string filename) {
    const uint64_t array_size = space_per_function<D, A, uint8_t>::of_type;
    std::vector<array_function<D, A, uint8_t>> vec;

    std::ifstream in;
    in.open(filename, std::ifstream::ate);
    std::streampos size = in.tellg();
    in.seekg(std::ios_base::beg);

    if (size % array_size != 0) {
        std::cerr << "File contains incomplete function." << std::endl;
    }

    vec.resize(size / array_size);

    uint64_t counter = 0;

    while (!in.eof()) {
        in.read(reinterpret_cast<char *>(vec[counter].storage.data()), array_size);

        if (in.gcount() == 0) {
            break;
        }

        if (in.fail()) {
            std::cerr << "Failed to extract function number " << vec.size() << "."
                      << std::endl;
            exit(1);
        }

        ++counter;
    }

    return vec;
}

template <uint64_t D, uint64_t A1, uint64_t A2>
void commutation_test(std::vector<array_function<D, A1, uint8_t>>& vec1,
                      std::vector<array_function<D, A2, uint8_t>>& vec2) {
    for (uint64_t i = 0; i < vec1.size(); ++i) {
        for (uint64_t j = A1 != A2 ? 0 : i; j < vec2.size(); ++j) {
            std::string id1 = to_string(i) + "/" + to_string(A1);
            std::string id2 = to_string(j) + "/" + to_string(A2);

            if (solver<D, A1, A2>::commutes(vec1[i], vec2[j])) {
                matches[id1].insert(id2);
                matches[id2].insert(id1);
            }
        }
    }
}

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
    std::string infilenames[MAX_ARITY];

    while (!argparse.eof()) {
        argparse >> s;

        if (argparse.bad() || argparse.eof()) {
            break;
        }

        if (s == "-functions-1" || s == "-1") {
            if (infilenames[0].empty()) {
                argparse >> infilenames[0];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "-functions-2" || s == "-2") {
            if (infilenames[1].empty()) {
                argparse >> infilenames[1];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "-functions-3" || s == "-3") {
            if (infilenames[2].empty()) {
                argparse >> infilenames[2];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        } else if (s == "-functions-4" || s == "-4") {
            if (infilenames[3].empty()) {
                argparse >> infilenames[3];
            } else {
                std::cerr << "Please specify only one function file per arity."
                          << std::endl;
                return 0;
            }
        }
    }

    std::vector<array_function<D, 1, uint8_t>> v1;
    std::vector<array_function<D, 2, uint8_t>> v2;
    std::vector<array_function<D, 3, uint8_t>> v3;
    std::vector<array_function<D, 4, uint8_t>> v4;

    if (!infilenames[0].empty()) {
        v1 = read_functions<4, 1>(infilenames[0]);
    }

    if (!infilenames[1].empty()) {
        v2 = read_functions<4, 2>(infilenames[1]);
    }

    if (!infilenames[2].empty()) {
        v3 = read_functions<4, 3>(infilenames[2]);
    }

    if (!infilenames[3].empty()) {
        v4 = read_functions<4, 4>(infilenames[3]);
    }

    if (!infilenames[0].empty()) {
        commutation_test(v1, v1);

        if (!infilenames[1].empty()) {
            commutation_test(v1, v2);
        }

        if (!infilenames[2].empty()) {
            commutation_test(v1, v3);
        }

        if (!infilenames[3].empty()) {
            commutation_test(v1, v4);
        }
    }

    if (!infilenames[1].empty()) {
        commutation_test(v2, v2);

        if (!infilenames[2].empty()) {
            commutation_test(v2, v3);
        }

        if (!infilenames[3].empty()) {
            commutation_test(v2, v4);
        }
    }

    if (!infilenames[2].empty()) {
        commutation_test(v3, v3);

        if (!infilenames[3].empty()) {
            commutation_test(v3, v4);
        }
    }

    if (!infilenames[3].empty()) {
        commutation_test(v4, v4);
    }

    std::cout << matches.size() << std::endl;

    for (auto& k : matches) {
        std::cout << k.first;
        std::cout << ":";

        for (auto& c : k.second) {
            std::cout << c << " ";
        }

        std::cout << std::endl;
    }

    return 0;
}
