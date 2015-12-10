#include <array>
#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <string>

#include "constants.hpp"
#include "bitset_function.hpp"
#include "primitive_solving_policy.hpp"
#include "simd_solving_policy.hpp"
#include "misc_tools.hpp"

std::map<std::string, std::set<std::string>> matches;

template <uint64_t D, uint64_t A>
std::vector<bitset_function<D, A>> read_functions(std::string filename) {
    const uint64_t array_size = space_per_function<D, A, uint8_t>::of_type;
    std::vector<bitset_function<D, A>> vec;
    std::array<uint8_t, array_size> arr;

    std::ifstream in;
    in.open(filename);

    while (!in.eof()) {
        in.read(reinterpret_cast<char *>(arr.data()), array_size);

        if (in.gcount() == 0) {
            break;
        }

        if (in.fail()) {
            std::cerr << "Failed to extract function number " << vec.size() << "."
                      << std::endl;
            exit(1);
        }

        vec.push_back(array_to_bitset_function<D, A, uint8_t>(arr));
    }

    return vec;
}

template <uint64_t D, uint64_t A1, uint64_t A2>
void commutation_test(std::vector<bitset_function<D, A1>>& vec1,
                      std::vector<bitset_function<D, A2>>& vec2) {
    for (uint64_t i = 0; i < vec1.size(); ++i) {
        for (uint64_t j = 0; j < vec2.size(); ++j) {
            std::string id1 = to_string(i) + "/" + to_string(A1);
            std::string id2 = to_string(j) + "/" + to_string(A2);

            if (primitive_solving_policy::commutes(vec1[i], vec2[j])) {
                matches[id1].insert(id2);
                matches[id2].insert(id1);
            }
        }
    }
}

int main() {
    const uint64_t D = 4;
    const uint64_t A1 = 4;
    const uint64_t A2 = 4;

#ifdef __clang__
    typedef simd_solving_policy<
    D, A1, A2, array_function, uint64_t,
    incremental_matrix_generation_policy<D, A1, A2>,
    incremental_transposed_matrix_generation_policy<D, A1, A2>,
    brute_force_evaluation_policy<D, A1>,
    brute_force_evaluation_policy<D, A2>,
    accumulating_result_handling_policy<D, A1, A2>> solver;
#else
    typedef simd_solving_policy<
    D, A1, A2, array_function, uint64_t,
    incremental_matrix_generation_policy<D, A1, A2>,
    incremental_transposed_matrix_generation_policy<D, A1, A2>,
    selective_evaluation_policy<D, A1>,
    selective_transposed_evaluation_policy<D, A2>,
    selective_accumulating_result_handling_policy<D, A1, A2>> solver;
#endif

    array_function<D, A1, uint64_t> f1;
    array_function<D, A2, uint64_t> f2;

    f1.storage.fill(-1);
    f2.storage.fill(-1);

    std::cout << solver::commutes(f1, f2) << std::endl;
    return 0;
}
