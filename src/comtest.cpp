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
    const uint64_t D = 2;
    //    const uint64_t A = 4;

    auto vec1 = read_functions<D, 1>("tests/data/all_functions.2.1.bin");
    auto vec2 = read_functions<D, 2>("tests/data/all_functions.2.2.bin");
    auto vec3 = read_functions<D, 3>("tests/data/all_functions.2.3.bin");
    auto vec4 = read_functions<3, 3>("tests/data/rand000.1.3.3.bin");

    auto f = vec3[183];

    auto fc =
        primitive_solving_policy::find_next_commuting<decltype(f), decltype(f)>(
            f);

    std::cout << fc.storage.to_string() << " "
              << primitive_solving_policy::commutes(f, fc) << std::endl;

    commutation_test<D, 1, 1>(vec1, vec1);
    commutation_test<D, 1, 2>(vec1, vec2);
    commutation_test<D, 1, 3>(vec1, vec3);
    commutation_test<D, 2, 2>(vec2, vec2);
    commutation_test<D, 2, 3>(vec2, vec3);
    commutation_test<D, 3, 3>(vec3, vec3);

    for (const auto& it : matches) {
        std::cout << it.first << ": ";

        for (const auto& ot : it.second) {
            std::cout << ot << " ";
        }

        std::cout << std::endl;
    }

    return 0;
}
