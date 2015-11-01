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
    const uint64_t A = 4;

    array_function<D, A, uint64_t> f1;
    array_function<D, A, uint64_t> f2;

    f1.storage.fill(0);
    f2.storage.fill(0);

    std::array<uint64_t, A> args;
    args.fill(0);

    // for (uint64_t i = 0; i < cpow(D, A); ++i) {
    //    print_iterable(args, std::cout, false);
    //    std::cout << f1.eval(args) << std::endl;
    //    increment_array<D, A>(args);
    //}

    args.fill(0);

    //    for (uint64_t i = 0; i < cpow(D, A); ++i) {
    //        print_iterable(args, std::cout, false);
    //        std::cout << f2.eval(args) << std::endl;
    //        increment_array<D, A>(args);
    //    }

    //    std::cout << "----" << std::endl;

    std::cout << simd_solving_policy<D, A, A, array_function, uint64_t>::commutes(
                  f1, f2)
              << std::endl;
    return 0;
}
