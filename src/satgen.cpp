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
#include "misc_tools.hpp"
#include "matrix_accessor.hpp"

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
    const uint64_t A1 = 2;
    const uint64_t A2 = 2;

    auto vec1 = read_functions<D, 1>("tests/data/all_functions.2.1.bin");
    auto vec2 = read_functions<D, 2>("tests/data/all_functions.2.2.bin");
    auto vec3 = read_functions<D, 3>("tests/data/all_functions.2.3.bin");
    auto vec4 = read_functions<3, 3>("tests/data/rand000.1.3.3.bin");

    auto f1 = vec2[3];

    const uint64_t cell_count = A1 * A2;
    const uint64_t matrix_count = pow(D, cell_count);

    std::array<uint64_t, cell_count> matrix;
    matrix.fill(0);
    std::array<uint64_t, A1> args1;
    std::array<uint64_t, A2> args2;

    std::array<uint64_t, A1> tmp;
    tmp.fill(0);
    typedef matrix_accessor<A1, A2, decltype(matrix)::value_type, row_policy> row;
    typedef matrix_accessor<A1, A2, decltype(matrix)::value_type, column_policy>
    column;


    std::cout << "p cnf " << space_per_function<D,A2>::bits << " " << matrix_count * cpow(D,A1) << std::endl; 

    for (uint64_t m = 0; m < matrix_count; ++m) {
        for (uint64_t i = 0; i < A2; ++i) {
            auto rw = row(matrix, i);
            args2[i] = f1.eval(rw);
            print_iterable(rw, std::cerr, false);
            std::cerr << args2[i] << std::endl;
        }

        for (uint64_t i = 0; i < A1; ++i) {
            std::cerr << get_pos<D, A2, uint64_t>(column(matrix, i));
            std::cerr << " ";
        }

        std::cerr << std::endl;
        std::cerr << std::endl;

        for (uint64_t i = 0; i < pow(D, A1); ++i) {
            std::cout << "c f_";
            print_iterable(tmp, std::cout, false, "");
            std::cout << " = " << f1.eval(tmp) << "   g_";
            print_iterable(args2, std::cout, true, "");

            for (uint64_t j = 0; j < A1; ++j) {
                if (!tmp[j]) {
                    std::cout << "-";
                }

                std::cout << get_pos<D, A2, uint64_t>(column(matrix, j)) + 1;
                std::cout << " ";
            }

            if (f1.eval(tmp)) {
                std::cout << "-";
            }

            std::cout << get_pos<D, A2, uint64_t>(args2) + 1 << " ";
            std::cout << "0" << std::endl;
            increment_array<D, A1>(tmp);
        }

        increment_array<D, cell_count>(matrix);
    }

    return 0;
}
