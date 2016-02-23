#include <array>
#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <string>

#include "cryptominisat4/cryptominisat.h"

#include "constants.hpp"
#include "sat_solving_policy.hpp"
#include "bitset_function.hpp"
#include "primitive_solving_policy.hpp"
#include "misc_tools.hpp"
#include "matrix_accessor.hpp"

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

template <uint64_t D, uint64_t A1, uint64_t A2, class ElementType>
struct dimacs_output_sat_policy {
    typedef std::nullptr_t state;

    static void setup(state&) {
        const uint64_t cell_count = A1 * A2;
        const uint64_t matrix_count = cpow(D, cell_count);

        const uint64_t var_count = space_per_function<D, A2>::bits;
        const uint64_t clause_count =
            matrix_count * cpow(D, A1) * space_per_element<D>::bits +
            cpow(D, A2) *
            value_restriction<D, dimacs_output_sat_policy>::clauses_per_var;

        std::cout << "p cnf " << var_count << " " << clause_count << std::endl;
    }

    static void add_var(uint64_t var, bool negated, state) {
        if (negated) {
            std::cout << "-";
        }

        std::cout << (var + 1) << " ";
    }

    static void end_clause(state) {
        std::cout << "0" << std::endl;
    }

    static bitset_function<D, A2> calculate_function(state) {
        bitset_function<D, A2> f;
        std::cerr << "Warning: method not implemented; returned function is "
                  "constant zero."
                  << std::endl;
        return f;
    }
};

int main() {
    const uint64_t D = 3;
    const uint64_t A1 = 2;
    const uint64_t A2 = 3;

    //    auto vec1 = read_functions<D, 1>("tests/data/all_functions.2.1.bin");
    //    auto vec2 = read_functions<D, 2>("tests/data/all_functions.2.2.bin");
    //    auto vec3 = read_functions<D, 3>("tests/data/all_functions.2.3.bin");
    //    auto vec4 = read_functions<3, 3>("tests/data/rand000.1.3.3.bin");
    auto vec4 = read_functions<3, 2>("fun.2.3.k");

    auto vec5 = read_functions<3, 2>("tests/data/all_functions.3.2.bin");
    auto f1 = vec4[15799];

    sat_solving_policy<3, A1, A2, dimacs_output_sat_policy,
                       uint64_t>::find_commuting(f1);
    auto f2 = sat_solving_policy<3, A1, A2, cryptominisat_sat_policy,
         uint64_t>::find_commuting(f1);
    std::cout << "Commutes?: " << primitive_solving_policy::commutes(f1, f2)
              << std::endl;

    std::array<uint64_t, A2> args;
    args.fill(0);

    for (uint64_t l = 0; l < pow(D, A2); ++l) {
        print_iterable(args, std::cout, false);

        std::cout << f1.eval(args) << std::endl;
        increment_array<D, A2, uint64_t>(args);
    }

    return 0;
}
