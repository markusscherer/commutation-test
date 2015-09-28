#include <array>
#include <iostream>

#include "misc_tools.hpp"
#include "constants.hpp"

int main() {
    const uint64_t D = 3;
    const uint64_t A = 2;
    const uint64_t num_lines = cpow(D, A);
    const uint64_t num_funcs = cpow(D, num_lines);

    std::array<uint8_t, num_lines> function;
    std::array<uint8_t, A> args;

    function.fill(0);
    args.fill(0);
    std::cout << "count " << num_funcs << std::endl;
    std::cout << "domain_size " << D << std::endl;
    std::cout << "arity " << A << std::endl;

    for (uint64_t f = 0; f < num_funcs; ++f) {
        for (uint64_t i = 0; i < num_lines; ++i) {
            print_iterable(args, std::cout, false);
            std::cout << (uint64_t)function[num_lines - i - 1] << std::endl;
            increment_array<D, A>(args);
        }

        std::cout << std::endl;
        args.fill(0);
        increment_array<D, num_lines>(function);
    }
}
