#ifndef PRIMITIVE_SOLVING_POLICY_HPP_MSDA
#define PRIMITIVE_SOLVING_POLICY_HPP_MSDA

#include <type_traits>
#include <iostream>
#include <array>
#include <cmath>

#include "misc_tools.hpp"
#include "bitset_function.hpp"

struct primitive_solving_policy {
    template <class F1, class F2> static bool commutes(F1 f1, F2 f2) {
        static_assert(F1::domain_size == F2::domain_size,
                      "Domain sizes must match.");
        static_assert(std::is_same<typename F1::element_type,
                      typename F2::element_type>::value,
                      "Element types must match.");

        typedef typename F1::element_type element_type;
        const uint64_t domain_size = F1::domain_size;
        const uint64_t cell_count = F1::arity * F2::arity;
        const uint64_t matrix_count = pow(domain_size, cell_count);

        std::array<element_type, cell_count> matrix;
        matrix.fill(0);
        std::array<element_type, F1::arity> args1;
        std::array<element_type, F2::arity> args2;
        element_type r1;
        element_type r2;
        //        args1.fill(0);
        //        args2.fill(0);

        for (uint64_t m = 0; m < matrix_count; ++m) {
            for (uint64_t i = 0; i < F2::arity; ++i) {
                for (uint64_t j = 0; j < F1::arity; ++j) {
                    args1[j] = matrix[i + F2::arity * j];
                }

                args2[i] = f1.eval(args1);
                print_iterable(args1, std::cout, false);
                std::cout << args2[i] << std::endl;
            }

            std::cout << std::endl;

            r2 = f2.eval(args2);

            for (uint64_t i = 0; i < F1::arity; ++i) {
                for (uint64_t j = 0; j < F2::arity; ++j) {
                    args2[j] = matrix[j + F2::arity * i];
                }

                args1[i] = f2.eval(args2);
                //                print_iterable(args2, std::cout, false);
                //                std::cout << args1[i] << std::endl;
            }

            //            std::cout << "......." << std::endl;

            r1 = f1.eval(args1);

            if (r1 != r2) {
                return false;
            }

            // generate next matrix
            increment_array<domain_size, cell_count, element_type>(matrix);
        }

        return true;
    }

    template <class F1, class F2>
    static F2 find_next_commuting(F1 f1, F2 f2 = F2{}) {
        F2 f = f2;

        do {
            bool b = primitive_solving_policy::commutes(f1, f);

            if (b) {
                return f;
            }

            increment_function(f);
        } while (!function_tools<F2>::is_max(f));

        return f2;
    }
};

#endif
