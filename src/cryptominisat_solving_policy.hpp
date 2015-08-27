#ifndef CRYPTOMINISAT_SOLVING_POLICY_HPP_MSDA
#define CRYPTOMINISAT_SOLVING_POLICY_HPP_MSDA

#include <array>
#include <vector>

#include "cryptominisat4/cryptominisat.h"

#include "constants.hpp"
#include "misc_tools.hpp"

struct cryptosat_solving_policy {
    template <class F1, class F2> static bool commutes(F1 f1, F2 f2) {
        check(false,
              "cryptosat_solving_policy::commutes(f1,f2) is not implemented");
        return false;
    }

    template <class F1, class F2>
    static F2 find_next_commuting(F1 f1, F2 f2 = F2{}) {
        typedef typename F1::element_type element_type;
        const uint64_t domain_size = F1::domain_size;
        const uint64_t width = space_per_element<domain_size>::bits;
        CMSat::SATSolver solver;
        solver.set_num_threads(4);
        std::vector<CMSat::Lit> clause;

        std::array<element_type, F1::arity> args1;
        // std::array<element_type, F2::arity> args2;
        args1.fill(0);

        for (uint64_t l = 0; l < pow(domain_size, F1::arity); ++l) {
            clause.clear();

            for (uint64_t i = 0; i < F1::arity; ++i) {
                for (uint64_t j = i * width; j < (i + 1) * width; ++j) {
                    clause.push_back(Lit(j, !f1.storage[j]));
                }
            }

            element_type res = f1.eval(args1);
            uint64_t mask = 1;
            uint64_t offset = width * F1::arity;

            for (uint64_t j = 0; j < width; ++j) {
                clause.push_back(Lit(j + offset, mask & res));
                mask = mask << 1;
            }

            increment_array(args1);
        }

        CMSat::lbool b = solver.solve();

        std::cout << b << std::endl;

        return f2;
    }
};

#endif
