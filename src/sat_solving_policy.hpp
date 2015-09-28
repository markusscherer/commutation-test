#ifndef SAT_SOLVING_POLICY_HPP_MSDA
#define SAT_SOLVING_POLICY_HPP_MSDA

#include <array>
#include <vector>

#include "cryptominisat4/cryptominisat.h"

#include "bitset_function.hpp"
#include "constants.hpp"
#include "matrix_accessor.hpp"
#include "misc_tools.hpp"

template <uint64_t D, uint64_t A1, uint64_t A2, class ElementType>
struct cryptominisat_sat_policy {
    class cryptominisat_sat_policy_state {
        CMSat::SATSolver solver;
        std::vector<CMSat::Lit> current_clause;

        friend struct cryptominisat_sat_policy<D, A1, A2, ElementType>;
    };

    typedef cryptominisat_sat_policy_state state;

    static void setup(state& s) {
        s.solver.set_num_threads(4);
        s.solver.new_vars(cpow(D, A2) * space_per_element<D>::bits);
    }

    static void add_var(uint64_t var, bool negated, state& s) {
        s.current_clause.push_back(CMSat::Lit(var, negated));
    }

    static void end_clause(state& s) {
        s.solver.add_clause(s.current_clause);
        s.current_clause.clear();
    }

    static bitset_function<D, A2> calculate_function(state& s) {
        bitset_function<D, A2> f;
        s.solver.solve();

        auto model = s.solver.get_model();

        for (uint64_t i = 0; i < space_per_function<D, A2>::bits; ++i) {
            if (model[i] == CMSat::l_True) {
                f.storage.set(i);
            }
        }

        std::cout << f.storage.to_string() << std::endl;
        return f;
    }
};

template <uint64_t D, class P> struct value_restriction {
    static void add_restriction(uint64_t var, typename P::state& s) {
        static_assert(D <= 4,
                      "Value restrictions for domain sizes > 4 not implemented!");
    }

    static const uint64_t clauses_per_var;
};

template <class P> struct value_restriction<3, P> {
    static void add_restriction(uint64_t var, typename P::state& s) {
        const uint64_t x1 = var * space_per_element<3>::bits;
        const uint64_t x2 = x1 + 1;
        P::add_var(x1, true, s);
        P::add_var(x2, true, s);
        P::end_clause(s);
    }
    static const uint64_t clauses_per_var;
};

template <uint64_t D, class P>
const uint64_t value_restriction<D, P>::clauses_per_var = 0;

template <class P> const uint64_t value_restriction<3, P>::clauses_per_var = 1;

template <uint64_t D, uint64_t A1, uint64_t A2,
          template <uint64_t, uint64_t, uint64_t, class> class P,
          class ElementType = uint64_t>
struct sat_solving_policy {
    static const uint64_t cell_count = A1 * A2;
    static const uint64_t matrix_count = cpow(D, cell_count);
    typedef typename P<D, A1, A2, ElementType>::state state;

    template <class F1, class F2> static bool commutes(F1 f1, F2 f2) {
        check(false, "sat_solving_policy::commutes(f1,f2) is not implemented");
        return false;
    }

    template <class F1, class F2>
    static F2 find_next_commuting(F1 f1, F2 f2 = F2{}) {
        check(false,
              "sat_solving_policy::find_next_commuting(f1,f2) is not implemented");
        return f1;
    }

    static bitset_function<D, A2, ElementType>
    find_commuting(bitset_function<D, A1, ElementType> f1) {
        typename P<D, A1, A2, ElementType>::state s;
        P<D, A1, A2, ElementType>::setup(s);
        std::array<ElementType, cell_count> matrix;
        matrix.fill(0);

        for (uint64_t m = 0; m < matrix_count; ++m) {
            encode_implication_clause(f1, matrix, s);
            increment_array<D, cell_count>(matrix);
        }

        for (uint64_t i = 0; i < cpow(D, A2); ++i) {
            value_restriction<D, P<D, A1, A2, ElementType>>::add_restriction(i, s);
        }

        return P<D, A1, A2, ElementType>::calculate_function(s);
    }

private:
    static void encode_var(uint64_t var, ElementType val, bool invert, state& s) {
        const uint64_t start = var * space_per_element<D>::bits;
        const uint64_t end = start + space_per_element<D>::bits;

        ElementType mask = 1;

        for (uint64_t i = start; i < end; ++i) {
            bool negated = !(mask & val);

            if (invert) {
                negated = !negated;
            }

            P<D, A1, A2, ElementType>::add_var(i, negated, s);

            mask = mask << 1;
        }
    }

    static inline void
    encode_implication_clause(bitset_function<D, A1, ElementType>& f,
                              std::array<ElementType, A1 * A2>& matrix,
                              state& s) {

        typedef matrix_accessor<A1, A2, ElementType, row_policy> row;
        typedef matrix_accessor<A1, A2, ElementType, column_policy> column;

        std::array<ElementType, A1> tmp;
        std::array<ElementType, A1> args1;
        std::array<ElementType, A2> args2;
        tmp.fill(0);

        for (uint64_t j = 0; j < A1; ++j) {
            args1[j] = get_pos<D, A2, uint64_t>(column(matrix, j));
        }

        for (uint64_t j = 0; j < A2; ++j) {
            args2[j] = f.eval(row(matrix, j));
        }

        const uint64_t gid = get_pos<D, A2, ElementType>(args2);

        for (uint64_t a = 0; a < pow(D, A1); ++a) {
            const uint64_t start = gid * space_per_element<D>::bits;
            const uint64_t end = start + space_per_element<D>::bits;

            const ElementType val = f.eval(tmp);
            ElementType mask = 1;

            for (uint64_t res = start; res < end; ++res) {
                bool negated = !(mask & val);

                for (uint64_t i = 0; i < A1; ++i) {
                    encode_var(args1[i], tmp[i], true, s);
                }

                mask = mask << 1;

                P<D, A1, A2, ElementType>::add_var(res, negated, s);
                P<D, A1, A2, ElementType>::end_clause(s);
            }

            increment_array<D, A1>(tmp);
        }
    }
};

#endif
