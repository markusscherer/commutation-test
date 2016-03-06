#include <cstdint>
#include <type_traits>

#include "code_generators.hpp"

#include "primitive_solving_policy.hpp"
#include "single_threaded_commutation_tester.hpp"

#ifndef SOLVER_DEFINED
template <uint64_t D, uint64_t A1, uint64_t A2> struct solver {
    static_assert(A1 <= A2, "Please always instantiate with A1 <= A2!");
    typedef uint8_t ElementType;

    static inline bool commutes(const bitset_function<D, A1, ElementType>& f1,
                                const bitset_function<D, A2, ElementType>& f2) {
        return primitive_solving_policy<D, A1, A2>::commutes(f1, f2);
    }
};
#endif

#ifndef TESTER_DEFINED
typedef single_threaded_commutation_tester<
match_map, bitset_function_single_threaded_tester_policy> tester;
#endif
