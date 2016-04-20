#include <cstdint>
#include <type_traits>

#include "code_generators.hpp"

#include "cuda_solving_policy.hpp"
#include "cuda_commutation_tester.hpp"

#ifndef SOLVER_DEFINED
template <uint64_t D, uint64_t A1, uint64_t A2> struct solver {
    static_assert(A1 <= A2, "Please always instantiate with A1 <= A2!");

    static inline void init(uint64_t thread_count = 256,
                            uint64_t group_size = (1 << 30)) {
        cuda_solving_policy<4, 4, 4>::init(thread_count, group_size);
    }

    static inline void deinit() {
        cuda_solving_policy<4, 4, 4>::deinit();
    }

    static inline bool commutes(const sparse_array_function<D, A1>& f1,
                                const sparse_array_function<D, A2>& f2) {
        return cuda_solving_policy<4, 4, 4>::commutes(f1, f2);
    }
};
#endif

#ifndef TESTER_DEFINED
typedef cuda_commutation_tester<match_map, cuda_tester_policy> tester;
#endif
