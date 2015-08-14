#ifndef CODE_GENERATORS_HPP_MSDA
#define CODE_GENERATORS_HPP_MSDA

#include <cstdint>

#include "misc_tools.hpp"

template <uint64_t A, uint64_t D, template <uint64_t, uint64_t> class C>
struct arity_select;

template <uint64_t D, template <uint64_t, uint64_t> class C>
struct arity_select<0, D, C> {
    static void select(uint64_t arity, uint64_t count) {
        check(false, "Arity 0 is not allowed!");
    }
};

template <uint64_t A, uint64_t D, template <uint64_t, uint64_t> class C>
struct arity_select {
    static void select(uint64_t arity, uint64_t count) {
        if (arity == A) {
            C<D, A>::run(count);
        } else {
            arity_select<A - 1, D, C>::select(arity, count);
        }
    }
};

template <uint64_t D, uint64_t A, template <uint64_t, uint64_t> class C>
struct domain_size_select;

template <uint64_t A, template <uint64_t, uint64_t> class C>
struct domain_size_select<0, A, C> {
    static void select(uint64_t domain_size, uint64_t arity, uint64_t count) {
        check(false, "Domain size 0 is not allowed!");
    }
};

template <uint64_t A, template <uint64_t, uint64_t> class C>
struct domain_size_select<1, A, C> {
    static void select(uint64_t domain_size, uint64_t arity, uint64_t count) {
        check(false, "Domain size 1 is not allowed!");
    }
};

template <uint64_t D, uint64_t A, template <uint64_t, uint64_t> class C>
struct domain_size_select {
    static void select(uint64_t domain_size, uint64_t arity, uint64_t count) {
        if (domain_size == D) {
            arity_select<A, D, C>::select(arity, count);
        } else {
            domain_size_select<D - 1, A, C>::select(domain_size, arity, count);
        }
    }
};

#endif
