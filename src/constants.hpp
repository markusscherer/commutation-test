#ifndef CONSTANTS_HPP_MSDA
#define CONSTANTS_HPP_MSDA

#include <cstdint>

template <uint64_t D> struct space_per_element {
    static const uint64_t bits;
};

template <typename A> constexpr A cpow(A base, A exp) {
    return exp == 0 ? 1 : base * cpow(base, exp - 1);
}

template <uint64_t D, uint64_t A, typename T = uint8_t>
struct space_per_function {
    static const uint64_t bits = space_per_element<D>::bits * cpow(D, A);
    static const uint64_t of_type =
        space_per_element<D>::bits * cpow(D, A) / (sizeof(T) * 8) +
        // round up instead of truncation
        ((space_per_element<D>::bits * cpow(D, A) / (sizeof(T) * 8)) *
         (sizeof(T) * 8) ==
         space_per_element<D>::bits * cpow(D, A)
         ? 0
         : 1);
};

template <> const uint64_t space_per_element<2>::bits = 1;
template <> const uint64_t space_per_element<3>::bits = 2;
template <> const uint64_t space_per_element<4>::bits = 2;
// 4-bit-encoding, since the code for now doesn't handle non-multiples of 8 well
template <> const uint64_t space_per_element<5>::bits = 4;
#endif
