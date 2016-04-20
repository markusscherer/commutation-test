#pragma once

#include <cstdint>
#include <array>

#include "array_function.hpp"
#include "constants.hpp"

template <uint64_t D, uint64_t A> struct sparse_array_function {
    static const uint64_t domain_size = D;
    static const uint64_t arity = A;

    typedef uint8_t element_type;

    std::array<uint8_t, cpow(D, A)> storage;

    template <typename ElementType>
    sparse_array_function(const array_function<D, A, ElementType>& f) {
        for (uint64_t i = 0; i < f.storage.size(); ++i) {
            ElementType mask = ~((~0) << space_per_element<D>::bits);

            const uint64_t steps =
                sizeof(ElementType) * 8 / space_per_element<D>::bits;

            for (uint64_t j = 0; j < steps; ++j) {
                /* std::cerr << i << " " << j << " " << (i*steps + j) << std::endl; */
                storage[i * steps + j] =
                    (f.storage[i] & mask) >> (j * space_per_element<D>::bits);
                mask <<= space_per_element<D>::bits;
            }
        }
    }

    template <class T> inline uint8_t eval(T args) const {
        return storage[get_pos<D, A, uint8_t>(args)];
    }
};
