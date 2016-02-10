#ifndef ARRAY_FUNCTION_HPP_MSDA
#define ARRAY_FUNCTION_HPP_MSDA

#include <cstdint>
#include <array>

#include "constants.hpp"
#include "bitset_function.hpp"

template <uint64_t D, uint64_t A, typename ElementType> class array_function {
public:
    typedef ElementType element_type;
    static const uint64_t domain_size = D;
    static const uint64_t arity = A;
    static const uint64_t storage_elements =
        space_per_function<D, A, element_type>::of_type;

    std::array<ElementType, storage_elements> storage;

    array_function() {};

    template <typename T>
    explicit array_function(bitset_function<D, A, T>& in)
        : storage(bitset_function_to_array(in)) {
    }
};
#endif
