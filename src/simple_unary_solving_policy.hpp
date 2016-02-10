#ifndef SIMPLE_UNARY_SOLVING_POLICY_HPP_MSDA
#define SIMPLE_UNARY_SOLVING_POLICY_HPP_MSDA

#include <cstdint>
#include <iostream>

#include "constants.hpp"
#include "array_function.hpp"

template <uint64_t D, class ElementType> struct simple_unary_solving_policy {
    static bool commutes(array_function<D, 1, ElementType> f1,
                         array_function<D, 1, ElementType> f2) {
        ElementType v1 = f1.storage[0];
        ElementType v2 = f2.storage[0];

        uint8_t mask;

        if (D == 2) {
            mask = 0x01;
        } else {
            mask = 0x03;
        }

        const auto space = space_per_element<D>::bits;

        for (uint32_t i = 0; i < D; ++i) {
            uint8_t r1 = (v1 >> (((v2 >> (i * space)) & mask) * space)) & mask;
            uint8_t r2 = (v2 >> (((v1 >> (i * space)) & mask) * space)) & mask;

            if (r1 != r2) {
                return false;
            }
        }

        return true;
    }
};

#endif
