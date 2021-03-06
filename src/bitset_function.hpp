#ifndef BITSET_FUNCTION_HPP_MSDA
#define BITSET_FUNCTION_HPP_MSDA

#include <cstdint>
#include <algorithm>
#include <bitset>
#include <array>
#include <cmath>
#include <initializer_list>
#include <iostream>

#include "constants.hpp"
#include "misc_tools.hpp"

template <uint64_t D, uint64_t A, typename ElementType, class C>
ElementType get_pos(C args) {
    uint64_t pos = 0;
    auto it = args.begin();

    for (uint64_t i = 0; i < A; ++i) {
        pos += pow(D, A - i - 1) * (*it);
        ++it;
    }

    return pos;
}

template <uint64_t D, uint64_t A, typename ElementType = uint64_t>
class bitset_function {
public:
    std::bitset<space_per_function<D, A>::bits> storage;
    typedef ElementType element_type;
    static const uint64_t domain_size = D;
    static const uint64_t arity = A;

    template <class T> inline ElementType eval(T args) const {
        const uint64_t width = space_per_element<D>::bits;
        const uint64_t pos = get_pos<D, A, ElementType>(args) * width;

        uint64_t mask = 1;
        uint64_t res = 0;

        for (uint64_t i = pos; i < pos + width; ++i) {
            res += mask * storage[i];
            mask = mask << 1;
        }

        return res;
    }

    template <class T> inline void set(T args, ElementType res) {
        const uint64_t width = space_per_element<D>::bits;
        const uint64_t pos = get_pos<D, A, ElementType>(args) * width;

        uint64_t mask = 1;

        for (uint64_t i = pos; i < pos + width; ++i) {
            storage[i] = mask & res;
            mask = mask << 1;
        }
    }
    bool operator==(const bitset_function<D, A, ElementType>& other) const {
        return this->storage == other.storage;
    }
    bool operator!=(const bitset_function<D, A, ElementType>& other) const {
        return this->storage != other.storage;
    }
};

template <uint64_t D, uint64_t A, typename T, typename ElementType = uint64_t>
std::array<T, space_per_function<D, A, T>::of_type>
bitset_function_to_array(bitset_function<D, A, ElementType>& f) {
    const uint64_t width = sizeof(T) * 8;
    T res = 0;
    std::array<T, space_per_function<D, A, T>::of_type> ret;

    int j = 0;

    for (uint64_t pos = 0; pos < f.storage.size(); pos += width) {
        res = 0;
        uint64_t mask = 1;

        const int end = std::min(pos + width, f.storage.size());

        for (int i = pos; i < end; ++i) {
            res += mask * f.storage[i];
            mask = mask << 1;
        }

        ret[j] = res;
        ++j;
    }

    return ret;
}

template <uint64_t D, uint64_t A, typename T, typename ElementType = uint64_t>
bitset_function<D, A, ElementType> array_to_bitset_function(
    const std::array<T, space_per_function<D, A, T>::of_type>& arr) {
    const uint64_t width = sizeof(T) * 8;
    bitset_function<D, A, ElementType> f;

    for (uint64_t pos = 0; pos < arr.size(); ++pos) {
        uint64_t mask = 1;

        for (uint64_t i = pos * width; i < (pos + 1) * width; ++i) {
            f.storage[i] = mask & arr[pos];
            mask = mask << 1;
        }
    }

    return f;
}

template <uint64_t D, uint64_t A, typename ElementType = uint64_t>
void increment_function(bitset_function<D, A, ElementType>& f) {
    std::array<ElementType, A> args;
    args.fill(0);

    for (uint64_t l = 0; l < pow(D, A); ++l) {
        ElementType c = f.eval(args);

        if (c != D - 1) {
            f.set(args, c + 1);

            args.fill(0);

            for (uint64_t k = 0; k < l; k++) {
                f.set(args, 0);
                increment_array<D, A, ElementType>(args);
            }

            break;
        }

        increment_array<D, A, ElementType>(args);
    }
}

template <typename F> struct function_tools {
    static bool is_max(F& f);
};

template <uint64_t A, typename ElementType>
struct function_tools<bitset_function<4, A, ElementType>> {
    static bool is_max(bitset_function<4, A, ElementType>& f) {
        return f.storage.all();
    }
};

template <uint64_t D, uint64_t A, typename ElementType>
struct function_tools<bitset_function<D, A, ElementType>> {
    static bool is_max(bitset_function<D, A, ElementType>& f) {
        std::array<ElementType, A> args;
        args.fill(0);

        for (uint64_t l = 0; l < pow(D, A); ++l) {
            ElementType c = f.eval(args);

            if (c != D - 1) {
                return false;
            }

            increment_array<D, A, ElementType>(args);
        }

        return true;
    }
};

#endif
