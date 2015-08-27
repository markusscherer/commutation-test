#ifndef MATRIX_ACCESSOR_HPP_MSDA
#define MATRIX_ACCESSOR_HPP_MSDA

#include <array>

template <uint64_t C, uint64_t R> struct row_policy {
    inline static uint64_t index(uint64_t i, uint64_t j) {
        return i + R * j;
    }
    inline static uint64_t end() {
        return C;
    }
};

template <uint64_t C, uint64_t R> struct column_policy {
    inline static uint64_t index(uint64_t i, uint64_t j) {
        return j + R * i;
    }
    inline static uint64_t end() {
        return R;
    }
};

template <uint64_t C, uint64_t R, class T,
          template <uint64_t, uint64_t> class P>
class matrix_accessor {
    const std::array<T, C * R>& matrix;
    uint64_t i;

    class iterator {
        const matrix_accessor& p;
        uint64_t j;

    public:
        iterator(const matrix_accessor& p, uint64_t j = 0) : p(p), j(j) {
        }

        T operator*() {
            return p.matrix[P<C, R>::index(p.i, j)];
        }

        bool operator==(const iterator& other) {
            return j == other.j && &p == &other.p;
        }
        bool operator!=(const iterator& other) {
            return !(*this == other);
        }
        iterator operator++() {
            ++j;
            return *this;
        }
    };

public:
    iterator begin() {
        return iterator(*this);
    }

    iterator end() {
        return iterator(*this, P<C, R>::end());
    }

public:
    matrix_accessor(std::array<T, C * R>& matrix, uint64_t in)
        : matrix(matrix), i(in) {
    }
};

#endif
