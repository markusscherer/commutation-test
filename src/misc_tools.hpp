#ifndef MISC_TOOLS_HPP_MSDA
#define MISC_TOOLS_HPP_MSDA

#include <iostream>
#include <array>
#include <sstream>

template <typename T> std::string to_string(T t) {
    std::stringstream s;
    s << t;
    return s.str();
}

void check(bool b, std::string s) {
    if (!b) {
        std::cerr << "Critical Error: " << s << std::endl;
        exit(1);
    }
}

void expect_string(std::string expected) {
    std::string s;
    std::cin >> s;
    check(expected == s,
          "Expected \"" + expected + "\", got \"" + s + "\" instead.");
}

template <typename T> bool safe_read(T &x, std::istream &in = std::cin) {
    in >> x;
    check(in.good(), "In safe_read (most likely expected numeral).");
    return true;
}

template <typename T> bool try_read(T &x) {
    std::cin >> x;

    if (!std::cin.good()) {
        std::cin.clear();
        return false;
    }

    return true;
}

template <typename T, typename C = uint64_t>
void print_iterable(T x, std::ostream &out = std::cout) {
    for (auto it = x.begin(); it != x.end(); ++it) {
        out << static_cast<uint64_t>(*it) << " ";
    }

    out << std::endl;
}

template <uint64_t D, uint64_t A, typename ElementType = uint64_t>
void increment_array(std::array<ElementType, A> &args) {
    for (int64_t i = A - 1; i >= 0; --i) {
        ++args[i];

        if (args[i] < D) {
            break;
        }
    }

    for (uint64_t i = 0; i < A; ++i) {
        if (args[i] == D) {
            args[i] = 0;
        }
    }
}
#endif
