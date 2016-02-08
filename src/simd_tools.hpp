#ifndef SIMD_TOOLS_HPP_MSDA
#define SIMD_TOOLS_HPP_MSDA

#include <iostream>
#include <iomanip>
#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

template <class T, uint64_t S, uint64_t C>
inline void array_to_si128_impl(std::array<T, S>& arr) {
    static_assert(S == C, "Array must have the right size!");
}

template <class T, uint64_t S, uint64_t C, class R, class... RR>
inline void array_to_si128_impl(std::array<T, S>& arr, R& reg, RR& ... rest) {
    static_assert(sizeof(T) * (S - C) == sizeof(R) * (sizeof...(rest) + 1),
                  "Array must have the right size!");

    const uint64_t elements_per_register = sizeof(R) / sizeof(T);

    reg = _mm_load_si128(reinterpret_cast<__m128i *>(arr.data() + C));
    array_to_si128_impl<T, S, C + elements_per_register, RR...>(arr, rest...);
}

template <class T, uint64_t S, class R, class... RR>
inline void array_to_si128(std::array<T, S>& arr, R& reg, RR& ... rest) {
    static_assert(sizeof(T) * S == sizeof(R) * (sizeof...(rest) + 1),
                  "Array must have the right size!");

    array_to_si128_impl<T, S, 0, R, RR...>(arr, reg, rest...);
}

template <typename T> void print_register(__m128i r) {
    const int size = 16 / sizeof(T);
    T buff[size];
    __m128i *dd = reinterpret_cast<__m128i *>(buff);

    _mm_storeu_si128(dd, r);

    for (int i = size - 1; i >= 0; --i) {
        std::cout << (uint64_t)buff[i] << " ";
    }

    std::cout << std::endl;
}

inline uint32_t ems2b(uint8_t i) {
    return ((i & 0x0C) >> 2);
}

inline uint32_t els2b(uint8_t i) {
    return (i & 0x03);
}

template <uint64_t A1, uint64_t A2>
void print_matrices(__m128i matl, __m128i math) {
    uint8_t b1[16];
    uint8_t b2[16];
    __m128i *p1 = reinterpret_cast<__m128i *>(b1);
    __m128i *p2 = reinterpret_cast<__m128i *>(b2);

    _mm_storeu_si128(p1, math);
    _mm_storeu_si128(p2, matl);

    std::cout << std::setw(2);

    switch (A1) {
        case 4:
            for (int i = 15; i >= 0; --i) {
                if (i % 4 == 3) {
                    std::cout << "   ";
                }

                if (static_cast<uint32_t>(i) % 4 >= A2) {
                    std::cout << "  ";
                }

                std::cout << ems2b(b1[i]) << " ";
            }

            std::cout << std::endl;

        case 3:
            for (int i = 15; i >= 0; --i) {
                if (i % 4 == 3) {
                    std::cout << "   ";
                }

                if (static_cast<uint32_t>(i) % 4 >= A2) {
                    std::cout << "  ";
                } else {
                    std::cout << els2b(b1[i]) << " ";
                }
            }

            std::cout << std::endl;

        case 2:
            for (int i = 15; i >= 0; --i) {
                if (i % 4 == 3) {
                    std::cout << "   ";
                }

                if (static_cast<uint32_t>(i) % 4 >= A2) {
                    std::cout << "  ";
                } else {
                    std::cout << ems2b(b2[i]) << " ";
                }
            }

            std::cout << std::endl;

        case 1:
            for (int i = 15; i >= 0; --i) {
                if (i % 4 == 3) {
                    std::cout << "   ";
                }

                if (static_cast<uint32_t>(i) % 4 >= A2) {
                    std::cout << "  ";
                } else {
                    std::cout << els2b(b2[i]) << " ";
                }
            }

            std::cout << std::endl;
    }
}

template <uint64_t A1, uint64_t A2>
void print_transposed_matrices(__m128i matl, __m128i math) {
    uint8_t b1[16];
    uint8_t b2[16];
    __m128i *p1 = reinterpret_cast<__m128i *>(b1);
    __m128i *p2 = reinterpret_cast<__m128i *>(b2);

    _mm_storeu_si128(p1, math);
    _mm_storeu_si128(p2, matl);

    std::cout << std::setw(2);

    for (int j = (A1 - 1); j >= 0; --j) {
        for (int i = 3; i >= 0; --i) {
            const int c = i * 4 + j;
            std::cout << "   ";

            for (int l = A2; l < 4; ++l) {
                std::cout << "  ";
            }

            switch (A2) {
                case 4:
                    std::cout << ems2b(b1[c]);
                    std::cout << " ";

                case 3:
                    std::cout << els2b(b1[c]);
                    std::cout << " ";

                case 2:
                    std::cout << ems2b(b2[c]);
                    std::cout << " ";

                case 1:
                    std::cout << els2b(b2[c]);
                    std::cout << " ";
            }
        }

        std::cout << std::endl;
    }
}

void print_result(__m128i res) {
    uint8_t b1[16];
    __m128i *p1 = reinterpret_cast<__m128i *>(b1);
    _mm_storeu_si128(p1, res);

    std::cout << std::setw(2);

    for (int i = 15; i >= 0; i--) {
        if (i % 4 == 3) {
            std::cout << "   ";
        }

        std::cout << static_cast<uint32_t>(b1[i]) << " ";
    }

    std::cout << std::endl;
}

template <uint64_t A1, uint64_t A2>
void print_matrices_and_results(__m128i res, __m128i matl, __m128i math) {
    print_matrices<A1, A2>(matl, math);
    std::cout << "-----------------------------------------------" << std::endl;
    print_result(res);
}

#endif
