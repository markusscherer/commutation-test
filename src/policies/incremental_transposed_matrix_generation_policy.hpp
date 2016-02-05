#ifndef INCREMENTAL_TRANSPOSED_MATRIX_GENERATION_POLICY_HPP_MSDA
#define INCREMENTAL_TRANSPOSED_MATRIX_GENERATION_POLICY_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

template <uint64_t D, uint64_t A1, uint64_t A2>
struct incremental_transposed_matrix_generation_policy {};

template <> struct incremental_transposed_matrix_generation_policy<4, 3, 3> {
    struct constants {
        const __m128i const0001;
        const __m128i const0010;
        const __m128i const00_44;
        const __m128i const001_4;
        const __m128i const004_16;
        const __m128i const00_160;
        const __m128i const0_1600;
        const __m128i const0100;
        const __m128i const00_40;

        constants()
            : const0001(_mm_set_epi32(1, 1, 1, 1)),
              const0010(
                  _mm_set_epi8(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)),
              const00_44(
                  _mm_set_epi8(0, 0, -4, 4, 0, 0, -4, 4, 0, 0, -4, 4, 0, 0, -4, 4)),
              const001_4(
                  _mm_set_epi8(0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4)),
              const004_16(_mm_set_epi8(0, 0, 4, -16, 0, 0, 4, -16, 0, 0, 4, -16, 0,
                                       0, 4, -16)),
              const00_160(_mm_set_epi8(0, 0, -16, 0, 0, 0, -16, 0, 0, 0, -16, 0, 0,
                                       0, -16, 0)),
              const0_1600(_mm_set_epi8(0, -16, 0, 0, 0, -16, 0, 0, 0, -16, 0, 0, 0,
                                       -16, 0, 0)),
              const0100(
                  _mm_set_epi8(0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)),
              const00_40(_mm_set_epi8(0, 0, -4, 0, 0, 0, -4, 0, 0, 0, -4, 0, 0, 0,
                                      -4, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl, __m128i& math) {
        const constants c;
        next_matrix(0, matl, math, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        math = _mm_sub_epi8(_mm_setzero_si128(), math);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math, const constants& c) {
        matl = _mm_add_epi8(matl, c.const0010);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const00_44);

            if (matcount % (1 << 6) == 0) {
                matl = _mm_add_epi8(matl, c.const004_16);

                if (matcount % (1 << 8) == 0) {
                    math = _mm_add_epi8(math, c.const0001);
                    matl = _mm_add_epi8(matl, c.const00_160);

                    if (matcount % (1 << 10) == 0) {
                        math = _mm_add_epi8(math, c.const001_4);

                        if (matcount % (1 << 12) == 0) {
                            math = _mm_add_epi8(math, c.const00_40);
                            matl = _mm_add_epi8(matl, c.const0100);

                            if (matcount % (1 << 16) == 0) {
                                matl = _mm_add_epi8(matl, c.const0_1600);
                                math = _mm_add_epi8(math, c.const0100);
                            }
                        }
                    }
                }
            }
        }
    }
};

template <> struct incremental_transposed_matrix_generation_policy<4, 2, 4> {
    struct constants {
        const __m128i const0001;
        const __m128i const0010;
        const __m128i const00_44;
        const __m128i const001_4;
        const __m128i const004_16;
        const __m128i const00_160;

        constants()
            : const0001(_mm_set_epi32(1, 1, 1, 1)),
              const0010(
                  _mm_set_epi8(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)),
              const00_44(
                  _mm_set_epi8(0, 0, -4, 4, 0, 0, -4, 4, 0, 0, -4, 4, 0, 0, -4, 4)),
              const001_4(
                  _mm_set_epi8(0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4)),
              const004_16(_mm_set_epi8(0, 0, 4, -16, 0, 0, 4, -16, 0, 0, 4, -16, 0,
                                       0, 4, -16)),
              const00_160(_mm_set_epi8(0, 0, -16, 0, 0, 0, -16, 0, 0, 0, -16, 0, 0,
                                       0, -16, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl, __m128i& math) {
        const constants c;
        next_matrix(0, matl, math, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        math = _mm_sub_epi8(_mm_setzero_si128(), math);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math, const constants& c) {
        matl = _mm_add_epi8(matl, c.const0010);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const00_44);

            if (matcount % (1 << 6) == 0) {
                matl = _mm_add_epi8(matl, c.const004_16);

                if (matcount % (1 << 8) == 0) {
                    math = _mm_add_epi8(math, c.const0001);
                    matl = _mm_add_epi8(matl, c.const00_160);

                    if (matcount % (1 << 10) == 0) {
                        math = _mm_add_epi8(math, c.const001_4);

                        if (matcount % (1 << 12) == 0) {
                            math = _mm_add_epi8(math, c.const00_44);

                            if (matcount % (1 << 14) == 0) {
                                math = _mm_add_epi8(math, c.const004_16);
                            }
                        }
                    }
                }
            }
        }
    }
};

template <> struct incremental_transposed_matrix_generation_policy<4, 3, 4> {
    struct constants {
        const __m128i const0001;
        const __m128i const0010;
        const __m128i const00_44;
        const __m128i const001_4;
        const __m128i const004_16;
        const __m128i const00_160;
        const __m128i const0_1600;
        const __m128i const0100;

        constants()
            : const0001(_mm_set_epi32(1, 1, 1, 1)),
              const0010(
                  _mm_set_epi8(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)),
              const00_44(
                  _mm_set_epi8(0, 0, -4, 4, 0, 0, -4, 4, 0, 0, -4, 4, 0, 0, -4, 4)),
              const001_4(
                  _mm_set_epi8(0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4)),
              const004_16(_mm_set_epi8(0, 0, 4, -16, 0, 0, 4, -16, 0, 0, 4, -16, 0,
                                       0, 4, -16)),
              const00_160(_mm_set_epi8(0, 0, -16, 0, 0, 0, -16, 0, 0, 0, -16, 0, 0,
                                       0, -16, 0)),
              const0_1600(_mm_set_epi8(0, -16, 0, 0, 0, -16, 0, 0, 0, -16, 0, 0, 0,
                                       -16, 0, 0)),
              const0100(
                  _mm_set_epi8(0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl, __m128i& math) {
        const constants c;
        next_matrix(0, matl, math, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        math = _mm_sub_epi8(_mm_setzero_si128(), math);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math, const constants& c) {
        matl = _mm_add_epi8(matl, c.const0010);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const00_44);

            if (matcount % (1 << 6) == 0) {
                matl = _mm_add_epi8(matl, c.const004_16);

                if (matcount % (1 << 8) == 0) {
                    math = _mm_add_epi8(math, c.const0001);
                    matl = _mm_add_epi8(matl, c.const00_160);

                    if (matcount % (1 << 10) == 0) {
                        math = _mm_add_epi8(math, c.const001_4);

                        if (matcount % (1 << 12) == 0) {
                            math = _mm_add_epi8(math, c.const00_44);

                            if (matcount % (1 << 14) == 0) {
                                math = _mm_add_epi8(math, c.const004_16);

                                if (matcount % (1 << 16) == 0) {
                                    matl = _mm_add_epi8(matl, c.const0100);
                                    math = _mm_add_epi8(math, c.const00_160);

                                    if (matcount % (1 << 20) == 0) {
                                        matl = _mm_add_epi8(matl, c.const0_1600);
                                        math = _mm_add_epi8(math, c.const0100);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

template <> struct incremental_transposed_matrix_generation_policy<4, 4, 4> {
    struct constants {
        const __m128i const0001;
        const __m128i const0010;
        const __m128i const00_44;
        const __m128i const001_4;
        const __m128i const1_400;
        const __m128i const004_16;
        const __m128i const00_160;
        const __m128i const_4400;
        const __m128i const4_1600;
        const __m128i const_16000;
        const __m128i const0100;

        constants()
            : const0001(_mm_set_epi32(1, 1, 1, 1)),
              const0010(
                  _mm_set_epi8(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)),
              const00_44(
                  _mm_set_epi8(0, 0, -4, 4, 0, 0, -4, 4, 0, 0, -4, 4, 0, 0, -4, 4)),
              const001_4(
                  _mm_set_epi8(0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4)),
              const1_400(
                  _mm_set_epi8(1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0)),
              const004_16(_mm_set_epi8(0, 0, 4, -16, 0, 0, 4, -16, 0, 0, 4, -16, 0,
                                       0, 4, -16)),
              const00_160(_mm_set_epi8(0, 0, -16, 0, 0, 0, -16, 0, 0, 0, -16, 0, 0,
                                       0, -16, 0)),
              const_4400(
                  _mm_set_epi8(-4, 4, 0, 0, -4, 4, 0, 0, -4, 4, 0, 0, -4, 4, 0, 0)),
              const4_1600(_mm_set_epi8(4, -16, 0, 0, 4, -16, 0, 0, 4, -16, 0, 0, 4,
                                       -16, 0, 0)),
              const_16000(_mm_set_epi8(-16, 0, 0, 0, -16, 0, 0, 0, -16, 0, 0, 0,
                                       -16, 0, 0, 0)),
              const0100(
                  _mm_set_epi8(0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl, __m128i& math) {
        const constants c;
        next_matrix(0, matl, math, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        math = _mm_sub_epi8(_mm_setzero_si128(), math);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math, const constants& c) {
        matl = _mm_add_epi8(matl, c.const0010);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const00_44);

            if (matcount % (1 << 6) == 0) {
                matl = _mm_add_epi8(matl, c.const004_16);

                if (matcount % (1 << 8) == 0) {
                    math = _mm_add_epi8(math, c.const0001);
                    matl = _mm_add_epi8(matl, c.const00_160);

                    if (matcount % (1 << 10) == 0) {
                        math = _mm_add_epi8(math, c.const001_4);

                        if (matcount % (1 << 12) == 0) {
                            math = _mm_add_epi8(math, c.const00_44);

                            if (matcount % (1 << 14) == 0) {
                                math = _mm_add_epi8(math, c.const004_16);

                                if (matcount % (1 << 16) == 0) {
                                    matl = _mm_add_epi8(matl, c.const0100);
                                    math = _mm_add_epi8(math, c.const00_160);

                                    if (matcount % (1 << 18) == 0) {
                                        matl = _mm_add_epi8(matl, c.const1_400);

                                        if (matcount % (1 << 20) == 0) {
                                            matl = _mm_add_epi8(matl, c.const_4400);

                                            if (matcount % (1 << 22) == 0) {
                                                matl = _mm_add_epi8(matl, c.const4_1600);

                                                if (matcount % (1 << 24) == 0) {
                                                    math = _mm_add_epi8(math, c.const0100);
                                                    matl = _mm_add_epi8(matl, c.const_16000);

                                                    if (matcount % (1 << 26) == 0) {
                                                        math = _mm_add_epi8(math, c.const1_400);

                                                        if (matcount % (1 << 28) == 0) {
                                                            math = _mm_add_epi8(math, c.const_4400);

                                                            if (matcount % (1 << 30) == 0) {
                                                                math = _mm_add_epi8(math, c.const4_1600);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

#endif
