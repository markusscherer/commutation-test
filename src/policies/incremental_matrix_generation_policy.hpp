#ifndef INCREMENTAL_MATRIX_GENERATION_POLICY_HPP_MSDA
#define INCREMENTAL_MATRIX_GENERATION_POLICY_HPP_MSDA

#include <cstdint>

#include "emmintrin.h"
#include "immintrin.h"
#include "tmmintrin.h"

#include "../constants.hpp"

template <uint64_t D, uint64_t A1, uint64_t A2>
struct incremental_matrix_generation_policy {};

template <> struct incremental_matrix_generation_policy<3, 1, 2> {
    const static uint64_t matrices_per_step = 3;
    struct constants {
        const __m128i const0010;

        constants()
            : const0010(
                  _mm_set_epi8(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi32(0, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {

        matl = _mm_add_epi8(matl, c.const0010);
    }
};

template <> struct incremental_matrix_generation_policy<3, 2, 2> {
    const static uint64_t matrices_per_step = 3;
    struct constants {
        const __m128i const0004;
        const __m128i const001_12;
        const __m128i const0010;
        constants()
            : const0004(
                  _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4)),
              const001_12(_mm_set_epi8(0, 0, 0, 0, 0, 0, 1, -12, 0, 0, 1, -12, 0, 0,
                                       1, -12)),
              const0010(
                  _mm_set_epi8(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi32(0, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {

        matl = _mm_add_epi8(matl, c.const0004);

        if (matcount % cpow(3, 2) == 0) {
            matl = _mm_add_epi8(matl, c.const001_12);

            if (matcount % cpow(3, 3) == 0) {
                matl = _mm_add_epi8(matl, c.const0010);
            }
        }
    }
};

template <> struct incremental_matrix_generation_policy<3, 1, 3> {
    const static uint64_t matrices_per_step = 3;
    struct constants {
        const __m128i const0010;
        const __m128i const001_3;
        const __m128i const0100;

        constants()
            : const0010(
                  _mm_set_epi8(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)),
              const001_3(
                  _mm_set_epi8(0, 0, 0, 0, 0, 1, -3, 0, 0, 1, -3, 0, 0, 1, -3, 0)),
              const0100(
                  _mm_set_epi8(0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi32(0, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {

        matl = _mm_add_epi8(matl, c.const0010);

        if (matcount % cpow(3, 2) == 0) {
            matl = _mm_add_epi8(matl, c.const001_3);

            if (matcount % cpow(3, 3) == 0) {
                matl = _mm_add_epi8(matl, c.const0100);
            }
        }
    }
};

template <> struct incremental_matrix_generation_policy<3, 2, 3> {
    const static uint64_t matrices_per_step = 3;
    struct constants {
        const __m128i const0010;
        const __m128i const0004;
        const __m128i const001_12;
        const __m128i const01_120;
        const __m128i const0100;

        constants()
            : const0010(
                  _mm_set_epi8(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)),
              const0004(
                  _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4)),
              const001_12(_mm_set_epi8(0, 0, 0, 0, 0, 0, 1, -12, 0, 0, 1, -12, 0, 0,
                                       1, -12)),
              const01_120(_mm_set_epi8(0, 0, 0, 0, 0, 1, -12, 0, 0, 1, -12, 0, 0, 1,
                                       -12, 0)),
              const0100(
                  _mm_set_epi8(0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi32(0, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {

        matl = _mm_add_epi8(matl, c.const0004);

        if (matcount % cpow(3, 2) == 0) {
            matl = _mm_add_epi8(matl, c.const001_12);

            if (matcount % cpow(3, 3) == 0) {
                matl = _mm_add_epi8(matl, c.const0010);

                if (matcount % cpow(3, 4) == 0) {
                    matl = _mm_add_epi8(matl, c.const01_120);

                    if (matcount % cpow(3, 5) == 0) {
                        matl = _mm_add_epi8(matl, c.const0100);
                    }
                }
            }
        }
    }
};

template <> struct incremental_matrix_generation_policy<3, 3, 3> {
    const static uint64_t matrices_per_step = 3;
    struct constants {
        const __m128i const0010;
        const __m128i const0004;
        const __m128i const001_12;
        const __m128i const01_120;
        const __m128i const0100;
        const __m128i const0_1200;
        const __m128i const0001;
        const __m128i const001_3;
        const __m128i const01_30;

        constants()
            : const0010(
                  _mm_set_epi8(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)),
              const0004(
                  _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4)),
              const001_12(_mm_set_epi8(0, 0, 0, 0, 0, 0, 1, -12, 0, 0, 1, -12, 0, 0,
                                       1, -12)),
              const01_120(_mm_set_epi8(0, 0, 0, 0, 0, 1, -12, 0, 0, 1, -12, 0, 0, 1,
                                       -12, 0)),
              const0100(
                  _mm_set_epi8(0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)),
              const0_1200(_mm_set_epi8(0, 0, 0, 0, 0, -12, 0, 0, 0, -12, 0, 0, 0,
                                       -12, 0, 0)),
              const0001(
                  _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)),

              const001_3(
                  _mm_set_epi8(0, 0, 0, 0, 0, 0, 1, -3, 0, 0, 1, -3, 0, 0, 1, -3)),
              const01_30(
                  _mm_set_epi8(0, 0, 0, 0, 0, 1, -3, 0, 0, 1, -3, 0, 0, 1, -3, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl, __m128i& math) {
        const constants c;
        matl = _mm_setzero_si128();
        math = _mm_setzero_si128();
        next_matrix(0, matl, math, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        math = _mm_sub_epi8(_mm_setzero_si128(), math);
        matl = _mm_add_epi8(matl, _mm_set_epi32(0, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math, const constants& c) {

        matl = _mm_add_epi8(matl, c.const0004);

        if (matcount % cpow(3, 2) == 0) {
            matl = _mm_add_epi8(matl, c.const001_12);

            if (matcount % cpow(3, 3) == 0) {
                matl = _mm_add_epi8(matl, c.const0010);

                if (matcount % cpow(3, 4) == 0) {
                    matl = _mm_add_epi8(matl, c.const01_120);

                    if (matcount % cpow(3, 5) == 0) {
                        matl = _mm_add_epi8(matl, c.const0100);

                        if (matcount % cpow(3, 6) == 0) {
                            matl = _mm_add_epi8(matl, c.const0_1200);
                            math = _mm_add_epi8(math, c.const0001);

                            if (matcount % cpow(3, 7) == 0) {
                                math = _mm_add_epi8(math, c.const001_3);

                                if (matcount % cpow(3, 8) == 0) {
                                    math = _mm_add_epi8(math, c.const01_30);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

template <> struct incremental_matrix_generation_policy<4, 1, 2> {
    const static uint64_t matrices_per_step = 4;
    struct constants {
        const __m128i const0010;

        constants()
            : const0010(
                  _mm_set_epi8(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {
        matl = _mm_add_epi8(matl, c.const0010);
    }
};

template <> struct incremental_matrix_generation_policy<4, 2, 2> {
    const static uint64_t matrices_per_step = 4;
    struct constants {
        const __m128i const0004;
        const __m128i const0010;

        constants()
            : const0004(_mm_set_epi32(4, 4, 4, 4)),
              const0010(_mm_set_epi8(0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0,
                                     1, -16)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {
        matl = _mm_add_epi8(matl, c.const0004);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const0010);
        }
    }
};

template <> struct incremental_matrix_generation_policy<4, 1, 3> {
    const static uint64_t matrices_per_step = 4;
    struct constants {
        const __m128i const0010;
        const __m128i const01_40;

        constants()
            : const0010(
                  _mm_set_epi8(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)),
              const01_40(_mm_set_epi8(0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1,
                                      -4, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {
        matl = _mm_add_epi8(matl, c.const0010);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const01_40);
        }
    }
};

template <> struct incremental_matrix_generation_policy<4, 2, 3> {
    const static uint64_t matrices_per_step = 4;
    struct constants {
        const __m128i const0004;
        const __m128i const0010;
        const __m128i const0100;

        constants()
            : const0004(_mm_set_epi32(4, 4, 4, 4)),
              const0010(_mm_set_epi8(0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0,
                                     1, -16)),
              const0100(_mm_set_epi8(0, 1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1,
                                     -16, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {
        matl = _mm_add_epi8(matl, c.const0004);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const0010);

            if (matcount % (1 << 8) == 0) {
                matl = _mm_add_epi8(matl, c.const0100);
            }
        }
    }
};

template <> struct incremental_matrix_generation_policy<4, 3, 3> {
    const static uint64_t matrices_per_step = 4;
    struct constants {
        const __m128i const0004;
        const __m128i const0001;
        const __m128i const0010;
        const __m128i const0100;
        const __m128i const0_1600;
        const __m128i const001_4;
        const __m128i const01_40;

        constants()
            : const0004(_mm_set_epi32(4, 4, 4, 4)),
              const0001(_mm_set_epi32(1, 1, 1, 1)),
              const0010(_mm_set_epi8(0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0,
                                     1, -16)),
              const0100(_mm_set_epi8(0, 1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1,
                                     -16, 0)),
              const0_1600(_mm_set_epi8(0, -16, 0, 0, 0, -16, 0, 0, 0, -16, 0, 0, 0,
                                       -16, 0, 0)),
              const001_4(
                  _mm_set_epi8(0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4)),
              const01_40(_mm_set_epi8(0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1,
                                      -4, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl, __m128i& math) {
        const constants c;
        matl = _mm_setzero_si128();
        math = _mm_setzero_si128();
        next_matrix(0, matl, math, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        math = _mm_sub_epi8(_mm_setzero_si128(), math);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math, const constants& c) {
        matl = _mm_add_epi8(matl, c.const0004);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const0010);

            if (matcount % (1 << 8) == 0) {
                matl = _mm_add_epi8(matl, c.const0100);

                if (matcount % (1 << 12) == 0) {
                    matl = _mm_add_epi8(matl, c.const0_1600);
                    math = _mm_add_epi8(math, c.const0001);

                    if (matcount % (1 << 14) == 0) {
                        math = _mm_add_epi8(math, c.const001_4);

                        if (matcount % (1 << 16) == 0) {
                            math = _mm_add_epi8(math, c.const01_40);
                        }
                    }
                }
            }
        }
    }
};

template <> struct incremental_matrix_generation_policy<4, 1, 4> {
    const static uint64_t matrices_per_step = 4;
    struct constants {
        const __m128i const0010;
        const __m128i const01_40;
        const __m128i const1_400;

        constants()
            : const0010(
                  _mm_set_epi8(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)),
              const01_40(
                  _mm_set_epi8(0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0)),
              const1_400(_mm_set_epi8(1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4,
                                      0, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {
        matl = _mm_add_epi8(matl, c.const0010);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const01_40);

            if (matcount % (1 << 6) == 0) {
                matl = _mm_add_epi8(matl, c.const1_400);
            }
        }
    }
};

template <> struct incremental_matrix_generation_policy<4, 2, 4> {
    const static uint64_t matrices_per_step = 4;
    struct constants {
        const __m128i const0004;
        const __m128i const0010;
        const __m128i const0100;
        const __m128i const1000;

        constants()
            : const0004(_mm_set_epi32(4, 4, 4, 4)),
              const0010(_mm_set_epi8(0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0,
                                     1, -16)),
              const0100(_mm_set_epi8(0, 1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1,
                                     -16, 0)),
              const1000(_mm_set_epi8(1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1,
                                     -16, 0, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl) {
        const constants c;
        matl = _mm_setzero_si128();
        next_matrix(0, matl, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   const constants& c) {
        matl = _mm_add_epi8(matl, c.const0004);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const0010);

            if (matcount % (1 << 8) == 0) {
                matl = _mm_add_epi8(matl, c.const0100);

                if (matcount % (1 << 12) == 0) {
                    matl = _mm_add_epi8(matl, c.const1000);
                }
            }
        }
    }
};

template <> struct incremental_matrix_generation_policy<4, 3, 4> {
    const static uint64_t matrices_per_step = 4;
    struct constants {
        const __m128i const0004;
        const __m128i const0001;
        const __m128i const0010;
        const __m128i const0100;
        const __m128i const1000;
        const __m128i const_16000;
        const __m128i const001_4;
        const __m128i const01_40;
        const __m128i const1_400;

        constants()
            : const0004(_mm_set_epi32(4, 4, 4, 4)),
              const0001(_mm_set_epi32(1, 1, 1, 1)),
              const0010(_mm_set_epi8(0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0,
                                     1, -16)),
              const0100(_mm_set_epi8(0, 1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1,
                                     -16, 0)),
              const1000(_mm_set_epi8(1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1,
                                     -16, 0, 0)),
              const_16000(_mm_set_epi8(-16, 0, 0, 0, -16, 0, 0, 0, -16, 0, 0, 0,
                                       -16, 0, 0, 0)),
              const001_4(
                  _mm_set_epi8(0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4)),
              const01_40(
                  _mm_set_epi8(0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0)),
              const1_400(_mm_set_epi8(1, -4, 0, 0, 1, -4, 0, 0, 1, -4, 0, 0, 1, -4,
                                      0, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl, __m128i& math) {
        const constants c;
        matl = _mm_setzero_si128();
        math = _mm_setzero_si128();
        next_matrix(0, matl, math, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        math = _mm_sub_epi8(_mm_setzero_si128(), math);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math, const constants& c) {
        matl = _mm_add_epi8(matl, c.const0004);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const0010);

            if (matcount % (1 << 8) == 0) {
                matl = _mm_add_epi8(matl, c.const0100);

                if (matcount % (1 << 12) == 0) {
                    matl = _mm_add_epi8(matl, c.const1000);

                    if (matcount % (1 << 16) == 0) {
                        math = _mm_add_epi8(math, c.const0001);
                        matl = _mm_add_epi8(matl, c.const_16000);

                        if (matcount % (1 << 18) == 0) {
                            math = _mm_add_epi8(math, c.const001_4);

                            if (matcount % (1 << 20) == 0) {
                                math = _mm_add_epi8(math, c.const01_40);

                                if (matcount % (1 << 22) == 0) {
                                    math = _mm_add_epi8(math, c.const1_400);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

template <> struct incremental_matrix_generation_policy<4, 4, 4> {
    const static uint64_t matrices_per_step = 4;
    struct constants {
        const __m128i const0004;
        const __m128i const0001;
        const __m128i const0010;
        const __m128i const0100;
        const __m128i const1000;
        const __m128i const_16000;

        constants()
            : const0004(_mm_set_epi32(4, 4, 4, 4)),
              const0001(_mm_set_epi32(1, 1, 1, 1)),
              const0010(_mm_set_epi8(0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0,
                                     1, -16)),
              const0100(_mm_set_epi8(0, 1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1,
                                     -16, 0)),
              const1000(_mm_set_epi8(1, -16, 0, 0, 1, -16, 0, 0, 1, -16, 0, 0, 1,
                                     -16, 0, 0)),
              const_16000(_mm_set_epi8(-16, 0, 0, 0, -16, 0, 0, 0, -16, 0, 0, 0,
                                       -16, 0, 0, 0)) {
        }
    };

    inline static void init_matrix(__m128i& matl, __m128i& math) {
        const constants c;
        matl = _mm_setzero_si128();
        math = _mm_setzero_si128();
        next_matrix(0, matl, math, c);
        matl = _mm_sub_epi8(_mm_setzero_si128(), matl);
        math = _mm_sub_epi8(_mm_setzero_si128(), math);
        matl = _mm_add_epi8(matl, _mm_set_epi32(3, 2, 1, 0));
    }

    inline static void next_matrix(uint64_t matcount, __m128i& matl,
                                   __m128i& math, const constants& c) {
        matl = _mm_add_epi8(matl, c.const0004);

        if (matcount % (1 << 4) == 0) {
            matl = _mm_add_epi8(matl, c.const0010);

            if (matcount % (1 << 8) == 0) {
                matl = _mm_add_epi8(matl, c.const0100);

                if (matcount % (1 << 12) == 0) {
                    matl = _mm_add_epi8(matl, c.const1000);

                    if (matcount % (1 << 16) == 0) {
                        math = _mm_add_epi8(math, c.const0001);
                        matl = _mm_add_epi8(matl, c.const_16000);

                        if (matcount % (1 << 20) == 0) {
                            math = _mm_add_epi8(math, c.const0010);

                            if (matcount % (1 << 24) == 0) {
                                math = _mm_add_epi8(math, c.const0100);

                                if (matcount % (1 << 28) == 0) {
                                    math = _mm_add_epi8(math, c.const1000);
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
