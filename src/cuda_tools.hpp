#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << ":"
                  << line << std::endl;

        if (abort) {
            exit(code);
        }
    }
}
