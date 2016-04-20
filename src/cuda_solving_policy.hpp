#pragma once

#include <cstdint>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cuda_tools.hpp"

#include "constants.hpp"
#include "sparse_array_function.hpp"

#include "misc_tools.hpp"

typedef uint32_t result_t;

__constant__ char df1[cpow(4, 4)];
__constant__ char df2[cpow(4, 4)];

__global__ void apply_function(uint8_t *C, uint64_t offset) {
    const unsigned int result_position = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int mat = result_position + offset;

    unsigned int r0 = ((mat >> 12) & 0xF0) | (mat & 0x0F);
    unsigned int r1 = ((mat >> 16) & 0xF0) | ((mat >> 4) & 0x0F);
    unsigned int r2 = ((mat >> 20) & 0xF0) | ((mat >> 8) & 0x0F);
    unsigned int r3 = ((mat >> 24) & 0xF0) | ((mat >> 12) & 0x0F);

    r0 = df1[r0];
    r1 = df1[r1];
    r2 = df1[r2];
    r3 = df1[r3];

    unsigned int ra = r0 | (r1 << 2) | (r2 << 4) | (r3 << 6);
    ra = df2[ra];

    r0 = ((mat >> 6) & 0xC0) | ((mat >> 4) & 0x30) | ((mat >> 2) & 0x0C) |
         (mat & 0x03);
    r1 = ((mat >> 8) & 0xC0) | ((mat >> 6) & 0x30) | ((mat >> 4) & 0x0C) |
         ((mat >> 2) & 0x03);
    r2 = ((mat >> 22) & 0xC0) | ((mat >> 20) & 0x30) | ((mat >> 18) & 0x0C) |
         ((mat >> 16) & 0x03);
    r3 = ((mat >> 24) & 0xC0) | ((mat >> 22) & 0x30) | ((mat >> 20) & 0x0C) |
         ((mat >> 18) & 0x03);

    r0 = df2[r0];
    r1 = df2[r1];
    r2 = df2[r2];
    r3 = df2[r3];

    unsigned int rb = r0 | (r1 << 2) | (r2 << 4) | (r3 << 6);
    rb = df1[rb];

    C[result_position] = (ra != rb);
}

template <uint64_t D, uint64_t A1, uint64_t A2> struct cuda_solving_policy {};

uint64_t apply_time = 0;
uint64_t reduce_time = 0;

template <> struct cuda_solving_policy<4, 4, 4> {

    static inline void init(uint64_t thread_count = 256,
                            uint64_t group_size = (1 << 30)) {
        cuda_solving_policy<4, 4, 4>::thread_count = thread_count;
        cuda_solving_policy<4, 4, 4>::group_size = group_size;

        host_result.resize(group_size / sizeof(result_t), 0);
        device_result = new thrust::device_vector<result_t>();
        *device_result = host_result;
    }

    static inline void deinit() {
        delete device_result;
    }

    static inline bool commutes(const sparse_array_function<4, 4>& f1,
                                const sparse_array_function<4, 4>& f2) {
        uint64_t runs = cpow<uint64_t>(4, 4 * 4) / group_size;

        gpuErrchk(cudaMemcpyToSymbol(df1, f1.storage.data(), cpow(4, 4)));
        gpuErrchk(cudaMemcpyToSymbol(df2, f2.storage.data(), cpow(4, 4)));

        for (uint64_t r = 0; r < runs; ++r) {
            const uint64_t offset = r * group_size;
            auto start = std::chrono::high_resolution_clock::now();
            apply_function<<<group_size / thread_count, thread_count>>>(
                reinterpret_cast<uint8_t *>(
                    thrust::raw_pointer_cast(device_result->data())),
                offset);
            auto end = std::chrono::high_resolution_clock::now();
            apply_time +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();

            start = std::chrono::high_resolution_clock::now();
            result_t red =
                thrust::reduce(device_result->begin(), device_result->end(), 0,
                               thrust::bit_or<result_t>());
            end = std::chrono::high_resolution_clock::now();
            reduce_time +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();

            if (red) {
                return false;
            }
        }

        return true;
    }

private:
    static uint64_t thread_count;
    static uint64_t group_size;
    static thrust::host_vector<result_t> host_result;
    static thrust::device_vector<result_t> *device_result;
};

uint64_t cuda_solving_policy<4, 4, 4>::thread_count;
uint64_t cuda_solving_policy<4, 4, 4>::group_size;
thrust::host_vector<result_t> cuda_solving_policy<4, 4, 4>::host_result;
thrust::device_vector<result_t> *cuda_solving_policy<4, 4, 4>::device_result;
