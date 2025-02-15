#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

#include "common/tensor.hpp"
#include "common/time_utils.hpp"

void matmul(const float *A, const float *B, float *C,
            int M, int K, int N);

#endif // __MATMUL_HPP__