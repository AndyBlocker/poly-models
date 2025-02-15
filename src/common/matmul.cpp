#include "common/matmul.hpp"

void matmul(const float *A, const float *B, float *C,
            int M, int K, int N)
{
    ScopedTimer timer(OpType::MATMUL);
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            double sum = 0.0;
            for (int k = 0; k < K; k++)
            {
                sum += (double)A[m * K + k] * (double)B[k * N + n];
            }
            C[m * N + n] = (float)sum;
        }
    }
}