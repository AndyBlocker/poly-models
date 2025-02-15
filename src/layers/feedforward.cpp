#include "layers/feedforward.hpp"
#include "common/time_utils.hpp"
#include "common/matmul.hpp" // use global matmul
#include <cmath>

/**
 * feed_forward:
 *  shape: x [N, S, D]
 *  out = ( (x*W1 + b1) relu ) * W2 + b2
 */
Tensor<float> feed_forward(const Tensor<float> &x, const FFParam &param)
{
    ScopedTimer t_ff(OpType::OTHERS);

    int N = x.shape()[0];
    int S = x.shape()[1];
    int D = x.shape()[2];
    int D4 = param.W1.shape()[1]; // 4*D

    // flatten => [N*S, D]
    Tensor<float> inp2d({N * S, D});
    for (int n = 0; n < N; n++)
    {
        for (int s = 0; s < S; s++)
        {
            for (int d = 0; d < D; d++)
            {
                inp2d.at4d(n * S + s, d, 0, 0) = x.at4d(n, s, d, 0);
            }
        }
    }

    // first matmul => [N*S, D4]
    Tensor<float> hidden({N * S, D4});
    {
        // param.W1: [D, D4]
        const float *Ap = inp2d.data();
        const float *Bp = param.W1.data();
        float *Cp = hidden.data();
        ::matmul(Ap, Bp, Cp, N * S, D, D4);

        // add b1 + relu
        for (int i = 0; i < (N * S); i++)
        {
            for (int c = 0; c < D4; c++)
            {
                float v = hidden.at4d(i, c, 0, 0) + param.b1[c];
                if (v < 0.f)
                    v = 0.f; // ReLU
                hidden.at4d(i, c, 0, 0) = v;
            }
        }
    }

    // second matmul => [N*S, D]
    Tensor<float> out2d({N * S, D});
    {
        const float *Ap = hidden.data();
        const float *Bp = param.W2.data();
        float *Cp = out2d.data();
        ::matmul(Ap, Bp, Cp, N * S, D4, D);

        // add b2
        for (int i = 0; i < (N * S); i++)
        {
            for (int c = 0; c < D; c++)
            {
                float v = out2d.at4d(i, c, 0, 0) + param.b2[c];
                out2d.at4d(i, c, 0, 0) = v;
            }
        }
    }

    // reshape => [N,S,D]
    Tensor<float> out({N, S, D});
    for (int n = 0; n < N; n++)
    {
        for (int s = 0; s < S; s++)
        {
            for (int d = 0; d < D; d++)
            {
                out.at4d(n, s, d, 0) = out2d.at4d(n * S + s, d, 0, 0);
            }
        }
    }
    return out;
}