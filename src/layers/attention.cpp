#include "layers/attention.hpp"
#include "common/time_utils.hpp"
#include "common/matmul.hpp" // 使用全局matmul
#include <cmath>
#include <vector>

/**
 * multi_head_self_attention:
 *  1) Q/K/V
 *  2) split heads
 *  3) scaled dot-product attention
 *  4) concat
 *  5) final linear
 */
Tensor<float> multi_head_self_attention(const Tensor<float> &input,
                                        const MHAParam &param)
{
    ScopedTimer t_attn(OpType::OTHERS);

    int N = input.shape()[0]; // batch
    int S = input.shape()[1]; // seq_len
    int D = input.shape()[2]; // hidden_dim
    int h = param.num_heads;
    int d_h = D / h;

    // flatten input => [N*S, D]
    Tensor<float> inp2d({N * S, D});
    for (int n = 0; n < N; n++)
    {
        for (int s = 0; s < S; s++)
        {
            for (int d = 0; d < D; d++)
            {
                inp2d.at4d(n * S + s, d, 0, 0) = input.at4d(n, s, d, 0);
            }
        }
    }

    // inline lambda for linear transform
    auto linear_transform = [&](const Tensor<float> &x,
                                const Tensor<float> &W,
                                const std::vector<float> &b)
    {
        // x: [N*S, D], W: [D,D], b:[D]
        Tensor<float> out({N * S, D});
        for (int i = 0; i < (N * S); i++)
        {
            for (int c = 0; c < D; c++)
            {
                out.at4d(i, c, 0, 0) = 0.f;
            }
        }
        // call matmul => out2d
        // A= x.data(), shape(M= N*S, K=D)
        // B= W.data(), shape(K=D, N=D)
        // C= out.data(), shape(M= N*S, N= D)
        const float *Ap = x.data();
        const float *Bp = W.data();
        float *Cp = out.data();
        ::matmul(Ap, Bp, Cp, N * S, D, D);

        // add bias
        for (int i = 0; i < (N * S); i++)
        {
            for (int c = 0; c < D; c++)
            {
                float val = out.at4d(i, c, 0, 0) + b[c];
                out.at4d(i, c, 0, 0) = val;
            }
        }
        return out;
    };

    // Q,K,V
    Tensor<float> Q = linear_transform(inp2d, param.Wq, param.bq);
    Tensor<float> K = linear_transform(inp2d, param.Wk, param.bk);
    Tensor<float> V = linear_transform(inp2d, param.Wv, param.bv);

    // output buffer
    Tensor<float> out2d({N * S, D});
    // accumulate each head
    for (int head = 0; head < h; head++)
    {
        // slice Qh,Kh,Vh => [N*S, d_h]
        Tensor<float> Qh({N * S, d_h});
        Tensor<float> Kh({N * S, d_h});
        Tensor<float> Vh({N * S, d_h});
        for (int i = 0; i < (N * S); i++)
        {
            for (int dd = 0; dd < d_h; dd++)
            {
                Qh.at4d(i, dd, 0, 0) = Q.at4d(i, head * d_h + dd, 0, 0);
                Kh.at4d(i, dd, 0, 0) = K.at4d(i, head * d_h + dd, 0, 0);
                Vh.at4d(i, dd, 0, 0) = V.at4d(i, head * d_h + dd, 0, 0);
            }
        }
        // scores = Qh * Kh^T => [N*S, N*S]
        // 1) Kh^T => [d_h, N*S]
        Tensor<float> KhT({d_h, N * S});
        for (int i = 0; i < (N * S); i++)
        {
            for (int dd = 0; dd < d_h; dd++)
            {
                KhT.at4d(dd, i, 0, 0) = Kh.at4d(i, dd, 0, 0);
            }
        }
        // matmul => scores
        Tensor<float> scores({N * S, N * S});
        {
            const float *Ap = Qh.data();
            const float *Bp = KhT.data();
            float *Cp = scores.data();
            ::matmul(Ap, Bp, Cp, N * S, d_h, N * S);
        }
        // scale + softmax
        float scale = 1.0f / std::sqrt((float)d_h);
        for (int i = 0; i < (N * S); i++)
        {
            for (int j = 0; j < (N * S); j++)
            {
                float v = scores.at4d(i, j, 0, 0) * scale;
                scores.at4d(i, j, 0, 0) = v;
            }
        }
        // row-wise softmax
        for (int i = 0; i < (N * S); i++)
        {
            float m = -1e30f;
            for (int j = 0; j < (N * S); j++)
            {
                float v = scores.at4d(i, j, 0, 0);
                if (v > m)
                    m = v;
            }
            double sum_exp = 0.0;
            for (int j = 0; j < (N * S); j++)
            {
                double e = std::exp(scores.at4d(i, j, 0, 0) - m);
                sum_exp += e;
            }
            for (int j = 0; j < (N * S); j++)
            {
                double e = std::exp(scores.at4d(i, j, 0, 0) - m);
                float soft = (float)(e / sum_exp);
                scores.at4d(i, j, 0, 0) = soft;
            }
        }
        // multiply => scores * Vh => [N*S, d_h]
        // Vh^T => [d_h, N*S]
        Tensor<float> VhT({d_h, N * S});
        for (int i = 0; i < (N * S); i++)
        {
            for (int dd = 0; dd < d_h; dd++)
            {
                VhT.at4d(dd, i, 0, 0) = Vh.at4d(i, dd, 0, 0);
            }
        }
        Tensor<float> head_out({N * S, d_h});
        {
            const float *Ap = scores.data();
            const float *Bp = VhT.data();
            float *Cp = head_out.data();
            ::matmul(Ap, Bp, Cp, N * S, N * S, d_h);
        }
        // add to out2d
        for (int i = 0; i < (N * S); i++)
        {
            for (int dd = 0; dd < d_h; dd++)
            {
                float oldv = out2d.at4d(i, head * d_h + dd, 0, 0);
                float newv = oldv + head_out.at4d(i, dd, 0, 0);
                out2d.at4d(i, head * d_h + dd, 0, 0) = newv;
            }
        }
    }

    // final linear (Wo)
    {
        Tensor<float> tmp({N * S, D});
        {
            // matmul out2d * Wo => tmp
            // Wo: [D,D]
            const float *Ap = out2d.data();
            const float *Bp = param.Wo.data();
            float *Cp = tmp.data();
            ::matmul(Ap, Bp, Cp, N * S, D, D);
        }
        // add bo
        for (int i = 0; i < (N * S); i++)
        {
            for (int c = 0; c < D; c++)
            {
                float v = tmp.at4d(i, c, 0, 0) + param.bo[c];
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