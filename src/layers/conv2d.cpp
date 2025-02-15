#include "layers/conv2d.hpp"
#include "common/time_utils.hpp"
#include <cmath>
#include <cstring>

namespace
{
    // im2col
    Tensor<float> im2col(const Tensor<float> &input,
                         int kernel_h, int kernel_w,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w)
    {
        ScopedTimer timer(OpType::IM2COL);

        int N = input.shape()[0];
        int C = input.shape()[1];
        int H = input.shape()[2];
        int W = input.shape()[3];

        int out_h = (H + 2 * pad_h - kernel_h) / stride_h + 1;
        int out_w = (W + 2 * pad_w - kernel_w) / stride_w + 1;

        // [N, C*kernel_h*kernel_w, out_h*out_w]
        Tensor<float> col(std::vector<int>{N, C * kernel_h * kernel_w, out_h * out_w});

        for (int n = 0; n < N; n++)
        {
            float *col_data_n = col.data() + n * (col.shape()[1] * col.shape()[2]);

            for (int c_in = 0; c_in < C; c_in++)
            {
                for (int kh = 0; kh < kernel_h; kh++)
                {
                    for (int kw = 0; kw < kernel_w; kw++)
                    {
                        int row_idx = c_in * kernel_h * kernel_w + kh * kernel_w + kw;
                        for (int oh = 0; oh < out_h; oh++)
                        {
                            int ih = oh * stride_h + kh - pad_h;
                            for (int ow = 0; ow < out_w; ow++)
                            {
                                int iw = ow * stride_w + kw - pad_w;
                                float val = 0.f;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                {
                                    val = input.at4d(n, c_in, ih, iw);
                                }
                                int col_idx = row_idx * (out_h * out_w) + (oh * out_w + ow);
                                col_data_n[col_idx] = val;
                            }
                        }
                    }
                }
            }
        }
        return col;
    }
}

Tensor<float> conv2d(const Tensor<float> &input,
                     const Tensor<float> &weight,
                     const std::vector<float> &bias,
                     const Conv2DParam &param)
{
    int N = input.shape()[0];
    int C_in = input.shape()[1];
    int H_in = input.shape()[2];
    int W_in = input.shape()[3];

    int C_out = weight.shape()[0];
    int kH = weight.shape()[2];
    int kW = weight.shape()[3];

    int out_h = (H_in + 2 * param.pad_h - kH) / param.stride_h + 1;
    int out_w = (W_in + 2 * param.pad_w - kW) / param.stride_w + 1;

    // 1) im2col
    Tensor<float> col = im2col(input, kH, kW,
                               param.stride_h, param.stride_w,
                               param.pad_h, param.pad_w);

    // K = C_in * kH * kW
    int K = C_in * kH * kW;
    Tensor<float> w2d({C_out, K});
    {
        const float *wp = weight.data();
        float *w2dp = w2d.data();
        for (int co = 0; co < C_out; co++)
        {
            for (int ci = 0; ci < C_in; ci++)
            {
                for (int kh = 0; kh < kH; kh++)
                {
                    for (int kw = 0; kw < kW; kw++)
                    {
                        int idx_in = ((co * C_in + ci) * kH + kh) * kW + kw;
                        int idx_out = co * K + (ci * kH * kW + kh * kW + kw);
                        w2dp[idx_out] = wp[idx_in];
                    }
                }
            }
        }
    }

    // [C_out x K] * [K x out_hw] => [C_out x out_hw]
    Tensor<float> output({N, C_out, out_h, out_w});
    {

        for (int n_i = 0; n_i < N; n_i++)
        {
            const float *B = col.data() + n_i * (K * (out_h * out_w));
            float *Out = output.data() + n_i * (C_out * out_h * out_w);

            matmul(w2d.data(), B, Out, C_out, K, out_h * out_w);

            // bias
            for (int co = 0; co < C_out; co++)
            {
                float b = bias[co];
                float *outptr = Out + co * (out_h * out_w);
                for (int idx = 0; idx < out_h * out_w; idx++)
                {
                    outptr[idx] += b;
                }
            }
        }
    }

    return output;
}

Tensor<float> depthwise_conv2d_im2col(const Tensor<float> &input,
                                      const Tensor<float> &weight,
                                      const std::vector<float> &bias,
                                      int stride_h, int stride_w,
                                      int pad_h, int pad_w)
{
    // input shape: [N, C_in, H_in, W_in]
    int N = input.shape()[0];
    int C_in = input.shape()[1];
    int H_in = input.shape()[2];
    int W_in = input.shape()[3];

    // weight shape: [C_in, 1, kH, kW]
    int kH = weight.shape()[2];
    int kW = weight.shape()[3];

    // 卷积输出大小
    int out_h = (H_in + 2 * pad_h - kH) / stride_h + 1;
    int out_w = (W_in + 2 * pad_w - kW) / stride_w + 1;

    // im2col => col shape = [N, (C_in*kH*kW), (out_h*out_w)]
    Tensor<float> col = im2col(input, kH, kW, stride_h, stride_w, pad_h, pad_w);

    // col.shape = [N, (C_in*kH*kW), (out_h*out_w)]

    //    weight[c_in, 0, :, :] => flatten => W_c( [1, kH*kW] )
    //    col_for_c: shape [N, kH*kW, out_h*out_w]

    Tensor<float> output(std::vector<int>{N, C_in, out_h, out_w});

    std::vector<float> w_c;
    w_c.resize(kH * kW);

    for (int c = 0; c < C_in; c++)
    {

        const float *wptr = weight.data() + c * kH * kW;
        for (int i = 0; i < kH * kW; i++)
        {
            w_c[i] = wptr[i];
        }

        float b_c = bias[c];

        for (int n_i = 0; n_i < N; n_i++)
        {
            int row_offset = c * kH * kW;
            const float *col_base_n = col.data() + n_i * (col.shape()[1] * col.shape()[2]);
            const float *col_for_c = col_base_n + row_offset * (out_h * out_w);

            float *out_ptr = output.data() + (n_i * C_in * out_h * out_w + c * out_h * out_w);

            // [1, kH*kW] * [kH*kW, out_h*out_w] => [1, out_h*out_w]
            std::vector<float> temp_out(out_h * out_w, 0.f);

            matmul(w_c.data(), col_for_c, temp_out.data(), 1, kH * kW, out_h * out_w);
            // bias
            for (int idx = 0; idx < out_h * out_w; idx++)
                temp_out[idx] += b_c;

            // out(n_i,c,oh,ow) => out_ptr[ oh*out_w + ow ]
            for (int idx = 0; idx < out_h * out_w; idx++)
                out_ptr[idx] = temp_out[idx];
        }
    }

    return output;
}