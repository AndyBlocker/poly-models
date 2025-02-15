#ifndef __CONV2D_HPP__
#define __CONV2D_HPP__

#include "common/tensor.hpp"
#include "common/matmul.hpp"
#include <vector>

struct Conv2DParam
{
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 0;
    int pad_w = 0;
};

struct DepthwiseConv2DParam
{
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 0;
    int pad_w = 0;
};

Tensor<float> conv2d(const Tensor<float> &input,
                     const Tensor<float> &weight,
                     const std::vector<float> &bias,
                     const Conv2DParam &param);

Tensor<float> depthwise_conv2d_im2col(const Tensor<float> &input,
                                      const Tensor<float> &weight,
                                      const std::vector<float> &bias,
                                      int stride_h, int stride_w,
                                      int pad_h, int pad_w);

#endif