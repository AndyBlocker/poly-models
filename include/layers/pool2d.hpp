#ifndef __POOL2D_HPP__
#define __POOL2D_HPP__

#include "common/tensor.hpp"

struct Pool2DParam {
    int kernel_h = 2;
    int kernel_w = 2;
    int stride_h = 2;
    int stride_w = 2;
    int pad_h = 0;
    int pad_w = 0;
};

Tensor<float> max_pool2d(const Tensor<float> &input, const Pool2DParam &param);

Tensor<float> avg_pool2d(const Tensor<float> &input, const Pool2DParam &param);

#endif