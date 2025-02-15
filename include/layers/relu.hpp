#ifndef __RELU_HPP__
#define __RELU_HPP__

#include "common/tensor.hpp"

// ReLU
Tensor<float> relu(const Tensor<float> &input);
Tensor<float> relu6(const Tensor<float> &input);

#endif