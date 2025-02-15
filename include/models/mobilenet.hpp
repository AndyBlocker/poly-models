#ifndef __MOBILENETV2_HPP__
#define __MOBILENETV2_HPP__

#include "common/tensor.hpp"
#include "layers/conv2d.hpp"
#include "layers/batchnorm.hpp"
#include "layers/linear.hpp"
#include "layers/softmax.hpp"
#include "layers/relu.hpp"
#include "layers/pool2d.hpp"
#include <vector>

struct InvertedResidual {
    int in_channels;
    int out_channels;
    int stride;
    int expand_ratio;

    Tensor<float> w_expand;
    std::vector<float> b_expand;
    BNParam bn_expand;

    Tensor<float> w_dwise;
    std::vector<float> b_dwise;
    BNParam bn_dwise;

    Tensor<float> w_project;
    std::vector<float> b_project;
    BNParam bn_project;

    Tensor<float> forward(const Tensor<float>& x) const;
};

class MobileNetV2 {
public:
    MobileNetV2();
    Tensor<float> forward(const Tensor<float> &input);

private:
    Tensor<float> first_conv_w_;
    std::vector<float> first_conv_b_;
    BNParam first_conv_bn_;

    std::vector<InvertedResidual> blocks_;

    Tensor<float> last_conv_w_;
    std::vector<float> last_conv_b_;
    BNParam last_conv_bn_;

    LinearParam fc_;

    int current_channels_;

    InvertedResidual make_inverted_residual(int in_c,int out_c,int stride,int expand_ratio);
};

#endif