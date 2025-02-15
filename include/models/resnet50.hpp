#ifndef __RESNET50_HPP__
#define __RESNET50_HPP__

#include "common/tensor.hpp"
#include "layers/conv2d.hpp"
#include "layers/pool2d.hpp"
#include "layers/batchnorm.hpp"
#include "layers/relu.hpp"
#include "layers/linear.hpp"
#include "layers/softmax.hpp"
#include <vector>
#include <memory>

struct Bottleneck {
    Tensor<float> w1;       // shape [planes, in_planes, 1, 1]
    std::vector<float> b1;
    BNParam bn1;

    Tensor<float> w2;       // shape [planes, planes, 3, 3]
    std::vector<float> b2;
    BNParam bn2;

    Tensor<float> w3;       // shape [planes*4, planes, 1, 1]
    std::vector<float> b3;
    BNParam bn3;

    bool use_downsample = false;
    Tensor<float> w_down;
    std::vector<float> b_down;
    BNParam bn_down;

    int stride = 1;

    // 前向
    Tensor<float> forward(const Tensor<float> &x) const;
};

class ResNet50 {
public:
    ResNet50();  

    Tensor<float> forward(const Tensor<float> &input);

private:
    Tensor<float> conv1_w_;
    std::vector<float> conv1_b_;
    BNParam bn1_;

    std::vector<Bottleneck> layer1_;  // 3 blocks
    std::vector<Bottleneck> layer2_;  // 4 blocks
    std::vector<Bottleneck> layer3_;  // 6 blocks
    std::vector<Bottleneck> layer4_;  // 3 blocks

    LinearParam fc_;   // [1000, 2048], bias[1000]

    Bottleneck make_bottleneck(int inplanes, int planes, int stride, bool downsample);
    std::vector<Bottleneck> make_layer(int inplanes, int planes, int blocks, int stride);

    int current_inplanes_;
};

#endif