#include "models/resnet50.hpp"
#include <iostream>
#include <cmath>

Tensor<float> Bottleneck::forward(const Tensor<float> &x) const
{
    // branch
    // 1x1 conv
    Conv2DParam p1; // stride=1
    Tensor<float> out = conv2d(x, w1, b1, p1);
    out = batchnorm2d(out, bn1);
    out = relu(out);

    // 3x3 conv
    Conv2DParam p2;
    p2.stride_h = stride;
    p2.stride_w = stride;
    p2.pad_h = 1;
    p2.pad_w = 1;
    out = conv2d(out, w2, b2, p2);
    out = batchnorm2d(out, bn2);
    out = relu(out);

    // 1x1 conv
    Conv2DParam p3; // stride=1
    Tensor<float> out3 = conv2d(out, w3, b3, p3);
    out3 = batchnorm2d(out3, bn3);

    // shortcut
    Tensor<float> shortcut = x;
    if(use_downsample) {
        Conv2DParam pd;
        pd.stride_h = stride;
        pd.stride_w = stride;
        Tensor<float> sc = conv2d(x, w_down, b_down, pd);
        sc = batchnorm2d(sc, bn_down);
        shortcut = sc;
    }

    // add
    // out3 + shortcut => out3
    int N = out3.shape()[0];
    int C = out3.shape()[1];
    int H = out3.shape()[2];
    int W = out3.shape()[3];
    for(int n=0; n<N; n++){
        for(int c=0; c<C; c++){
            for(int h=0; h<H; h++){
                for(int w=0; w<W; w++){
                    float val = out3.at4d(n,c,h,w) + shortcut.at4d(n,c,h,w);
                    out3.at4d(n,c,h,w) = val;
                }
            }
        }
    }

    // relu
    out3 = relu(out3);
    return out3;
}

ResNet50::ResNet50()
{
    // ========== conv1 ==========
    conv1_w_ = Tensor<float>(std::vector<int>{64, 3, 7, 7});
    conv1_b_.resize(64, 0.f);
    bn1_.gamma.resize(64, 1.f);
    bn1_.beta.resize(64, 0.f);
    bn1_.running_mean.resize(64, 0.f);
    bn1_.running_var.resize(64, 1.f);

    current_inplanes_ = 64;

    // layer1: inplanes = 64, planes = 64, blocks=3, stride=1
    layer1_ = make_layer(64, 64, 3, 1);

    // layer2: inplanes=256, planes=128, blocks=4, stride=2
    layer2_ = make_layer(256, 128, 4, 2);

    // layer3: inplanes=128*4=512, planes=256, blocks=6, stride=2
    layer3_ = make_layer(512, 256, 6, 2);

    // layer4: inplanes=256*4=1024, planes=512, blocks=3, stride=2
    layer4_ = make_layer(1024, 512, 3, 2);

    // 最终 fc => [1000, 2048], bias [1000]
    fc_.weight = Tensor<float>({1000, 2048});
    fc_.bias.resize(1000, 0.f);

}

Bottleneck ResNet50::make_bottleneck(int inplanes, int planes, int stride, bool downsample)
{
    // expansion=4 => outplanes = planes*4
    int outplanes = planes*4;

    Bottleneck block;
    block.stride = stride;

    // conv1: 1x1, (planes, inplanes)
    block.w1 = Tensor<float>(std::vector<int>{planes, inplanes, 1, 1});
    block.b1.resize(planes, 0.f);
    block.bn1.gamma.resize(planes, 1.f);
    block.bn1.beta.resize(planes, 0.f);
    block.bn1.running_mean.resize(planes, 0.f);
    block.bn1.running_var.resize(planes, 1.f);

    // conv2: 3x3, (planes, planes)
    block.w2 = Tensor<float>(std::vector<int>{planes, planes, 3, 3});
    block.b2.resize(planes, 0.f);
    block.bn2.gamma.resize(planes, 1.f);
    block.bn2.beta.resize(planes, 0.f);
    block.bn2.running_mean.resize(planes, 0.f);
    block.bn2.running_var.resize(planes, 1.f);

    // conv3: 1x1, (outplanes, planes)
    block.w3 = Tensor<float>(std::vector<int>{outplanes, planes, 1, 1});
    block.b3.resize(outplanes, 0.f);
    block.bn3.gamma.resize(outplanes, 1.f);
    block.bn3.beta.resize(outplanes, 0.f);
    block.bn3.running_mean.resize(outplanes, 0.f);
    block.bn3.running_var.resize(outplanes, 1.f);

    block.use_downsample = downsample;
    if(downsample){
        // 1x1 conv, stride
        block.w_down = Tensor<float>(std::vector<int>{outplanes, inplanes, 1, 1});
        block.b_down.resize(outplanes, 0.f);
        block.bn_down.gamma.resize(outplanes, 1.f);
        block.bn_down.beta.resize(outplanes, 0.f);
        block.bn_down.running_mean.resize(outplanes, 0.f);
        block.bn_down.running_var.resize(outplanes, 1.f);
    }

    return block;
}

std::vector<Bottleneck> ResNet50::make_layer(int inplanes, int planes, int blocks, int stride)
{
    std::vector<Bottleneck> layer;
    bool downsample = (stride != 1 || inplanes != planes*4);

    Bottleneck b0 = make_bottleneck(inplanes, planes, stride, downsample);
    layer.push_back(b0);

    for(int i=1; i<blocks; i++){
        Bottleneck bN = make_bottleneck(planes*4, planes, 1, false);
        layer.push_back(bN);
    }

    current_inplanes_ = planes*4;
    return layer;
}

Tensor<float> ResNet50::forward(const Tensor<float> &input)
{
    // 1) conv1 (7x7, stride=2, pad=3)
    Conv2DParam p;
    p.stride_h=2; p.stride_w=2;
    p.pad_h=3;    p.pad_w=3;
    auto x = conv2d(input, conv1_w_, conv1_b_, p);

    // bn + relu
    x = batchnorm2d(x, bn1_);
    x = relu(x);

    // 2) maxpool(3x3, stride=2, pad=1)
    Pool2DParam poolp;
    poolp.kernel_h=3; poolp.kernel_w=3;
    poolp.stride_h=2; poolp.stride_w=2;
    poolp.pad_h=1;    poolp.pad_w=1;
    x = max_pool2d(x, poolp);

    // 3) layer1..4
    // layer1_
    for(auto &b : layer1_){
        x = b.forward(x);
    }
    // layer2_
    for(auto &b : layer2_){
        x = b.forward(x);
    }
    // layer3_
    for(auto &b : layer3_){
        x = b.forward(x);
    }
    // layer4_
    for(auto &b : layer4_){
        x = b.forward(x);
    }

    {
        int N = x.shape()[0];
        int C = x.shape()[1];
        int H = x.shape()[2];
        int W = x.shape()[3];
        Tensor<float> pooled(std::vector<int>{N, C, 1, 1});
        for(int n=0; n<N; n++){
            for(int c=0; c<C; c++){
                float sum=0.f;
                for(int hh=0; hh<H; hh++){
                    for(int ww=0; ww<W; ww++){
                        sum += x.at4d(n,c,hh,ww);
                    }
                }
                sum /= (H*W);
                pooled.at4d(n,c,0,0)= sum;
            }
        }
        x=pooled;
    }

    // 5) FC
    // x: [N, 2048, 1, 1] => reshape to [N, 2048]
    {
        int N = x.shape()[0];
        int C = x.shape()[1];
        // flatten
        Tensor<float> flatten(std::vector<int>{N, C});
        for(int n=0; n<N; n++){
            for(int c=0; c<C; c++){
                flatten.at4d(n,c,0,0) = x.at4d(n,c,0,0);
            }
        }
        auto logits = linear(flatten, fc_); // [N, 1000]
        // softmax
        auto probs = softmax(logits);      // [N, 1000]
        // reshape to [N, 1000, 1, 1]
        Tensor<float> out(std::vector<int>{N, 1000, 1, 1});
        for(int n=0; n<N; n++){
            for(int c=0; c<1000; c++){
                out.at4d(n,c,0,0) = probs.at4d(n,c,0,0);
            }
        }
        return out;
    }
}