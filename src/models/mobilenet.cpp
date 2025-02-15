#include "models/mobilenet.hpp"
#include <cmath>
#include <vector>

Tensor<float> InvertedResidual::forward(const Tensor<float> &x) const
{
    // expand
    Tensor<float> out = x;
    if (expand_ratio != 1)
    {
        Conv2DParam p1;
        auto tmp = conv2d(x, w_expand, b_expand, p1);
        tmp = batchnorm2d(tmp, bn_expand);
        tmp = relu6(tmp);
        out = tmp;
    }
    // depthwise
    Tensor<float> dw = depthwise_conv2d_im2col(out, w_dwise, b_dwise,
                                               stride, stride, 1, 1);
    dw = batchnorm2d(dw, bn_dwise);
    dw = relu6(dw);
    // project
    Conv2DParam p2;
    auto proj = conv2d(dw, w_project, b_project, p2);
    proj = batchnorm2d(proj, bn_project);
    // residual
    if (stride == 1 && in_channels == out_channels)
    {
        // proj + x
        int N = proj.shape()[0];
        int C = proj.shape()[1];
        int H = proj.shape()[2];
        int W = proj.shape()[3];
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int h = 0; h < H; h++)
                {
                    for (int w = 0; w < W; w++)
                    {
                        float val = proj.at4d(n, c, h, w) + x.at4d(n, c, h, w);
                        proj.at4d(n, c, h, w) = val;
                    }
                }
            }
        }
    }
    return proj;
}

InvertedResidual MobileNetV2::make_inverted_residual(int in_c, int out_c, int stride, int expand_ratio)
{
    InvertedResidual block;
    block.in_channels = in_c;
    block.out_channels = out_c;
    block.stride = stride;
    block.expand_ratio = expand_ratio;

    int hidden_c = in_c * expand_ratio;

    if (expand_ratio != 1)
    {
        block.w_expand = Tensor<float>(std::vector<int>{hidden_c, in_c, 1, 1});
        block.b_expand.resize(hidden_c, 0.f);
        block.bn_expand.gamma.resize(hidden_c, 1.f);
        block.bn_expand.beta.resize(hidden_c, 0.f);
        block.bn_expand.running_mean.resize(hidden_c, 0.f);
        block.bn_expand.running_var.resize(hidden_c, 1.f);
    }

    block.w_dwise = Tensor<float>(std::vector<int>{hidden_c, 1, 3, 3});
    block.b_dwise.resize(hidden_c, 0.f);
    block.bn_dwise.gamma.resize(hidden_c, 1.f);
    block.bn_dwise.beta.resize(hidden_c, 0.f);
    block.bn_dwise.running_mean.resize(hidden_c, 0.f);
    block.bn_dwise.running_var.resize(hidden_c, 1.f);

    block.w_project = Tensor<float>(std::vector<int>{out_c, hidden_c, 1, 1});
    block.b_project.resize(out_c, 0.f);
    block.bn_project.gamma.resize(out_c, 1.f);
    block.bn_project.beta.resize(out_c, 0.f);
    block.bn_project.running_mean.resize(out_c, 0.f);
    block.bn_project.running_var.resize(out_c, 1.f);

    return block;
}

MobileNetV2::MobileNetV2()
{
    // first conv
    first_conv_w_ = Tensor<float>(std::vector<int>{32, 3, 3, 3});
    first_conv_b_.resize(32, 0.f);
    first_conv_bn_.gamma.resize(32, 1.f);
    first_conv_bn_.beta.resize(32, 0.f);
    first_conv_bn_.running_mean.resize(32, 0.f);
    first_conv_bn_.running_var.resize(32, 1.f);

    current_channels_ = 32;

    // blocks
    // (t=1,c=16,n=1,s=1)
    blocks_.push_back(make_inverted_residual(32, 16, 1, 1));
    current_channels_ = 16;

    // (t=6,c=24,n=2,s=2 then 1)
    {
        blocks_.push_back(make_inverted_residual(16, 24, 2, 6));
        blocks_.push_back(make_inverted_residual(24, 24, 1, 6));
        current_channels_ = 24;
    }

    // (t=6,c=32,n=3,s=2,1,1)
    {
        blocks_.push_back(make_inverted_residual(24, 32, 2, 6));
        blocks_.push_back(make_inverted_residual(32, 32, 1, 6));
        blocks_.push_back(make_inverted_residual(32, 32, 1, 6));
        current_channels_ = 32;
    }

    // (t=6,c=64,n=4,s=2,1,1,1)
    {
        blocks_.push_back(make_inverted_residual(32, 64, 2, 6));
        blocks_.push_back(make_inverted_residual(64, 64, 1, 6));
        blocks_.push_back(make_inverted_residual(64, 64, 1, 6));
        blocks_.push_back(make_inverted_residual(64, 64, 1, 6));
        current_channels_ = 64;
    }

    // (t=6,c=96,n=3,s=1)
    {
        blocks_.push_back(make_inverted_residual(64, 96, 1, 6));
        blocks_.push_back(make_inverted_residual(96, 96, 1, 6));
        blocks_.push_back(make_inverted_residual(96, 96, 1, 6));
        current_channels_ = 96;
    }

    // (t=6,c=160,n=3,s=2,1,1)
    {
        blocks_.push_back(make_inverted_residual(96, 160, 2, 6));
        blocks_.push_back(make_inverted_residual(160, 160, 1, 6));
        blocks_.push_back(make_inverted_residual(160, 160, 1, 6));
        current_channels_ = 160;
    }

    // (t=6,c=320,n=1,s=1)
    {
        blocks_.push_back(make_inverted_residual(160, 320, 1, 6));
        current_channels_ = 320;
    }

    // last conv => 1280
    last_conv_w_ = Tensor<float>(std::vector<int>{1280, 320, 1, 1});
    last_conv_b_.resize(1280, 0.f);
    last_conv_bn_.gamma.resize(1280, 1.f);
    last_conv_bn_.beta.resize(1280, 0.f);
    last_conv_bn_.running_mean.resize(1280, 0.f);
    last_conv_bn_.running_var.resize(1280, 1.f);

    // fc => 1000
    fc_.weight = Tensor<float>({1000, 1280});
    fc_.bias.resize(1000, 0.f);
}

Tensor<float> MobileNetV2::forward(const Tensor<float> &input)
{
    // first conv
    Conv2DParam p;
    p.stride_h = 2; 
    p.stride_w = 2; 
    p.pad_h = 1; 
    p.pad_w = 1;
    auto x = conv2d(input, first_conv_w_, first_conv_b_, p);
    x = batchnorm2d(x, first_conv_bn_);
    x = relu6(x);

    // inverted residual blocks
    for (auto &b : blocks_) {
        x = b.forward(x);
    }

    // last 1x1 conv
    {
        Conv2DParam p2;
        p2.stride_h = 1; 
        p2.stride_w = 1; 
        p2.pad_h = 0; 
        p2.pad_w = 0;
        auto tmp = conv2d(x, last_conv_w_, last_conv_b_, p2);
        tmp = batchnorm2d(tmp, last_conv_bn_);
        tmp = relu6(tmp);
        x = tmp;
    }

    // global average pool via avg_pool2d
    {
        int H = x.shape()[2];
        int W = x.shape()[3];
        Pool2DParam pool_param;
        pool_param.kernel_h = H;
        pool_param.kernel_w = W;
        pool_param.stride_h = H;  
        pool_param.stride_w = W;  
        pool_param.pad_h = 0;     
        pool_param.pad_w = 0;
        x = avg_pool2d(x, pool_param);  // => [N, 1280, 1, 1] if width multiplier=1
    }

    // flatten => linear => softmax
    {
        int N = x.shape()[0];
        int C = x.shape()[1]; // 1280
        Tensor<float> flat({N, C});
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                flat.at4d(n, c, 0, 0) = x.at4d(n, c, 0, 0);
            }
        }
        auto logits = linear(flat, fc_);
        auto probs  = softmax(logits);
        Tensor<float> out({N, 1000, 1, 1});
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < 1000; c++) {
                out.at4d(n, c, 0, 0) = probs.at4d(n, c, 0, 0);
            }
        }
        return out;
    }
}