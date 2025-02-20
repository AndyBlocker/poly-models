#include "layers/batchnorm.hpp"
#include "common/time_utils.hpp"
#include <cmath>

Tensor<float> batchnorm2d(const Tensor<float> &input, const BNParam &param)
{
    ScopedTimer timer(OpType::NORMALIZATION);

    int N = input.shape()[0];
    int C = input.shape()[1];
    int H = input.shape()[2];
    int W = input.shape()[3];

    Tensor<float> output({N, C, H, W});

    for(int n=0; n<N; n++){
        for(int c=0; c<C; c++){
            float gamma = param.gamma[c];
            float beta = param.beta[c];
            float mean = param.running_mean[c];
            float var = param.running_var[c];
            float denom = 1.0f / std::sqrt(var + param.eps);

            for(int hh=0; hh<H; hh++){
                for(int ww=0; ww<W; ww++){
                    float x = input.at4d(n,c,hh,ww);
                    float x_hat = (x - mean)*denom;
                    float y = gamma * x_hat + beta;
                    output.at4d(n,c,hh,ww) = y;
                }
            }
        }
    }
    return output;
}