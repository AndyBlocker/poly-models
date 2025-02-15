#include "layers/linear.hpp"
#include "common/time_utils.hpp"

Tensor<float> linear(const Tensor<float> &input, const LinearParam &param)
{
    // input shape: [N, in_features], weight: [out_features, in_features]
    // out: [N, out_features]
    ScopedTimer timer(OpType::MATMUL);

    int N = input.shape()[0];
    int in_features = input.shape()[1];
    int out_features= param.weight.shape()[0];

    Tensor<float> output({N, out_features});
    const float* wptr = param.weight.data();
    const float* inptr= input.data();
    float* outptr = output.data();

    for(int n_i=0; n_i<N; n_i++){
        for(int oc=0; oc<out_features; oc++){
            double sum = 0.0;
            for(int ic=0; ic<in_features; ic++){
                sum += (double)inptr[n_i*in_features + ic] * (double)wptr[oc*in_features + ic];
            }
            sum += param.bias[oc];
            outptr[n_i*out_features + oc] = (float)sum;
        }
    }

    return output;
}