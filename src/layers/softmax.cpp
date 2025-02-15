#include "layers/softmax.hpp"
#include "common/time_utils.hpp"
#include <cmath>

Tensor<float> softmax(const Tensor<float> &input)
{
    ScopedTimer timer(OpType::OTHERS);

    int N = input.shape()[0];
    int C = input.shape()[1];

    Tensor<float> output(std::vector<int>{N, C});
    for(int n=0; n<N; n++){
        float max_val = -1e30f;
        for(int c=0; c<C; c++){
            float v = input.at4d(n,c,0,0);
            if(v>max_val) max_val=v;
        }
        double sum_exp = 0.0;
        for(int c=0; c<C; c++){
            float v = input.at4d(n,c,0,0);
            double e = std::exp((double)v - (double)max_val);
            sum_exp += e;
        }
        for(int c=0; c<C; c++){
            float v = input.at4d(n,c,0,0);
            double e = std::exp((double)v - (double)max_val);
            output.at4d(n,c,0,0) = (float)(e / sum_exp);
        }
    }
    return output;
}