#include "layers/relu.hpp"
#include "common/time_utils.hpp"

Tensor<float> relu(const Tensor<float> &input)
{
    ScopedTimer timer(OpType::RELU);

    Tensor<float> output = input; // 拷贝
    float* ptr = output.data();
    int total = output.total_size();
    for(int i=0; i<total; i++){
        if(ptr[i] < 0.f) {
            ptr[i] = 0.f;
        }
    }
    return output;
}

Tensor<float> relu6(const Tensor<float> &input)
{
    ScopedTimer timer(OpType::RELU);

    Tensor<float> output = input;
    float* ptr = output.data();
    int total = output.total_size();
    for(int i=0; i<total; i++){
        if(ptr[i] < 0.f) {
            ptr[i] = 0.f;
        } else if(ptr[i] > 6.f) {
            ptr[i] = 6.f;
        }
    }
    return output;
}