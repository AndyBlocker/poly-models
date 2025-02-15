#ifndef __LAYERNORM_HPP__
#define __LAYERNORM_HPP__

#include "common/tensor.hpp"
#include <vector>

/**
 * LayerNormParam:
 *   - gamma, beta: shape=[hidden_dim]
 *   - eps
 * We do LN over last dimension: e.g. [N, seq_len, hidden_dim].
 */
struct LayerNormParam {
    std::vector<float> gamma; 
    std::vector<float> beta; 
    float eps=1e-5f;
};

/**
 * layernorm:
 *  input: [N, seq_len, hidden_dim]
 *  output: same shape
 */
Tensor<float> layernorm(const Tensor<float> &input,
                        const LayerNormParam &param);

#endif // MY_DEMO_LAYERS_LAYERNORM_HPP_