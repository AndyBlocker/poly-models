#ifndef __ATTENTION_HPP__
#define __ATTENTION_HPP__

#include "common/tensor.hpp"
#include <vector>

/**
 * Multi-head self-attention param
 *   Typically we have "Q,K,V" linear transforms
 *   Then we split heads, do scaled dot-product attention, 
 *   then a final linear out
 */
struct MHAParam {
    // Q, K, V: [hidden_dim, hidden_dim]
    // out: [hidden_dim, hidden_dim]
    Tensor<float> Wq;
    std::vector<float> bq;
    Tensor<float> Wk;
    std::vector<float> bk;
    Tensor<float> Wv;
    std::vector<float> bv;
    Tensor<float> Wo;
    std::vector<float> bo;
    int num_heads=8;
};

/**
 * multi_head_self_attention:
 *  input: [N, seq_len, hidden_dim]
 *  output: same shape
 */
Tensor<float> multi_head_self_attention(const Tensor<float> &input,
                                        const MHAParam &param);

#endif // MY_DEMO_LAYERS_ATTENTION_HPP_