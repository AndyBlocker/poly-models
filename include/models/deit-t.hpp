#ifndef __DEIT_TINY_HPP__
#define __DEIT_TINY_HPP__

#include "layers/embedding.hpp"
#include "layers/attention.hpp"
#include "layers/feedforward.hpp"
#include "layers/layernorm.hpp"
#include "layers/linear.hpp"
#include <vector>

/**
 * DeiTEncoderLayer:
 *  - multi-head self-attn
 *  - LN1
 *  - feedforward
 *  - LN2
 */
struct DeiTEncoderLayer
{
    MHAParam mha;
    LayerNormParam ln1;
    FFParam ff;
    LayerNormParam ln2;

    // forward
    Tensor<float> forward(const Tensor<float> &x) const;
};

/**
 * DeiT-Tiny:
 *   - patch_size=16, embed_dim=192, num_heads=3, depth=12
 *   - patch_embed
 *   - cls_token, dist_token (each shape [1, embed_dim])
 *   - posEmbed shape [1, 1+1+num_patches, embed_dim]
 *   - N=12 DeiTEncoderLayer
 *   - final ln
 *   - head => [embed_dim, 1000], dist_head => [embed_dim, 1000]
 */
class DeiTTiny
{
public:
    DeiTTiny();
    // forward
    // input: [N,3,224,224], output => {cls_logits, dist_logits}
    // we can return a pair or 2 Tensors
    // here just return a vector: out[0]=cls, out[1]=dist
    std::vector<Tensor<float>> forward(const Tensor<float> &input);

private:
    // patch embed
    PatchEmbedParam patch_;
    // cls_token, dist_token
    Tensor<float> cls_token_;  // shape [1,1,embed_dim]
    Tensor<float> dist_token_; // shape [1,1,embed_dim]
    // posEmbed => shape [1, 1+1+num_patches, embed_dim]
    Tensor<float> pos_embed_;
    // LN
    LayerNormParam ln_;
    // depth=12
    std::vector<DeiTEncoderLayer> layers_;
    // heads
    LinearParam head_;      // for cls
    LinearParam dist_head_; // for dist
    // some config
    int embed_dim_;
    int depth_;
    int num_heads_;
    int num_patches_;
};

#endif // MY_MODELS_DEIT_TINY_HPP_