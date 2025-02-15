#ifndef __EMBEDDING_HPP__
#define __EMBEDDING_HPP__

#include "common/tensor.hpp"

struct EmbeddingParam
{
    Tensor<float> weight;
};

/**
 * Embedding:
 *  input: [N, seq_len], each element is an id in [0, vocab_size-1].
 *  output: [N, seq_len, embedding_dim].
 *  Implementation uses simple indexing into weight.
 */
Tensor<float> embedding_forward(const Tensor<float> &input_ids,
                                const EmbeddingParam &param);

/**
 * PatchEmbedParam:
 *   - patch_size: typically 16 for DeiT-Tiny
 *   - in_ch: 3
 *   - embed_dim: 192 for DeiT-Tiny
 *
 *   weight: [embed_dim, in_ch*patch_size*patch_size]
 *   bias: [embed_dim]
 */
struct PatchEmbedParam
{
    int patch_size;
    int in_ch;
    int embed_dim;

    Tensor<float> weight;
    std::vector<float> bias;
};

/**
 * patch_embed_forward:
 *  input: [N, in_ch, H, W]
 *  output: [N, num_patches, embed_dim]
 */
Tensor<float> patch_embed_forward(const Tensor<float> &input,
                                  const PatchEmbedParam &param);

#endif // MY_DEMO_LAYERS_EMBEDDING_HPP_