#ifndef __BERT_HPP__
#define __BERT_HPP__

#include "layers/embedding.hpp"
#include "layers/layernorm.hpp"
#include "layers/attention.hpp"
#include "layers/feedforward.hpp"
#include <vector>

struct BertEncoderLayer
{
    MHAParam mha;
    LayerNormParam ln1;
    FFParam ff;
    LayerNormParam ln2;

    Tensor<float> forward(const Tensor<float> &x) const;
};

class BertModel
{
public:
    BertModel();
    Tensor<float> forward(const Tensor<float> &token_ids,
                          const Tensor<float> &pos_ids,
                          const Tensor<float> &seg_ids);

private:
    EmbeddingParam word_emb_;
    EmbeddingParam pos_emb_;
    EmbeddingParam seg_emb_;
    LayerNormParam emb_ln_;
    std::vector<BertEncoderLayer> layers_;
    int hidden_dim_;
    int num_layers_;
};

#endif