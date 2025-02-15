#include "models/bert.hpp"
#include <stdexcept>

Tensor<float> BertEncoderLayer::forward(const Tensor<float> &x) const
{
    auto attn_out = multi_head_self_attention(x, mha);

    int N = x.shape()[0];
    int S = x.shape()[1];
    int D = x.shape()[2];

    // residual
    Tensor<float> res1({N, S, D});
    for (int n = 0; n < N; n++)
    {
        for (int s = 0; s < S; s++)
        {
            for (int d = 0; d < D; d++)
            {
                float v = attn_out.at4d(n, s, d, 0) + x.at4d(n, s, d, 0);
                res1.at4d(n, s, d, 0) = v;
            }
        }
    }
    auto ln1_out = layernorm(res1, ln1);

    auto ff_out = feed_forward(ln1_out, ff);
    Tensor<float> res2({N, S, D});
    for (int n = 0; n < N; n++)
    {
        for (int s = 0; s < S; s++)
        {
            for (int d = 0; d < D; d++)
            {
                float v = ff_out.at4d(n, s, d, 0) + ln1_out.at4d(n, s, d, 0);
                res2.at4d(n, s, d, 0) = v;
            }
        }
    }
    auto ln2_out = layernorm(res2, ln2);
    return ln2_out;
}

BertModel::BertModel()
{
    hidden_dim_ = 768;
    num_layers_ = 12;

    word_emb_.weight = Tensor<float>({30522, hidden_dim_});
    pos_emb_.weight = Tensor<float>({512, hidden_dim_});
    seg_emb_.weight = Tensor<float>({2, hidden_dim_});

    emb_ln_.gamma.resize(hidden_dim_, 1.f);
    emb_ln_.beta.resize(hidden_dim_, 0.f);

    for (int i = 0; i < num_layers_; i++)
    {
        BertEncoderLayer layer;
        layer.mha.Wq = Tensor<float>({hidden_dim_, hidden_dim_});
        layer.mha.bq.resize(hidden_dim_, 0.f);
        layer.mha.Wk = Tensor<float>({hidden_dim_, hidden_dim_});
        layer.mha.bk.resize(hidden_dim_, 0.f);
        layer.mha.Wv = Tensor<float>({hidden_dim_, hidden_dim_});
        layer.mha.bv.resize(hidden_dim_, 0.f);
        layer.mha.Wo = Tensor<float>({hidden_dim_, hidden_dim_});
        layer.mha.bo.resize(hidden_dim_, 0.f);
        layer.mha.num_heads = 12;

        layer.ln1.gamma.resize(hidden_dim_, 1.f);
        layer.ln1.beta.resize(hidden_dim_, 0.f);

        layer.ff.W1 = Tensor<float>({hidden_dim_, 4 * hidden_dim_});
        layer.ff.b1.resize(4 * hidden_dim_, 0.f);
        layer.ff.W2 = Tensor<float>({4 * hidden_dim_, hidden_dim_});
        layer.ff.b2.resize(hidden_dim_, 0.f);

        layer.ln2.gamma.resize(hidden_dim_, 1.f);
        layer.ln2.beta.resize(hidden_dim_, 0.f);

        layers_.push_back(layer);
    }
}

Tensor<float> BertModel::forward(const Tensor<float> &token_ids,
                                 const Tensor<float> &pos_ids,
                                 const Tensor<float> &seg_ids)
{
    auto w_embed = embedding_forward(token_ids, word_emb_);
    auto p_embed = embedding_forward(pos_ids, pos_emb_);
    auto s_embed = embedding_forward(seg_ids, seg_emb_);

    int N = w_embed.shape()[0];
    int S = w_embed.shape()[1];
    int D = w_embed.shape()[2];

    Tensor<float> sum_emb({N, S, D});
    for (int n = 0; n < N; n++)
    {
        for (int s = 0; s < S; s++)
        {
            for (int d = 0; d < D; d++)
            {
                float val = w_embed.at4d(n, s, d, 0) + p_embed.at4d(n, s, d, 0) + s_embed.at4d(n, s, d, 0);
                sum_emb.at4d(n, s, d, 0) = val;
            }
        }
    }

    auto emb_out = layernorm(sum_emb, emb_ln_);

    auto x = emb_out;
    for (auto &layer : layers_)
    {
        x = layer.forward(x);
    }
    return x;
}