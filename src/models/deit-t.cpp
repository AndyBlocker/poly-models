#include "models/deit-t.hpp"
#include "layers/layernorm.hpp"
#include "layers/feedforward.hpp"
#include "common/matmul.hpp"
#include "common/time_utils.hpp"
#include <cmath>
#include <stdexcept>

// ----- DeiTEncoderLayer -----
Tensor<float> DeiTEncoderLayer::forward(const Tensor<float> &x) const
{
    // self-attn
    auto attn_out = multi_head_self_attention(x, mha);
    // add+ln
    int N = x.shape()[0];
    int L = x.shape()[1];
    int D = x.shape()[2];
    Tensor<float> res1({N, L, D});
    for (int n = 0; n < N; n++)
    {
        for (int l = 0; l < L; l++)
        {
            for (int d = 0; d < D; d++)
            {
                float v = attn_out.at4d(n, l, d, 0) + x.at4d(n, l, d, 0);
                res1.at4d(n, l, d, 0) = v;
            }
        }
    }
    auto ln1_out = layernorm(res1, ln1);

    // feedforward
    auto ff_out = feed_forward(ln1_out, ff);
    // add+ln
    Tensor<float> res2({N, L, D});
    for (int n = 0; n < N; n++)
    {
        for (int l = 0; l < L; l++)
        {
            for (int d = 0; d < D; d++)
            {
                float v = ff_out.at4d(n, l, d, 0) + ln1_out.at4d(n, l, d, 0);
                res2.at4d(n, l, d, 0) = v;
            }
        }
    }
    auto ln2_out = layernorm(res2, ln2);
    return ln2_out;
}

// ----- DeiTTiny -----
DeiTTiny::DeiTTiny()
{
    // config
    embed_dim_ = 192;
    depth_ = 12;
    num_heads_ = 3;
    int patch_size = 16;
    patch_.patch_size = patch_size;
    patch_.in_ch = 3;
    patch_.embed_dim = embed_dim_;
    // patch weight shape => [embed_dim, 3*16*16] => [192,768]
    patch_.weight = Tensor<float>({embed_dim_, patch_.in_ch * patch_size * patch_size});
    patch_.bias.resize(embed_dim_, 0.f);

    // we expect input 224x224 => H_out=14, W_out=14 => 14*14=196 patches
    num_patches_ = 14 * 14;

    // cls_token, dist_token => shape [1,1,embed_dim]
    cls_token_ = Tensor<float>({1, 1, embed_dim_});
    dist_token_ = Tensor<float>({1, 1, embed_dim_});

    // pos_embed => shape [1, 1+1+num_patches, embed_dim] => [1,198,192]
    pos_embed_ = Tensor<float>({1, 2 + num_patches_, embed_dim_});

    // create 12 layers
    for (int i = 0; i < depth_; i++)
    {
        DeiTEncoderLayer layer;
        // MHA
        layer.mha.Wq = Tensor<float>({embed_dim_, embed_dim_});
        layer.mha.bq.resize(embed_dim_, 0.f);
        layer.mha.Wk = Tensor<float>({embed_dim_, embed_dim_});
        layer.mha.bk.resize(embed_dim_, 0.f);
        layer.mha.Wv = Tensor<float>({embed_dim_, embed_dim_});
        layer.mha.bv.resize(embed_dim_, 0.f);
        layer.mha.Wo = Tensor<float>({embed_dim_, embed_dim_});
        layer.mha.bo.resize(embed_dim_, 0.f);
        layer.mha.num_heads = num_heads_;

        layer.ln1.gamma.resize(embed_dim_, 1.f);
        layer.ln1.beta.resize(embed_dim_, 0.f);

        // FF => hidden= embed_dim, intermediate= embed_dim*4=768
        layer.ff.W1 = Tensor<float>({embed_dim_, 4 * embed_dim_});
        layer.ff.b1.resize(4 * embed_dim_, 0.f);
        layer.ff.W2 = Tensor<float>({4 * embed_dim_, embed_dim_});
        layer.ff.b2.resize(embed_dim_, 0.f);

        layer.ln2.gamma.resize(embed_dim_, 1.f);
        layer.ln2.beta.resize(embed_dim_, 0.f);

        layers_.push_back(layer);
    }

    // final LN
    ln_.gamma.resize(embed_dim_, 1.f);
    ln_.beta.resize(embed_dim_, 0.f);

    // head => [1000, embed_dim], dist_head => [1000, embed_dim]
    head_.weight = Tensor<float>({1000, embed_dim_});
    head_.bias.resize(1000, 0.f);
    dist_head_.weight = Tensor<float>({1000, embed_dim_});
    dist_head_.bias.resize(1000, 0.f);
}

std::vector<Tensor<float>> DeiTTiny::forward(const Tensor<float> &input)
{
    // input: [N,3,224,224]
    ScopedTimer t_deit(OpType::OTHERS);

    int N = input.shape()[0];
    int C = input.shape()[1];
    int H = input.shape()[2];
    int W = input.shape()[3];
    if (C != 3 || H != 224 || W != 224)
    {
        throw std::runtime_error("DeiT-Tiny expects input [N,3,224,224]");
    }

    // 1) patch embed => [N, num_patches, embed_dim]
    auto x = patch_embed_forward(input, patch_);

    // 2) cat cls_token + dist_token => shape [N, 2+num_patches, embed_dim]
    //    cls_token_, dist_token_ are [1,1,embed_dim], so we broadcast along batch
    int L = 2 + num_patches_;
    Tensor<float> x_cat({N, L, embed_dim_});
    for (int n = 0; n < N; n++)
    {
        // first row => cls_token
        for (int d = 0; d < embed_dim_; d++)
        {
            float v = cls_token_.at4d(0, 0, d, 0);
            x_cat.at4d(n, 0, d, 0) = v;
        }
        // second row => dist_token
        for (int d = 0; d < embed_dim_; d++)
        {
            float v = dist_token_.at4d(0, 0, d, 0);
            x_cat.at4d(n, 1, d, 0) = v;
        }
        // then copy x => patch embeddings
        for (int p = 0; p < num_patches_; p++)
        {
            for (int d = 0; d < embed_dim_; d++)
            {
                float v = x.at4d(n, p, d, 0);
                x_cat.at4d(n, 2 + p, d, 0) = v;
            }
        }
    }

    // 3) add pos_embed => shape [1, L, embed_dim_], broadcast batch
    // pos_embed_.shape= [1,L,embed_dim_]
    for (int n = 0; n < N; n++)
    {
        for (int l = 0; l < L; l++)
        {
            for (int d = 0; d < embed_dim_; d++)
            {
                float val = x_cat.at4d(n, l, d, 0) + pos_embed_.at4d(0, l, d, 0);
                x_cat.at4d(n, l, d, 0) = val;
            }
        }
    }

    // 4) pass through 12 transformer layers
    auto z = x_cat;
    for (auto &layer : layers_)
    {
        z = layer.forward(z);
    }

    // 5) final LN
    // shape => [N,L,embed_dim_]
    // LN over last dim => reuse layernorm
    // create a new LN param or use ln_
    auto ln_out = layernorm(z, ln_);

    // 6) cls_token => ln_out[:,0,:], dist_token => ln_out[:,1,:]
    // then feed each into linear head => logits => [N,1000]
    Tensor<float> cls_in({N, embed_dim_});
    Tensor<float> dist_in({N, embed_dim_});
    for (int n = 0; n < N; n++)
    {
        for (int d = 0; d < embed_dim_; d++)
        {
            cls_in.at4d(n, d, 0, 0) = ln_out.at4d(n, 0, d, 0);
            dist_in.at4d(n, d, 0, 0) = ln_out.at4d(n, 1, d, 0);
        }
    }
    // linear => logits
    // head_.weight => [1000, embed_dim_], bias=>[1000]
    // => out => [N,1000]
    auto cls_logits = Tensor<float>({N, 1000});
    {
        const float *Ap = cls_in.data();
        const float *Bp = head_.weight.data();
        float *Cp = cls_logits.data();
        // matmul => (N,embed_dim_)*(embed_dim_,1000) => (N,1000)
        ::matmul(Ap, Bp, Cp, N, embed_dim_, 1000);

        // add bias
        for (int n_i = 0; n_i < N; n_i++)
        {
            for (int c = 0; c < 1000; c++)
            {
                float val = cls_logits.at4d(n_i, c, 0, 0) + head_.bias[c];
                cls_logits.at4d(n_i, c, 0, 0) = val;
            }
        }
    }
    // dist logits
    auto dist_logits = Tensor<float>({N, 1000});
    {
        const float *Ap = dist_in.data();
        const float *Bp = dist_head_.weight.data();
        float *Cp = dist_logits.data();
        ::matmul(Ap, Bp, Cp, N, embed_dim_, 1000);

        for (int n_i = 0; n_i < N; n_i++)
        {
            for (int c = 0; c < 1000; c++)
            {
                float val = dist_logits.at4d(n_i, c, 0, 0) + dist_head_.bias[c];
                dist_logits.at4d(n_i, c, 0, 0) = val;
            }
        }
    }

    // return [cls_logits, dist_logits]
    std::vector<Tensor<float>> outs;
    outs.push_back(cls_logits);
    outs.push_back(dist_logits);
    return outs;
}