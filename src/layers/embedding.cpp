#include "layers/embedding.hpp"
#include "common/time_utils.hpp"
#include "common/matmul.hpp"
#include <stdexcept>

/**
 * embedding_forward:
 *  - input_ids shape: [N, seq_len]
 *  - param.weight: [vocab_size, embedding_dim]
 *  - output: [N, seq_len, embedding_dim]
 */
Tensor<float> embedding_forward(const Tensor<float> &input_ids,
                                const EmbeddingParam &param)
{
    // time stats
    ScopedTimer t_embed(OpType::OTHERS);

    int N = input_ids.shape()[0];
    int seq_len = input_ids.shape()[1];
    int vocab_size = param.weight.shape()[0];
    int emb_dim = param.weight.shape()[1];

    // output
    Tensor<float> out({N, seq_len, emb_dim}); // 3D if we wish, or 4D with last=1

    for (int n = 0; n < N; n++)
    {
        for (int s = 0; s < seq_len; s++)
        {
            // input_ids.at4d(n,s,0,0) is an id
            int idx = (int)input_ids.at4d(n, s, 0, 0);
            if (idx < 0 || idx >= vocab_size)
            {
                throw std::runtime_error("Embedding index out of range");
            }
            // copy embedding vector
            const float *wptr = param.weight.data() + idx * emb_dim;
            for (int d = 0; d < emb_dim; d++)
            {
                out.at4d(n, s, d, 0) = wptr[d];
            }
        }
    }

    return out;
}

/**
 * patch_embed_forward:
 *   1) H_out = H/patch_size, W_out = W/patch_size
 *   2) for each patch => flatten => matmul with param.weight => add bias
 *   3) out shape: [N, H_out*W_out, embed_dim]
 */
Tensor<float> patch_embed_forward(const Tensor<float> &input,
                                  const PatchEmbedParam &param)
{
    ScopedTimer t_patch(OpType::OTHERS);

    int N = input.shape()[0];
    int C = input.shape()[1]; // should be param.in_ch
    int H = input.shape()[2];
    int W = input.shape()[3];

    if (C != param.in_ch)
    {
        throw std::runtime_error("PatchEmbed: input channel mismatch");
    }
    if (H % param.patch_size != 0 || W % param.patch_size != 0)
    {
        throw std::runtime_error("PatchEmbed: image not divisible by patch_size");
    }
    int patch_h = param.patch_size;
    int patch_w = param.patch_size;
    int H_out = H / patch_h;
    int W_out = W / patch_w;
    int num_patches = H_out * W_out;
    int in_size = C * patch_h * patch_w; // e.g. 3*16*16=768
    int embed_dim = param.embed_dim;

    // output => [N, num_patches, embed_dim]
    Tensor<float> out({N, num_patches, embed_dim});

    // flatten each patch => [1, in_size], then matmul => [1, embed_dim]
    for (int n = 0; n < N; n++)
    {
        for (int ph = 0; ph < H_out; ph++)
        {
            for (int pw = 0; pw < W_out; pw++)
            {
                // flatten patch
                std::vector<float> patch_flat(in_size, 0.f);
                int patch_idx = ph * W_out + pw;
                // read from input
                // patch top-left => (ph*patch_h, pw*patch_w)
                for (int c_i = 0; c_i < C; c_i++)
                {
                    for (int kh = 0; kh < patch_h; kh++)
                    {
                        for (int kw = 0; kw < patch_w; kw++)
                        {
                            float val = input.at4d(n, c_i, ph * patch_h + kh, pw * patch_w + kw);
                            int idx = c_i * (patch_h * patch_w) + kh * (patch_w) + kw;
                            patch_flat[idx] = val;
                        }
                    }
                }
                // matmul => [1, embed_dim]
                std::vector<float> embed_vec(embed_dim, 0.f);
                // A= patch_flat(1 x in_size)
                // B= param.weight.data() (in_size x embed_dim)
                // C= embed_vec(1 x embed_dim)
                // => matmul(A,B,C, M=1, K=in_size, N=embed_dim)
                ::matmul(patch_flat.data(), param.weight.data(), embed_vec.data(),
                         1, in_size, embed_dim);

                // add bias
                for (int e = 0; e < embed_dim; e++)
                {
                    embed_vec[e] += param.bias[e];
                }
                // fill out => out(n, patch_idx, e)
                for (int e = 0; e < embed_dim; e++)
                {
                    out.at4d(n, patch_idx, e, 0) = embed_vec[e];
                }
            }
        }
    }
    return out;
}