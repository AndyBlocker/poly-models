#include "layers/layernorm.hpp"
#include "common/time_utils.hpp"
#include <cmath>

/**
 * layernorm:
 *  We sum over the last dim (hidden_dim) to get mean, var, then
 *  out[i] = gamma*(x[i]-mean)/sqrt(var+eps)+beta
 */
Tensor<float> layernorm(const Tensor<float> &input,
                        const LayerNormParam &param)
{
    ScopedTimer t_ln(OpType::OTHERS);

    // shape e.g. [N, seq_len, hidden_dim]
    int N = input.shape()[0];
    int seq_len = input.shape()[1];
    int hidden_dim = input.shape()[2]; 
    // if shape has 4 dims, adjust accordingly

    Tensor<float> out({N, seq_len, hidden_dim});
    for(int n=0; n<N; n++){
        for(int s=0; s<seq_len; s++){
            // compute mean/var for this row
            double sum=0.0, sum_sq=0.0;
            for(int h=0; h<hidden_dim; h++){
                float v = input.at4d(n,s,h,0);
                sum+= (double)v;
                sum_sq += (double)v*(double)v;
            }
            double mean = sum/hidden_dim;
            double var  = sum_sq/hidden_dim - mean*mean;
            double denom = 1.0/std::sqrt(var+param.eps);

            // apply LN
            for(int h=0; h<hidden_dim; h++){
                float x = input.at4d(n,s,h,0);
                float xhat = (float)((x-mean)*denom);
                float g = param.gamma[h];
                float b = param.beta[h];
                float y = g*xhat + b;
                out.at4d(n,s,h,0)=y;
            }
        }
    }
    return out;
}