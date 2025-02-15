#include "layers/pool2d.hpp"
#include "common/time_utils.hpp"
#include <algorithm>

Tensor<float> max_pool2d(const Tensor<float> &input, const Pool2DParam &param)
{
    ScopedTimer timer(OpType::POOL);

    int N = input.shape()[0];
    int C = input.shape()[1];
    int H = input.shape()[2];
    int W = input.shape()[3];

    int out_h = (H + 2*param.pad_h - param.kernel_h) / param.stride_h + 1;
    int out_w = (W + 2*param.pad_w - param.kernel_w) / param.stride_w + 1;

    Tensor<float> output({N, C, out_h, out_w});

    for(int n=0; n<N; n++){
        for(int c=0; c<C; c++){
            for(int oh=0; oh<out_h; oh++){
                int hstart = oh*param.stride_h - param.pad_h;
                for(int ow=0; ow<out_w; ow++){
                    int wstart = ow*param.stride_w - param.pad_w;
                    float max_val = -1e30f;
                    for(int kh=0; kh<param.kernel_h; kh++){
                        int ih = hstart + kh;
                        for(int kw=0; kw<param.kernel_w; kw++){
                            int iw = wstart + kw;
                            if(ih>=0 && ih<H && iw>=0 && iw<W){
                                float v = input.at4d(n,c,ih,iw);
                                max_val = std::max(max_val, v);
                            }
                        }
                    }
                    output.at4d(n,c,oh,ow) = max_val;
                }
            }
        }
    }
    return output;
}

Tensor<float> avg_pool2d(const Tensor<float> &input, const Pool2DParam &param)
{
    ScopedTimer timer(OpType::POOL);

    int N = input.shape()[0];
    int C = input.shape()[1];
    int H = input.shape()[2];
    int W = input.shape()[3];

    int out_h = (H + 2*param.pad_h - param.kernel_h) / param.stride_h + 1;
    int out_w = (W + 2*param.pad_w - param.kernel_w) / param.stride_w + 1;

    Tensor<float> output({N, C, out_h, out_w});

    for(int n=0; n<N; n++){
        for(int c=0; c<C; c++){
            for(int oh=0; oh<out_h; oh++){
                int hstart = oh*param.stride_h - param.pad_h;
                for(int ow=0; ow<out_w; ow++){
                    int wstart = ow*param.stride_w - param.pad_w;
                    float sum_val = 0.f;
                    int count = 0;
                    for(int kh=0; kh<param.kernel_h; kh++){
                        int ih = hstart + kh;
                        for(int kw=0; kw<param.kernel_w; kw++){
                            int iw = wstart + kw;
                            if(ih>=0 && ih<H && iw>=0 && iw<W){
                                float v = input.at4d(n,c,ih,iw);
                                sum_val += v;
                                count++;
                            }
                        }
                    }
                    if(count > 0) {
                        sum_val /= (float)count;
                    }
                    output.at4d(n,c,oh,ow) = sum_val;
                }
            }
        }
    }
    return output;
}