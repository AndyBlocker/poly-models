#include <iostream>
#include "models/resnet50.hpp"
#include "models/mobilenet.hpp"
#include "models/bert.hpp"
#include "models/deit-t.hpp"
#include "common/time_utils.hpp"

void output_time(int freq)
{
    double t_im2col = GlobalProfiler::instance().get_time(OpType::IM2COL);
    double t_matmul = GlobalProfiler::instance().get_time(OpType::MATMUL);
    double t_pool = GlobalProfiler::instance().get_time(OpType::POOL);
    double t_relu = GlobalProfiler::instance().get_time(OpType::RELU);
    double t_norm = GlobalProfiler::instance().get_time(OpType::NORMALIZATION);
    double t_overall = GlobalProfiler::instance().get_time(OpType::OVERALL);
    double t_others = t_overall - t_im2col - t_matmul - t_pool - t_relu - t_norm;
    std::cout << "===== Operator Time (ms, cycles) =====\n"
              << "im2col : " << t_im2col << ", " << t_im2col * freq << "\n"
              << "matmul : " << t_matmul << ", " << t_matmul * freq << "\n"
              << "pooling: " << t_pool << ", " << t_pool * freq << "\n"
              << "relu   : " << t_relu << ", " << t_relu * freq << "\n"
              << "norm   : " << t_norm << ", " << t_norm * freq << "\n"
              << "others : " << t_others << ", " << t_others * freq << "\n"
              << "overall: " << t_overall << ", " << t_overall * freq << "\n"
              << "==============================" << std::endl;
}

int main(int argc, char **argv)
{
    int freq = 80000; // Cycle per ms
    if (argc > 1)
    {
        freq = std::stoi(argv[1]);
    }
    printf("CPU frequency: %d\n", freq);
    printf("===== Start Inference =====\n");

    printf("\n====== (1) ResNet50 ======\n");
    GlobalProfiler::instance().reset();

    ResNet50 model;

    Tensor<float> input(std::vector<int>{1, 3, 224, 224});

    {
        ScopedTimer t(OpType::OVERALL);
        auto output = model.forward(input);

        std::cout << "Output shape: ("
                  << output.shape()[0] << ", "
                  << output.shape()[1] << ", "
                  << output.shape()[2] << ", "
                  << output.shape()[3] << ")\n";
    }
    output_time(freq);

    printf("\n====== (2) MobileNetV2 ======\n");
    GlobalProfiler::instance().reset();

    MobileNetV2 model2;

    Tensor<float> input2(std::vector<int>{1, 3, 224, 224});

    {
        ScopedTimer t(OpType::OVERALL);
        auto output2 = model2.forward(input2);

        std::cout << "Output shape: ("
                  << output2.shape()[0] << ", "
                  << output2.shape()[1] << ", "
                  << output2.shape()[2] << ", "
                  << output2.shape()[3] << ")\n";
    }
    output_time(freq);

    // printf("\n====== (3) BERT ======\n");
    // GlobalProfiler::instance().reset();

    // BertModel bert;

    // Tensor<float> token_ids({1, 128});
    // Tensor<float> pos_ids({1, 128});
    // Tensor<float> seg_ids({1, 128});
    // for (int i = 0; i < 128; i++)
    // {
    //     token_ids.at4d(0, i, 0, 0) = (float)(100 + i);
    //     pos_ids.at4d(0, i, 0, 0) = (float)i;
    //     seg_ids.at4d(0, i, 0, 0) = 0.f;
    // }

    // {
    //     ScopedTimer t(OpType::OVERALL);
    //     auto out = bert.forward(token_ids, pos_ids, seg_ids);
    //     std::cout << "BERT output shape: ("
    //               << out.shape()[0] << ","
    //               << out.shape()[1] << ","
    //               << out.shape()[2] << ")\n";
    // }
    // output_time(freq);

    // printf("\n====== (4) DeiT-Tiny ======\n");
    // GlobalProfiler::instance().reset();

    // // create DeiT-Tiny
    // DeiTTiny model3;

    // // construct dummy input => [N=1, C=3, H=224, W=224]
    // Tensor<float> input3({1, 3, 224, 224});
    // // fill some random or zero data
    // // forward => get {cls_logits, dist_logits}
    // {
    //     ScopedTimer t(OpType::OVERALL);
    //     auto outs = model3.forward(input3);
    //     // outs[0].shape= [1,1000], outs[1].shape= [1,1000]

    //     std::cout << "cls_logits shape: ("
    //               << outs[0].shape()[0] << ", " << outs[0].shape()[1] << ")\n"
    //               << "dist_logits shape: ("
    //               << outs[1].shape()[0] << ", " << outs[1].shape()[1] << ")\n";
    // }
    // output_time(freq);

    printf("===== End Inference =====\n");

    return 0;
}