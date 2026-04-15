/*
 * Copyright (C) Vicoretek, Inc - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by tianling.yang <tianling.yang@vicoretek.com>, May 2023
 */
#include <chrono>
#include <iostream>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <unordered_map>

#include "Vknn.h"
#include "method.hpp"
using namespace std;

extern vector<string> COCO_ClASSES;
extern vector<string> CLASSES;
static int H_, W_;
std::vector<float> softmax(const float* input, int size) {
    std::vector<float> output(size);
    float sum = 0.0;

    // 指数化和求和
    for (int i = 0; i < size; i++) {
        float expValue = std::exp(input[i]);
        output[i] = expValue;
        sum += expValue;
    }

    // 归一化
    for (float& value : output) {
        value /= sum;
    }

    return output;
}

void preprocess_yolov5(const char* path, float* buffer) {
    cv::Mat image = cv::imread(path);
    image = resize_letterbox(image, H_, W_);
    copyMatToBuffer(image, (uchar*)buffer);

}

void preprocess_yolov7(const char* path, float* buffer) {
    cv::Mat image = cv::imread(path);
    image = resize_letterbox(image, H_, W_);
    copyMatToBuffer(image, (uchar*)buffer);

}

void preprocess_centernet(const char* path, float* buffer) {
    cv::Mat image = cv::imread(path);
    cvtColor(image, image, CV_BGR2RGB);
    cv::resize(image, image, cv::Size(H_, W_), 0, 0, cv::INTER_CUBIC);
    copyMatToBuffer(image, (uchar*)buffer);
}

void preprocess_detr(const char* path, float* buffer) {
    cv::Mat image = cv::imread(path);
    image = resize_letterbox(image, H_, W_);
    copyMatToBuffer(image, (uchar*)buffer);
}

bool CompareByScore(const Detection& a, const Detection& b) {
    return a.score > b.score;
}

#if 1
std::vector<Detection> postprocess_yolov5(const vknn_output* output)
{
    unsigned int m_numClasses = 80;
    auto elements_offset = 5 + m_numClasses;
    int h_ = H_;
    int w_ = W_;
    std::unordered_map<int, std::vector<int>> s_anchors{
    {8, {10, 13, 16, 30, 33, 23}},
    {16, {30, 61, 62, 45, 59, 119}},
    {32, {116, 90, 156, 198, 373, 326}} };
    int stride[3] = { 8, 16, 32 };
    float m_objectThreshold = 0.4;  //set 0.001 when calculate map
    float m_ClsThreshold = 0.4;     //set 0.001 when calculate map
    float m_NmsThreshold = 0.6;
    float m_NmsThreshold_all = 0.6;
    int id_num = -1;
    int anchor_num = 3;
    std::vector<Detection> detections;
    Detection detectedObject;
    detectedObject.box.resize(4);
    std::vector<std::vector<float>> networkResults(3);
    for (int i = 0; i < 3; ++i) {
        networkResults[i] = std::vector<float>((float*)output[i].buffer, (float*)output[i].buffer + output[i].size);
    }
    for (int id_output = 0; id_output < 3; ++id_output)
    {
        int m_gridHeight = h_ / stride[id_output];
        int m_gridWidth = w_ / stride[id_output];
        for (int id_grid_h = 0; id_grid_h < m_gridHeight; ++id_grid_h)
        {
            for (int id_grid_w = 0; id_grid_w < m_gridWidth; ++id_grid_w)
            {
                for (int id_anchor = 0; id_anchor < anchor_num; ++id_anchor)
                {
                    id_num += 1;
                    float conf = networkResults[id_output][(elements_offset * id_anchor + 4) * m_gridHeight * m_gridWidth + id_grid_h * m_gridWidth + id_grid_w];
                    conf = sigmoid(conf);
                    if ((conf) > m_objectThreshold)
                    {
                        for (unsigned int classIndex = 0; classIndex < m_numClasses; ++classIndex)
                        {
                            float class_prob = networkResults[id_output][(elements_offset * id_anchor + 5 + classIndex) * m_gridHeight * m_gridWidth + id_grid_h * m_gridWidth + id_grid_w];
                            class_prob = sigmoid(class_prob) * conf;
                            if (class_prob > m_ClsThreshold)
                            {
                                float x, y, w, h;
                                x = sigmoid(networkResults[id_output][(elements_offset * id_anchor) * m_gridHeight * m_gridWidth + id_grid_h * m_gridWidth + id_grid_w]) * 2 - 0.5;
                                y = sigmoid(networkResults[id_output][(elements_offset * id_anchor) * m_gridHeight * m_gridWidth + id_grid_h * m_gridWidth + id_grid_w + 1 * m_gridHeight * m_gridWidth]) * 2 - 0.5;
                                h = sigmoid(networkResults[id_output][(elements_offset * id_anchor) * m_gridHeight * m_gridWidth + id_grid_h * m_gridWidth + id_grid_w + 3 * m_gridHeight * m_gridWidth]) * 2;
                                w = sigmoid(networkResults[id_output][(elements_offset * id_anchor) * m_gridHeight * m_gridWidth + id_grid_h * m_gridWidth + id_grid_w + 2 * m_gridHeight * m_gridWidth]) * 2;

                                x = (x + id_grid_w) * static_cast<float>(stride[id_output]);
                                y = (y + id_grid_h) * static_cast<float>(stride[id_output]);
                                w = w * w * static_cast<float>(s_anchors[stride[id_output]][id_anchor * 2]);
                                h = h * h * static_cast<float>(s_anchors[stride[id_output]][id_anchor * 2 + 1]);
                                float topLeftX = (x - w / 2.0);
                                if (topLeftX < 0)
                                {
                                    topLeftX = 0;
                                }
                                float topLeftY = (y - h / 2.0);
                                if (topLeftY < 0)
                                {
                                    topLeftY = 0;
                                }
                                float botRightX = (x + w / 2.0);
                                float botRightY = (y + h / 2.0);
                                assert(botRightX > topLeftX);
                                assert(botRightY > topLeftY);
                                detectedObject.box[0] = topLeftX;
                                detectedObject.box[1] = topLeftY;
                                detectedObject.box[2] = botRightX;
                                detectedObject.box[3] = botRightY;
                                detectedObject.score = class_prob;
                                detectedObject.label = classIndex;
                                detections.emplace_back(detectedObject);

                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<Detection> resultsAfterNMS = NMS(detections, m_NmsThreshold, m_NmsThreshold_all);
    return resultsAfterNMS;
}
#else
std::vector<Detection> postprocess_yolov5(const vknn_output* output)
{
    std::vector<Detection> resultsAfterNMS;
    return resultsAfterNMS;
}
#endif

std::vector<Detection> postprocess_centernet(const vknn_output* output)
{

    std::vector<Detection> detections;
    Detection detection;
    detection.box.resize(4);
    int width = 128;
    int height = 128;
    int class_num = 20;
    float score_threshold = 0.02;
    float iou = 0.3;
    float heat_map[width * height * class_num];
    float class_conf[width * height], class_index[width * height]; //heat_map maxpooling过滤得到的置信度与类别索引
    int mask[width * height]; //置信度过滤的mask;
    float* pred_wh, * pred_offset;
    int vx[width * height], vy[width * height];

    PoolNms((float*)output[0].buffer, heat_map, height, width, class_num);

    cwh_to_whc(heat_map, height, width, class_num);

    keep_last_dim_max(heat_map, class_conf, class_index, height, width, class_num);

    filter_by_confidence(class_conf, width * height, score_threshold, mask);

    pred_wh = (float*)output[1].buffer;

    cwh_to_whc(pred_wh, height, width, 2);
    pred_offset = (float*)output[2].buffer;
    cwh_to_whc(pred_offset, height, width, 2);
    meshgrid(vx, vy, width, height);

    int box_index = 0;
    for (int m = 0;m < width * height;m++) {
        if (mask[m] == 1) {
            float vx_mask = pred_offset[m * 2] + vx[m];
            float vy_mask = pred_offset[m * 2 + 1] + vy[m];
            float conf_mask = class_conf[m];
            float index_mask = class_index[m];
            float half_w = pred_wh[m * 2] / 2.0;
            float half_h = pred_wh[m * 2 + 1] / 2.0;

            detection.box[0] = (vx_mask - half_w) / 128 * W_;
            detection.box[1] = (vy_mask - half_h) / 128 * H_;
            detection.box[2] = (vx_mask + half_w) / 128 * W_;
            detection.box[3] = (vy_mask + half_h) / 128 * H_;
            detection.score = conf_mask;
            detection.label = index_mask;
            if (detection.box[0] < 0) {
                detection.box[0] = 0;
            }
            if (detection.box[1] < 0) {
                detection.box[1] = 0;
            }
            detections.push_back(detection);
            box_index++;
        }
    }
    std::vector<Detection> resultsAfterNMS = NMS(detections, iou, 1);
    return resultsAfterNMS;
}

std::vector<Detection> postprocess_detr(const vknn_output* output)
{

    std::vector<Detection> detections;
    Detection detection;
    int num_rows = 100;
    int num_cols = 92;
    float threshold = 0.9;
    const float* pred_logits = (float*)output[0].buffer;
    const float* pred_boxes = (float*)output[1].buffer;
    for (int i = 0;i < num_rows;i++) {
        const float* logits_row = pred_logits + i * num_cols;
        const float* boxes_row = pred_boxes + i * 4;
        std::vector<float> scores = softmax(logits_row, num_cols);
        std::vector<float> boxes(boxes_row, boxes_row + 4);

        std::vector<float> keep_scores;
        std::vector<float> keep_boxes;

        for (int j = 0; j < num_cols - 1; j++) {
            if (scores[j] > threshold) {
                detection.score = scores[j];
                for (unsigned int k = 0; k < COCO_ClASSES.size(); k++) {
                    if (COCO_ClASSES[k] == CLASSES[j]) {
                        detection.label = k;
                    }
                }

                // 提取对应的边界框
                float x = boxes[0];
                float y = boxes[1];
                float w = boxes[2];
                float h = boxes[3];
                boxes[0] = ((x - w / 2.0) * W_ > 0) ? (x - w / 2.0) * W_ : 0;
                boxes[1] = ((y - h / 2.0) * H_ > 0) ? (y - h / 2.0) * H_ : 0;
                boxes[2] = ((x + w / 2.0) * W_ < W_) ? (x + w / 2.0) * W_ : W_;
                boxes[3] = ((y + h / 2.0) * H_ < H_) ? (y + h / 2.0) * H_ : H_;
                detection.box = boxes;
                detections.push_back(detection);
            }

        }

    }
    sort(detections.begin(), detections.end(), CompareByScore);
    return detections;
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        cout << "Usage: ./od_test [model_path] [model_name] [img_path] [output_path]" << endl;
        return 0;
    }

    void (*preprocess)(const char*, float* buffer) = NULL;
    //std::vector<Detection> (*postprocess)(const vknn_output* output) = NULL;
    int n_input, n_output;
    if (strcmp(argv[2], "detr") == 0) {
        H_ = 480;
        W_ = 640;
        n_input = 1;
        n_output = 2;
        preprocess = &preprocess_detr;
        //postprocess = &postprocess_detr;
    }

    else if (strcmp(argv[2], "yolov5") == 0) {
        H_ = 640;
        W_ = 640;
        n_input = 1;
        n_output = 3;
        preprocess = &preprocess_yolov5;
        //postprocess = &postprocess_yolov5;
    }
    else if (strcmp(argv[2], "centernet") == 0) {
        H_ = 512;
        W_ = 512;
        n_input = 1;
        n_output = 3;
        preprocess = &preprocess_centernet;
        //postprocess = &postprocess_centernet;
    }
    else if (strcmp(argv[2], "yolov7") == 0) {
        H_ = 224;
        W_ = 960;
        n_input = 2;
        n_output = 1;
        preprocess = &preprocess_yolov7;
        //postprocess = &postprocess_yolov5;
    }
    else {
        cout << "model_name: detr or yolov5 or centernet or yolov7" << endl;
        return 0;
    }

    std::string argv4 = argv[4];
    std::string saveimg = argv4 + "/" + "img";
    std::string savelabel = argv4 + "/" + "label";
    std::filesystem::create_directories(saveimg);
    std::filesystem::create_directories(savelabel);
    vknn_context ctx;
    vknn_input input[2];
    vknn_output output[3];
    input[0].buffer = new float[H_ * W_ * 3];
    input[0].type = VKNN_TENSOR_UINT8;
    input[1].buffer = new float[H_ * W_ * 3];
    input[1].type = VKNN_TENSOR_UINT8;
    for (int i = 0; i < 3; i++) {
        output[i].user_allocated = false;
        output[i].original_tensor = true;
    }
    cout << "start vknn_init" << endl;
    vknn_init(&ctx, argv[1], 0, NULL);
    cout << "end vknn_init" << endl;
    std::vector<Detection> result;
    for (int i = 0; i < 200; i++)
    {
        for (const auto& entry : filesystem::directory_iterator(argv[3])) {
            std::string extension = entry.path().extension().string();
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
                std::string absoluteImagePath = std::filesystem::absolute(entry.path()).string();
                std::string absoluteImagePath1 = absoluteImagePath + "right";
                std::string imageName = entry.path().stem().string();
                std::string labelName = imageName + ".txt";
                std::string absoluteLabelPath = std::filesystem::absolute(savelabel + "/" + labelName).string();
                std::string absoluteSaveImgPath = std::filesystem::absolute(saveimg + "/" + imageName + ".jpg").string();
                std::ifstream file(absoluteLabelPath);
                if (file) {
                    std::cout << "jump" << imageName << std::endl;
                    continue;
                }
                cout << imageName << endl;
                preprocess(absoluteImagePath.c_str(), (float*)input[0].buffer);
                preprocess(absoluteImagePath1.c_str(), (float*)input[1].buffer);
                //vknn_process_input
                auto start = std::chrono::high_resolution_clock::now();
                vknn_process_input(ctx, input, n_input);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "vknn_process_input: " << duration.count() << " ms" << std::endl;
                // vknn_inference
                start = std::chrono::high_resolution_clock::now();
                vknn_inference(ctx, NULL);
                end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "vknn_inference: " << duration.count() << " ms" << std::endl;
                // vknn_process_output
                start = std::chrono::high_resolution_clock::now();
                vknn_process_output(ctx, output, n_output);
                end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "vknn_process_output: " << duration.count() << " ms" << std::endl;
                /*result = postprocess(output);

                FILE *fpw = NULL;
                fpw = fopen(absoluteLabelPath.c_str(),"w");
                cv::Mat saveimg = cv::imread(absoluteImagePath);
                if (strcmp(argv[2], "centernet") == 0){
                    cv::resize(saveimg,saveimg,cv::Size(H_,W_));
                }
                else{
                    saveimg = resize_letterbox(saveimg,H_,W_);
                }
                for(long unsigned int i = 0;i<result.size();i++){
                    rectangle(saveimg,
                        cv::Point(result[i].box[0],result[i].box[1]),
                        cv::Point(result[i].box[2],result[i].box[3]),
                        cv::Scalar(255, 0, 0, 255), 1);
                    std::string text = std::to_string(result[i].label);
                    putText(saveimg, text.c_str(),
                        cv::Point(result[i].box[0], result[i].box[1] + 12), 1, 2,
                        cv::Scalar(0, 0, 255, 0), 1);
                    fprintf(fpw,"%d %.6f %.6f %.6f %.6f %.6f\n",result[i].label,result[i].score,result[i].box[0],result[i].box[1],result[i].box[2],result[i].box[3]);
            }
                fclose(fpw);
                cv::imwrite(absoluteSaveImgPath,saveimg);*/
            }
        }
    }
    if (input[0].buffer != nullptr) {
        delete[](float*)input[0].buffer;
        input[0].buffer = nullptr;
    }
    vknn_release(ctx);
    return 0;
}
