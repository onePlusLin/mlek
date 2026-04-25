#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h> 
#include <vector>
#include <string>
#include "log_macros.h"
#include "post_process.hpp"

#define INPUT_W         640
#define INPUT_H         480
#define NUM_BOXES       6300    
#define NUM_CLASSES     80
#define INFO_PER_BOX    84      


#define CONF_THRESH     0.25f          
#define NMS_THRESH      0.7f         
#define MAX_CANDIDATES  200           




static inline float dequantize(int8_t val, float scale, int zero_point) {
    return (float)((int)val - zero_point) * scale;
}

int compare_boxes(const void* a, const void* b) {
    DetectionBox* boxA = (DetectionBox*)a;
    DetectionBox* boxB = (DetectionBox*)b;
    if (boxB->score > boxA->score) return 1;
    if (boxB->score < boxA->score) return -1;
    return 0;
}


static inline float calc_iou(const DetectionBox* a, const DetectionBox* b) {
    float inter_x1 = fmaxf(a->x1, b->x1);
    float inter_y1 = fmaxf(a->y1, b->y1);
    float inter_x2 = fminf(a->x2, b->x2);
    float inter_y2 = fminf(a->y2, b->y2);

    float w = fmaxf(0.0f, inter_x2 - inter_x1);
    float h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = w * h;


    float union_area = a->area + b->area - inter_area;

    if (union_area <= 0.00001f) return 0.0f;
    return inter_area / union_area;
}


// void yolov8s_post_process(int8_t* input_data, const std::vector<std::string>& labels, float scale, int zero_point) {
std::vector<DetectionBox> yolov8s_post_process(int8_t* input_data, const std::vector<std::string>& labels, float scale, int zero_point) {

    DetectionBox candidates[MAX_CANDIDATES];
    int candidate_count = 0;
    std::vector<DetectionBox> dected_vector;

    //  roundf() 防止浮点截断导致阈值小幅偏差
    int thresh_calc = (int)roundf(CONF_THRESH / scale) + zero_point;
    if (thresh_calc < -128) thresh_calc = -128;
    if (thresh_calc > 127)  thresh_calc = 127;
    int8_t conf_thresh_quant = (int8_t)thresh_calc;

    info("\n\nStarting YOLOv8 post-processing... Thresh(int8): %d\n\n", conf_thresh_quant);

    // 遍历所有 8400 个框
    for (int i = 0; i < NUM_BOXES; i++) {

        int8_t max_class_score = -128;
        int class_id = -1;

        // 优化5：指针步进代替乘法。跳过前4个坐标，将指针定位到第0个类别的起始地址
        const int8_t* class_ptr = input_data + (4 * NUM_BOXES) + i;

        // 寻找 80 个类别中的最大得分
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (*class_ptr > max_class_score) {
                max_class_score = *class_ptr;
                class_id = c;
            }
            // 每次向前跳跃 8400 步长，到达下一个类别
            class_ptr += NUM_BOXES;
        }

        // 使用整数阈值进行第一轮淘汰
        if (max_class_score < conf_thresh_quant) {
            continue;
        }

        // 检查缓冲区
        if (candidate_count >= MAX_CANDIDATES) {
            info("\t候选框缓冲区已满，停止添加...");
            break;
        }

        // 提取坐标并反量化
        float cx = dequantize(input_data[0 * NUM_BOXES + i], scale, zero_point);
        float cy = dequantize(input_data[1 * NUM_BOXES + i], scale, zero_point);
        float w = dequantize(input_data[2 * NUM_BOXES + i], scale, zero_point);
        float h = dequantize(input_data[3 * NUM_BOXES + i], scale, zero_point);

        cx *= INPUT_W; cy *= INPUT_H;
        w *= INPUT_W; h *= INPUT_H;

        float half_w = w / 2.0f;
        float half_h = h / 2.0f;

        candidates[candidate_count].x1 = cx - half_w;
        candidates[candidate_count].y1 = cy - half_h;
        candidates[candidate_count].x2 = cx + half_w;
        candidates[candidate_count].y2 = cy + half_h;

        // 预计算面积，供 NMS 高速使用
        candidates[candidate_count].area = w * h;

        candidates[candidate_count].score = dequantize(max_class_score, scale, zero_point);
        candidates[candidate_count].class_id = class_id;

        candidate_count++;
    }

    // ==========================================
    // 第二步：NMS 处理 
    // ==========================================
    if (candidate_count == 0)
    {
        return dected_vector;
    }

    qsort(candidates, candidate_count, sizeof(DetectionBox), compare_boxes);

    uint8_t keep[MAX_CANDIDATES];
    // 使用指针偏移初始化数组，比循环赋值稍微快一点点
    __builtin_memset(keep, 1, sizeof(keep));

    for (int i = 0; i < candidate_count; i++) {
        if (!keep[i]) continue;

        for (int j = i + 1; j < candidate_count; j++) {
            if (!keep[j]) continue;
            if (candidates[i].class_id != candidates[j].class_id) continue;

            float iou = calc_iou(&candidates[i], &candidates[j]);
            if (iou > NMS_THRESH) {
                keep[j] = 0;
            }
        }
    }


    // ==========================================
    // 第三步：输出最终结果
    // ==========================================
    for (int i = 0; i < candidate_count; i++) {
        if (keep[i]) {
            const DetectionBox* b = &candidates[i];

            dected_vector.push_back(*b);
            info("Detected: Class %d, \tScore %.2f, \tBox [%d, %d, %d, %d],\t\tobject %s\n",
                b->class_id, b->score,
                (int)b->x1, (int)b->y1, (int)b->x2, (int)b->y2, labels[b->class_id].c_str());
        }
    }

    return dected_vector;
    // 返回boxes 坐标、类别

}
