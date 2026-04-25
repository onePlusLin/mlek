#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h> 
#include <vector>
#include <string>
#include "log_macros.h"


#define NUM_BOXES       8400    
#define NUM_CLASSES     80
#define INFO_PER_BOX    84      


#define CONF_THRESH     0.25f          
#define NMS_THRESH      0.7f         
#define MAX_CANDIDATES  200           


typedef struct {
    float x1, y1, x2, y2;
    float score;
    float area;
    int class_id;
} DetectionBox;



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
void yolov8s_post_process(
    const float* input_data,
    const std::vector<std::string>& labels,
    float ratio,
    float pad_w,
    float pad_h,
    uint32_t input_w,
    uint32_t input_h
) {

    DetectionBox candidates[MAX_CANDIDATES];
    int candidate_count = 0;

    // 输入数据已经是反量化后的 float，直接使用 CONF_THRESH
    info("\n\nStarting YOLOv8 post-processing... Thresh(float): %.2f\n\n", CONF_THRESH);

    // 遍历所有 8400 个框
    for (int i = 0; i < NUM_BOXES; i++) {

        float max_class_score = -128.0f;
        int class_id = -1;

        // 优化5：指针步进代替乘法。跳过前4个坐标，将指针定位到第0个类别的起始地址
        const float* class_ptr = input_data + (4 * NUM_BOXES) + i;

        // 寻找 80 个类别中的最大得分
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (*class_ptr > max_class_score) {
                max_class_score = *class_ptr;
                class_id = c;
            }
            // 每次向前跳跃 8400 步长，到达下一个类别
            class_ptr += NUM_BOXES;
        }

        // 使用浮点阈值进行第一轮淘汰
        if (max_class_score < CONF_THRESH) {
            continue;
        }

        // 检查缓冲区
        if (candidate_count >= MAX_CANDIDATES) {
            info("\t候选框缓冲区已满，停止添加...");
            break;
        }

        // 提取坐标（输入已经是反量化的 float）
        float cx = input_data[0 * NUM_BOXES + i];
        float cy = input_data[1 * NUM_BOXES + i];
        float w = input_data[2 * NUM_BOXES + i];
        float h = input_data[3 * NUM_BOXES + i];

        // 1. 去归一化：乘以图片的宽高
        cx *= input_w;
        cy *= input_h;
        w *= input_w;
        h *= input_h;

        // 2. 对 cx, cy 去掉 padding
        cx -= pad_w;
        cy -= pad_h;

        // 3. 将坐标除以 ratio 还原到原始图片尺寸
        cx /= ratio;
        cy /= ratio;
        w /= ratio;
        h /= ratio;

        float half_w = w / 2.0f;
        float half_h = h / 2.0f;

        candidates[candidate_count].x1 = cx - half_w;
        candidates[candidate_count].y1 = cy - half_h;
        candidates[candidate_count].x2 = cx + half_w;
        candidates[candidate_count].y2 = cy + half_h;


        // 预计算面积，供 NMS 高速使用
        candidates[candidate_count].area = w * h;

        candidates[candidate_count].score = max_class_score;
        candidates[candidate_count].class_id = class_id;


        candidate_count++;
    }

    // ==========================================
    // 第二步：NMS 处理 
    // ==========================================
    if (candidate_count == 0) return; // 提前退出，没有检测到物体

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
    std::vector<std::vector<int>> dected_vector;

    // ==========================================
    // 第三步：输出最终结果
    // ==========================================
    for (int i = 0; i < candidate_count; i++) {
        if (keep[i]) {
            const DetectionBox* b = &candidates[i];

            dected_vector.push_back({ b->class_id, (int)b->x1, (int)b->y1, (int)b->x2, (int)b->y2 });
            info("Detected: Class %d, \tScore %.2f, \tBox [%d, %d, %d, %d],\t\tobject %s\n",
                b->class_id, b->score,
                (int)b->x1, (int)b->y1, (int)b->x2, (int)b->y2, labels[b->class_id].c_str());
        }
    }

    // return dected_vector;
    // // 返回boxes 坐标、类别

}