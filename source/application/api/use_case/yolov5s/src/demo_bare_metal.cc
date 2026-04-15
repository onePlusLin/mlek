
#include <stdio.h>
//#include <tvm_runtime.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h> // for qsort
#include <vector>
#include <string>
#include "log_macros.h"

#define INPUT_W         640
#define INPUT_H         640
#define NUM_BOXES       25200
#define NUM_CLASSES     80
#define INFO_PER_BOX    (5 + NUM_CLASSES) // 85


#define QP_SCALE        0.00456139f 
#define QP_ZERO_POINT   3     

// 阈值设置
#define CONF_THRESH     0.25f          // 置信度阈值 (过滤垃圾框)
#define NMS_THRESH      0.7f         // IOU 阈值 (去除重复框)
#define MAX_CANDIDATES  200           // 内存限制：由于是裸机，我们限制最多保留80个候选框

// ================= 数据结构定义 =================
typedef struct {
  float x1, y1, x2, y2; // 左上角和右下角坐标 
  float score;          // 最终得分 (obj_conf * class_conf)
  int class_id;         // 类别索引
} DetectionBox;

// ================= 辅助函数 =================
// 1. 反量化函数: uint8 -> float
static inline float dequantize(uint8_t val) {
  return (float)((float)val - QP_ZERO_POINT) * QP_SCALE;
}

// 2. 比较函数 (用于 qsort 排序，按分数降序)
int compare_boxes(const void* a, const void* b) {
  DetectionBox* boxA = (DetectionBox*)a;
  DetectionBox* boxB = (DetectionBox*)b;
  if (boxB->score > boxA->score) return 1;
  if (boxB->score < boxA->score) return -1;
  return 0;
}

// 3. 计算两个框的 IOU (交并比)
float calc_iou(DetectionBox* a, DetectionBox* b) {
  // 计算交集区域坐标
  float inter_x1 = fmaxf(a->x1, b->x1);
  float inter_y1 = fmaxf(a->y1, b->y1);
  float inter_x2 = fminf(a->x2, b->x2);
  float inter_y2 = fminf(a->y2, b->y2);

  // 计算交集面积 (如果不想交，宽高会小于0，取0)
  float w = fmaxf(0.0f, inter_x2 - inter_x1);
  float h = fmaxf(0.0f, inter_y2 - inter_y1);
  float inter_area = w * h;

  // 计算各自面积
  float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
  float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);

  // 计算并集面积
  float union_area = area_a + area_b - inter_area;

  if (union_area <= 0.00001f) return 0.0f; // 防止除零
  return inter_area / union_area;
}

// ================= 核心后处理函数 =================
// uint8[1,25200,85]
void yolo_post_process(uint8_t* input_data, const std::vector <std::string>& labels) {

  // 静态数组作为候选框缓冲区 (避免 malloc)
  static DetectionBox candidates[MAX_CANDIDATES]; // 80 个框
  int candidate_count = 0;

  // 1. 计算整数阈值 (优化：避免在循环中每次都做反量化比较)
  uint8_t conf_thresh_quant = (uint8_t)(CONF_THRESH / QP_SCALE + QP_ZERO_POINT);

  info("\n\nStarting post-processing... Thresh(int): %d\n\n", conf_thresh_quant);


  for (int i = 0; i < NUM_BOXES; i++) {
    // 获取当前框的内存指针
    uint8_t* ptr = &input_data[i * INFO_PER_BOX];

    // 过滤有效框
    if ((ptr[4] < conf_thresh_quant))
      continue;

    // 反量化框的置信度
    float obj_conf = dequantize(ptr[4]);

    // 框的类别置信度
    uint8_t max_class_score = 0;
    int class_id = -1;
    for (int c = 0; c < NUM_CLASSES; c++) {
      uint8_t score = ptr[5 + c];
      if (score > max_class_score) {
        max_class_score = score;
        class_id = c;
      }
    }

    // 最终得分 = 置信度 * 类别概率
    float final_score = obj_conf * (dequantize(max_class_score));


    // 再次过滤
    if (final_score < CONF_THRESH)
      continue;

    // 如果缓冲区满了，就不加了
    if (candidate_count >= MAX_CANDIDATES)
    {
      info("\t候选框缓冲区已满，停止添加...");
      break;
    }

    // 解码坐标 (xywh -> xyxy)
    // 解归一化
    float cx = dequantize(ptr[0]) * INPUT_W; // 如果模型输出没乘宽高，这里要乘
    float cy = dequantize(ptr[1]) * INPUT_H;
    float w = dequantize(ptr[2]) * INPUT_W;
    float h = dequantize(ptr[3]) * INPUT_H;

    candidates[candidate_count].x1 = cx - w / 2;
    candidates[candidate_count].y1 = cy - h / 2;
    candidates[candidate_count].x2 = cx + w / 2;
    candidates[candidate_count].y2 = cy + h / 2;
    candidates[candidate_count].score = final_score;
    candidates[candidate_count].class_id = class_id;

    candidate_count++;
  }

  // info("Found %d candidates. Running NMS...\n", candidate_count);

  // ==========================================
  // 第二步：NMS 处理
  // ==========================================

  // 1. 排序 (按分数从高到低)
  qsort(candidates, candidate_count, sizeof(DetectionBox), compare_boxes);

  // 2. IOU 过滤
  // 使用一个简单的 bool 数组标记是否保留
  uint8_t keep[MAX_CANDIDATES];
  for (int k = 0; k < MAX_CANDIDATES; k++) keep[k] = 1; // 初始化全保留
  // memset(keep, 1, sizeof(keep));

  for (int i = 0; i < candidate_count; i++) {
    if (keep[i] == 0) continue; // 已经被抑制的框，跳过

    for (int j = i + 1; j < candidate_count; j++) {
      if (keep[j] == 0) continue;

      // // 优化：只对同类别的物体做 NMS (如果想做跨类别NMS，去掉这个if)
      if (candidates[i].class_id != candidates[j].class_id) continue;

      // 计算 IOU
      float iou = calc_iou(&candidates[i], &candidates[j]);

      // 如果重叠严重，删除分数较低的那个 (j)
      if (iou > NMS_THRESH) {
        // printf("舍弃掉这个框%d\n", j);
        keep[j] = 0;
      }
    }
  }

  // ==========================================
  // 第三步：输出最终结果
  // ==========================================
  for (int i = 0; i < candidate_count; i++) {
    if (keep[i]) {
      DetectionBox* b = &candidates[i];
      info("Detected: Class %d, \tScore %.2f, \tBox [%d, %d, %d, %d],\t\tobject %s\n",
        b->class_id, b->score,
        (int)b->x1, (int)b->y1, (int)b->x2, (int)b->y2, labels[b->class_id].c_str());
    }
  }
}


