#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h> 
#include <vector>
#include <string>
#include "log_macros.h"

typedef struct {
    float x1, y1, x2, y2;
    float score;
    float area;
    int class_id;
} DetectionBox;

class Yolov8sPostProcessor {
    public:
    Yolov8sPostProcessor(
        int num_boxes = 8400,
        int num_classes = 80,
        int info_per_box = 84,
        float conf_thresh = 0.25f,
        float nms_thresh = 0.7f,
        int max_candidates = 200
    );

    void set_num_boxes(int num_boxes) { m_num_boxes = num_boxes; }
    void set_num_classes(int num_classes) { m_num_classes = num_classes; }
    void set_info_per_box(int info_per_box) { m_info_per_box = info_per_box; }
    void set_conf_thresh(float conf_thresh) { m_conf_thresh = conf_thresh; }
    void set_nms_thresh(float nms_thresh) { m_nms_thresh = nms_thresh; }
    void set_max_candidates(int max_candidates) { m_max_candidates = max_candidates; }

    std::vector<std::vector<int>> process(
        const float* input_data,
        const std::vector<std::string>& labels,
        float ratio,
        float pad_w,
        float pad_h,
        uint32_t input_w,
        uint32_t input_h
    );

    bool dump_output_tensor(const int8_t* tensor_data, size_t size, const char* filename);

    private:
    int compare_boxes(const void* a, const void* b);
    float calc_iou(const DetectionBox* a, const DetectionBox* b);

    int m_num_boxes;
    int m_num_classes;
    int m_info_per_box;
    float m_conf_thresh;
    float m_nms_thresh;
    int m_max_candidates;
};