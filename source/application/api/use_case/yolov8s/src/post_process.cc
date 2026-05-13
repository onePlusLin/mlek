#include "post_process.hpp"


static int compare_boxes_wrapper(const void* a, const void* b) {
    DetectionBox* boxA = (DetectionBox*)a;
    DetectionBox* boxB = (DetectionBox*)b;
    if (boxB->score > boxA->score) return 1;
    if (boxB->score < boxA->score) return -1;
    return 0;
}


int Yolov8sPostProcessor::compare_boxes(const void* a, const void* b) {
    DetectionBox* boxA = (DetectionBox*)a;
    DetectionBox* boxB = (DetectionBox*)b;
    if (boxB->score > boxA->score) return 1;
    if (boxB->score < boxA->score) return -1;
    return 0;
}


float Yolov8sPostProcessor::calc_iou(const DetectionBox* a, const DetectionBox* b) {
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


Yolov8sPostProcessor::Yolov8sPostProcessor(
    int num_boxes,
    int num_classes,
    int info_per_box,
    float conf_thresh,
    float nms_thresh,
    int max_candidates
) : m_num_boxes(num_boxes),
m_num_classes(num_classes),
m_info_per_box(info_per_box),
m_conf_thresh(conf_thresh),
m_nms_thresh(nms_thresh),
m_max_candidates(max_candidates) {
}


std::vector<std::vector<int>> Yolov8sPostProcessor::process(
    const float* input_data,
    const std::vector<std::string>& labels,
    float ratio,
    float pad_w,
    float pad_h,
    uint32_t input_w,
    uint32_t input_h
) {

    DetectionBox candidates[200];
    int candidate_count = 0;

    info("\n\nStarting YOLOv8 post-processing... Thresh(float): %.2f\n\n", m_conf_thresh);

    for (int i = 0; i < m_num_boxes; i++) {

        float max_class_score = -128.0f;
        int class_id = -1;

        const float* class_ptr = input_data + (4 * m_num_boxes) + i;

        for (int c = 0; c < m_num_classes; c++) {
            if (*class_ptr > max_class_score) {
                max_class_score = *class_ptr;
                class_id = c;
            }
            class_ptr += m_num_boxes;
        }

        if (max_class_score < m_conf_thresh) {
            continue;
        }

        if (candidate_count >= m_max_candidates) {
            info("\t候选框缓冲区已满，停止添加...");
            break;
        }

        float cx = input_data[0 * m_num_boxes + i];
        float cy = input_data[1 * m_num_boxes + i];
        float w = input_data[2 * m_num_boxes + i];
        float h = input_data[3 * m_num_boxes + i];

        cx *= input_w;
        cy *= input_h;
        w *= input_w;
        h *= input_h;

        cx -= pad_w;
        cy -= pad_h;

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


        candidates[candidate_count].area = w * h;

        candidates[candidate_count].score = max_class_score;
        candidates[candidate_count].class_id = class_id;


        candidate_count++;
    }

    if (candidate_count == 0) {
        std::vector<std::vector<int>> empty_result;
        return empty_result;
    }

    qsort(candidates, candidate_count, sizeof(DetectionBox), compare_boxes_wrapper);

    uint8_t keep[200];
    __builtin_memset(keep, 1, sizeof(keep));

    for (int i = 0; i < candidate_count; i++) {
        if (!keep[i]) continue;

        for (int j = i + 1; j < candidate_count; j++) {
            if (!keep[j]) continue;
            if (candidates[i].class_id != candidates[j].class_id) continue;

            float iou = calc_iou(&candidates[i], &candidates[j]);
            if (iou > m_nms_thresh) {
                keep[j] = 0;
            }
        }
    }
    std::vector<std::vector<int>> dected_vector;

    for (int i = 0; i < candidate_count; i++) {
        if (keep[i]) {
            const DetectionBox* b = &candidates[i];

            dected_vector.push_back({ b->class_id, (int)b->x1, (int)b->y1, (int)b->x2, (int)b->y2 });
            info("new Detected: Class %d, \tScore %.2f, \tBox [%.2f, %.2f, %.2f, %.2f],\t\tobject %s\n",
                b->class_id, b->score,
                b->x1, b->y1, b->x2-b->x1, b->y2-b->y1, labels[b->class_id].c_str());
        }
    }

    return dected_vector;
}


bool Yolov8sPostProcessor::dump_output_tensor(const int8_t* tensor_data, size_t size, const char* filename) {
    if (tensor_data == nullptr || filename == nullptr) {
        printf_err("Invalid parameters for dump_output_tensor\n");
        return false;
    }

    FILE* fp = fopen(filename, "wb");
    if (fp == nullptr) {
        printf_err("Failed to open file %s for writing\n", filename);
        return false;
    }

    size_t written = fwrite(tensor_data, sizeof(int8_t), size, fp);
    fclose(fp);

    if (written != size) {
        printf_err("Failed to write complete data to file %s\n", filename);
        return false;
    }

    info("Successfully dumped %zu int8_t values to file %s\n", size, filename);
    return true;
}