#include "post_process.hpp"
#include "log_macros.h"

namespace arm {
namespace app {

Yolov8sNppPostProcessor::Yolov8sNppPostProcessor(
    float conf_thresh,
    float nms_thresh,
    int max_candidates)
    : m_conf_thresh(conf_thresh),
      m_nms_thresh(nms_thresh),
      m_max_candidates(max_candidates) {}

OutputRole Yolov8sNppPostProcessor::identify_role(int channels) {
    if (channels == 64) {
        return OUTPUT_ROLE_BOX;
    } else if (channels == YOLOV8SNPP_OBJ_CLASS_NUM) {
        return OUTPUT_ROLE_CLS;
    } else if (channels == 1) {
        return OUTPUT_ROLE_SCORE_SUM;
    }
    return OUTPUT_ROLE_UNKNOWN;
}

float Yolov8sNppPostProcessor::dequantize(int8_t val, QuantParams params) {
    return ((float)val - (float)params.offset) * params.scale;
}

void Yolov8sNppPostProcessor::compute_dfl(float* tensor, int dfl_len, float* box) {
    for (int b = 0; b < 4; b++) {
        float exp_sum = 0.0f;
        float acc_sum = 0.0f;
        for (int i = 0; i < dfl_len; i++) {
            float exp_t = expf(tensor[i + b * dfl_len]);
            exp_sum += exp_t;
        }
        for (int i = 0; i < dfl_len; i++) {
            float exp_t = expf(tensor[i + b * dfl_len]);
            acc_sum += (exp_t / exp_sum) * i;
        }
        box[b] = acc_sum;
    }
}
int Yolov8sNppPostProcessor::process_scale(
    TfLiteTensor* box_tensor,
    TfLiteTensor* cls_tensor,
    TfLiteTensor* score_sum_tensor,
    int grid_h,
    int grid_w,
    int stride,
    std::vector<float>& boxes,
    std::vector<float>& scores,
    std::vector<int>& class_ids) {
    
    int validCount = 0;
    
    QuantParams box_params = GetTensorQuantParams(box_tensor);
    QuantParams cls_params = GetTensorQuantParams(cls_tensor);
    
    int8_t* box_data = tflite::GetTensorData<int8_t>(box_tensor);
    int8_t* cls_data = tflite::GetTensorData<int8_t>(cls_tensor);
    int8_t* score_sum_data = score_sum_tensor ? tflite::GetTensorData<int8_t>(score_sum_tensor) : nullptr;
    
    QuantParams score_sum_params;
    if (score_sum_tensor) {
        score_sum_params = GetTensorQuantParams(score_sum_tensor);
    } else {
        info("WARNING: score_sum_tensor is nullptr use default params.\n");
        score_sum_params.scale = 1.0f;
        score_sum_params.offset = 0;
    }
    
    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            // NHWC 空间一维索引
            int spatial_idx = i * grid_w + j;
            int max_class_id = -1;
            
            // 1. 处理 SCORE_SUM (由于通道数为1，NCHW和NHWC的索引刚好一致)
            float score_sum_val = 1.0f;  // default if no score_sum tensor
            if (score_sum_data != nullptr) {
                score_sum_val = dequantize(score_sum_data[spatial_idx], score_sum_params);

                // 【注意】如果 RKNN 导出时去掉了 Sigmoid，这里的值是 Logit！
                // 如果是 Logit，需要放开下面这行代码进行 Sigmoid 激活：
                // score_sum_val = 1.0f / (1.0f + expf(-score_sum_val));

                if (score_sum_val < m_conf_thresh) {
                    continue; // 快速过滤背景
                }
            }
            
            // 2. 处理分类分数 (NHWC排布：通道连续)
            float max_score = -1000.0f; // 初始化为一个很小的负数，兼容Logit
            int cls_base_offset = spatial_idx * YOLOV8SNPP_OBJ_CLASS_NUM;
            for (int c = 0; c < YOLOV8SNPP_OBJ_CLASS_NUM; c++) {
                float score_val = dequantize(cls_data[cls_base_offset + c], cls_params);
                
                // 【注意】如果模型输出的是未经过 Sigmoid 的 Logits，需要放开下面这行：
                // score_val = 1.0f / (1.0f + expf(-score_val));

                if (score_val > m_conf_thresh && score_val > max_score) {
                    max_score = score_val;
                    max_class_id = c;
                }
            }
            
            // 3. 处理边界框 (NHWC排布：通道连续)
            if (max_class_id >= 0) {
                float final_score = max_score * score_sum_val;
                if (final_score < m_conf_thresh) {
                    continue;
                }

                float box[4];
                float before_dfl[YOLOV8SNPP_DFL_LEN * 4];

                int box_base_offset = spatial_idx * (YOLOV8SNPP_DFL_LEN * 4);
                for (int k = 0; k < YOLOV8SNPP_DFL_LEN * 4; k++) {
                    before_dfl[k] = dequantize(box_data[box_base_offset + k], box_params);
                }
                compute_dfl(before_dfl, YOLOV8SNPP_DFL_LEN, box);

                // 还原回原图尺度
                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(x2);
                boxes.push_back(y2);
                scores.push_back(final_score);
                class_ids.push_back(max_class_id);
                validCount++;
            }
        }
    }
    
    return validCount;
}
static float calculate_iou(
    float x1, float y1, float x2, float y2,
    float x1b, float y1b, float x2b, float y2b) {
    
    float w = fmax(0.0f, fmin(x2, x2b) - fmax(x1, x1b));
    float h = fmax(0.0f, fmin(y2, y2b) - fmax(y1, y1b));
    float inter = w * h;
    float area1 = (x2 - x1) * (y2 - y1);
    float area2 = (x2b - x1b) * (y2b - y1b);
    float union_area = area1 + area2 - inter;
    return union_area <= 0.0f ? 0.0f : (inter / union_area);
}

static int quick_sort_indices(std::vector<float>& scores, int left, int right, std::vector<int>& indices) {
    if (left < right) {
        float pivot = scores[left];
        int pivot_idx = indices[left];
        int low = left, high = right;
        
        while (low < high) {
            while (low < high && scores[high] <= pivot) high--;
            scores[low] = scores[high];
            indices[low] = indices[high];
            
            while (low < high && scores[low] >= pivot) low++;
            scores[high] = scores[low];
            indices[high] = indices[low];
        }
        
        scores[low] = pivot;
        indices[low] = pivot_idx;
        
        quick_sort_indices(scores, left, low - 1, indices);
        quick_sort_indices(scores, low + 1, right, indices);
    }
    return 0;
}

void Yolov8sNppPostProcessor::nms(
    std::vector<float>& boxes,
    std::vector<float>& scores,
    std::vector<int>& class_ids,
    std::vector<int>& indices) {
    
    int count = scores.size();
    for (int i = 0; i < count; i++) {
        int n = indices[i];
        if (n == -1) continue;
        
        float x1a = boxes[n * 4 + 0];
        float y1a = boxes[n * 4 + 1];
        float x2a = boxes[n * 4 + 2];
        float y2a = boxes[n * 4 + 3];
        
        for (int j = i + 1; j < count; j++) {
            int m = indices[j];
            if (m == -1 || class_ids[m] != class_ids[n]) continue;
            
            float x1b = boxes[m * 4 + 0];
            float y1b = boxes[m * 4 + 1];
            float x2b = boxes[m * 4 + 2];
            float y2b = boxes[m * 4 + 3];
            
            float iou = calculate_iou(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b);
            if (iou > m_nms_thresh) {
                indices[j] = -1;
            }
        }
    }
}

bool Yolov8sNppPostProcessor::classify_outputs(TfLiteTensor** outputs, int num_outputs) {
    m_output_info.clear();
    
    for (int i = 0; i < num_outputs; i++) {
        TfLiteTensor* tensor = outputs[i];
        TfLiteIntArray* dims = tensor->dims;
        
        int channels = 0;
        int grid_h = 0, grid_w = 0;
        
        if (dims->size == 4) {
            channels = dims->data[3];
            grid_h = dims->data[1];
            grid_w = dims->data[2];
        } else if (dims->size == 3) {
            channels = dims->data[2];
            grid_h = dims->data[1];
            grid_w = 1;
        } else {
            printf_err("Unsupported output tensor dimension: %d\n", dims->size);
            return false;
        }
        
        OutputRole role = identify_role(channels);
        if(role == OUTPUT_ROLE_UNKNOWN) 
        {
            printf_err("output tensor %d has unknown role.\n", i);
            return false;
        }
            
        
        int scale_idx = 0;
        
        for (const auto& info : m_output_info) {
            if (info.role == role) {
                scale_idx++;
                if(scale_idx > 2)
                {
                    printf_err("output tensor %d has more than 3 scales which is wrong.\n", i);
                    return false;
                }
            }
        }
        
        OutputTensorInfo info;
        info.tensor = tensor;
        info.role = role;
        info.scale_idx = scale_idx;
        info.grid_h = grid_h;
        info.grid_w = grid_w;
        info.stride = 640 / grid_h;         // 如果是减低分辨率的话，这里应该是规定是H除以grid_h，应该传进来input H
        info.quant_params = GetTensorQuantParams(tensor);
        
        m_output_info.push_back(info);
        ::info("Output tensor %d: role=%d, channels=%d, grid=%dx%d, stride=%d\n", 
               i, role, channels, grid_h, grid_w, info.stride);
    }
    
    return true;
}

bool Yolov8sNppPostProcessor::process(
    TfLiteTensor** outputs,
    int num_outputs,
    int model_input_w,
    int model_input_h,
    float ratio,
    float pad_w,
    float pad_h,
    DetectionResultList& results) {
    
    memset(&results, 0, sizeof(DetectionResultList));
    
    if (!classify_outputs(outputs, num_outputs)) {
        printf_err("Failed to classify outputs\n");
        return false;
    }
    
    std::vector<float> all_boxes;       // role = OUTPUT_ROLE_BOX         1 
    std::vector<float> all_scores;      // role = OUTPUT_ROLE_SCORE_SUM   2      
    std::vector<int> all_class_ids;     // role = OUTPUT_ROLE_CLS         3  
    
    for (int scale_idx = 0; scale_idx < 3; scale_idx++) {
        TfLiteTensor* box_tensor = nullptr;
        TfLiteTensor* cls_tensor = nullptr;
        TfLiteTensor* score_sum_tensor = nullptr;
        int grid_h = 0, grid_w = 0, stride = 0;
        
        for (const auto& info : m_output_info) {
            if (info.scale_idx == scale_idx) {
                if (info.role == OUTPUT_ROLE_BOX) {
                    box_tensor = info.tensor;
                    grid_h = info.grid_h;
                    grid_w = info.grid_w;
                    stride = info.stride;
                } else if (info.role == OUTPUT_ROLE_CLS) {
                    cls_tensor = info.tensor;
                } else if (info.role == OUTPUT_ROLE_SCORE_SUM) {
                    score_sum_tensor = info.tensor;
                }
            }
        }
        
        if (!box_tensor || !cls_tensor) {
            printf_err("Missing box or cls tensor for scale %d\n", scale_idx);
            continue;
        }
        
        process_scale(box_tensor, cls_tensor, score_sum_tensor,
                      grid_h, grid_w, stride,
                      all_boxes, all_scores, all_class_ids);
    }
    
    int validCount = all_scores.size();
    if (validCount == 0) {
        info("No detections found\n");
        return true;
    }
    
    std::vector<int> indices(validCount);
    for (int i = 0; i < validCount; i++) {
        indices[i] = i;
    }
    
    quick_sort_indices(all_scores, 0, validCount - 1, indices);
    
    nms(all_boxes, all_scores, all_class_ids, indices);
    
    int result_count = 0;
    for (int i = 0; i < validCount && result_count < m_max_candidates; i++) {
        int n = indices[i];
        if (n == -1) continue;
        
        float x1 = all_boxes[n * 4 + 0] - pad_w;
        float y1 = all_boxes[n * 4 + 1] - pad_h;
        float x2 = all_boxes[n * 4 + 2] - pad_w;
        float y2 = all_boxes[n * 4 + 3] - pad_h;
        
        x1 = x1 < 0 ? 0 : x1;
        y1 = y1 < 0 ? 0 : y1;
        x2 = x2 > model_input_w ? model_input_w : x2;
        y2 = y2 > model_input_h ? model_input_h : y2;
        
        results.results[result_count].x1 = x1 / ratio;
        results.results[result_count].y1 = y1 / ratio;
        results.results[result_count].x2 = x2 / ratio;
        results.results[result_count].y2 = y2 / ratio;
        results.results[result_count].score = all_scores[i];
        results.results[result_count].cls_id = all_class_ids[n];
        result_count++;
    }
    
    results.count = result_count;
    
    return true;
}

bool Yolov8sNppPostProcessor::dump_outputs(
    TfLiteTensor** outputs,
    int num_outputs,
    int image_id,
    const char* dump_dir) {
    
    const char* dir = dump_dir ? dump_dir : "/home/linzejia/app/mlek_lee/dump_outputs";
    
    for (int i = 0; i < num_outputs; i++) {
        TfLiteTensor* tensor = outputs[i];
        TfLiteIntArray* dims = tensor->dims;
        
        int channels = dims->data[3];
        int grid_h = dims->data[1];
        int grid_w = dims->data[2];
        
        OutputRole role = identify_role(channels);
        const char* role_name = "unknown";
        if (role == OUTPUT_ROLE_BOX) role_name = "box";
        else if (role == OUTPUT_ROLE_CLS) role_name = "cls";
        else if (role == OUTPUT_ROLE_SCORE_SUM) role_name = "score_sum";
        
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/output_%d_img%d_%s_%dx%d.npy",
                 dir, i, image_id, role_name, grid_h, grid_w);
        
        FILE* fp = fopen(filename, "wb");
        if (!fp) {
            printf_err("Failed to open %s for writing\n", filename);
            return false;
        }
        
        const char magic[] = "\x93NUMPY";
        fwrite(magic, 1, 6, fp);
        
        uint8_t major_version = 1;
        uint8_t minor_version = 0;
        fwrite(&major_version, 1, 1, fp);
        fwrite(&minor_version, 1, 1, fp);
        
        std::ostringstream header;
        header << "{'descr': '<i1', 'fortran_order': False, 'shape': (" 
               << dims->data[0] << ", " << dims->data[1] << ", " 
               << dims->data[2] << ", " << dims->data[3] << "), }";
        std::string header_str = header.str();
        
        size_t header_len = header_str.length();
        size_t padding = (16 - (header_len + 1) % 16) % 16;
        header_str.append(padding, ' ');
        header_str.push_back('\n');
        
        uint16_t header_len_le = static_cast<uint16_t>(header_str.length());
        fwrite(&header_len_le, 2, 1, fp);
        fwrite(header_str.c_str(), 1, header_str.length(), fp);
        
        int8_t* data = tflite::GetTensorData<int8_t>(tensor);
        size_t data_size = tensor->bytes;
        fwrite(data, 1, data_size, fp);
        fclose(fp);
        
        info("Dumped output %d to %s\n", i, filename);
    }
    
    return true;
}

void Yolov8sNppPostProcessor::print_results(const DetectionResultList& results,
                                            const std::vector<std::string>* labels) {
    info("Detection results (count: %d):\n", results.count);
    for (int i = 0; i < results.count; i++) {
        const DetectionBox& box = results.results[i];
        const char* label = "unknown";
        if (labels && box.cls_id >= 0 && box.cls_id < (int)labels->size()) {
            label = labels->at(box.cls_id).c_str();
        }
        info("  [%d] Class: %s (%d), Box: (%.2f, %.2f, %.2f, %.2f), Score: %.4f\n",
             i, label, box.cls_id, box.x1, box.y1, box.x2-box.x1, box.y2-box.y1, box.score);
    }
}

} /* namespace app */
} /* namespace arm */