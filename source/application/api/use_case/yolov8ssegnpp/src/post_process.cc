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
      m_max_candidates(max_candidates),
      m_proto_tensor(nullptr),
      m_proto_h(0),
      m_proto_w(0)
{
    m_proto_quant.scale = 1.0f;
    m_proto_quant.offset = 0;
    m_ratio = 1.0f;
    m_pad_w = 0.0f;
    m_pad_h = 0.0f;
}

float Yolov8sNppPostProcessor::sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

OutputRole Yolov8sNppPostProcessor::identify_role(int channels, int grid_h, int grid_w)
{
    if (channels == 64) {
        return OUTPUT_ROLE_BOX;
    } else if (channels == YOLOV8SNPP_OBJ_CLASS_NUM) {
        return OUTPUT_ROLE_CLS;
    } else if (channels == 1) {
        return OUTPUT_ROLE_SCORE_SUM;
    } else if (channels == PROTO_CHANNEL) {
        if (grid_h == PROTO_HEIGHT && grid_w == PROTO_WEIGHT) {
            return OUTPUT_ROLE_PROTO;
        }
        return OUTPUT_ROLE_MASK;
    }
    return OUTPUT_ROLE_UNKNOWN;
}

float Yolov8sNppPostProcessor::dequantize(int8_t val, QuantParams params)
{
    return ((float)val - (float)params.offset) * params.scale;
}

void Yolov8sNppPostProcessor::compute_dfl(float* tensor, int dfl_len, float* box)
{
    for (int b = 0; b < 4; b++) {
        float exp_t[YOLOV8SNPP_DFL_LEN];
        float exp_sum = 0.0f;
        for (int i = 0; i < dfl_len; i++) {
            exp_t[i] = expf(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }
        float acc_sum = 0.0f;
        for (int i = 0; i < dfl_len; i++) {
            acc_sum += (exp_t[i] / exp_sum) * i;
        }
        box[b] = acc_sum;
    }
}

int Yolov8sNppPostProcessor::process_scale(
    TfLiteTensor* box_tensor,
    TfLiteTensor* cls_tensor,
    TfLiteTensor* score_sum_tensor,
    TfLiteTensor* mask_tensor,
    int grid_h,
    int grid_w,
    int stride,
    std::vector<float>& boxes,
    std::vector<float>& scores,
    std::vector<int>& class_ids,
    std::vector<float>& segments)
{
    int validCount = 0;

    QuantParams box_params = GetTensorQuantParams(box_tensor);
    QuantParams cls_params = GetTensorQuantParams(cls_tensor);

    int8_t* box_data = tflite::GetTensorData<int8_t>(box_tensor);
    int8_t* cls_data = tflite::GetTensorData<int8_t>(cls_tensor);
    int8_t* score_sum_data = score_sum_tensor ? tflite::GetTensorData<int8_t>(score_sum_tensor) : nullptr;
    int8_t* mask_data = mask_tensor ? tflite::GetTensorData<int8_t>(mask_tensor) : nullptr;

    QuantParams score_sum_params;
    if (score_sum_tensor) {
        score_sum_params = GetTensorQuantParams(score_sum_tensor);
    } else {
        score_sum_params.scale = 1.0f;
        score_sum_params.offset = 0;
    }

    QuantParams mask_params;
    if (mask_tensor) {
        mask_params = GetTensorQuantParams(mask_tensor);
    } else {
        mask_params.scale = 1.0f;
        mask_params.offset = 0;
    }

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int spatial_idx = i * grid_w + j;
            int max_class_id = -1;
            float score_sum_val = 1.0f;

            if (score_sum_data != nullptr) {
                score_sum_val = dequantize(score_sum_data[spatial_idx], score_sum_params);
                if (score_sum_val < m_conf_thresh) {
                    continue;
                }
            }

            float max_score = -1000.0f;
            int cls_base_offset = spatial_idx * YOLOV8SNPP_OBJ_CLASS_NUM;
            for (int c = 0; c < YOLOV8SNPP_OBJ_CLASS_NUM; c++) {
                float score_val = dequantize(cls_data[cls_base_offset + c], cls_params);
                if (score_val > m_conf_thresh && score_val > max_score) {
                    max_score = score_val;
                    max_class_id = c;
                }
            }

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

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;

                if (x2 <= x1 || y2 <= y1) {
                    continue;
                }

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(x2);
                boxes.push_back(y2);
                scores.push_back(final_score);
                class_ids.push_back(max_class_id);

                if (mask_data != nullptr) {
                    int mask_base_offset = spatial_idx * PROTO_CHANNEL;
                    for (int k = 0; k < PROTO_CHANNEL; k++) {
                        segments.push_back(dequantize(mask_data[mask_base_offset + k], mask_params) * score_sum_val);
                    }
                } else {
                    for (int k = 0; k < PROTO_CHANNEL; k++) {
                        segments.push_back(0.0f);
                    }
                }

                validCount++;
            }
        }
    }
    return validCount;
}

static float calculate_iou(
    float x1, float y1, float x2, float y2,
    float x1b, float y1b, float x2b, float y2b)
{
    float w = fmax(0.0f, fmin(x2, x2b) - fmax(x1, x1b));
    float h = fmax(0.0f, fmin(y2, y2b) - fmax(y1, y1b));
    float inter = w * h;
    float area1 = (x2 - x1) * (y2 - y1);
    float area2 = (x2b - x1b) * (y2b - y1b);
    float union_area = area1 + area2 - inter;
    return union_area <= 0.0f ? 0.0f : (inter / union_area);
}

static int quick_sort_indices(std::vector<float>& scores, int left, int right, std::vector<int>& indices)
{
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
    std::vector<int>& indices)
{
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

bool Yolov8sNppPostProcessor::classify_outputs(TfLiteTensor** outputs, int num_outputs,
                                               int model_input_h)
{
    m_output_info.clear();
    m_proto_tensor = nullptr;

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

        OutputRole role = identify_role(channels, grid_h, grid_w);
        if (role == OUTPUT_ROLE_UNKNOWN) {
            printf_err("Output tensor %d has unknown role (channels=%d, grid=%dx%d).\n",
                       i, channels, grid_h, grid_w);
            return false;
        }

        if (role == OUTPUT_ROLE_PROTO) {
            m_proto_tensor = tensor;
            m_proto_h = grid_h;
            m_proto_w = grid_w;
            m_proto_quant = GetTensorQuantParams(tensor);
            info("Proto tensor %d: channels=%d, grid=%dx%d (int8, on-the-fly dequant)\n",
                 i, channels, grid_h, grid_w);
            continue;
        }

        OutputTensorInfo info;
        info.tensor = tensor;
        info.role = role;
        info.grid_h = grid_h;
        info.grid_w = grid_w;
        info.stride = model_input_h / grid_h;
        info.quant_params = GetTensorQuantParams(tensor);

        m_output_info.push_back(info);
        info("Output tensor %d: role=%d, channels=%d, grid=%dx%d, stride=%d\n",
             i, role, channels, grid_h, grid_w, info.stride);
    }

    if (!m_proto_tensor) {
        printf_err("Proto tensor not found in outputs.\n");
        return false;
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
    DetectionResultList& results)
{
    memset(&results, 0, sizeof(DetectionResultList));

    m_ratio = ratio;
    m_pad_w = pad_w;
    m_pad_h = pad_h;

    if (!classify_outputs(outputs, num_outputs, model_input_h)) {
        printf_err("Failed to classify outputs\n");
        return false;
    }

    std::vector<float> all_boxes;
    std::vector<float> all_scores;
    std::vector<int>   all_class_ids;
    std::vector<float> all_segments;

    std::set<int> unique_strides;
    for (const auto& info : m_output_info) {
        if (info.role == OUTPUT_ROLE_BOX) {
            unique_strides.insert(info.stride);
        }
    }

    for (int stride : unique_strides) {
        TfLiteTensor* box_tensor = nullptr;
        TfLiteTensor* cls_tensor = nullptr;
        TfLiteTensor* score_sum_tensor = nullptr;
        TfLiteTensor* mask_tensor = nullptr;
        int grid_h = 0, grid_w = 0;

        for (const auto& info : m_output_info) {
            if (info.stride == stride) {
                if (info.role == OUTPUT_ROLE_BOX) {
                    box_tensor = info.tensor;
                    grid_h = info.grid_h;
                    grid_w = info.grid_w;
                } else if (info.role == OUTPUT_ROLE_CLS) {
                    cls_tensor = info.tensor;
                } else if (info.role == OUTPUT_ROLE_SCORE_SUM) {
                    score_sum_tensor = info.tensor;
                } else if (info.role == OUTPUT_ROLE_MASK) {
                    mask_tensor = info.tensor;
                }
            }
        }

        if (!box_tensor || !cls_tensor) {
            printf_err("Missing box or cls tensor for stride %d\n", stride);
            continue;
        }
        process_scale(box_tensor, cls_tensor, score_sum_tensor, mask_tensor,
                      grid_h, grid_w, stride,
                      all_boxes, all_scores, all_class_ids, all_segments);
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

        DetectionBox& det = results.results[result_count];

        float x1 = all_boxes[n * 4 + 0] - pad_w;
        float y1 = all_boxes[n * 4 + 1] - pad_h;
        float x2 = all_boxes[n * 4 + 2] - pad_w;
        float y2 = all_boxes[n * 4 + 3] - pad_h;

        x1 = x1 < 0 ? 0 : x1;
        y1 = y1 < 0 ? 0 : y1;
        x2 = x2 < 0 ? 0 : (x2 > model_input_w ? model_input_w : x2);
        y2 = y2 < 0 ? 0 : (y2 > model_input_h ? model_input_h : y2);

        if (x2 <= x1 || y2 <= y1) continue;

        det.x1 = x1 / ratio;
        det.y1 = y1 / ratio;
        det.x2 = x2 / ratio;
        det.y2 = y2 / ratio;
        det.score = all_scores[i];
        det.cls_id = all_class_ids[n];

        for (int k = 0; k < PROTO_CHANNEL; k++) {
            det.mask_coeffs[k] = all_segments[n * PROTO_CHANNEL + k];
        }

        result_count++;
    }

    results.count = result_count;

    return true;
}

bool Yolov8sNppPostProcessor::dump_outputs(
    TfLiteTensor** outputs,
    int num_outputs,
    int image_id,
    const char* dump_dir)
{
    const char* dir = dump_dir ? dump_dir : "/home/linzejia/app/mlek_lee/dump_outputs";

    for (int i = 0; i < num_outputs; i++) {
        TfLiteTensor* tensor = outputs[i];
        TfLiteIntArray* dims = tensor->dims;

        int channels = 0, grid_h = 0, grid_w = 0;
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

        OutputRole role = identify_role(channels, grid_h, grid_w);
        const char* role_name = "unknown";
        if (role == OUTPUT_ROLE_BOX) role_name = "box";
        else if (role == OUTPUT_ROLE_CLS) role_name = "cls";
        else if (role == OUTPUT_ROLE_SCORE_SUM) role_name = "score_sum";
        else if (role == OUTPUT_ROLE_MASK) role_name = "mask_coeffs";
        else if (role == OUTPUT_ROLE_PROTO) role_name = "proto";

        char filename[256];
        snprintf(filename, sizeof(filename), "%s/output_%d_img%d_%s_%dx%dx%d.npy",
                 dir, i, image_id, role_name, grid_h, grid_w, channels);

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
        if (dims->size == 4) {
            header << "{'descr': '<i1', 'fortran_order': False, 'shape': ("
                   << dims->data[0] << ", " << dims->data[1] << ", "
                   << dims->data[2] << ", " << dims->data[3] << "), }";
        } else {
            header << "{'descr': '<i1', 'fortran_order': False, 'shape': ("
                   << dims->data[0] << ", " << dims->data[1] << ", "
                   << dims->data[2] << "), }";
        }
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

void Yolov8sNppPostProcessor::print_mask_polygon(
    const DetectionResultList& results,
    uint8_t* scratch_buf,
    int model_w, int model_h)
{
    if (!m_proto_tensor || !scratch_buf || results.count == 0) return;

    int8_t* proto_data = tflite::GetTensorData<int8_t>(m_proto_tensor);

    static int contour_x[YOLOV8SNPP_MAX_CONTOUR_VERTS];
    static int contour_y[YOLOV8SNPP_MAX_CONTOUR_VERTS];

    static const int moore_dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    static const int moore_dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};

    for (int d = 0; d < results.count; d++) {
        const DetectionBox& det = results.results[d];

        float x1 = det.x1 * m_ratio + m_pad_w;
        float y1 = det.y1 * m_ratio + m_pad_h;
        float x2 = det.x2 * m_ratio + m_pad_w;
        float y2 = det.y2 * m_ratio + m_pad_h;

        int ix1 = static_cast<int>(x1);
        int iy1 = static_cast<int>(y1);
        int ix2 = static_cast<int>(x2);
        int iy2 = static_cast<int>(y2);

        ix1 = ix1 < 0 ? 0 : (ix1 >= model_w ? model_w - 1 : ix1);
        iy1 = iy1 < 0 ? 0 : (iy1 >= model_h ? model_h - 1 : iy1);
        ix2 = ix2 < 0 ? 0 : (ix2 >= model_w ? model_w - 1 : ix2);
        iy2 = iy2 < 0 ? 0 : (iy2 >= model_h ? model_h - 1 : iy2);

        int bw = ix2 - ix1;
        int bh = iy2 - iy1;
        if (bw <= 0 || bh <= 0) continue;

        /* Compute binary mask within bbox: dot(coeffs, proto) → sigmoid → >0.5 */
        memset(scratch_buf, 0, model_w * model_h);

        for (int y = iy1; y < iy2; y++) {
            for (int x = ix1; x < ix2; x++) {
                float sum = 0.0f;
                for (int k = 0; k < PROTO_CHANNEL; k++) {
                    int proto_idx = k * m_proto_h * m_proto_w + y * m_proto_w + x;
                    float pv = (static_cast<float>(proto_data[proto_idx]) - m_proto_quant.offset) * m_proto_quant.scale;
                    sum += det.mask_coeffs[k] * pv;
                }
                if (sigmoid(sum) > 0.5f) {
                    scratch_buf[y * model_w + x] = 1;
                }
            }
        }

        /* Moore-neighbor contour tracing */
        int mx1 = ix1;
        int my1 = iy1;

        printf("det %d: cls=%d %.1f %.1f %.1f %.1f",
               d, det.cls_id, det.x1, det.y1, det.x2, det.y2);

        for (int y = iy1; y < iy2; y++) {
            for (int x = ix1; x < ix2; x++) {
                if (scratch_buf[y * model_w + x] == 0) continue;

                int cx = x, cy = y;
                int nv = 0;

                int start_dir = 7;
                for (int dir = 0; dir < 8; dir++) {
                    int nx = cx + moore_dx[dir];
                    int ny = cy + moore_dy[dir];
                    if (nx >= 0 && nx < model_w && ny >= 0 && ny < model_h &&
                        scratch_buf[ny * model_w + nx] == 0) {
                        start_dir = dir;
                        break;
                    }
                }

                int dir = start_dir;
                int px = cx, py = cy;

                do {
                    if (nv < YOLOV8SNPP_MAX_CONTOUR_VERTS) {
                        contour_x[nv] = px;
                        contour_y[nv] = py;
                        nv++;
                    }
                    scratch_buf[py * model_w + px] = 2;

                    int found = -1;
                    for (int k = 0; k < 8; k++) {
                        int nd = (dir + 5 + k) % 8;
                        int nx = px + moore_dx[nd];
                        int ny = py + moore_dy[nd];
                        if (nx >= 0 && nx < model_w && ny >= 0 && ny < model_h &&
                            scratch_buf[ny * model_w + nx] == 1) {
                            found = nd;
                            px = nx;
                            py = ny;
                            break;
                        }
                    }

                    if (found < 0) break;
                    dir = found;
                } while (px != cx || py != cy);

                if (nv >= 3) {
                    printf(" %d", nv);
                    for (int v = 0; v < nv; v++) {
                        float orig_x = (static_cast<float>(contour_x[v]) - m_pad_w) / m_ratio;
                        float orig_y = (static_cast<float>(contour_y[v]) - m_pad_h) / m_ratio;
                        printf(" %g %g", orig_x, orig_y);
                    }
                }
                printf("\n\n");

                goto next_det;
            }
        }
        printf("\n\n");
        next_det:;
    }
}

void Yolov8sNppPostProcessor::print_results(const DetectionResultList& results)
{
    printf("count: %d\n", results.count);
    for (int i = 0; i < results.count; i++) {
        const DetectionBox& det = results.results[i];
        printf("%d %.4f %.4f %.4f %.4f %.4f [",
               det.cls_id, det.x1, det.y1, det.x2, det.y2, det.score);
        for (int j = 0; j < PROTO_CHANNEL; j++) {
            printf("%.4f%s", det.mask_coeffs[j],
                   j < PROTO_CHANNEL - 1 ? ", " : "");
        }
        printf("]\n");
    }
}

} /* namespace app */
} /* namespace arm */
