#ifndef YOLOV8SSEG_NPP_POST_PROCESS_HPP
#define YOLOV8SSEG_NPP_POST_PROCESS_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <set>
#include <sstream>

#include "tensorflow/lite/c/common.h"
#include "TensorFlowLiteMicro.hpp"

namespace arm {
namespace app {

/* Constants */
#define YOLOV8SNPP_OBJ_CLASS_NUM 80
#define YOLOV8SNPP_OBJ_NUMB_MAX_SIZE 200
#define YOLOV8SNPP_NMS_THRESH 0.45f
#define YOLOV8SNPP_BOX_THRESH 0.25f
#define YOLOV8SNPP_DFL_LEN 16
#define YOLOV8SNPP_MAX_CONTOUR_VERTS 1024

#define PROTO_CHANNEL 32
#define PROTO_HEIGHT   160
#define PROTO_WEIGHT   160

/**
 * @brief Detection box structure with segmentation info
 */
struct DetectionBox {
    float x1;               // top-left x
    float y1;               // top-left y
    float x2;               // bottom-right x
    float y2;               // bottom-right y
    float score;            // confidence score
    int cls_id;             // class ID
    float mask_coeffs[PROTO_CHANNEL]; // mask coefficients (32 floats)
};

/**
 * @brief Detection result list
 */
struct DetectionResultList {
    int count;
    DetectionBox results[YOLOV8SNPP_OBJ_NUMB_MAX_SIZE];
};

/**
 * @brief Output tensor role enum
 */
typedef enum {
    OUTPUT_ROLE_UNKNOWN = 0,
    OUTPUT_ROLE_BOX,       // box tensor with 64 channels (4*16 DFL)
    OUTPUT_ROLE_CLS,       // class tensor with 80 channels
    OUTPUT_ROLE_SCORE_SUM, // score sum tensor with 1 channel
    OUTPUT_ROLE_MASK,      // mask coefficients with 32 channels
    OUTPUT_ROLE_PROTO      // proto mask: 32 channels, 160x160 grid
} OutputRole;

/**
 * @brief Output tensor info structure
 */
struct OutputTensorInfo {
    TfLiteTensor* tensor;
    OutputRole role;
    int grid_h;
    int grid_w;
    int stride;
    QuantParams quant_params;
};

/**
 * @brief YOLOv8s-seg NPP post processor class
 *
 * No heap allocations — proto is kept as int8 and dequantized on-the-fly.
 * Mask coefficients are stored per detection (32 floats each).
 */
class Yolov8sNppPostProcessor {
public:
    Yolov8sNppPostProcessor(
        float conf_thresh = YOLOV8SNPP_BOX_THRESH,
        float nms_thresh = YOLOV8SNPP_NMS_THRESH,
        int max_candidates = YOLOV8SNPP_OBJ_NUMB_MAX_SIZE);

    /**
     * @brief Process multiple output tensors and generate detection results
     * @param outputs          List of output tensors
     * @param num_outputs      Number of output tensors
     * @param model_input_w    Model input width
     * @param model_input_h    Model input height
     * @param ratio            Image scaling ratio
     * @param pad_w            Width padding
     * @param pad_h            Height padding
     * @param results          Output detection results (with mask coefficients)
     * @return true if successful
     */
    bool process(
        TfLiteTensor** outputs,
        int num_outputs,
        int model_input_w,
        int model_input_h,
        float ratio,
        float pad_w,
        float pad_h,
        DetectionResultList& results);

    /**
     * @brief Dump output tensors to npy files
     * @param outputs          List of output tensors
     * @param num_outputs      Number of output tensors
     * @param image_id         Image ID for filename
     * @param dump_dir         Dump directory path
     * @return true if successful
     */
    bool dump_outputs(
        TfLiteTensor** outputs,
        int num_outputs,
        int image_id,
        const char* dump_dir = nullptr);

    /**
     * @brief Print detection count (brief, to info log).
     */
    void print_results(const DetectionResultList& results);

    /**
     * @brief Compute per-detection binary masks and print box + polygon.
     *
     * For each detection:
     *   1. Compute mask within bbox at model resolution:
     *      dot(coeffs[32], proto) → sigmoid → >0.5
     *   2. Moore-neighbor contour tracing → vertices in model coords
     *   3. Map vertices to original image coords via letterbox reverse
     *   4. printf: "cl x1 y1 x2 y2 vx vy ...\n"
     *
     * @param results          Detection results (after NMS, with mask_coeffs).
     * @param scratch_buf      Scratch buffer (model_w × model_h bytes), pre-zeroed.
     * @param model_w          Model input width.
     * @param model_h          Model input height.
     */
    void print_mask_polygon(const DetectionResultList& results,
                            uint8_t* scratch_buf,
                            int model_w, int model_h);

private:
    /**
     * @brief Identify output role by channel count and grid dimensions
     */
    OutputRole identify_role(int channels, int grid_h, int grid_w);

    /**
     * @brief Classify outputs by role and scale
     */
    bool classify_outputs(TfLiteTensor** outputs, int num_outputs,
                          int model_input_h);

    /**
     * @brief Dequantize int8 to float
     */
    float dequantize(int8_t val, QuantParams params);

    /**
     * @brief Compute DFL (Distribution Focal Loss) decoding
     */
    void compute_dfl(float* tensor, int dfl_len, float* box);

    /**
     * @brief Sigmoid activation
     */
    static float sigmoid(float x);

    /**
     * @brief Process single scale outputs
     */
    int process_scale(
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
        std::vector<float>& segments);

    /**
     * @brief NMS (Non-Maximum Suppression)
     */
    void nms(
        std::vector<float>& boxes,
        std::vector<float>& scores,
        std::vector<int>& class_ids,
        std::vector<int>& indices);

    float m_conf_thresh;
    float m_nms_thresh;
    int m_max_candidates;
    std::vector<OutputTensorInfo> m_output_info;

    /* Proto mask — kept as int8 ref, dequantized on-the-fly */
    TfLiteTensor* m_proto_tensor;
    QuantParams m_proto_quant;
    int m_proto_h;
    int m_proto_w;

    /* Letterbox params saved during process(), used by print_mask_polygon() */
    float m_ratio;
    float m_pad_w;
    float m_pad_h;
};

} /* namespace app */
} /* namespace arm */

#endif /* YOLOV8SSEG_NPP_POST_PROCESS_HPP */
