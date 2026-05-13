#ifndef YOLOV8SNPP_POST_PROCESS_HPP
#define YOLOV8SNPP_POST_PROCESS_HPP

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

/**
 * @brief Detection box structure
 */
struct DetectionBox {
    float x1;      // top-left x
    float y1;      // top-left y
    float x2;      // bottom-right x
    float y2;      // bottom-right y
    float score;   // confidence score
    int cls_id;    // class ID
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
    OUTPUT_ROLE_SCORE_SUM  // score sum tensor with 1 channel
} OutputRole;

/**
 * @brief Output tensor info structure
 */
struct OutputTensorInfo {
    TfLiteTensor* tensor;
    OutputRole role;
    int scale_idx;         // scale index (0, 1, 2 for 3 scales)
    int grid_h;
    int grid_w;
    int stride;
    QuantParams quant_params;
};

/**
 * @brief YOLOv8sNpp post processor class
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
     * @param results          Output detection results                       
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
     * @brief Print detection results
     * @param results          Detection results
     * @param labels           Class labels (optional)
     */
    void print_results(const DetectionResultList& results, 
                       const std::vector<std::string>* labels = nullptr);

private:
    /**
     * @brief Identify output role by channel count
     */
    OutputRole identify_role(int channels);

    /**
     * @brief Classify outputs by role and scale
     */
    bool classify_outputs(TfLiteTensor** outputs, int num_outputs);

    /**
     * @brief Dequantize int8 to float
     */
    float dequantize(int8_t val, QuantParams params);

    /**
     * @brief Compute DFL (Distribution Focal Loss) decoding
     */
    void compute_dfl(float* tensor, int dfl_len, float* box);

    /**
     * @brief Process single scale outputs
     */
    int process_scale(
        TfLiteTensor* box_tensor,
        TfLiteTensor* cls_tensor,
        TfLiteTensor* score_sum_tensor,
        int grid_h,
        int grid_w,
        int stride,
        std::vector<float>& boxes,
        std::vector<float>& scores,
        std::vector<int>& class_ids);

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
};

} /* namespace app */
} /* namespace arm */

#endif /* YOLOV8SNPP_POST_PROCESS_HPP */