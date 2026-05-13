/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef IMG_CLASS_PROCESSING_HPP
#define IMG_CLASS_PROCESSING_HPP

#include "BaseProcessing.hpp"
#include "Classifier.hpp"
#include "post_process.hpp"

namespace arm {
    namespace app {

        /**
         * @brief   Pre-processing class for Image Classification use case.
         *          Implements methods declared by BasePreProcess and anything else needed
         *          to populate input tensors ready for inference.
         */
        class ImgClassPreProcess : public BasePreProcess {

            public:
            /**
             * @brief       Constructor
             * @param[in]   inputTensor     Pointer to the TFLite Micro input Tensor.
             * @param[in]   convertToInt8   Should the image be converted to Int8 range.
             **/
            explicit ImgClassPreProcess(TfLiteTensor* inputTensor, bool convertToInt8);

            /**
             * @brief       Should perform pre-processing of 'raw' input image data and load it into
             *              TFLite Micro input tensors ready for inference
             * @param[in]   input      Pointer to the data that pre-processing will work on.
             * @param[in]   inputSize  Size of the input data.
             * @return      true if successful, false otherwise.
             **/
            bool DoPreProcess(const void* input, size_t inputSize) override;

            private:
            TfLiteTensor* m_inputTensor;
            bool m_convertToInt8;
        };

        /**
         * @brief   Post-processing class for Image Classification use case.
         *          Implements methods declared by BasePostProcess and anything else needed
         *          to populate result vector.
         */
        class ImgClassPostProcess : public BasePostProcess {

            public:
            /**
             * @brief       Constructor
             * @param[in]   outputTensor       Pointer to the TFLite Micro output Tensor.
             * @param[in]   classifier         Classifier object used to get top N results from classification.
             * @param[in]   labels             Vector of string labels to identify each output of the model.
             * @param[in]   results            Vector of classification results to store decoded outputs.
             * @param[in]   output_num_boxes   Number of output boxes (e.g., 8400 or 6300).
             * @param[in]   output_num_classes Number of output classes (e.g., 84).
             * @param[in]   conf_thresh        Confidence threshold for detection.
             * @param[in]   nms_thresh         NMS IoU threshold.
             * @param[in]   max_candidates     Maximum number of candidate boxes.
             **/
            ImgClassPostProcess(
                TfLiteTensor* outputTensor,
                Classifier& classifier,
                const std::vector<std::string>& labels,
                std::vector<ClassificationResult>& results,
                uint32_t output_num_boxes = 8400,
                uint8_t output_num_classes = 84,
                float conf_thresh = 0.25f,
                float nms_thresh = 0.7f,
                int max_candidates = 200);

            /**
             * @brief       Should perform post-processing of the result of inference then
             *              populate classification result data for any later use.
             * @return      true if successful, false otherwise.
             **/
            bool DoPostProcess() override;

            /**
             * @brief       Dump output tensor data to file
             * @param[in]   filename  Path to the output file
             * @return      true if successful, false otherwise.
             **/
            bool DumpOutputTensor(const char* filename);

            /**
             * @brief       Set image ID
             * @param[in]   imageId    Image ID extracted from filename.
             **/
            void SetModelInputShape(uint32_t input_w, uint32_t input_h);
            void SetImageId(int imageId);
            void SetRatio(float ratio);
            void SetPadW(float pad_w);
            void SetPadH(float pad_h);

            private:
            TfLiteTensor* m_outputTensor;
            Classifier& m_imgClassifier;
            const std::vector<std::string>& m_labels;
            std::vector<ClassificationResult>& m_results;
            int m_image_id = 0;
            float m_ratio = 0.0f;              // 图片预处理的缩放比例
            float m_pad_w = 0.0f;              // 图片预处理时候的宽度填充值
            float m_pad_h = 0.0f;              // 图片预处理时候的高度填充值
            uint32_t m_input_w = 0;         // 模型输入的宽度
            uint32_t m_input_h = 0;         // 模型输入的高度
            uint32_t m_output_num_boxes = 8400;    // 输出的框数量（如8400或6300）
            uint8_t m_output_num_classes = 84;     // 输出的类别数（如84）
            Yolov8sPostProcessor m_post_processor;  // YOLOv8s 后处理器
        };

    } /* namespace app */
} /* namespace arm */

#endif /* IMG_CLASS_PROCESSING_HPP */