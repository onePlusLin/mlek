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
             * @param[in]   outputTensors     Vector of TFLite Micro output Tensors.
             * @param[in]   classifier        Classifier object used to get top N results from classification.
             * @param[in]   labels            Vector of string labels to identify each output of the model.
             * @param[in]   results           Vector of classification results to store decoded outputs.
             * @param[in]   model_input_w     Model input width.
             * @param[in]   model_input_h     Model input height.
             * @param[in]   conf_thresh       Confidence threshold for detection.
             * @param[in]   nms_thresh        NMS IoU threshold.
             **/
            ImgClassPostProcess(
                std::vector<TfLiteTensor*> outputTensors,
                Classifier& classifier,
                const std::vector<std::string>& labels,
                std::vector<ClassificationResult>& results,
                uint32_t model_input_w = 640,
                uint32_t model_input_h = 640,
                float conf_thresh = 0.25f,
                float nms_thresh = 0.7f);

            /**
             * @brief       Should perform post-processing of the result of inference then
             *              populate classification result data for any later use.
             * @return      true if successful, false otherwise.
             **/
            bool DoPostProcess() override;

            /**
             * @brief       Dump output tensor data to files
             * @param[in]   imageId  Image ID for filename
             * @param[in]   dumpDir  Dump directory path
             * @return      true if successful, false otherwise.
             **/
            bool DumpOutputTensors(int imageId, const char* dumpDir = nullptr);

            /**
             * @brief       Set image ID
             * @param[in]   imageId    Image ID extracted from filename.
             **/
            void SetImageId(int imageId);

            /**
             * @brief       Set image scaling ratio
             * @param[in]   ratio    Image scaling ratio.
             **/
            void SetRatio(float ratio);

            /**
             * @brief       Set width padding
             * @param[in]   pad_w    Width padding.
             **/
            void SetPadW(float pad_w);

            /**
             * @brief       Set height padding
             * @param[in]   pad_h    Height padding.
             **/
            void SetPadH(float pad_h);

            /**
             * @brief       Set model input shape
             * @param[in]   input_w    Model input width.
             * @param[in]   input_h    Model input height.
             **/
            void SetModelInputShape(uint32_t input_w, uint32_t input_h);

            private:
            std::vector<TfLiteTensor*> m_outputTensors;
            Classifier& m_imgClassifier;
            const std::vector<std::string>& m_labels;
            std::vector<ClassificationResult>& m_results;
            int m_imageId = 0;
            float m_ratio = 1.0f;           // Image scaling ratio
            float m_padW = 0.0f;            // Width padding
            float m_padH = 0.0f;            // Height padding
            uint32_t m_modelInputW = 640;   // Model input width
            uint32_t m_modelInputH = 640;   // Model input height
            Yolov8sNppPostProcessor m_postProcessor;  // YOLOv8sNpp post processor
        };

    } /* namespace app */
} /* namespace arm */

#endif /* IMG_CLASS_PROCESSING_HPP */