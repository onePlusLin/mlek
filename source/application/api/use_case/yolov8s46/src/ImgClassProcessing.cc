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
#include "ImgClassProcessing.hpp"

#include "ImageUtils.hpp"
#include "log_macros.h"
#include <stdio.h>

namespace arm {
    namespace app {

        ImgClassPreProcess::ImgClassPreProcess(TfLiteTensor* inputTensor, bool convertToInt8)
            : m_inputTensor{ inputTensor },
            m_convertToInt8{ convertToInt8 }
        {
        }

        bool ImgClassPreProcess::DoPreProcess(const void* data, size_t inputSize)
        {
            if (data == nullptr) {
                printf_err("Data pointer is null");
                return false;
            }

            auto input = static_cast<const uint8_t*>(data);

            std::memcpy(this->m_inputTensor->data.data, input, inputSize);
            debug("Input tensor populated \n");

            if (this->m_convertToInt8) {
                image::ConvertImgToInt8(this->m_inputTensor->data.data, this->m_inputTensor->bytes);
            }

            return true;
        }

        ImgClassPostProcess::ImgClassPostProcess(TfLiteTensor* outputTensor, Classifier& classifier,
            const std::vector<std::string>& labels,
            std::vector<ClassificationResult>& results)
            :m_outputTensor{ outputTensor },
            m_imgClassifier{ classifier },
            m_labels{ labels },
            m_results{ results },
            m_image_id{ 0 }
        {
        }

        void ImgClassPostProcess::SetImageId(int imageId)
        {
            m_image_id = imageId;
        }

        bool ImgClassPostProcess::DoPostProcess()
        {
            if (this->m_outputTensor == nullptr) {
                printf_err("Output vector is null pointer.\n");
                return false;
            }

            uint32_t totalOutputSize = 1;
            for (int inputDim = 0; inputDim < this->m_outputTensor->dims->size; inputDim++) {
                totalOutputSize *= this->m_outputTensor->dims->data[inputDim];
                info("totalOutputSize calc %" PRIu32 ", %" PRIu32 "\n", totalOutputSize, this->m_outputTensor->dims->data[inputDim]);
            }

            info("labels.size %" PRIu32 "\n", this->m_labels.size());

            /* De-Quantize Output Tensor */
            QuantParams quantParams = GetTensorQuantParams(this->m_outputTensor);
            info("outputTensor->type uint8 %f, %d\n", quantParams.scale, quantParams.offset);

            // 下面的是进行反量化的
            /* Floating point tensor data to be populated
             * NOTE: The assumption here is that the output tensor size isn't too
             * big and therefore, there's neglibible impact on heap usage. */
             //  std::vector<float> tensorData(totalOutputSize);

             /*uint8_t *tensor_buffer = tflite::GetTensorData<uint8_t>(this->m_outputTensor);
             for (size_t i = 0; i < totalOutputSize; ++i) {
                 tensorData[i] = quantParams.scale *
                     (static_cast<float>(tensor_buffer[i]) - quantParams.offset);
             }*/

            int8_t* tensor_buffer = tflite::GetTensorData<int8_t>(this->m_outputTensor);
            std::vector<DetectionBox> detected_vector;

            detected_vector = yolov8s_post_process(tensor_buffer, this->m_labels, quantParams.scale, quantParams.offset);

            m_all_detections.push_back(std::make_pair(m_image_id, detected_vector));

            return true;
            /*return this->m_imgClassifier.GetYolov8sResults(
                    this->m_outputTensor, this->m_results,
                    this->m_labels, 5, false);*/
        }


        bool ImgClassPostProcess::DumpAllResultsToJson()
        {
            FILE* fp = fopen("tflite.json", "w");
            if (fp == NULL) {
                printf_err("Failed to open tflite.json for writing\n");
                return false;
            }

            fprintf(fp, "[\n");
            bool first_entry = true;
            for (const auto& entry : m_all_detections) {
                int image_id = entry.first;
                const std::vector<DetectionBox>& detected_vector = entry.second;

                for (size_t i = 0; i < detected_vector.size(); i++) {
                    const DetectionBox& box = detected_vector[i];
                    float x = box.x1;
                    float y = box.y1;
                    float width = box.x2 - box.x1;
                    float height = box.y2 - box.y1;

                    if (!first_entry) {
                        fprintf(fp, ",\n");
                    }
                    fprintf(fp, "  {\n");
                    fprintf(fp, "    \"image_id\": %d,\n", image_id);
                    fprintf(fp, "    \"category_id\": %d,\n", box.class_id);
                    fprintf(fp, "    \"bbox\": [%.2f, %.2f, %.2f, %.2f],\n", x, y, width, height);
                    fprintf(fp, "    \"score\": %.4f\n", box.score);
                    fprintf(fp, "  }");
                    first_entry = false;
                }
            }
            fprintf(fp, "\n]\n");

            fclose(fp);
            info("Detection results written to tflite.json\n");
            return true;
        }



    } /* namespace app */
} /* namespace arm */