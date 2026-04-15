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
extern void yolo_post_process(uint8_t* input_data, const std::vector <std::string>& labels);

namespace arm {
    namespace app {

        ImgClassPreProcess::ImgClassPreProcess(TfLiteTensor* inputTensor, bool convertToInt8)
            :m_inputTensor{ inputTensor },
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
            m_results{ results }
        {
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
             //std::vector<float> tensorData(totalOutputSize);

            /*uint8_t *tensor_buffer = tflite::GetTensorData<uint8_t>(this->m_outputTensor);
            for (size_t i = 0; i < totalOutputSize; ++i) {
                tensorData[i] = quantParams.scale *
                    (static_cast<float>(tensor_buffer[i]) - quantParams.offset);
            }*/

            uint8_t* tensor_buffer = tflite::GetTensorData<uint8_t>(this->m_outputTensor);
            yolo_post_process(tensor_buffer, this->m_labels);

            return true;
            /*return this->m_imgClassifier.GetYolov5sResults(
                    this->m_outputTensor, this->m_results,
                    this->m_labels, 5, false);*/
        }

    } /* namespace app */
} /* namespace arm */