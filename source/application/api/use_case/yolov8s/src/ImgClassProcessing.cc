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
#include <algorithm>


extern void yolov8s_post_process(
    const float* input_data,
    const std::vector<std::string>& labels,
    float ratio,
    float pad_w,
    float pad_h,
    uint32_t input_w,
    uint32_t input_h);


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

            // // 归一化、量化
            // QuantParams quantParams = GetTensorQuantParams(this->m_inputTensor);

            // info("Input tensor quantization: scale=%f, zero_point=%d\n",
            //     quantParams.scale, quantParams.offset);
            // uint8_t* input_data = (uint8_t*)this->m_inputTensor->data.data;

            // for (size_t i = 0; i < inputSize; ++i) {
            //     // 归一化到 0.0-1.0
            //     float normalized = static_cast<float>(input[i]) / 255.0f;

            //     // 量化为 int32（中间计算）
            //     int32_t quantized = static_cast<int32_t>(
            //         round(normalized / quantParams.scale) + quantParams.offset);

            //     // 裁剪到 int8 范围
            //     quantized = std::max(static_cast<int32_t>(-128),
            //         std::min(static_cast<int32_t>(127), quantized));

            //     // 转换为 int8
            //     input_data[i] = static_cast<int8_t>(quantized);
            // }
            // if (!this->m_convertToInt8)
            // {
            //     info("DoPreProcess failed, convertToInt8 is false.\n");
            //     info("Input tensor is not int8 type.\n");
            //     return false;
            // }
            // // if (this->m_convertToInt8) {
            // //     image::ConvertImgToInt8(this->m_inputTensor->data.data, this->m_inputTensor->bytes);
            // // }

            // debug("Input tensor populated \n");

            return true;
        }

        ImgClassPostProcess::ImgClassPostProcess(
            TfLiteTensor* outputTensor,
            Classifier& classifier,
            const std::vector<std::string>& labels,
            std::vector<ClassificationResult>& results)
            :m_outputTensor{ outputTensor },
            m_imgClassifier{ classifier },
            m_labels{ labels },
            m_results{ results }
        {
        }

        void ImgClassPostProcess::SetImageId(int imageId)
        {
            this->m_image_id = imageId;
        }

        void ImgClassPostProcess::SetModelInputShape(uint32_t input_w, uint32_t input_h)
        {
            this->m_input_w = input_w;
            this->m_input_h = input_h;
        }

        void ImgClassPostProcess::SetRatio(float ratio)
        {
            this->m_ratio = ratio;
        }

        void ImgClassPostProcess::SetPadW(float pad_w)
        {
            this->m_pad_w = pad_w;
        }

        void ImgClassPostProcess::SetPadH(float pad_h)
        {
            this->m_pad_h = pad_h;
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

            // 使用静态缓冲区避免栈溢出,在这里进行反量化应该可以被优化掉
            // 使用 GCC 特有的 __attribute__ 将这个大数组强制分配到 DDR 中，同时保证 16 字节对齐
            __attribute__((section("activation_buf_dram"), aligned(16))) static float tensorDataBuffer[705600];
            float* tensorData = tensorDataBuffer;

            int8_t* tensor_buffer = tflite::GetTensorData<int8_t>(this->m_outputTensor);
            for (size_t i = 0; i < totalOutputSize; ++i) {
                tensorData[i] = quantParams.scale *
                    (static_cast<float>(tensor_buffer[i]) - quantParams.offset);
            }

            yolov8s_post_process(tensorData, this->m_labels, this->m_ratio, this->m_pad_w, this->m_pad_h, this->m_input_w, this->m_input_h);

            return true;
            /*return this->m_imgClassifier.GetYolov8sResults(
                    this->m_outputTensor, this->m_results,
                    this->m_labels, 5, false);*/
        }
    } /* namespace app */
} /* namespace arm */