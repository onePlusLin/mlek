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
#include <cstdio>
#include <cstring>
#include <sstream>


namespace arm {
    namespace app {

        ImgClassPreProcess::ImgClassPreProcess(TfLiteTensor* inputTensor, bool convertToInt8)
            :m_inputTensor{ inputTensor },
            m_convertToInt8{ convertToInt8 }
        {
        }

        bool ImgClassPreProcess::DoPreProcess(const void* data, size_t inputSize)
        {
            info("DoPreProcess start in yolov8ssegnpp.\n\n\n");
            if (data == nullptr) {
                printf_err("Data pointer is null");
                return false;
            }

            if (!this->m_convertToInt8)
            {
                info("DoPreProcess failed, convertToInt8 is false.\n");
                info("Input tensor is not int8 type.\n");
                return false;
            }

            auto input = static_cast<const uint8_t*>(data);
            int8_t* tensor_data = static_cast<int8_t*>(this->m_inputTensor->data.data);


            QuantParams quantParams = GetTensorQuantParams(this->m_inputTensor);
            info("Input tensor quantization: scale=%.12f, zero_point=%d\n", quantParams.scale, quantParams.offset);

            for (size_t i = 0; i < inputSize; ++i) {
                float normalized = static_cast<float>(input[i]) / 255.0f;

                int32_t quantized = static_cast<int32_t>(
                    (normalized / quantParams.scale) + quantParams.offset);

                quantized = std::max(static_cast<int32_t>(-128),
                    std::min(static_cast<int32_t>(127), quantized));

                tensor_data[i] = static_cast<int8_t>(quantized);
            }

            return true;
        }

        ImgClassPostProcess::ImgClassPostProcess(
            std::vector<TfLiteTensor*> outputTensors,
            Classifier& classifier,
            const std::vector<std::string>& labels,
            std::vector<ClassificationResult>& results,
            uint32_t model_input_w,
            uint32_t model_input_h,
            float conf_thresh,
            float nms_thresh)

            :m_outputTensors{ outputTensors },
            m_imgClassifier{ classifier },
            m_labels{ labels },
            m_results{ results },
            m_modelInputW{ model_input_w },
            m_modelInputH{ model_input_h },
            m_postProcessor(conf_thresh, nms_thresh)
        {
        }

        void ImgClassPostProcess::SetImageId(int imageId)
        {
            this->m_imageId = imageId;
        }

        void ImgClassPostProcess::SetRatio(float ratio)
        {
            this->m_ratio = ratio;
        }

        void ImgClassPostProcess::SetPadW(float pad_w)
        {
            this->m_padW = pad_w;
        }

        void ImgClassPostProcess::SetPadH(float pad_h)
        {
            this->m_padH = pad_h;
        }

        void ImgClassPostProcess::SetModelInputShape(uint32_t input_w, uint32_t input_h)
        {
            this->m_modelInputW = input_w;
            this->m_modelInputH = input_h;
        }

        bool ImgClassPostProcess::DoPostProcess()
        {
            info("DoPostProcess start in yolov8ssegnpp.\n\n\n");
            if (this->m_outputTensors.empty()) {
                printf_err("Output tensors vector is empty.\n");
                return false;
            }

            /* 这9个输出说明这里的输出不是按顺序的，是乱序的*/
            // info("Number of output tensors: %zu\n", m_outputTensors.size());
            // for (size_t i = 0; i < m_outputTensors.size(); i++) {
            //     TfLiteTensor* tensor = m_outputTensors[i];
            //     TfLiteIntArray* dims = tensor->dims;
            //     info("Output tensor %zu: type=%d, bytes=%zu, dims=[", i, tensor->type, tensor->bytes);
            //     for (int j = 0; j < dims->size; j++) {
            //         info("%d%s", dims->data[j], j < dims->size - 1 ? ", " : "]\n");
            //     }
            // }

            /* Static to avoid 30KB stack allocation (stack = 32KB total) */
            static DetectionResultList det_results;
            bool success = m_postProcessor.process(
                m_outputTensors.data(),
                m_outputTensors.size(),
                m_modelInputW,
                m_modelInputH,
                m_ratio,
                m_padW,
                m_padH,
                det_results);

            if (success) {
                m_postProcessor.print_results(det_results);
                
                // 这里的post_process mask 还有画图还不没校对，画图需要去学一下object_detection的mask画图
                // /* Print per-detection mask polygons + box in original image coords */
                // m_postProcessor.print_mask_polygon(det_results, m_scratchBuf,
                //                                    m_modelInputW, m_modelInputH);
            }

            return success;
        }

        void ImgClassPostProcess::SetScratchBuffer(
            uint8_t* buf, int w, int h)
        {
            m_scratchBuf = buf;
            m_scratchW = w;
            m_scratchH = h;
        }

        bool ImgClassPostProcess::DumpOutputTensors(int imageId, const char* dumpDir)
        {
            if (this->m_outputTensors.empty()) {
                printf_err("Output tensors vector is empty.\n");
                return false;
            }
            
            return m_postProcessor.dump_outputs(m_outputTensors.data(), m_outputTensors.size(), imageId, dumpDir);
        }

    } /* namespace app */
} /* namespace arm */