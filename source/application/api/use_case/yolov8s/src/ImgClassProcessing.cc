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

        namespace {
            bool SaveAsNpy(const int8_t* data, size_t size, const char* filename) {
                if (data == nullptr || filename == nullptr) {
                    printf_err("Invalid parameters for SaveAsNpy\n");
                    return false;
                }

                FILE* fp = fopen(filename, "wb");
                if (fp == nullptr) {
                    printf_err("Failed to open file %s for writing\n", filename);
                    return false;
                }

                const char magic[] = "\x93NUMPY";
                fwrite(magic, 1, 6, fp);

                uint8_t major_version = 1;
                uint8_t minor_version = 0;
                fwrite(&major_version, 1, 1, fp);
                fwrite(&minor_version, 1, 1, fp);

                std::ostringstream header;
                header << "{'descr': '<i1', 'fortran_order': False, 'shape': (" << size << ",), }";
                std::string header_str = header.str();

                size_t header_len = header_str.length();
                size_t padding = (16 - (header_len + 1) % 16) % 16;
                header_str.append(padding, ' ');
                header_str.push_back('\n');

                uint16_t header_len_le = static_cast<uint16_t>(header_str.length());
                fwrite(&header_len_le, 2, 1, fp);
                fwrite(header_str.c_str(), 1, header_str.length(), fp);

                size_t written = fwrite(data, sizeof(int8_t), size, fp);
                fclose(fp);

                if (written != size) {
                    printf_err("Failed to write complete data to file %s\n", filename);
                    return false;
                }

                info("Successfully saved %zu int8_t values to npy file %s\n", size, filename);
                return true;
            }
        }

        bool ImgClassPreProcess::DoPreProcess(const void* data, size_t inputSize)
        {
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



            // 归一化、量化
            QuantParams quantParams = GetTensorQuantParams(this->m_inputTensor);
            info("Input tensor quantization: scale=%.12f, zero_point=%d\n", quantParams.scale, quantParams.offset);

            for (size_t i = 0; i < inputSize; ++i) {
                // 归一化到 0.0-1.0
                float normalized = static_cast<float>(input[i]) / 255.0f;

                // 量化为 int32（中间计算）
                int32_t quantized = static_cast<int32_t>(
                    // round(normalized / quantParams.scale) + quantParams.offset);
                    (normalized / quantParams.scale) + quantParams.offset);
                // 对齐python的x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

            // 裁剪到 int8 范围
                quantized = std::max(static_cast<int32_t>(-128),
                    std::min(static_cast<int32_t>(127), quantized));

                // 转换为 int8 类型
                tensor_data[i] = static_cast<int8_t>(quantized);
                // tensor_data[i] = 0;
            }

            static int file_counter = 0;
            char filename[256];
            snprintf(filename, sizeof(filename), "/home/linzejia/app/mlek_lee/aMyWork/input_tensor_all0_%d.npy", file_counter++);
            SaveAsNpy(tensor_data, inputSize, filename);

            return true;
        }

        ImgClassPostProcess::ImgClassPostProcess(
            TfLiteTensor* outputTensor,
            Classifier& classifier,
            const std::vector<std::string>& labels,
            std::vector<ClassificationResult>& results,
            uint32_t output_num_boxes,
            uint8_t output_num_classes,
            float conf_thresh,
            float nms_thresh,
            int max_candidates)

            :m_outputTensor{ outputTensor },
            m_imgClassifier{ classifier },
            m_labels{ labels },
            m_results{ results },
            m_output_num_boxes{ output_num_boxes },
            m_output_num_classes{ output_num_classes },
            m_post_processor(output_num_boxes, output_num_classes - 4, output_num_classes, conf_thresh, nms_thresh, max_candidates)
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

            // check
            uint32_t totalOutputSize = m_output_num_boxes * m_output_num_classes;
            info("totalOutputSize calc %" PRIu32 "\n", totalOutputSize);

            info("labels.size %" PRIu32 "\n", this->m_labels.size());

            /* De-Quantize Output Tensor */
            QuantParams quantParams = GetTensorQuantParams(this->m_outputTensor);
            info("outputTensor->type int8 %f, %d\n", quantParams.scale, quantParams.offset);

            __attribute__((section("activation_buf_dram"), aligned(16))) static float tensorDataBuffer[705600];// ddr,后续应该改为totalOutputSize不是705600
            float* tensorData = tensorDataBuffer;

            int8_t* tensor_buffer = tflite::GetTensorData<int8_t>(this->m_outputTensor);
            for (size_t i = 0; i < totalOutputSize; ++i) {
                tensorData[i] = quantParams.scale *
                    (static_cast<float>(tensor_buffer[i]) - quantParams.offset);
            }

            std::vector<std::vector<int>> results = this->m_post_processor.process(
                tensorData,
                this->m_labels,
                this->m_ratio,
                this->m_pad_w,
                this->m_pad_h,
                this->m_input_w,
                this->m_input_h
            );

            return true;
        }

        bool ImgClassPostProcess::DumpOutputTensor(const char* filename)
        {
            if (this->m_outputTensor == nullptr) {
                printf_err("Output tensor is null pointer.\n");
                return false;
            }

            uint32_t totalOutputSize = m_output_num_boxes * m_output_num_classes;
            int8_t* tensor_buffer = tflite::GetTensorData<int8_t>(this->m_outputTensor);

            // // 置为0
            // for (size_t i = 0; i < totalOutputSize; ++i) {
            //     tensor_buffer[i] = 0;
            // }

            std::string npy_filename = std::string(filename);
            if (npy_filename.find(".npy") == std::string::npos) {
                size_t dot_pos = npy_filename.rfind('.');
                if (dot_pos != std::string::npos) {
                    npy_filename = npy_filename.substr(0, dot_pos) + ".npy";
                }
                else {
                    npy_filename = npy_filename + ".npy";
                }
            }

            FILE* fp = fopen(npy_filename.c_str(), "wb");
            if (fp == nullptr) {
                printf_err("Failed to open file %s for writing\n", npy_filename.c_str());
                return false;
            }

            const char magic[] = "\x93NUMPY";
            fwrite(magic, 1, 6, fp);

            uint8_t major_version = 1;
            uint8_t minor_version = 0;
            fwrite(&major_version, 1, 1, fp);
            fwrite(&minor_version, 1, 1, fp);

            std::ostringstream header;
            header << "{'descr': '<i1', 'fortran_order': False, 'shape': (" << totalOutputSize << ",), }";
            std::string header_str = header.str();

            size_t header_len = header_str.length();
            size_t padding = (16 - (header_len + 1) % 16) % 16;
            header_str.append(padding, ' ');
            header_str.push_back('\n');

            uint16_t header_len_le = static_cast<uint16_t>(header_str.length());
            fwrite(&header_len_le, 2, 1, fp);
            fwrite(header_str.c_str(), 1, header_str.length(), fp);

            size_t written = fwrite(tensor_buffer, sizeof(int8_t), totalOutputSize, fp);
            fclose(fp);

            if (written != totalOutputSize) {
                printf_err("Failed to write complete data to file %s\n", npy_filename.c_str());
                return false;
            }

            info("Successfully saved %" PRIu32 " int8_t values to npy file %s\n", totalOutputSize, npy_filename.c_str());
            return true;
        }
    } /* namespace app */
} /* namespace arm */