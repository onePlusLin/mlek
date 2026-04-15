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
#include "Am_Test_Processing.hpp"
#include "log_macros.h"

namespace arm {
namespace app {

    AM_Test_PreProcess::AM_Test_PreProcess(TfLiteTensor* inputTensor0, TfLiteTensor* inputTensor1)
    :   m_inputTensor0{inputTensor0},  m_inputTensor1{inputTensor1}
    {
        return;
    }

    bool AM_Test_PreProcess::DoPreProcess(const void* data, size_t inputSize)
    {
        if (data == nullptr) {
            printf_err("Data pointer is null");
            return false;
        }

        auto input = static_cast<const int8_t*>(data);

        std::memcpy(this->m_inputTensor0->data.data, input, inputSize);
        std::memcpy(this->m_inputTensor1->data.data, input, inputSize);
        debug("Input tensor populated 0x%x, 0x%x, 0x%x \n", this->m_inputTensor0->data.data, this->m_inputTensor1->data.data, input);

        return true;
    }

    AM_Test_PostProcess::AM_Test_PostProcess(TfLiteTensor* outputTensor, TfLiteTensor* inputTensor0, TfLiteTensor* inputTensor1)
    :   m_outputTensor{outputTensor},  m_inputTensor0{inputTensor0},  m_inputTensor1{inputTensor1}
        {
            
        }

    bool AM_Test_PostProcess::DoPostProcess(const void* data, size_t inputSize)
    {
        if (data == nullptr) {
            printf_err("Data pointer is null");
            return false;
        }

        const int8_t* input0 = static_cast<const int8_t*>(this->m_inputTensor0->data.data);
        const int8_t* input1 = static_cast<const int8_t*>(this->m_inputTensor1->data.data);
        const int8_t* output = static_cast<const int8_t*>(this->m_outputTensor->data.data);

        /* De-Quantize Output Tensor */
        QuantParams quantParams = GetTensorQuantParams(m_outputTensor);
        /* Floating point tensor data to be populated
         * NOTE: The assumption here is that the output tensor size isn't too
         * big and therefore, there's neglibible impact on heap usage. */
        std::vector<float> tensorData(inputSize);

        int8_t *tensor_buffer = tflite::GetTensorData<int8_t>(this->m_outputTensor);
                /*for (size_t i = 0; i < inputSize; ++i) {
                    tensorData[i] = quantParams.scale *
                        (static_cast<float>(tensor_buffer[i]) - quantParams.offset);
                }*/


        
        for(size_t i = 0; i < inputSize; i++) {
            if (((*(input0 +i) +  *(input1 +i)) - tensor_buffer[i]) > 1.0) {
                printf_err("Data err %d %d %d %d\n\r", *(input0 +i), *(input0 +i), tensor_buffer[i], i);
                if (i > 100)
                    return false;
            }
        }
        
        return true;
    }

} /* namespace app */
} /* namespace arm */
