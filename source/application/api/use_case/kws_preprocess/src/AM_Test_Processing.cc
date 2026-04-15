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

    AM_Test_PreProcess::AM_Test_PreProcess(TfLiteTensor* inputTensor)
    :   m_inputTensor{inputTensor}
    {
        return;
    }

    bool AM_Test_PreProcess::DoPreProcess(const void* data, size_t inputSize)
    {
        if (data == nullptr) {
            printf_err("Data pointer is null");
            return false;
        }

        auto input = static_cast<const uint8_t*>(data);

        std::memcpy(this->m_inputTensor->data.data, input, inputSize);
        debug("Input tensor populated \n");

        return true;
    }

    AM_Test_PostProcess::AM_Test_PostProcess(TfLiteTensor* outputTensor)
    :   m_outputTensor{outputTensor}
        {
            
        }

   bool AM_Test_PostProcess::DoPostProcess(const void* data, size_t inputSize)
    {
        if (data == nullptr) {
            printf_err("Data pointer is null");
            return false;
        }
        
        const int8_t* input = static_cast<const int8_t*>(data);
        const int8_t* output = static_cast<const int8_t*>(this->m_outputTensor->data.data);
        

        for (size_t i = 0; i < inputSize; i++) {
           if (*(input +i) != *(output+i))
               printf_err("Data err %d %d %d", *(input +i), *(output+i), i);
        }
        
        return true;
    }

} /* namespace app */
} /* namespace arm */
