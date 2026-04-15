/*
 * SPDX-FileCopyrightText: Copyright 2021-2022, 2024-2025 Arm Limited and/or its
 * affiliates <open-source-office@arm.com>
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
#include "UseCaseHandler.hpp"
#include "Am_Test_Processing.hpp"
#include "Kws_Preprocess_Model.hpp"
#include "yes_30ms_data.h"
#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "log_macros.h"


namespace arm {
namespace app {

    /* Noise reduction inference handler. */
    bool Kws_Preprocess_Handler(ApplicationContext& ctx)
    {
        auto& profiler               = ctx.Get<Profiler&>("profiler");

        /* Get model reference. */
        auto& model = ctx.Get<Kws_Preprocess_Model&>("model");
        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return false;
        }

        TfLiteTensor* inputTensor  = model.GetInputTensor(0);
        TfLiteTensor* outputTensor = model.GetOutputTensor(0);
        if (!inputTensor->dims) {
            printf_err("Invalid input tensor dims\n");
            return false;
        } else if (inputTensor->dims->size < 4) {
            printf_err("Input tensor dimension should be = 4\n");
            return false;
        }
        
        /* Set up pre and post-processing. */
        AM_Test_PreProcess preProcess = AM_Test_PreProcess(inputTensor);
        
        
        /* Run the pre-processing, inference and post-processing. */
        if (!preProcess.DoPreProcess(yes_30ms_wav, yes_30ms_wav_len)) {
            printf_err("Pre-processing failed.");
            return false;
        }
        
        {   
            bool ret = model.RunInference();
           
        }
        
        AM_Test_PostProcess postProcess = AM_Test_PostProcess(outputTensor);

        return postProcess.DoPostProcess(expected_feature, kFeatureSize);
    }
} /* namespace app */
} /* namespace arm */
