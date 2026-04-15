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
#ifndef RNNOISE_PROCESSING_HPP
#define RNNOISE_PROCESSING_HPP

//#include "BaseProcessing.hpp"
#include "Model.hpp"

namespace arm {
namespace app {

    /**
     * @brief   Pre-processing class for Noise Reduction use case.
     *          Implements methods declared by BasePreProcess and anything else needed
     *          to populate input tensors ready for inference.
     */
    class AM_Test_PreProcess {

    public:
        /**
         * @brief           Constructor
         * @param[in]       inputTensor        Pointer to the TFLite Micro input Tensor.
         * @param[in/out]   featureProcessor   RNNoise specific feature extractor object.
         * @param[in/out]   frameFeatures      RNNoise specific features shared between pre & post-processing.
         *
         **/
        AM_Test_PreProcess(TfLiteTensor* inputTensor0, TfLiteTensor* inputTensor1);

        /**
         * @brief       Should perform pre-processing of 'raw' input audio data and load it into
         *              TFLite Micro input tensors ready for inference
         * @param[in]   input      Pointer to the data that pre-processing will work on.
         * @param[in]   inputSize  Size of the input data.
         * @return      true if successful, false otherwise.
         **/
        bool DoPreProcess(const void* input, size_t inputSize);

    private:
        TfLiteTensor* m_inputTensor0;                        /* Model input tensor. */
        TfLiteTensor* m_inputTensor1;                        /* Model input tensor. */
    };

    /**
     * @brief   Post-processing class for Noise Reduction use case.
     *          Implements methods declared by BasePostProcess and anything else needed
     *          to populate result vector.
     */
    class AM_Test_PostProcess {

    public:
        /**
         * @brief           Constructor
         * @param[in]       outputTensor         Pointer to the TFLite Micro output Tensor.
         * @param[out]      denoisedAudioFrame   Vector to store the final denoised audio frame.
         * @param[in/out]   featureProcessor     RNNoise specific feature extractor object.
         * @param[in/out]   frameFeatures        RNNoise specific features shared between pre & post-processing.
         **/
        AM_Test_PostProcess(TfLiteTensor* outputTensor, TfLiteTensor* inputTensor0, TfLiteTensor* inputTensor1);

        /**
         * @brief       Should perform post-processing of the result of inference then
         *              populate result data for any later use.
         * @return      true if successful, false otherwise.
         **/
        bool DoPostProcess(const void* input, size_t outputSize);

    private:
        TfLiteTensor* m_outputTensor;                       /* Model output tensor. */
        TfLiteTensor* m_inputTensor0;                        /* Model input tensor. */
        TfLiteTensor* m_inputTensor1;                        /* Model input tensor. */

    };

} /* namespace app */
} /* namespace arm */

#endif /* RNNOISE_PROCESSING_HPP */
