/*
 * SPDX-FileCopyrightText: Copyright 2021 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "Memory_Bandwidth_Model.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver& arm::app::Memory_Bandwidth_Model::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::Memory_Bandwidth_Model::EnlistOperations()
{
    this->m_opResolver.AddUnpack();
    this->m_opResolver.AddFullyConnected();
    this->m_opResolver.AddSplit();
    this->m_opResolver.AddSplitV();
    this->m_opResolver.AddAdd();
    this->m_opResolver.AddLogistic();
    this->m_opResolver.AddMul();
    this->m_opResolver.AddSub();
    this->m_opResolver.AddTanh();
    this->m_opResolver.AddPack();
    this->m_opResolver.AddReshape();
    this->m_opResolver.AddQuantize();
    this->m_opResolver.AddConcatenation();
    this->m_opResolver.AddRelu();

    if (kTfLiteOk == this->m_opResolver.AddEthosU()) {
        info("Added %s support to op resolver\n",
            tflite::GetString_ETHOSU());
    } else {
        printf_err("Failed to add Arm NPU support to op resolver.");
        return false;
    }
    return true;
}

bool arm::app::Memory_Bandwidth_Model::RunInference()
{
    return Model::RunInference();
}




