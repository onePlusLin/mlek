#----------------------------------------------------------------------------
#  SPDX-FileCopyrightText: Copyright 2021, 2024 Arm Limited and/or its
#  affiliates <open-source-office@arm.com>
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#----------------------------------------------------------------------------
# Append the API to use for this use case
list(APPEND ${use_case}_API_LIST "memory_bandwidth_test")

USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "Activation buffer size for the chosen model"
    0x00080000
    STRING)

if (ETHOS_U_NPU_ENABLED)
    set(DEFAULT_MODEL_PATH      ${DEFAULT_MODEL_DIR}/memory_bandwidth_test_vela_${ETHOS_U_NPU_CONFIG_ID}.tflite)
else()
    set(DEFAULT_MODEL_PATH      ${DEFAULT_MODEL_DIR}/memory_bandwidth_test.tflite)
endif()

USER_OPTION(${use_case}_MODEL_TFLITE_PATH "NN models file to be used in the evaluation application. Model files must be in tflite format."
    ${DEFAULT_MODEL_PATH}
    FILEPATH)

set_input_file_path_user_option(".bin" ${use_case})


# Generate model file.
generate_tflite_code(
    MODEL_PATH ${${use_case}_MODEL_TFLITE_PATH}
    DESTINATION ${SRC_GEN_DIR}
    EXPRESSIONS ${EXTRA_MODEL_CODE}
    NAMESPACE   "arm" "app" "rnn")


# For MPS3, allow dumping of output data to memory, based on these parameters:
if (TARGET_PLATFORM STREQUAL mps3)
    USER_OPTION(${use_case}_MEM_DUMP_BASE_ADDR
        "Inference output dump address for ${use_case}"
        0x80000000 # DDR bank 2
        STRING)

    USER_OPTION(${use_case}_MEM_DUMP_LEN
        "Inference output dump buffer size for ${use_case}"
        0x00100000 # 1 MiB
        STRING)

    # Add special compile definitions for this use case files:
    set(${use_case}_COMPILE_DEFS
        "MEM_DUMP_BASE_ADDR=${${use_case}_MEM_DUMP_BASE_ADDR}"
        "MEM_DUMP_LEN=${${use_case}_MEM_DUMP_LEN}")

    file(GLOB_RECURSE SRC_FILES
        "${SRC_USE_CASE}/${use_case}/src/*.cpp"
        "${SRC_USE_CASE}/${use_case}/src/*.cc")

    set_source_files_properties(
        ${SRC_FILES}
        PROPERTIES COMPILE_DEFINITIONS
        "${${use_case}_COMPILE_DEFS}")
endif()
