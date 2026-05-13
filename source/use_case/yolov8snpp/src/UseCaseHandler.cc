/*
 * usecaseHandler.cc 任务的入口
   获取输入、调用处理函数进行输入、输出的处理！
   一般这些文件是api，定义在application里面
 */
#include "UseCaseHandler.hpp"

#include "Classifier.hpp"
#include "ImageUtils.hpp"
#include "ImgClassProcessing.hpp"
#include "Yolov8sNppModel.hpp"
#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "log_macros.h"
#include "hal_camera_static_external.h"

#include <cinttypes>
#include <cstring>
#include <cstdlib>
#include <vector>

using Yolov8sNppClassifier = arm::app::Classifier;

namespace arm {
    namespace app {

        /* Image classification inference handler. */
        bool ClassifyImageHandler(ApplicationContext& ctx)
        {
            auto& profiler = ctx.Get<Profiler&>("profiler");
            auto& model = ctx.Get<Model&>("model");

            // 640 /3 = 213,size_lcd = 320 240 --(w,h) path:./source/hal/source/components/lcd/source/glcd.h
            constexpr uint32_t dataPsnImgDownscaleFactor = 3;
            constexpr uint32_t dataPsnImgStartX = 10;
            constexpr uint32_t dataPsnImgStartY = 35;

            constexpr uint32_t dataPsnTxtInfStartX = 150;
            constexpr uint32_t dataPsnTxtInfStartY = 40;

            if (!model.IsInited()) {
                printf_err("Model is not initialised! Terminating processing.\n");
                return false;
            }

            TfLiteTensor* inputTensor = model.GetInputTensor(0);
            if (!inputTensor->dims) {
                printf_err("Invalid input tensor dims\n");
                return false;
            }
            else if (inputTensor->dims->size < 4) {
                printf_err("Input tensor dimension should be = 4\n");
                return false;
            }

            /* Get input shape for displaying the image. */
            TfLiteIntArray* inputShape = model.GetInputShape(0);
            const uint32_t nCols = inputShape->data[arm::app::Yolov8sNppModel::ms_inputColsIdx];
            const uint32_t nRows = inputShape->data[arm::app::Yolov8sNppModel::ms_inputRowsIdx];
            const uint32_t nChannels = inputShape->data[arm::app::Yolov8sNppModel::ms_inputChannelsIdx];
            info("input shape: %" PRIu32 ", %" PRIu32 ", %" PRIu32 ", %d\n", nCols, nRows, nChannels, inputShape->data[0]);
            info("yolov8snpp signed %d.\n", model.IsDataSigned());

            /* Get number of outputs */
            int numOutputs = model.GetNumOutputs();
            info("Number of model outputs: %d\n", numOutputs);

            /* Get all output tensors */
            std::vector<TfLiteTensor*> outputTensors;
            for (int i = 0; i < numOutputs; i++) {
                TfLiteTensor* outputTensor = model.GetOutputTensor(i);
                if (outputTensor) {
                    outputTensors.push_back(outputTensor);
                    TfLiteIntArray* outShape = model.GetOutputShape(i);
                    info("Output %d shape: [", i);
                    for (int j = 0; j < outShape->size; j++) {
                        info("%d%s", outShape->data[j], j < outShape->size - 1 ? ", " : "]\n");
                    }
                }
            }

            /* Set up pre and post-processing. */
            ImgClassPreProcess preProcess = ImgClassPreProcess(inputTensor, model.IsDataSigned());

            std::vector<ClassificationResult> results;
            ImgClassPostProcess postProcess =
                ImgClassPostProcess(outputTensors,
                    ctx.Get<Yolov8sNppClassifier&>("classifier"),
                    ctx.Get<std::vector<std::string>&>("labels"),
                    results,
                    0.25f,
                    0.45f,
                    128);

            postProcess.SetModelInputShape(nCols, nRows);

            hal_camera_init();
            auto bCamera = hal_camera_configure(
                nCols,
                nRows,
                HAL_CAMERA_MODE_SINGLE_FRAME,
                HAL_CAMERA_COLOUR_FORMAT_RGB888);
            if (!bCamera) {
                printf_err("Failed to configure camera.\n");
                return false;
            }

            while (true) {
#ifdef INTERACTIVE_MODE
                AwaitUserInput();
#endif /* INTERACTIVE_MODE */

                hal_lcd_clear(COLOR_BLACK);
                hal_camera_start();

                std::string str_inf{ "Running inference... " };

                uint32_t capturedFrameSize = 0;
                static uint32_t frame_idx = 0;
                const uint8_t* imgSrc = hal_camera_get_captured_frame(&capturedFrameSize);
                if (!imgSrc || !capturedFrameSize) {
                    break;
                }

                info("capturedFrameSize %" PRIu32 ".\n", capturedFrameSize);
                info("inputTensor->bytes %zu.\n", inputTensor->bytes);

                int img_id = 0;
                const char* filename = get_sample_data_filename(frame_idx);
                if (filename) {
                    char numeric_part[15];
                    int i = 0;
                    const char* ptr = filename;
                    while (*ptr && *ptr != '.' && i < 14) {
                        if (*ptr >= '0' && *ptr <= '9') {
                            numeric_part[i++] = *ptr;
                        }
                        ptr++;
                    }
                    numeric_part[i] = '\0';
                    img_id = atoi(numeric_part);
                    info("Extracted image ID: %d from filename: %s\n", img_id, filename);
                }

                float ratio = get_sample_img_ratio(frame_idx);
                float pad_w = get_sample_img_dw(frame_idx);
                float pad_h = get_sample_img_dh(frame_idx);
                postProcess.SetImageId(img_id);
                postProcess.SetRatio(ratio);
                postProcess.SetPadW(pad_w);
                postProcess.SetPadH(pad_h);
                info("get img_id %d ratio %f padding %f x %f.\n", img_id, ratio, pad_w, pad_h);

                frame_idx++;
                if (frame_idx >= get_sample_n_elements()) {
#if defined(HAL_CAMERA_LOOP)
                    frame_idx = 0;
#endif /* HAL_CAMERA_LOOP */
                }

                hal_lcd_display_image(imgSrc,
                    nCols,
                    nRows,
                    nChannels,
                    dataPsnImgStartX,
                    dataPsnImgStartY,
                    dataPsnImgDownscaleFactor);

                hal_lcd_display_text(
                    str_inf.c_str(), str_inf.size(), dataPsnTxtInfStartX, dataPsnTxtInfStartY, false);

                const size_t imgSz =
                    inputTensor->bytes < capturedFrameSize ? inputTensor->bytes : capturedFrameSize;

                info("use input shape %zu \n", imgSz);

                if (!preProcess.DoPreProcess(imgSrc, imgSz)) {
                    printf_err("Pre-processing failed.");
                    return false;
                }
                info("pre-processing done.\n");

                if (!RunInference(model, profiler)) {
                    printf_err("Inference failed.");
                    return false;
                }
                info("inference done.\n");

                if (!postProcess.DumpOutputTensors(img_id)) {
                    printf_err("Dump output tensors failed.");
                    return false;
                }
                info("dump output tensors done.\n");

                if (!postProcess.DoPostProcess()) {
                    printf_err("Post-processing failed.");
                    return false;
                }
                info("post-processing done.\n");

                str_inf = std::string(str_inf.size(), ' ');
                hal_lcd_display_text(
                    str_inf.c_str(), str_inf.size(), dataPsnTxtInfStartX, dataPsnTxtInfStartY, false);

                ctx.Set<std::vector<ClassificationResult>>("results", results);

#if VERIFY_TEST_OUTPUT
                for (int i = 0; i < numOutputs; i++) {
                    arm::app::DumpTensor(outputTensors[i]);
                }
#endif /* VERIFY_TEST_OUTPUT */

                if (!PresentInferenceResult(results)) {
                    return false;
                }
            }

            return true;
        }

    } /* namespace app */
} /* namespace arm */