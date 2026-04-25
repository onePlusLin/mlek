/*
 * usecaseHandler.cc 任务的入口
   获取输入、调用处理函数进行输入、输出的处理！
   一般这些文件是api，定义在application里面
 */
#include "UseCaseHandler.hpp"

#include "Classifier.hpp"
#include "ImageUtils.hpp"
#include "ImgClassProcessing.hpp"
#include "Yolov8s46Model.hpp"
#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "log_macros.h"
#include "sample_files.h"
 // #include "hal_camera_static_external.h"

#include <cinttypes>
#include <cstring>
#include <cstdlib>

using Yolov8s46Classifier = arm::app::Classifier;

namespace arm {
    namespace app {

        /* Image classification inference handler. */
        bool ClassifyImageHandler(ApplicationContext& ctx)
        {
            auto& profiler = ctx.Get<Profiler&>("profiler");
            auto& model = ctx.Get<Model&>("model");

            constexpr uint32_t dataPsnImgDownscaleFactor = 2;//？
            constexpr uint32_t dataPsnImgStartX = 10;//？
            constexpr uint32_t dataPsnImgStartY = 35;//？

            constexpr uint32_t dataPsnTxtInfStartX = 150;//？
            constexpr uint32_t dataPsnTxtInfStartY = 40;//？

            if (!model.IsInited()) {
                printf_err("Model is not initialised! Terminating processing.\n");
                return false;
            }

            TfLiteTensor* inputTensor = model.GetInputTensor(0);
            TfLiteTensor* outputTensor = model.GetOutputTensor(0);
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
            const uint32_t nCols = inputShape->data[arm::app::Yolov8s46Model::ms_inputColsIdx];
            const uint32_t nRows = inputShape->data[arm::app::Yolov8s46Model::ms_inputRowsIdx];
            const uint32_t nChannels = inputShape->data[arm::app::Yolov8s46Model::ms_inputChannelsIdx];

            info("yolov8s46 signed %d.\n", model.IsDataSigned());// 是否有符号整形

            // 可以在这里实现一个是否输入变形

            TfLiteIntArray* outputShape = model.GetOutputShape(0);
            const uint32_t output_boxes_num = outputShape->data[2];     // 获取输出的8400 还是6300？后后续的降低分辨率
            const uint8_t output_classes_num = outputShape->data[1];    // 获取输出的84


            /* Set up pre and post-processing. */
            ImgClassPreProcess preProcess = ImgClassPreProcess(inputTensor, model.IsDataSigned());

            std::vector<ClassificationResult> results;
            ImgClassPostProcess postProcess =
                ImgClassPostProcess(outputTensor,
                    ctx.Get<Yolov8s46Classifier&>("classifier"),
                    ctx.Get<std::vector<std::string>&>("labels"),
                    results);
            hal_camera_init();
            auto bCamera = hal_camera_configure(//？lcd_display_image不能显示
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
                AwaitUserInput(); // Wait for user input before moving forward.
#endif /* INTERACTIVE_MODE */

                hal_lcd_clear(COLOR_BLACK);
                hal_camera_start();

                /* Strings for presentation/logging. */
                std::string str_inf{ "Running inference... " };

                uint32_t capturedFrameSize = 0;
                static uint32_t frame_idx = 0;
                const uint8_t* imgSrc = hal_camera_get_captured_frame(&capturedFrameSize);//？获取捕获的帧
                if (!imgSrc || !capturedFrameSize) {
                    break;
                }

                /* Get image filename and extract image ID */
                int img_id = 0;
                const char* filename = get_sample_data_filename(frame_idx);
                if (filename) {
                    /* Extract numeric part from filename (e.g., "000000524280.bmp" -> "524280") */
                    char numeric_part[15]; // Assuming max 9 digits
                    int i = 0;
                    const char* ptr = filename;
                    while (*ptr && *ptr != '.' && i < 14) { // Stop at the extension
                        if (*ptr >= '0' && *ptr <= '9') {
                            numeric_part[i++] = *ptr;
                        }
                        ptr++;
                    }
                    numeric_part[i] = '\0';
                    img_id = atoi(numeric_part); // atoi automatically ignores leading zeros
                    info("Extracted image ID: %d from filename: %s\n", img_id, filename);
                }
                frame_idx++;
                if (frame_idx >= get_sample_n_elements()) {
#if defined(HAL_CAMERA_LOOP)
                    frame_idx = 0;
#endif /* HAL_CAMERA_LOOP */
                }

                /* Set image ID for post-processing */
                postProcess.SetImageId(img_id);

                /* Display this image on the LCD. */
                hal_lcd_display_image(imgSrc,
                    nCols,
                    nRows,
                    nChannels,
                    dataPsnImgStartX,
                    dataPsnImgStartY,
                    dataPsnImgDownscaleFactor);

                /* Display message on the LCD - inference running. */
                hal_lcd_display_text(
                    str_inf.c_str(), str_inf.size(), dataPsnTxtInfStartX, dataPsnTxtInfStartY, false);

                const size_t imgSz =
                    inputTensor->bytes < capturedFrameSize ? inputTensor->bytes : capturedFrameSize;// 图片大小

                /* Run the pre-processing, inference and post-processing. */
                // 获取输入到->m_inputTensor和是否转为int8
                if (!preProcess.DoPreProcess(imgSrc, imgSz)) {
                    printf_err("Pre-processing failed.");
                    return false;
                }

                if (!RunInference(model, profiler)) {
                    printf_err("Inference failed.");
                    return false;
                }

                if (!postProcess.DoPostProcess()) {
                    printf_err("Post-processing failed.");
                    return false;
                }

                /* Erase. */
                str_inf = std::string(str_inf.size(), ' ');
                hal_lcd_display_text(
                    str_inf.c_str(), str_inf.size(), dataPsnTxtInfStartX, dataPsnTxtInfStartY, false);

                /* Add results to context for access outside handler. */
                ctx.Set<std::vector<ClassificationResult>>("results", results);

#if VERIFY_TEST_OUTPUT // how to open this function?
                arm::app::DumpTensor(outputTensor);
#endif /* VERIFY_TEST_OUTPUT */

                if (!PresentInferenceResult(results)) {
                    return false;
                }

                // profiler.PrintProfilingResult();直接在RunInference里面打印了
            }

            if (!postProcess.DumpAllResultsToJson()) {
                printf_err("Failed to dump all results to JSON file.");
                return false;
            }

            return true;
        }

    } /* namespace app */
} /* namespace arm */
