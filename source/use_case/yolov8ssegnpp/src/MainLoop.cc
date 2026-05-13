/*
 * MainLoop.cc定义了对应demo的激活张量缓冲区、命名空间、组件的通信、性能分析根据以及主入口！
 */
#include "hal.h"                    /* Brings in platform definitions. */
#include "Classifier.hpp"           /* Classifier. */
#include "Labels.hpp"               /* For label strings. */
#include "log_macros.h"             /* Logging functions */
#include "Yolov8ssegNppModel.hpp"         /* Model class for running inference. */
#include "UseCaseHandler.hpp"       /* Handlers for different user options. */
#include "UseCaseCommonUtils.hpp"   /* Utils functions. */
#include "BufAttributes.hpp"        /* Buffer attributes to be applied */

namespace arm {
    namespace app {
        static uint8_t tensorArena[0x00F00000] ACTIVATION_BUF_ATTRIBUTE;
        namespace yolov8ssegnpp {
            extern uint8_t* GetModelPointer();
            extern size_t GetModelLen();
        } /* namespace yolov8ssegnpp */
    } /* namespace app */
} /* namespace arm */

using Yolov8sNppClassifier = arm::app::Classifier;

void MainLoop()
{
    arm::app::Yolov8ssegNppModel model;  /* Model wrapper object. */

    /* Load the model. */
    if (!model.Init(arm::app::tensorArena,
        sizeof(arm::app::tensorArena),
        arm::app::yolov8ssegnpp::GetModelPointer(),
        arm::app::yolov8ssegnpp::GetModelLen())) {
        printf_err("Failed to initialise model\n");
        return;
    }

    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;

    arm::app::Profiler profiler{ "yolov8ssegnpp" };
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<arm::app::Model&>("model", model);

    Yolov8sNppClassifier classifier;  /* Classifier wrapper object. */
    caseContext.Set<arm::app::Classifier&>("classifier", classifier);

    std::vector <std::string> labels;
    GetLabelsVector(labels);

    caseContext.Set<const std::vector <std::string>&>("labels", labels);

    /* Loop. */
    bool executionSuccessful = ClassifyImageHandler(caseContext);
    info("Main loop terminated %s.\n",
        executionSuccessful ? "successfully" : "with failure");
}
