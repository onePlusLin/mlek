# 我需要你帮我完善一下yolov8ssegnpp示例代码：
## 我建议你先了解一下rknn去掉后处理的工作原理：[修改的脚本在](/home/linzejia/app/u01/u01/model/script/export/export_no_head_seg.py)

我的yolov8ssegnpp的项目目录结构是:  
[定义了模型和主要的流程](/home/linzejia/app/mlek_lee/source/use_case/yolov8ssegnpp)  
[预处理和后处理](/home/linzejia/app/mlek_lee/source/application/api/use_case/yolov8ssegnpp)   
[rknn的例子](/home/linzejia/app/mlek_lee/source/application/api/use_case/yolov8ssegnpp/rknn/postprocess.cc)  
[rknn的例子](/home/linzejia/app/mlek_lee/source/application/api/use_case/yolov8ssegnpp/rknn/postprocess.h)  
[图片输入的处理]这个不需要管，已经做好了；

我最主要的是要你完善后处理的代码：
我需要你按照rknn的例子，修改为yolov8ssegnpp的后处理 ,输出结果包括：类别、框坐标、置信度、segment mask

[postprocess.h](/home/linzejia/app/mlek_lee/source/application/api/use_case/yolov8ssegnpp/include/post_process.hpp)  
[postprocess.cpp](/home/linzejia/app/mlek_lee/source/application/api/use_case/yolov8ssegnpp/include/post_process.cpp)  
你也可以自己新添加代码文件，但是注意要在
/home/linzejia/app/mlek_lee/source/application/api/use_case/yolov8ssegnpp/这个目录下面
并注意添加其路径到/home/linzejia/app/mlek_lee/source/application/api/use_case/yolov8ssegnpp/CMakeLists.txt里面

我的模型结构大致上和rknn一样是经过修改head结构的：  
[修改的脚本在](/home/linzejia/app/u01/u01/model/script/export/export_no_head_seg.py)
并且经过onnx2tf，将其转换为int8的tflite模型：

你在完善这个示例的时候需要注意：
1.输入是int8的tflite模型，shape是[1, 640, 640, 3]（但是我需要你按照模型的输入shape来处理，因为我的模型有可能是[1,480,640,3]）  
2.rknn在识别这个13个输出的时候是按照顺序的，但是我的输出不一定是顺序的，所以我希望你通过通道数来判断；  
3.rknn里面有很多它自定义的数据结构，你需要重新自己进行编写；  
4.rknn示例里面很多是用的自己的库，需要改变；  
5.我还希望你在实现一个dump_outputs的函数，用于dump模型的输出；  
6.最后检测出来的结果，需要你打印出来：类别、框坐标、置信度还有关于segment的信息  
7.注意这里的tensor是NHWC格式，模型的输出可能是flatten的，需要注意优化时间复杂度和空间复杂度，不要申请太多的内存为优！  
8.在迁移代码的时候应注意内存区域的分配，避免内存泄漏，内存配置文件按：/home/linzejia/app/mlek_lee/scripts/cmake/platforms/mps3/sse-300/mps3-sse-300-release.gnu.ld，请依据此文件里面的内存配置来分配内存。

## 编译验证：
bash:
```
cd /home/linzejia/app/mlek_lee/

mkdir build_yolov8ssegnpp_test

cd build_yolov8ssegnpp_test

cmake ../ \
 -DUSE_CASE_BUILD=yolov8ssegnpp \
 -DETHOSU_TARGET_NPU_CONFIG=ethos-u65-512 \
 -DETHOS_U_NPU_CONFIG_ID=Y512 \
 -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram \
 -Dyolov8ssegnpp_MODEL_TFLITE_PATH=/home/linzejia/app/u01/u01/model/tfliteModel/v8sseg/seg66npp/yolov8s-seg_npp_66_full_integer_quant_vela.tflite \
 -DETHOS_U_NPU_ID=U65 \
 -DETHOS_U_NPU_CACHE_SIZE=2097152 \
 -DSEMIHOSTING_ENABLED=ON \
 -DUSE_SINGLE_INPUT=ON

 make -j4

```
运行FVP仿真：
启动环境依赖库：
bash:
```
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/fvp39/lib:$LD_LIBRARY_PATH
```
运行FVP仿真：
```
FVP_Corstone_SSE-300_Ethos-U65 \
 -C cpu0.CFGDTCMSZ=15 \
 -C cpu0.CFGITCMSZ=15 \
 -C mps3_board.visualisation.disable-visualisation=1 \
 -C mps3_board.uart0.out_file="-" \
 -C mps3_board.uart0.shutdown_tag="EXITTHESIM" \
 -C ethosu.num_macs=512 \
 -C mps3_board.FPGA_SRAM_SIZE=2 \
 -C ethosu.extra_args="--fast" \
 -a ./bin/ethos-u-yolov8ssegnpp.axf \
  -C cpu0.semihosting-enable=1
```
观察能否检测出来框和掩码的坐标；