# YOLOv8s Demo 部署说明

本文档介绍如何在评估套件上部署和运行YOLOv8s目标检测模型。

## 目录结构

```
source/
├── application/api/use_case/yolov8s/    # API层核心实现
│   ├── include/
│   │   ├── ImgClassProcessing.hpp       # 预处理和后处理接口
│   │   ├── Yolov8sModel.hpp            # 模型接口
│   │   └── post_process.hpp            # 后处理算法接口
│   ├── src/
│   │   ├── ImgClassProcessing.cc       # 预处理和后处理实现
│   │   ├── Yolov8sModel.cc             # 模型初始化和算子注册
│   │   └── post_process.cc             # NMS后处理算法实现
│   └── CMakeLists.txt
├── use_case/yolov8s/                    # 用例层实现
│   ├── include/
│   │   └── UseCaseHandler.hpp          # 用例处理器接口
│   ├── src/
│   │   ├── MainLoop.cc                 # 主循环入口
│   │   └── UseCaseHandler.cc           # 用例处理器实现
│   ├── usecase.cmake                    # 用例CMake配置文件
│   └── readme
└── resources/yolov8s/                   # 资源文件
    ├── labels/
    │   ├── labels_yolov8s.txt          # 类别标签
    │   └── labels.h                    # 标签头文件
    └── samples/
        ├── 000000000139.jpg
        ├── 000000000285.jpg
        ├── 000000000632.jpg
        └── bus.jpg

resources_downloaded/                     # 编译好的模型文件目录
└── yolov8s/
    ├── yolov8s.tflite                  # 原始TFLite模型
    └── yolov8s_vela_*.tflite         # Vela优化后的模型

scripts/
├── cmake/
│   └── source_gen_utils.cmake           # 源码生成工具CMake函数
└── py/
    ├── gen_img_cpp_yolo.py              # YOLO图片转C++数组生成脚本
    └── templates/                       # 模板目录（使用jinja2库)
        └── sample-data/
            └── yolo_images/             # C++代码模板文件
                ├── image.c.template     # 单个图片数组模板
                ├── images.c.template    # 图片集合实现模板
                └── images.h.template    # 图片集合头文件模板
```

## 主要工作流程

### 0. 图片的设置（输入大小，使用哪个编译脚本生成c文件）

**位置**: `source/use_case/yolov8s/usecase.cmake`

### 1. 图片预处理（编译时：JPG转C++数组）

**位置**: `scripts/py/gen_img_cpp_yolo.py`

这是编译时的预处理流程，将JPG图片转换为C++数组：
包括lettbox处理：将图片缩放，填充灰色(114, 114, 114)到目标尺寸，但是没有进行归一化和量化处理（因为此时还拿不到量化参数）

**调用位置**: `source/use_case/yolov8s/usecase.cmake`

```cmake
generate_yolo_images_code("${${use_case}_FILE_PATH}"
                         ${SAMPLES_GEN_DIR}
                         "${${use_case}_IMAGE_HEIGHT}"
                         "${${use_case}_IMAGE_WIDTH}")
```

**实现位置**: `scripts/cmake/source_gen_utils.cmake::generate_yolo_images_code()`

```cmake
function(generate_yolo_images_code input_dir gen_dir img_height img_width )
    execute_process(
        COMMAND ${PYTHON} ${MLEK_SCRIPTS_DIR}/py/gen_img_cpp_yolo.py
        --image_path ${input_dir_abs}
        --package_gen_dir ${gen_out_abs}
        --image_size ${img_height} ${img_width}
    )
endfunction()
```

**处理步骤**:
1. **读取JPG图片**: 使用PIL库读取RGB格式图片
2. **计算缩放比例**: `r = min(target_h / src_h, target_w / src_w)`
3. **调整大小**: 使用双线性插值将图片缩放到目标尺寸
4. **计算填充**: `dw = (target_w - new_w) / 2`, `dh = (target_h - new_h) / 2`
5. **边界填充**: 使用灰色(114, 114, 114)填充到640x640
6. **生成C++数组**: 将像素数据转换为uint8_t数组
7. **生成代码文件**: 使用Jinja2模板生成.c和.h文件

**生成的文件**:
- `sample_files.h`: 头文件，包含图片数组的声明和访问函数
- `sample_files.c`: 实现文件，包含图片数组数据和访问函数
- `xxx.c`: 单个图片的数组文件（如`bus.c`, `000000000139.c`）

**模板文件**:
- `scripts/py/templates/sample-data/yolo_images/image.c.template`: 单个图片数组模板
- `scripts/py/templates/sample-data/yolo_images/images.h.template`: 头文件模板
- `scripts/py/templates/sample-data/yolo_images/images.c.template`: 实现文件模板

**访问函数**:
```cpp
const char* get_sample_data_filename(const uint32_t idx);  // 获取文件名
const uint8_t* get_sample_data_ptr(const uint32_t idx);   // 获取图片数据指针
uint32_t get_sample_n_elements(void);                      // 获取图片数量
uint32_t get_sample_img_total_bytes(void);                  // 获取图片大小
uint32_t get_sample_img_width(void);                       // 获取图片宽度
uint32_t get_sample_img_height(void);                      // 获取图片高度
float get_sample_img_ratio(const uint32_t idx);            // 获取缩放比例
float get_sample_img_dw(const uint32_t idx);              // 获取宽度填充
float get_sample_img_dh(const uint32_t idx);              // 获取高度填充
```

### 2. 运行时图片预处理

**位置**: `source/application/api/use_case/yolov8s/src/ImgClassProcessing.cc`

运行时预处理流程包括：
- 从摄像头获取RGB888格式图像数据
- 归一化到[0.0, 1.0]范围
- 量化为int8格式（使用模型输入的scale和zero_point）
- 裁剪到int8范围[-128, 127]
- 将处理后的数据填充到输入张量

```cpp
// 归一化
float normalized = static_cast<float>(input[i]) / 255.0f;

// 量化
int32_t quantized = (normalized / quantParams.scale) + quantParams.offset;

// 裁剪
quantized = std::max(static_cast<int32_t>(-128),
    std::min(static_cast<int32_t>(127), quantized));
```

### 3. 模型初始化

**位置**: `source/application/api/use_case/yolov8s/src/Yolov8sModel.cc`

模型初始化包括：
- 加载TFLite模型文件（从`resources_downloaded/yolov8s/`）
- 注册Ethos-U NPU算子
- 配置算子解析器
- 初始化张量区域（tensorArena）

```cpp
bool arm::app::Yolov8sModel::EnlistOperations()
{
    if (kTfLiteOk == this->m_opResolver.AddEthosU()) {
        info("Added %s support to op resolver\n",
            tflite::GetString_ETHOSU());
    }
    return true;
}
```

### 4. LCD读取输入

**位置**: `source/use_case/yolov8s/src/UseCaseHandler.cc`

输入获取流程：
- 初始化摄像头：`hal_camera_init()`
- 配置摄像头参数（分辨率、颜色格式等）
- 启动摄像头：`hal_camera_start()`
- 获取捕获的帧：`hal_camera_get_captured_frame()`
- 在LCD上显示图像：`hal_lcd_display_image()`

```cpp
hal_camera_init();
auto bCamera = hal_camera_configure(
    nCols, nRows,
    HAL_CAMERA_MODE_SINGLE_FRAME,
    HAL_CAMERA_COLOUR_FORMAT_RGB888);
```

### 5. 主流程

**位置**: `source/use_case/yolov8s/src/MainLoop.cc`

主循环流程：
1. 创建模型实例并初始化
2. 设置应用上下文（ApplicationContext）
3. 加载标签文件
4. 进入用例处理器（ClassifyImageHandler）
5. 循环处理图像帧

```cpp
void MainLoop()
{
    arm::app::Yolov8sModel model;
    model.Init(arm::app::tensorArena,
               sizeof(arm::app::tensorArena),
               arm::app::yolov8s::GetModelPointer(),
               arm::app::yolov8s::GetModelLen());
    
    arm::app::ApplicationContext caseContext;
    ClassifyImageHandler(caseContext);
}
```

### 6. 输入预处理（运行时）

**位置**: `source/application/api/use_case/yolov8s/src/ImgClassProcessing.cc::DoPreProcess()`

预处理步骤：
- 验证输入数据有效性
- 获取输入张量的量化参数（scale、zero_point）
- 遍历每个像素进行归一化和量化
- 可选：保存预处理后的张量到npy文件用于调试

### 7. 输出后处理

**位置**: `source/application/api/use_case/yolov8s/src/post_process.cc`

后处理流程：
1. **反量化**：将int8输出转换为float格式
2. **置信度过滤**：筛选置信度大于阈值的检测框
3. **坐标转换**：将模型输出坐标映射回原图坐标（考虑缩放比例和填充）
4. **NMS（非极大值抑制）**：去除重叠的检测框
5. **结果输出**：返回最终的检测结果（类别ID、边界框坐标）

```cpp
// 反量化
tensorData[i] = quantParams.scale *
    (static_cast<float>(tensor_buffer[i]) - quantParams.offset);

// NMS处理
std::vector<std::vector<int>> results = this->m_post_processor.process(
    tensorData, labels, ratio, pad_w, pad_h, input_w, input_h);
```

### 8. Dump功能

**位置**: `source/application/api/use_case/yolov8s/src/ImgClassProcessing.cc::SaveAsNpy()`

支持将中间结果保存为NumPy格式文件：
- 输入张量dump：保存预处理后的int8数据
- 输出张量dump：保存模型推理后的int8数据
- 文件格式：符合NumPy .npy标准格式

## 运行命令

### 1. 创建构建目录

```bash
mkdir build
cd build
```

### 2. CMake配置

根据目标平台选择合适的配置：

**Corstone-300 (Ethos-U65)**:  
-注意修改模型路径：*Dyolov8s_MODEL_TFLITE_PATH*
```bash
cmake ../ \
 -DUSE_CASE_BUILD=yolov8s \
 -DETHOSU_TARGET_NPU_CONFIG=ethos-u65-512 \
 -DETHOS_U_NPU_CONFIG_ID=Y512 \
 -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram \
 -Dyolov8s_MODEL_TFLITE_PATH=/home/linzejia/app/ethos-u-vela/work/yolo_66/v8s/output/our/yolov8s_full_integer_quant_vela.tflite \
 -DETHOS_U_NPU_ID=U65 \
 -DETHOS_U_NPU_CACHE_SIZE=2097152
```


### 3. 编译

```bash
make -j$(nproc)
```

编译成功后，在`build/bin/`目录下生成可执行文件。

### 4. 使用FVP运行  
注意：*-C ethosu.extra_args="--fast"* 只在快速查看模型推理结果的正确性时使用，不建议在测量性能数据的时候使用！

**Corstone-300 FVP**:
```bash
FVP_Corstone_SSE-300_Ethos-U65 \
 -C cpu0.CFGDTCMSZ=15 \
 -C cpu0.CFGITCMSZ=15 \
 -C mps3_board.visualisation.disable-visualisation=1 \
 -C mps3_board.uart0.out_file="-" \
 -C mps3_board.uart0.shutdown_tag="EXITTHESIM" \
 -C ethosu.num_macs=512 \
 -C mps3_board.FPGA_SRAM_SIZE=2 \
 -C ethosu.extra_args="--fast" \
 -a ethos-u-yolov8s.axf
```


## 文件夹说明

### source/application/api/use_case/yolov8s/src

**功能**: YOLOv8s模型的核心API实现层

**主要文件**:
- `ImgClassProcessing.cc`: 图像预处理和后处理实现
  - `DoPreProcess()`: 输入图像归一化、量化
  - `DoPostProcess()`: 输出反量化、调用NMS
  - `DumpOutputTensor()`: 保存输出张量到文件
  
- `Yolov8sModel.cc`: 模型初始化和算子注册
  - `GetOpResolver()`: 获取算子解析器
  - `EnlistOperations()`: 注册Ethos-U NPU算子
  
- `post_process.cc`: YOLOv8s后处理算法实现
  - `process()`: 执行完整的后处理流程（置信度过滤、NMS）
  - `compare_boxes()`: 比较检测框置信度
  - `calc_iou()`: 计算检测框交并比
  - `dump_output_tensor()`: 保存原始输出数据

**设计特点**:
- 提供可复用的API接口
- 与具体平台解耦
- 支持量化模型推理
- 包含完整的YOLOv8s后处理算法

### source/use_case/yolov8s/src

**功能**: YOLOv8s用例的业务逻辑层

**主要文件**:
- `MainLoop.cc`: 主程序入口
  - 初始化模型和上下文
  - 设置性能分析器
  - 加载标签文件
  - 启动主循环
  
- `UseCaseHandler.cc`: 用例处理器
  - `ClassifyImageHandler()`: 图像分类/检测处理函数
  - 管理摄像头和LCD交互
  - 协调预处理、推理、后处理流程
  - 显示推理结果

**设计特点**:
- 实现具体的业务逻辑
- 管理硬件资源（摄像头、LCD）
- 协调各个处理阶段
- 提供用户交互界面

### source/use_case/yolov8s/usecase.cmake

**功能**: YOLOv8s用例的CMake配置文件

**主要配置**:
- 定义图像尺寸（默认640x640）
- 指定标签文件路径
- 调用`generate_yolo_images_code()`生成图片C++数组
- 调用`generate_labels_code()`生成标签C++代码
- 配置激活缓冲区大小
- 指定模型文件路径（支持Vela优化模型）

```cmake
USER_OPTION(${use_case}_IMAGE_HEIGHT "Image height in pixels."
    640 STRING)
USER_OPTION(${use_case}_IMAGE_WIDTH "Image width in pixels."
    640 STRING)

generate_yolo_images_code("${${use_case}_FILE_PATH}"
                         ${SAMPLES_GEN_DIR}
                         "${${use_case}_IMAGE_HEIGHT}"
                         "${${use_case}_IMAGE_WIDTH}")
```

### resources/yolov8s/

**功能**: 存放模型相关的资源文件

**主要内容**:
- `labels/labels_yolov8s.txt`: COCO数据集的80个类别标签
- `labels/labels.h`: 生成的标签头文件
- `samples/*.jpg`: 测试用的样本图像

### resources_downloaded/yolov8s/

**功能**: 存放编译好的模型文件

**主要内容**:
- `yolov8s.tflite`: 原始TFLite模型文件
- `yolov8s_vela_*.tflite`: 使用Vela编译器优化后的模型文件
  - 支持不同的NPU配置（如Ethos-U55、Ethos-U65）
  - 优化后的模型在NPU上运行效率更高

**模型选择**:  
**建议自定义模型路径**:
```cmake
if (ETHOS_U_NPU_ENABLED)
    set(DEFAULT_MODEL_PATH ${DEFAULT_MODEL_DIR}/yolov8s_vela_${ETHOS_U_NPU_CONFIG_ID}.tflite)
else()
    set(DEFAULT_MODEL_PATH ${DEFAULT_MODEL_DIR}/yolov8s.tflite)
endif()
```

### scripts/cmake/source_gen_utils.cmake

**功能**: 提供源码生成工具的CMake函数

**主要函数**:
- `generate_yolo_images_code()`: 调用Python脚本生成YOLO图片的C++数组
  - 输入：图片目录、生成目录、目标尺寸
  - 输出：C++源文件和头文件

```cmake
function(generate_yolo_images_code input_dir gen_dir img_height img_width )
    execute_process(
        COMMAND ${PYTHON} ${MLEK_SCRIPTS_DIR}/py/gen_img_cpp_yolo.py
        --image_path ${input_dir_abs}
        --package_gen_dir ${gen_out_abs}
        --image_size ${img_height} ${img_width}
    )
endfunction()
```

### scripts/py/gen_img_cpp_yolo.py

**功能**: 将YOLO图片转换为C++数组的Python脚本

**主要功能**:
1. 读取指定目录下的JPG图片
2. 对每张图片进行预处理（resize、pad）
3. 计算并保存缩放比例和填充参数
4. 生成C++数组代码
5. 使用Jinja2模板生成头文件和实现文件

**关键函数**:
- `resize_pad_rgb_image()`: 调整图片大小并填充
- `write_individual_img_cc_file()`: 生成单个图片的C++数组
- `write_hpp_file()`: 生成头文件和实现文件

**预处理细节**:
```python
# 计算缩放比例
r = min(image_size[0] / shape[0], image_size[1] / shape[1])

# 计算填充
dw, dh = (image_size[1] - new_unpad[0]) / 2, (image_size[0] - new_unpad[1]) / 2

# 调整大小
img = original_image.resize(new_unpad, Image.Resampling.BILINEAR)

# 边界填充（灰色）
img = ImageOps.expand(img, border=(left, top, right, bottom), fill=(114, 114, 114))
```

### scripts/py/templates/sample-data/yolo_images/

**功能**: 存放C++代码生成模板

**主要模板**:
- `image.c.template`: 单个图片数组模板
  - 生成包含图片数据的uint8_t数组
  - 使用`IFM_BUF_ATTRIBUTE`属性标记为输入特征图缓冲区

- `images.h.template`: 头文件模板
  - 声明图片数组
  - 定义常量（图片数量、大小、宽高）
  - 声明访问函数（获取文件名、数据指针、缩放比例等）

- `images.c.template`: 实现文件模板
  - 实现访问函数
  - 存储图片文件名数组
  - 存储图片数据指针数组
  - 存储缩放比例和填充参数数组

**生成的代码示例**:
```cpp
// images.h
#define NUMBER_OF_FILES (4U)
#define IMAGE_DATA_SIZE (1228800U)  // 640*640*3
extern const uint8_t im0[IMAGE_DATA_SIZE];
const char* get_sample_data_filename(const uint32_t idx);
const uint8_t* get_sample_data_ptr(const uint32_t idx);
float get_sample_img_ratio(const uint32_t idx);
float get_sample_img_dw(const uint32_t idx);
float get_sample_img_dh(const uint32_t idx);

// images.c
static const char* imgFilenames[] = {"bus.jpg", "000000000139.jpg", ...};
static const uint8_t* imgArrays[] = {im0, im1, ...};
static const float image_ratios[] = {1.0f, 0.8f, ...};
static const float image_dws[] = {0.0f, 64.0f, ...};
static const float image_dhs[] = {0.0f, 32.0f, ...};
```

## 配置参数

### 模型参数

- **输入尺寸**: 640x640x3 (RGB888)
- **输出形状**: [1, 84, 8400] 或 [1, 84, 6300]
- **类别数**: 80 (COCO数据集)
- **检测框数**: 8400 (默认) 或 6300 (低分辨率)

### 后处理参数

- **置信度阈值**: 0.25 (可调整)
- **NMS IoU阈值**: 0.7 (可调整)
- **最大候选框数**: 200

### 内存配置

- **Tensor Arena大小**: 0x00F00000 (15MB)
- **输出缓冲区**: 705600 floats (DDR)

## 调试技巧

### 启用输入/输出Dump

在`source/use_case/yolov8s/src/UseCaseHandler.cc`中取消注释：

```cpp
// Dump输出张量
std::string dump_name = std::to_string(img_id) + ".npy";
if (!postProcess.DumpOutputTensor(dump_name.c_str())) {
    printf_err("Dump output tensor failed.");
    return false;
}
```

### 查看推理性能

推理完成后，性能分析器会自动打印：
- 推理时间
- 各层执行时间
- NPU利用率

### 常见问题

1. **内存不足**: 增加`tensorArena`大小
2. **检测框过多**: 调整置信度阈值或NMS阈值
3. **精度下降**: 检查量化参数是否正确
4. **图片预处理错误**: 检查`gen_img_cpp_yolo.py`生成的数组是否正确
5. **模型加载失败**: 确保`resources_downloaded/yolov8s/`目录下有正确的模型文件

## 注意事项

1. 确保模型文件已正确编译为Vela优化版本并放在`resources_downloaded/yolov8s/`目录
2. 摄像头配置必须与模型输入尺寸匹配
3. 标签文件必须与模型输出类别数一致
4. FVP运行时需要足够的系统内存
5. 首次运行可能需要较长的初始化时间
6. 编译时会自动调用`gen_img_cpp_yolo.py`生成图片C++数组，确保Python环境正确配置

## 参考资料

- [Arm Ethos-U NPU文档](https://developer.arm.com/documentation/101888/0500/)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [YOLOv8模型架构](https://github.com/ultralytics/ultralytics)
- [Vela编译器文档](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela)