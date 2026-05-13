# Object Detection Use Case Demo 总结

## 一、项目目录结构

```
source/use_case/object_detection/
├── include/
│   └── UseCaseHandler.hpp          # 推理处理 handler 声明
├── src/
│   ├── MainLoop.cc                 # 入口: 初始化模型 + 进入 handler
│   └── UseCaseHandler.cc           # 核心: 预处理 → 推理 → 后处理 → LCD显示
└── usecase.cmake                   # 构建配置 (模型路径、anchors、图像尺寸等)

source/application/api/use_case/object_detection/
├── include/
│   ├── DetectionResult.hpp          # 检测结果数据结构
│   └── DetectorPostProcessing.hpp   # 后处理类声明 + PostProcessParams/Branch/Network 结构体
├── src/
│   ├── DetectorPreProcessing.cc    # 预处理实现
│   └── DetectorPostProcessing.cc   # 后处理实现 (核心: GetNetworkBoxes + NMS + 结果转换)

source/application/api/common/
├── include/ImageUtils.hpp          # Box/Detection 结构体 + NMS + IOU + 颜色常量
└── source/ImageUtils.cc            # NMS 实现、IOU/Intersect/Union 计算
```

**构建配置关键参数** (`usecase.cmake`):
- `IMAGE_SIZE`: 192 (输入图像尺寸)
- `ANCHOR_1`: {38, 77, 47, 97, 61, 126} (大特征图 anchor)
- `ANCHOR_2`: {14, 26, 19, 37, 28, 55} (小特征图 anchor)
- `CHANNELS_IMAGE_DISPLAYED`: 3 (RGB 显示)
- 模型: `yolo-fastest_192_face_v4.tflite`

---

## 二、整体处理流程

`MainLoop.cc` → `ObjectDetectionHandler` 是一次性的无限循环流程。

### MainLoop.cc (入口)

```
1. 分配 tensorArena (activation buffer, 0x82000 字节)
2. 创建 YoloFastestModel, 调用 model.Init() 加载 .tflite 模型
3. 创建 ApplicationContext, 注册 profiler 和 model
4. 调用 ObjectDetectionHandler(ctx)
```

### ObjectDetectionHandler (核心循环)

```
hal_lcd_clear(BLACK)                     # 清屏
  ↓
获取 input/output tensors               # 1个输入, 2个输出
  ↓
创建 DetectorPreProcess                 # 预处理 (当前为空操作, 仅填0)
创建 DetectorPostProcess                # 后处理 (绑定两个输出tensor + 参数)
  ↓
hal_camera_init() + configure           # 初始化摄像头 (RGB888, 单帧模式)
  ↓
┌─────────────────────────────────────────────┐
│  while(true) 主循环:                         │
│                                               │
│  1. results.clear()                          │
│  2. hal_camera_start()                       │
│  3. hal_camera_get_captured_frame()          │ # 捕获一帧
│  4. preProcess.DoPreProcess()                │ # 预处理
│  5. hal_lcd_display_image()                  │ # ★ LCD显示原图
│  6. hal_lcd_display_text("Running inference...") │ # ★ LCD显示文字
│  7. RunInference()                           │ # 推理
│  8. postProcess.DoPostProcess()              │ # 后处理 (含NMS)
│  9. DrawDetectionBoxes()                     │ # ★ LCD画检测框
│ 10. PresentInferenceResult()                 │ # 串口打印结果
│ 11. profiler.PrintProfilingResult()          │
└──────────────────────────────────────────────┘
```

---

## 三、LCD 显示详解 (`hal_lcd.h` + `lcd_img.h`)

所有 LCD 操作通过 `hal_lcd_*` 宏封装, 实际调用底层 `lcd_*` 函数。LCD 使用 **RGB565** 颜色格式。

### 1. 清屏 — `hal_lcd_clear(COLOR_BLACK)`

- 在推理开始前, 用黑色填充整个 LCD 屏幕
- `COLOR_BLACK = 0` (RGB565 格式)

### 2. 显示图片 — `hal_lcd_display_image(data, w, h, channels, pos_x, pos_y, downsample_factor)`

- 发生在推理前 (`UseCaseHandler.cc:142-149`)
- 显示的是摄像头捕获的原始图像 (`currImage`)
- 判断逻辑: 如果是 3 通道(RGB)直接显示原图, 否则显示预处理后的 tensor 数据
- 固定位置: `pos_x=10, pos_y=35`, `downsample_factor=1` (无缩放)

### 3. 显示文字 — `hal_lcd_display_text(str, str_len, pos_x, pos_y, allow_multiple_lines)`

两处使用:

- **推理前** (`UseCaseHandler.cc:152-153`): 在位置 (20, 28) 显示 `"Running inference... "` 提示
- **推理后** (`UseCaseHandler.cc:166-168`): 用空格覆盖擦除该文字 (写入等长空格字符串, 实现"清除"效果)
- 参数 `allow_multiple_lines=false`, 不换行

### 4. 设置文字颜色 — `hal_lcd_set_text_color(COLOR_GREEN)`

- 在 `PresentInferenceResult` 中调用, 将后续文字颜色设为绿色

### 5. 画检测框 — `hal_lcd_display_box(x, y, width, height, color)`

- 在 `DrawDetectionBoxes` 中调用, 这是核心的 **框绘制** 函数
- 每次调用画一个矩形填充区域, 通过组合 4 条线 (顶/底/左/右) 形成完整的检测框

**颜色常量定义** (`ImageUtils.hpp:31-39`):
```cpp
COLOR_BLACK  = RGB888_TO_RGB565(  0,   0,   0)
COLOR_GREEN  = RGB888_TO_RGB565(  0, 255,   0)
COLOR_YELLOW = RGB888_TO_RGB565(255, 255,   0)
COLOR_WHITE  = RGB888_TO_RGB565(255, 255, 255)
// ... 等
```

---

## 四、画框显示详解 — `DrawDetectionBoxes` (`UseCaseHandler.cc:217-253`)

```
函数签名: DrawDetectionBoxes(results, imgStartX, imgStartY, imgDownscaleFactor)
固定参数:  imgStartX=10, imgStartY=35, imgDownscaleFactor=1
```

**实现逻辑**: 对每个检测结果, 画 **4 条线** 构成一个矩形框, 颜色为 `COLOR_GREEN`:

```
① 顶线 (Top):
   hal_lcd_display_box(
       imgStartX + m_x0 / downscale,    // x 起点
       imgStartY + m_y0 / downscale,    // y 起点
       m_w / downscale,                 // 宽度 = 框宽
       1,                               // 高度 = 1 像素
       COLOR_GREEN)

② 底线 (Bottom):
   hal_lcd_display_box(
       imgStartX + m_x0 / downscale,
       imgStartY + (m_y0 + m_h) / downscale - 1,  // 注意减去 lineThickness
       m_w / downscale,
       1,
       COLOR_GREEN)

③ 左线 (Left):
   hal_lcd_display_box(
       imgStartX + m_x0 / downscale,
       imgStartY + m_y0 / downscale,
       1,                               // 宽度 = 1 像素
       m_h / downscale,                 // 高度 = 框高
       COLOR_GREEN)

④ 右线 (Right):
   hal_lcd_display_box(
       imgStartX + (m_x0 + m_w) / downscale - 1,  // 注意减去 lineThickness
       imgStartY + m_y0 / downscale,
       1,
       m_h / downscale,
       COLOR_GREEN)
```

**关键点**:
- `m_x0, m_y0, m_w, m_h` 是后处理输出的 **原图尺度** 坐标 (0~192)
- `imgStartX/Y` 是图像在 LCD 上的偏移 (10, 35), 保证框和图像对齐
- `imgDownscaleFactor=1` 意味着无缩放, 坐标 1:1 映射
- 线宽固定为 **1 像素**, 因此底线和右线需要 **各减 1** 防止画到框外
- 从底层看, `lcd_display_box` 是画一个填充矩形, 这里用极窄矩形 (宽或高为1) 来模拟线条

---

## 五、NMS 函数详解 (`ImageUtils.cc:75-97`)

### 函数签名

```cpp
void CalculateNMS(std::forward_list<Detection>& detections, int classes, float iouThreshold)
// classes = 1 (人脸检测只有一个类别)
// iouThreshold = 0.45 (默认值, 来自 PostProcessParams.nms)
```

### 依赖的辅助函数

**`Calculate1DOverlap`** (`ImageUtils.cc:25-36`):
- 计算两个一维区间 (中心+宽度) 的重叠长度
- 公式: `overlap = min(right1, right2) - max(left1, left2)`, 其中 `left = center - width/2`

**`CalculateBoxIntersect`** (`ImageUtils.cc:38-50`):
- 计算两个 Box 的交集面积
- 方法: 分别计算 x 和 y 方向的重叠宽度, 若任一方向 < 0 则交集为 0, 否则 `area = width * height`

**`CalculateBoxUnion`** (`ImageUtils.cc:53-58`):
- 计算两个 Box 的并集面积
- 公式: `union = area1 + area2 - intersection`

**`CalculateBoxIOU`** (`ImageUtils.cc:60-72`):
- 计算 `IOU = intersection / union`
- 若交集或并集为 0, 返回 0

### NMS 算法流程

```
输入: detections (forward_list of Detection), classes=1, iouThreshold=0.45
输出: 原地修改 detections, 被抑制的检测结果其 prob[class] 置为 0

逐类别循环 (仅 1 类):
  │
  ├── 1. 按当前类别的 prob 降序排列 detections
  │       (CompareProbs: prob1.prob[idxClass] > prob2.prob[idxClass])
  │
  └── 2. 从最高分开始遍历每个检测框 (设为 anchor 框):
         │
         ├── 如果 anchor.prob[class] == 0 → 跳过 (已被抑制)
         │
         └── 对每个后续检测框 (设为 candidate 框):
              │
              ├── 如果 candidate.prob[class] == 0 → 跳过
              │
              └── 如果 CalculateBoxIOU(anchor.bbox, candidate.bbox) > 0.45:
                   └── candidate.prob[class] = 0    ← 抑制该框
```

### 算法特点

- 使用 **`std::forward_list`** (单向链表), 排序和遍历效率较低但节省内存 (嵌入式场景)
- **贪心策略**: 得分最高的框保留, 与其 IOU 超过阈值的低分框被抑制 (prob 置 0)
- 被抑制的检测在后续 `DoPostProcess` 中因 `prob == 0` 被过滤掉 (`DetectorPostProcessing.cc:108`)
- 当 `classes=1` 时, 本质是单类别的标准 NMS

### Box 数据结构

```cpp
struct Box { float x, y, w, h; };  // 中心点坐标 + 宽高, 已映射到原图尺度

struct Detection {
    Box bbox;                    // 检测框
    std::vector<float> prob;    // 每个类别的置信度
    float objectness;           // 目标性得分
};
```

### 后处理中 NMS 的前置步骤 (`GetNetworkBoxes`, `DetectorPostProcessing.cc:139-228`)

NMS 之前, `GetNetworkBoxes` 已完成:

1. **双分支 (两个特征图尺度)**: branch 0 (分辨率 = 192/32 = 6×6), branch 1 (分辨率 = 192/16 = 12×12), 每个位置 3 个 anchor
2. **量化反量化**: 将 int8 输出通过 `(value - zeroPoint) * scale` 恢复为 float
3. **解码 bbox**: `x = sigmoid(tx) + grid_x`, `w = exp(tw) * anchor_w`, 再归一化到 0~1, 最后乘原图尺寸
4. **目标性过滤**: 只保留 `objectness > threshold(0.5)` 的候选框
5. **TopN 筛选**: 若 topN=0 则保留全部; 否则只保留最多 topN 个最高 objectness 的候选

NMS 之后, 再将 bbox 的 (center_x, center_y, w, h) 转为 (x0, y0, w, h) 即左上角坐标表示, 并进行边界裁剪 (clamp 到 [0, originalImageSize]).

---

## 六、DetectionResult 数据结构 (`DetectionResult.hpp`)

```cpp
class DetectionResult {
    double  m_normalisedVal;   // 置信度分数 (sig * objectness)
    int     m_x0;              // 左上角 x
    int     m_y0;              // 左上角 y
    int     m_w;               // 框宽
    int     m_h;               // 框高
};
```

---

## 七、关键设计特点

1. **预处理当前注释掉了实际逻辑** (`DetectorPreProcessing.cc:37-49`): 直接填零而非复制图像数据, 说明 `RunInference` 内部可能自行处理输入, 或当前处于调试状态
2. **双输出分支**: YOLO 模型有两个输出 head (stride=32 和 stride=16), 每个预测 3 个 anchor box, 共享同样的解码逻辑
3. **嵌入式适配**: 使用 `forward_list` 而非 `vector` 存储中间检测结果, 大量使用量化 int8 模型, 激活缓冲区仅 0.5MB
4. **DumpTensor**: 后处理会额外保存两份 `.npy` 格式的输出 tensor 到文件, 用于离线调试分析 (`DetectorPostProcessing.cc:70-71`)
