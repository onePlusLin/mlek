typedef struct {
    float x1, y1, x2, y2;
    float score;
    float area;
    int class_id;
} DetectionBox;

std::vector<DetectionBox> yolov8s_post_process(int8_t* input_data, const std::vector<std::string>& labels, float scale, int zero_point);
