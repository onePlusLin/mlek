#include <fstream>
#include <vector>
#include <iostream>

// 保存为纯文本格式（可读性好，但文件大）
void saveTensorToText(const std::vector<float>& tensor, 
                      const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    
    // 写入张量形状信息
    file << "shape: 1 25200 85\n";
    file << "total_elements: " << tensor.size() << "\n";
    
    // 写入数据
    for (size_t i = 0; i < tensor.size(); ++i) {
        file << tensor[i];
        if (i < tensor.size() - 1) {
            file << " ";
        }
        // 每行保存10个值，提高可读性
        if ((i + 1) % 10 == 0) {
            file << "\n";
        }
    }
    
    file.close();
    std::cout << "张量已保存到: " << filename << std::endl;
}