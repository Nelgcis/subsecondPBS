# 剂量分布分析说明

## 概述
这个工具可以从C++计算的剂量分布中提取数据，并进行高斯叠加验证分析。

## 使用方法

### 1. 运行C++测试生成剂量数据
```bash
./build/bin/wrapper_integration_test
```

这将生成 `dose_distribution.bin` 文件，包含：
- 剂量体积维度 (400 x 400 x 32)
- 空间分辨率 (0.1 cm)
- 原点坐标 (-20.0, -20.0, 0.0) cm
- 所有体素的剂量值（float数组）

### 2. 运行Python分析脚本
```bash
python3 analyze_dose_from_bin.py
```

这将：
- 读取 `dose_distribution.bin`
- 找到最大剂量所在的Z截面
- 对该截面进行2D高斯拟合
- 计算R²值验证高斯叠加特性
- 生成以下输出文件：
  - `dose_analysis_summary.csv` - 分析摘要
  - `dose_distribution_slice.csv` - 该截面的剂量分布数据
  - `dose_analysis.png` - 可视化图表

## Z坐标信息
- Z范围: 0.0 到 3.1 cm（32个切片，每片0.1 cm）
- 分析结果将显示最大剂量所在的Z坐标

## 二进制文件格式
```
Header (36 bytes):
  - dims.x (int, 4 bytes)
  - dims.y (int, 4 bytes)  
  - dims.z (int, 4 bytes)
  - spacing.x (float, 4 bytes)
  - spacing.y (float, 4 bytes)
  - spacing.z (float, 4 bytes)
  - origin.x (float, 4 bytes)
  - origin.y (float, 4 bytes)
  - origin.z (float, 4 bytes)

Data:
  - dose values (float array, dims.x * dims.y * dims.z * 4 bytes)
  - 存储顺序: x, y, z (最内层是x)
```

## 高斯拟合验证
脚本会：
1. 在最大剂量的Z截面进行2D高斯拟合
2. 计算拟合的R²值
3. 如果R² > 0.9，认为满足高斯叠加特性

## 注意事项
- 需要有CUDA设备才能运行C++测试
- 如果 `dose_distribution.bin` 不存在，Python脚本会自动运行C++测试
- 分析结果会显示具体的Z坐标值
