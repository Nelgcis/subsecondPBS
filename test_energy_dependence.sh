#!/bin/bash
# 测试不同能量下的剂量变化

cd /root/raytracedicom_updated1011

echo "=================================================================================="
echo "测试不同能量下的剂量变化"
echo "=================================================================================="

# 备份原始文件
cp src/tests/wrapper_integration_test.cu src/tests/wrapper_integration_test.cu.bak

# 测试1: 所有能量层设置为120 MeV
echo ""
echo "=== 测试1: 所有能量层设置为120 MeV ==="
sed -i 's/beam.energies = {120.0f, 140.0f, 160.0f};/beam.energies = {120.0f, 120.0f, 120.0f};/' src/tests/wrapper_integration_test.cu
cd build && make wrapper_integration_test -j$(nproc) > /dev/null 2>&1
echo "运行测试..."
timeout 60 ./bin/wrapper_integration_test 2>&1 | grep -E "(Layer|Ray IDD.*Total|Dose Analysis|Total dose|Max dose)" | head -10
cd ..

# 测试2: 所有能量层设置为140 MeV
echo ""
echo "=== 测试2: 所有能量层设置为140 MeV ==="
sed -i 's/beam.energies = {120.0f, 120.0f, 120.0f};/beam.energies = {140.0f, 140.0f, 140.0f};/' src/tests/wrapper_integration_test.cu
cd build && make wrapper_integration_test -j$(nproc) > /dev/null 2>&1
echo "运行测试..."
timeout 60 ./bin/wrapper_integration_test 2>&1 | grep -E "(Layer|Ray IDD.*Total|Dose Analysis|Total dose|Max dose)" | head -10
cd ..

# 测试3: 所有能量层设置为160 MeV
echo ""
echo "=== 测试3: 所有能量层设置为160 MeV ==="
sed -i 's/beam.energies = {140.0f, 140.0f, 140.0f};/beam.energies = {160.0f, 160.0f, 160.0f};/' src/tests/wrapper_integration_test.cu
cd build && make wrapper_integration_test -j$(nproc) > /dev/null 2>&1
echo "运行测试..."
timeout 60 ./bin/wrapper_integration_test 2>&1 | grep -E "(Layer|Ray IDD.*Total|Dose Analysis|Total dose|Max dose)" | head -10
cd ..

# 恢复原始文件
mv src/tests/wrapper_integration_test.cu.bak src/tests/wrapper_integration_test.cu

echo ""
echo "=================================================================================="
echo "测试完成"
echo "=================================================================================="
