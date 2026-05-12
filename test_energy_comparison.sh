#!/bin/bash
cd /root/raytracedicom_updated1011

echo "=================================================================================="
echo "测试不同能量下的最终总剂量"
echo "=================================================================================="

# 备份
cp src/tests/wrapper_integration_test.cu src/tests/wrapper_integration_test.cu.bak

# 测试1: 120 MeV
echo ""
echo "=== 测试1: 所有能量层 = 120 MeV ==="
sed -i 's/beam.energies = {120.0f, 140.0f, 160.0f};/beam.energies = {120.0f, 120.0f, 120.0f};/' src/tests/wrapper_integration_test.cu
cd build && make wrapper_integration_test -j$(nproc) > /dev/null 2>&1
DOSE1=$(timeout 60 ./bin/wrapper_integration_test 2>&1 | grep "Total dose:" | awk '{print $3}')
echo "总剂量: $DOSE1"
cd ..

# 测试2: 140 MeV  
echo ""
echo "=== 测试2: 所有能量层 = 140 MeV ==="
sed -i 's/beam.energies = {120.0f, 120.0f, 120.0f};/beam.energies = {140.0f, 140.0f, 140.0f};/' src/tests/wrapper_integration_test.cu
cd build && make wrapper_integration_test -j$(nproc) > /dev/null 2>&1
DOSE2=$(timeout 60 ./bin/wrapper_integration_test 2>&1 | grep "Total dose:" | awk '{print $3}')
echo "总剂量: $DOSE2"
cd ..

# 测试3: 160 MeV
echo ""
echo "=== 测试3: 所有能量层 = 160 MeV ==="
sed -i 's/beam.energies = {140.0f, 140.0f, 140.0f};/beam.energies = {160.0f, 160.0f, 160.0f};/' src/tests/wrapper_integration_test.cu
cd build && make wrapper_integration_test -j$(nproc) > /dev/null 2>&1
DOSE3=$(timeout 60 ./bin/wrapper_integration_test 2>&1 | grep "Total dose:" | awk '{print $3}')
echo "总剂量: $DOSE3"
cd ..

# 恢复
mv src/tests/wrapper_integration_test.cu.bak src/tests/wrapper_integration_test.cu

echo ""
echo "=================================================================================="
echo "结果汇总:"
echo "  120 MeV: $DOSE1"
echo "  140 MeV: $DOSE2"
echo "  160 MeV: $DOSE3"
echo "=================================================================================="

# 检查是否不同
if [ "$DOSE1" != "$DOSE2" ] && [ "$DOSE2" != "$DOSE3" ] && [ "$DOSE1" != "$DOSE3" ]; then
    echo "✓ 剂量随能量变化 - 正常"
else
    echo "✗ 剂量不随能量变化 - 有问题"
fi
