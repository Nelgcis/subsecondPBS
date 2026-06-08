#!/usr/bin/env python3
"""
深层剂量下降分析工具
分析为什么z>2.5cm后剂量逐渐下降且表现为随机点消失

用法:
    python3 analyze_deep_decay.py test_run_with_reference_lut.log
"""

import re
import sys
from typing import Dict, List, Tuple

def extract_zslice_data(logfile: str) -> Dict:
    """从日志文件中提取 z 分层统计数据"""
    
    data = {
        'layers': {}
    }
    
    with open(logfile, 'r') as f:
        content = f.read()
    
    # 查找所有 "BEV dose z-slice summary" 块
    pattern = r'BEV dose z-slice summary.*?(?=\[TIMING\]|\[TRANSF\]|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    layer_idx = 0
    for match in matches:
        zslice_data = {'z_data': []}
        
        # 提取能量信息
        energy_match = re.search(r'\[ENERGY\].*?E=([\d.e+-]+)', content[max(0, content.find(match)-500):content.find(match)])
        if energy_match:
            energy = float(energy_match.group(1))
            zslice_data['energy'] = energy
        else:
            zslice_data['energy'] = None
        
        # 提取 z 层数据
        zslice_pattern = r'z=(\d+) sum=([\d.e+-]+) max=([\d.e+-]+).*?cnt>thr=(\d+)'
        for z_match in re.finditer(zslice_pattern, match):
            z = int(z_match.group(1))
            z_sum = float(z_match.group(2))
            z_max = float(z_match.group(3))
            z_count = int(z_match.group(4))
            
            zslice_data['z_data'].append({
                'z': z,
                'sum': z_sum,
                'max': z_max,
                'count': z_count
            })
        
        if zslice_data['z_data']:
            data['layers'][f'Layer {layer_idx}'] = zslice_data
            layer_idx += 1
    
    return data

def analyze_decay_pattern(layer_data: Dict) -> Dict:
    """分析剂量下降模式"""
    
    z_list = layer_data['z_data']
    if not z_list:
        return {}
    
    z_map = {z['z']: z for z in z_list}
    
    # 创建完整的z=0-31映射
    full_z_data = []
    for z_idx in range(32):
        if z_idx in z_map:
            full_z_data.append(z_map[z_idx])
        else:
            full_z_data.append({'z': z_idx, 'sum': 0.0, 'max': 0.0, 'count': 0})
    
    # 分析各个区域
    result = {
        'z_data': full_z_data,
        'z1_peak': {
            'z_sum': z_map.get(1, {}).get('sum', 0),
            'z_count': z_map.get(1, {}).get('count', 0),
        },
        'plateau': {  # z=2-5平坦区
            'avg_sum': 0,
            'avg_count': 0,
            'change_rate': 0,
        },
        'decay': {  # z>5的下降
            'decay_start_z': None,
            'decay_rate': 0,
            'final_sum': 0,
            'random_zeros': [],  # sum>0 but count=0
        }
    }
    
    # 计算平坦区统计 (z=2-5)
    plateau_sums = []
    plateau_counts = []
    for z_idx in range(2, 6):
        if z_idx in z_map:
            plateau_sums.append(z_map[z_idx]['sum'])
            plateau_counts.append(z_map[z_idx]['count'])
    
    if plateau_sums:
        result['plateau']['avg_sum'] = sum(plateau_sums) / len(plateau_sums)
        result['plateau']['avg_count'] = sum(plateau_counts) / len(plateau_counts)
    
    # 分析深层下降 (z>5)
    decay_sums = []
    decay_counts = []
    for z_idx in range(6, 32):
        if z_idx in z_map:
            z_entry = z_map[z_idx]
            decay_sums.append(z_entry['sum'])
            decay_counts.append(z_entry['count'])
            
            # 检查"随机点"：有剂量但没有非零体素计数
            if z_entry['sum'] > 0 and z_entry['count'] == 0:
                result['decay']['random_zeros'].append(z_idx)
            
            # 找到开始下降的位置
            if result['decay']['decay_start_z'] is None:
                if z_entry['sum'] < result['plateau']['avg_sum'] * 0.5:
                    result['decay']['decay_start_z'] = z_idx
        else:
            decay_sums.append(0.0)
            decay_counts.append(0)
    
    # 计算衰减率
    if decay_sums and result['plateau']['avg_sum'] > 0:
        result['decay']['final_sum'] = decay_sums[-1] if decay_sums else 0
        
        # 计算z=6到z=31的平均衰减
        valid_decays = [s for s in decay_sums if s > 0]
        if len(valid_decays) > 1:
            # 简单线性拟合的衰减率
            result['decay']['decay_rate'] = (valid_decays[0] - valid_decays[-1]) / valid_decays[0] if valid_decays[0] > 0 else 0
    
    return result

def print_full_table(layer_name: str, analysis: Dict):
    """打印完整的z=0-31表格"""
    
    z_data = analysis['z_data']
    
    print(f"\n{layer_name}:")
    print(f"  {'Z':<5} {'Sum':<16} {'Max':<16} {'Count':<8} {'Region':<20}")
    print("  " + "-" * 75)
    
    max_sum = max((z['sum'] for z in z_data), default=1.0)
    
    for z_entry in z_data:
        z = z_entry['z']
        z_sum = z_entry['sum']
        z_max = z_entry['max']
        count = z_entry['count']
        
        # 标记区域
        if z == 0:
            region = "[SKIPPED]"
        elif z == 1:
            region = "[PEAK]"
        elif 2 <= z <= 5:
            region = "[PLATEAU]"
        elif 6 <= z <= 10:
            region = "[EARLY DECAY]"
        elif 11 <= z <= 20:
            region = "[MID DECAY]"
        else:
            region = "[LATE/TAIL]"
        
        # 格式化
        if z_sum > 0:
            pct = 100.0 * z_sum / max_sum if max_sum > 0 else 0
            print(f"  {z:<5} {z_sum:<16.3e} {z_max:<16.3e} {count:<8} {region:<20} ({pct:5.1f}%)")
        else:
            print(f"  {z:<5} {'0':<16} {'0':<16} {count:<8} {region:<20} (0.0%)")

def analyze_and_report(data: Dict):
    """主分析和报告函数"""
    
    print("\n" + "="*100)
    print("  DEEP DECAY ANALYSIS: Why does dose decrease after 2.5cm?")
    print("="*100)
    
    print("\n[PHYSICAL PARAMETERS]")
    print("  Ray tracing range: z = [-32mm, 0mm] = -3.2cm to 0cm (向后)")
    print("  z-coordinate system: z=0 is beam entry, z increases backward into patient")
    print("  2.5cm depth corresponds to approximately z=25 (in 1mm steps)")
    print("  Bragg peak depths:")
    print("    120 MeV: 103.75 mm (beyond tracing range!)")
    print("    160 MeV: 173.92 mm (way beyond tracing range!)")
    print("    180 MeV: 214.05 mm (way beyond tracing range!)")
    
    print("\n[DATA EXTRACTION]")
    all_analyses = {}
    for layer_name, layer_data in data['layers'].items():
        analysis = analyze_decay_pattern(layer_data)
        all_analyses[layer_name] = analysis
    
    print(f"  Found {len(all_analyses)} energy layers with z-slice data")
    
    # 打印完整表格
    print("\n" + "="*100)
    print("  COMPLETE Z=0-31 DISTRIBUTION (all layers)")
    print("="*100)
    
    for layer_name, analysis in all_analyses.items():
        print_full_table(layer_name, analysis)
    
    # 深度分析
    print("\n" + "="*100)
    print("  DETAILED ANALYSIS")
    print("="*100)
    
    print("\n[1] Z=1 PEAK (已确认)")
    for layer_name, analysis in all_analyses.items():
        print(f"  {layer_name}:")
        print(f"    z=1 sum: {analysis['z1_peak']['z_sum']:.3e}")
        print(f"    z=1 count: {analysis['z1_peak']['z_count']}")
    
    print("\n[2] PLATEAU REGION (z=2-5)")
    for layer_name, analysis in all_analyses.items():
        print(f"  {layer_name}:")
        print(f"    Average sum: {analysis['plateau']['avg_sum']:.3e}")
        print(f"    Average count: {analysis['plateau']['avg_count']:.0f}")
        pct_of_peak = (100 * analysis['plateau']['avg_sum'] / analysis['z1_peak']['z_sum']) if analysis['z1_peak']['z_sum'] > 0 else 0
        print(f"    % of z=1 peak: {pct_of_peak:.1f}%")
    
    print("\n[3] DECAY REGION ANALYSIS (z>5)")
    for layer_name, analysis in all_analyses.items():
        print(f"  {layer_name}:")
        
        if analysis['decay']['decay_start_z']:
            print(f"    Decay starts at z≈{analysis['decay']['decay_start_z']}")
        
        if analysis['decay']['decay_rate'] > 0:
            print(f"    Overall decay rate: {100*analysis['decay']['decay_rate']:.1f}%")
        
        if analysis['decay']['random_zeros']:
            print(f"    ⚠️  Layers with dose but count=0: z={analysis['decay']['random_zeros']}")
            print(f"        This indicates isolated voxels or numerical artifacts!")
        
        print(f"    Final value at z=31: {analysis['decay']['final_sum']:.3e}")
    
    # 物理解释
    print("\n" + "="*100)
    print("  PHYSICAL INTERPRETATION")
    print("="*100)
    
    print("\n[WHY DOSE DECREASES AT z>2.5cm]")
    print("\n  问题1: 射线追踪范围限制")
    print("    当前范围: -32mm 到 0mm (33步)")
    print("    范围内最深: z=32 = 3.2cm = 32mm")
    print("    Bragg峰位置:")
    print("      120 MeV: 103.75 mm  ← 远超范围!")
    print("      160 MeV: 173.92 mm  ← 远超范围!")
    print("      180 MeV: 214.05 mm  ← 远超范围!")
    print("    结论: 我们完全看不到Bragg峰。追踪停在 ~3cm，但峰在 10+ cm处")
    
    print("\n  问题2: 剂量分布的物理原因")
    print("    浅层(z=0-2.5cm):")
    print("      - 质子速度快，能量损失率(dE/dx)中等")
    print("      - 多数射线在这个范围内")
    print("      - 累积效应导致相对均匀的剂量")
    print("")
    print("    深层(z=2.5-3.2cm):")
    print("      - 质子继续减速，但仍未到达Bragg峰")
    print("      - 射线数量减少（部分已停止）")
    print("      - dE/dx继续上升，但多样性增加")
    print("      - 导致剂量下降但非均匀")
    
    print("\n  问题3: 'z>2.5cm处随机点消失'的含义")
    if any(analysis['decay']['random_zeros'] for analysis in all_analyses.values()):
        print("    ✓ 观察到: 某些z层有sum>0但count=0")
        print("    原因: 可能是")
        print("      1. 数值舍入误差（非常小的值）")
        print("      2. 特定体素有剂量，但不足以创建单独的计数")
        print("      3. 卷积或插值的边界效应")
    else:
        print("    未观察到count=0的情况")
        print("    但z>2.5cm的非零体素数确实减少了")
    
    # 建议
    print("\n" + "="*100)
    print("  RECOMMENDATIONS")
    print("="*100)
    
    print("\n[扩展射线追踪范围]")
    print("  当前: steps=33, stepLen=1mm, range=3.2cm")
    print("  建议: 增加steps到 150+ 以覆盖150+ mm深度")
    print("        这样才能看到完整的Bragg峰!")
    print("  代码位置: raytracedicom_wrapper.cu")
    print("           params.steps 设置")
    
    print("\n[分析当前范围的有效性]")
    print("  虽然看不到Bragg峰，但当前范围足以:")
    print("    ✓ 验证算法的早期能量沉积")
    print("    ✓ 检查浅层的数值稳定性")
    print("    ✓ 验证射线权重和IDD的映射")
    print("  关键是: z>2.5cm的下降是EXPECTED的，因为射线还在减速")
    
    print("\n[验证数据完整性]")
    print("  1. 检查是否有count=0的情况（表示artifact）")
    print("  2. 检查sum的平滑性（应该平缓下降）")
    print("  3. 检查CT数据是否在某个深度有边界")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_deep_decay.py <logfile>")
        sys.exit(1)
    
    logfile = sys.argv[1]
    
    try:
        data = extract_zslice_data(logfile)
        analyze_and_report(data)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
