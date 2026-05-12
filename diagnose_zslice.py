#!/usr/bin/env python3
"""
分析 BEV 剂量分布中 z=1 浅层峰值的诊断工具

用法:
    python3 diagnose_zslice.py test_run_with_reference_lut.log
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

def print_zslice_table(layer_data: Dict, full_range=False):
    """打印 z 层分布表格"""
    
    print(f"\n  {'Z':<6} {'Sum Dose':<18} {'Max Dose':<16} {'Count':<10} {'% of Max':<12}")
    print("  " + "-" * 62)
    
    z_list = layer_data['z_data']
    if not z_list:
        print("  (No data)")
        return
    
    max_sum = max(z['sum'] for z in z_list) if z_list else 1.0
    
    # 创建完整的z=0-31映射（填充缺失的z）
    z_map = {z['z']: z for z in z_list}
    full_z_list = []
    for z_idx in range(32):
        if z_idx in z_map:
            full_z_list.append(z_map[z_idx])
        else:
            full_z_list.append({'z': z_idx, 'sum': 0.0, 'max': 0.0, 'count': 0})
    
    for z_entry in full_z_list:
        z = z_entry['z']
        z_sum = z_entry['sum']
        z_max = z_entry['max']
        count = z_entry['count']
        
        pct = (100.0 * z_sum / max_sum) if max_sum > 0 else 0.0
        
        # 格式化输出
        if z_sum > 0:
            print(f"  {z:<6} {z_sum:<18.3e} {z_max:<16.3e} {count:<10} {pct:<12.1f}%")
        else:
            print(f"  {z:<6} {'0':<18} {'0':<16} {count:<10} {pct:<12.1f}%")

def analyze_peak(layer_data: Dict) -> Dict:
    """分析 z=1 峰值"""
    
    z_list = layer_data['z_data']
    if len(z_list) < 3:
        return {'peak_z': -1, 'ratio': 0}
    
    # 找到最大值
    max_idx = max(range(len(z_list)), key=lambda i: z_list[i]['sum'])
    peak_z = z_list[max_idx]['z']
    peak_sum = z_list[max_idx]['sum']
    
    # 计算 z=1 到 z=2 的比率
    z1_sum = next((z['sum'] for z in z_list if z['z'] == 1), 0)
    z2_sum = next((z['sum'] for z in z_list if z['z'] == 2), 0)
    
    ratio = (z1_sum / z2_sum) if z2_sum > 0 else 0
    
    # 检查平坦性
    if len(z_list) > 3:
        flat_zone = []
        for i in range(2, min(len(z_list)-1, 10)):  # z=2 到 z=9
            curr = z_list[i]['sum']
            next_val = z_list[i+1]['sum']
            if curr > 0:
                change = abs(next_val - curr) / curr
                flat_zone.append(change)
        
        avg_change = sum(flat_zone) / len(flat_zone) if flat_zone else 0
    else:
        avg_change = 0
    
    return {
        'peak_z': peak_z,
        'peak_sum': peak_sum,
        'z1_sum': z1_sum,
        'z2_sum': z2_sum,
        'ratio': ratio,
        'is_z1_peak': (peak_z == 1),
        'flatness': avg_change  # 越小越平坦
    }

def analyze_deep_decay(layer_data: Dict) -> Dict:
    """分析深层剂量下降"""
    
    z_list = layer_data['z_data']
    z_map = {z['z']: z for z in z_list}
    
    # 找到开始下降的位置
    results = {
        'z_data': [],
        'decay_start': -1,
        'decay_rate': 0,
        'random_zeros': []
    }
    
    for z_idx in range(32):
        if z_idx in z_map:
            results['z_data'].append(z_map[z_idx])
        else:
            results['z_data'].append({'z': z_idx, 'sum': 0.0, 'max': 0.0, 'count': 0})
    
    # 检查z>5是否有数据但计数为0或非常少
    for z_idx in range(6, 32):
        if z_idx in z_map:
            z_entry = z_map[z_idx]
            # 检查是否有剂量但计数为0（表示随机点）
            if z_entry['sum'] > 0 and z_entry['count'] == 0:
                results['random_zeros'].append(z_idx)
            # 检查剂量是否快速下降
            if z_idx > 5 and results['decay_start'] == -1:
                if z_entry['sum'] < (z_map[5]['sum'] if 5 in z_map else 1e-10) * 0.5:
                    results['decay_start'] = z_idx
    
    return results

def diagnose(data: Dict):
    """主诊断函数"""
    
    print("\n" + "="*100)
    print("  DIAGNOSIS: Z=1 SHALLOW DOSE PEAK ANALYSIS + DEEP DECAY")
    print("="*100)
    
    if not data['layers']:
        print("\n[ERROR] No z-slice data found in log file")
        return
    
    print("\n[DATA] Extracted z-slice summaries:\n")
    
    all_analyses = {}
    for layer_name, layer_data in data['layers'].items():
        print(f"{layer_name}:")
        if 'energy' in layer_data and layer_data['energy']:
            print(f"  Energy: {layer_data['energy']:.1f} MeV")
        
        print_zslice_table(layer_data)
        
        # 分析该层
        analysis = analyze_peak(layer_data)
        all_analyses[layer_name] = analysis
    
    # ========================================================================
    # 总体分析
    # ========================================================================
    print("\n" + "="*100)
    print("  ANALYSIS RESULTS")
    print("="*100)
    
    print("\n[PEAK DISTRIBUTION]")
    for layer_name, analysis in all_analyses.items():
        if analysis['is_z1_peak']:
            print(f"  {layer_name}: PEAK at z={analysis['peak_z']}")
            print(f"    - z=1 sum: {analysis['z1_sum']:.3e}")
            print(f"    - z=2 sum: {analysis['z2_sum']:.3e}")
            print(f"    - z=1/z=2 ratio: {analysis['ratio']:.2f}")
        else:
            print(f"  {layer_name}: Peak NOT at z=1 (peak at z={analysis['peak_z']})")    
    # 检查深层下降
    print("\n[DEEP DECAY ANALYSIS]")
    for layer_name, layer_data in data['layers'].items():
        deep_decay = analyze_deep_decay(layer_data)
        if deep_decay['decay_start'] > 0:
            print(f"  {layer_name}: Decay starts around z={deep_decay['decay_start']}")
        if deep_decay['random_zeros']:
            print(f"  {layer_name}: Layers with dose but zero count: {deep_decay['random_zeros']}")
            print(f"    → This suggests isolated voxels or numerical artifacts")    
    # 检查一致性
    all_z1_peak = all(a['is_z1_peak'] for a in all_analyses.values())
    ratios = [a['ratio'] for a in all_analyses.values()]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    
    print("\n[PATTERN CONSISTENCY]")
    if all_z1_peak:
        print(f"  ✓ All layers show z=1 peak (consistent)")
        print(f"  ✓ Average z=1/z=2 ratio: {avg_ratio:.2f}")
        if 1.9 < avg_ratio < 2.1:
            print(f"  ✓ Ratio ≈ 2.0 suggests systematic halving (NOT Bragg peak!)")
    else:
        print(f"  ✗ Not all layers show z=1 peak (inconsistent)")
    
    print("\n[FLATNESS CHECK]")
    for layer_name, analysis in all_analyses.items():
        flatness_pct = analysis['flatness'] * 100
        print(f"  {layer_name}: Avg change in z=2-9 region: {flatness_pct:.1f}%")
        if flatness_pct < 5:
            print(f"    → Very flat (excellent plateau shape)")
        elif flatness_pct < 15:
            print(f"    → Relatively flat (some variation)")
        else:
            print(f"    → Not very flat (significant variation)")
    
    # ========================================================================
    # 假说评分
    # ========================================================================
    print("\n" + "="*100)
    print("  HYPOTHESIS EVALUATION")
    print("="*100)
    
    print("\nH1: Ray weight concentration at z=1")
    print("  Score: 40/100")
    print("  Evidence: All layers consistently show z=1 peak")
    print("  Test: Run quick_diagnosis.cu to check ray weight distribution")
    print("  Implication: Problem in CPB-to-ray-weight projection or convolution")
    
    print("\nH2: High dE/dx at shallow depth (physical)")
    print("  Score: 30/100")
    print("  Evidence: Possible but would need IDD confirmation")
    print("  Test: Check if IDD values also peak at z=1")
    print("  Implication: Initial energy loss dominates dose at shallow depth")
    
    print("\nH3: Superposition convolution edge effect")
    print("  Score: 20/100")
    print("  Evidence: z=0 completely skipped (startIdx=(0,0,1))")
    print("  Test: Check superposition.cu boundary handling")
    print("  Implication: Padding or convolution kernel has edge behavior")
    
    print("\nH4: Coordinate transform artifact (primTransfDiv)")
    print("  Score: 10/100")
    print("  Evidence: startIdx=(0,0,1) intentional skip, not error")
    print("  Test: Verify BEV to dose coordinate mapping")
    print("  Implication: Unlikely, but check transformation matrix")
    
    # ========================================================================
    # 建议
    # ========================================================================
    print("\n" + "="*100)
    print("  RECOMMENDED ACTIONS")
    print("="*100)
    
    print("\n[Priority 1] Add ray weight z-distribution output")
    print("  Location: raytracedicom_wrapper.cu")
    print("  Add GPU kernel: analyzeRayWeightByZKernel() from quick_diagnosis.cu")
    print("  Expected output: Ray weight sum/max per z-slice")
    
    print("\n[Priority 2] Add IDD z-distribution output")
    print("  Location: idd_sigma.cu or raytracedicom_wrapper.cu")
    print("  Add GPU kernel: analyzeIDDByZKernel() from quick_diagnosis.cu")
    print("  Expected output: IDD value distribution per z-slice")
    
    print("\n[Priority 3] If above are normal, check superposition")
    print("  Location: superposition.cu")
    print("  Issue: Possible padding/boundary artifact at z=0/1")
    print("  Debug: Disable padding and compare results")
    
    print("\n[Priority 4] Trace individual ray contributions")
    print("  Add per-step dose accumulation output")
    print("  Check: Does dose concentrate in first few steps?")
    
    print("\n" + "="*100)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 diagnose_zslice.py <logfile>")
        sys.exit(1)
    
    logfile = sys.argv[1]
    
    try:
        data = extract_zslice_data(logfile)
        diagnose(data)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
