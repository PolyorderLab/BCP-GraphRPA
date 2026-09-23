"""
重新计算 ML 推理和 RPA 相图，并生成发表级图片
"""
import os
import sys
import numpy as np

sys.path.append("/home/zxy/project/TAPD/src")

from ml_predictor import ChiNPredictor
from rpa_calculator_julia import get_all_phase_boundaries

# 配置
MODEL_PATH = "/home/zxy/project/TAPD/Demo/outcome/chiNSTL_optimal/best_model.pth"
OUTPUT_DIR = "/home/zxy/project/TAPD/Demo/outcome/phase_diagrams"

# 拓扑结构列表
TOPOLOGIES = ['AB', 'ABA', 'ABAB', 'ABABA', 'AB2', 'AB3', 'M1', 'M2']

def main():
    print("=" * 60)
    print("ML 推理 + RPA 计算 + 相图绘制")
    print("=" * 60)

    # 1. 初始化 ML 预测器
    print("\n[1/3] 加载 ML 模型...")
    predictor = ChiNPredictor(MODEL_PATH)
    print("   模型加载完成")

    # 2. 计算 ML 预测
    print("\n[2/3] 计算 ML 预测...")

    # M1/M2 与 AB/ABA 等其他拓扑保持一致的全范围
    fA_range_all = (0.01, 0.99)
    num_points = 100

    ml_results = {}
    for topo in TOPOLOGIES:
        fA_range = fA_range_all
        fA_values = np.linspace(fA_range[0], fA_range[1], num_points)
        print(f"   推理 {topo} (fA={fA_range[0]:.2f}-{fA_range[1]:.2f})...")
        chiN_pred = predictor.predict_batch(topo, fA_values, batch_size=64)
        ml_results[topo] = (fA_values, chiN_pred)
    print("   ML 推理完成")

    # 3. 计算 RPA 相边界
    print("\n[3/3] 计算 RPA 相边界...")
    rpa_results = {}
    for topo in TOPOLOGIES:
        print(f"   计算 {topo} RPA (fA={fA_range_all[0]:.2f}-{fA_range_all[1]:.2f})...")
        from rpa_calculator_julia import compute_rpa_phase_boundary
        fA_vals, chiN_vals = compute_rpa_phase_boundary(
            topo, fA_range_all, num_points, chiN0=15.0, julia_threads=112
        )
        rpa_results[topo] = (fA_vals, chiN_vals)
    print("   RPA 计算完成")

    # 4. 绘图
    print("\n[4/4] 生成发表级图片...")
    from plot_publication import generate_publication_plots

    generate_publication_plots(rpa_results, ml_results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"所有图片已保存至: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
