"""
针对 m1_m2_comparison_publication.png 的高密度推理：
M1/M2 在 0.11-0.19 区间内取 100 个点，RPA 同范围
"""
import os
import sys
import numpy as np

sys.path.append("/home/zxy/project/TAPD/src")

from ml_predictor import ChiNPredictor
from plot_publication import save_data_for_julia, plot_m1_m2_with_makie

MODEL_PATH = "/home/zxy/project/TAPD/Demo/outcome/chiNSTL_optimal/best_model.pth"
OUTPUT_DIR = "/home/zxy/project/TAPD/Demo/outcome/phase_diagrams"

FA_RANGE = (0.11, 0.19)
NUM_POINTS = 100

def main():
    print("=" * 60)
    print(f"M1/M2 高密度推理 (fA={FA_RANGE[0]:.2f}-{FA_RANGE[1]:.2f}, N={NUM_POINTS})")
    print("=" * 60)

    predictor = ChiNPredictor(MODEL_PATH)
    print("模型加载完成")

    fA_values = np.linspace(FA_RANGE[0], FA_RANGE[1], NUM_POINTS)

    # ML 推理
    ml_results = {}
    for topo in ["M1", "M2"]:
        print(f"  ML 推理 {topo}...")
        chiN_pred = predictor.predict_batch(topo, fA_values, batch_size=64)
        ml_results[topo] = (fA_values, chiN_pred)

    # RPA 计算
    rpa_results = {}
    for topo in ["M1", "M2"]:
        print(f"  RPA 计算 {topo}...")
        from rpa_calculator_julia import compute_rpa_phase_boundary
        fA_vals, chiN_vals = compute_rpa_phase_boundary(
            topo, FA_RANGE, NUM_POINTS, chiN0=15.0, julia_threads=112
        )
        rpa_results[topo] = (fA_vals, chiN_vals)

    # 保存并绘图
    data_path = os.path.join(OUTPUT_DIR, "m1_m2_comparison_data.json")
    save_data_for_julia(rpa_results, ml_results, data_path)
    print("数据保存完成")

    print("生成 m1_m2_comparison_publication.png ...")
    plot_m1_m2_with_makie(data_path, os.path.join(OUTPUT_DIR, "m1_m2_comparison_publication.png"), publication=True)
    print("完成")

if __name__ == "__main__":
    main()
