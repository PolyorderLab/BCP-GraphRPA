#!/usr/bin/env python3
"""
统一打分脚本 V2 - 基于整体基线性能的综合评分

与 V1 的区别：
    V1: 每个嵌段数分别使用各自的 baseline，最后取平均
    V2: 直接取 baseline CSV 的最后一行（Overall）作为统一的基线

打分公式：
    Score = sqrt( (MAPE/MAPE_base)^2 + (RMSE/RMSE_base)^2 + ((1-R2)/(1-R2_base))^2 )

说明：
    - MAPE、RMSE：越小越好，直接与baseline做比值
    - R2：越大越好，转换为 (1-R2) 后也是越小越好
    - Score < 1 表示优于baseline，Score > 1 表示差于baseline
    - 基线取自 CSV 文件的 Overall 行（所有嵌段数取平均后的整体指标）

作者：zxy
日期：2026-04-29
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Tuple, Optional

# 项目路径
sys.path.append("/home/zxy/project/TAPD")
sys.path.append("/home/zxy/project/TAPD/src")
from src.viz_utils import TOPO_MAP_REV

# # ============ 配置路径 ============
# BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph_trainn"
# KSTL_BASE_DIR = os.path.join(BASE_RESULT_ROOT, "kSTL")
# CHINSTL_BASE_DIR = os.path.join(BASE_RESULT_ROOT, "chiNSTL")

# chiNlimited 专用基线路径
LIMITED_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph_trainnqc_limited"
LIMITED_KSTL_BASE_DIR = os.path.join(LIMITED_RESULT_ROOT, "kSTL_nqc1w2k")
LIMITED_CHINSTL_BASE_DIR = os.path.join(LIMITED_RESULT_ROOT, "chiNSTL_nqc1w2k")


# ============ 基线数据加载（V2: 统一取 Overall） ============
_baseline_cache = {}


def _load_baseline_overall(task_name: str, use_limited: bool = True) -> Dict[str, float]:
    """
    加载并缓存 STl 基线数据，返回 Overall 行（整体平均指标）。

    Args:
        task_name: 'kstar' 或 'chiN'
        use_limited: 是否使用 chiNlimited 专用基线（默认 True）

    Returns:
        {'mape': float, 'rmse': float, 'r2': float}
    """
    cache_key = f"{task_name}_limited" if use_limited else task_name
    if cache_key in _baseline_cache:
        return _baseline_cache[cache_key]

    if use_limited:
        base_dir = LIMITED_KSTL_BASE_DIR if task_name == 'kstar' else LIMITED_CHINSTL_BASE_DIR
    else:
        base_dir = KSTL_BASE_DIR if task_name == 'kstar' else CHINSTL_BASE_DIR

    csv_path = os.path.join(base_dir, "topology_mape_stats.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Baseline file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 取最后一行作为 Overall 基线
    overall_row = df.iloc[-1]

    # 确保是 Overall 行
    if str(overall_row['Topology']).strip() != 'Overall':
        raise ValueError(f"Expected last row to be 'Overall', got '{overall_row['Topology']}'")

    _baseline_cache[cache_key] = {
        'mape': float(overall_row['MAPE(%)']),
        'rmse': float(overall_row['RMSE']),
        'r2':   float(overall_row['R2'])
    }
    return _baseline_cache[cache_key]


def get_baseline_overall(task_name: str, use_limited: bool = True) -> Dict[str, float]:
    """
    获取指定任务的整体基线指标（Overall 行）。

    Args:
        task_name: 'kstar' 或 'chiN'
        use_limited: 是否使用 chiNlimited 专用基线（默认 True）

    Returns:
        {'mape': float, 'rmse': float, 'r2': float}
    """
    return _load_baseline_overall(task_name, use_limited)


# ============ 打分公式 ============
def compute_score(mape: float, rmse: float, r2: float,
                  mape_base: float, rmse_base: float, r2_base: float,
                  eps: float = 1e-9) -> float:
    """
    计算单一样本的打分。

    公式：score = sqrt( (MAPE/MAPE_base)^2 + (RMSE/RMSE_base)^2 + ((1-R2)/(1-R2_base))^2 )

    Args:
        mape, rmse, r2: 当前模型指标
        mape_base, rmse_base, r2_base: 基线模型指标
        eps: 防止除0的小量

    Returns:
        score: 综合得分（越小越好，<1 优于基线，>1 差于基线）
    """
    mape_base = max(mape_base, eps)
    rmse_base = max(rmse_base, eps)
    r2_base   = max(min(r2_base, 0.999999), eps)

    term1 = (mape / mape_base) ** 2
    term2 = (rmse / rmse_base) ** 2
    term3 = ((1 - r2) / (1 - r2_base)) ** 2

    return float(np.sqrt(term1 + term2 + term3))


# ============ 指标计算 ============
def compute_overall_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    计算整体的 MAPE, RMSE, R2（不对嵌段数分组，直接整体计算）。

    Args:
        y_pred: 预测值数组
        y_true: 真值数组

    Returns:
        {'mape': float, 'rmse': float, 'r2': float}
    """
    return {
        'mape': np.mean(np.abs((y_pred - y_true) / y_true)) * 100,
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2':   r2_score(y_true, y_pred)
    }


def compute_per_topo_metrics(model, dataloader, device, task_name: str) -> Dict[str, Dict[str, float]]:
    """
    从模型和dataloader计算每个topology的回归指标（MAPE, RMSE, R2）。

    Args:
        model: 训练好的模型
        dataloader: 测试集dataloader
        device: 计算设备
        task_name: 任务名 'chiN' 或 'kstar'

    Returns:
        {topo_name: {'mape': float, 'rmse': float, 'r2': float}, ...}
    """
    model.eval()
    preds, trues, topo_types = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out_chiN, out_kstar, out_cls, _ = model(batch)
            pred = out_chiN if task_name == 'chiN' else out_kstar
            true_val = batch.chiN_star if task_name == 'chiN' else batch.k_star

            preds.append(pred.cpu().numpy().reshape(-1, 1))
            trues.append(true_val.cpu().numpy().reshape(-1, 1))
            topo_types.extend(batch.topo_type.cpu().numpy().flatten())

    y_pred = 10 ** np.vstack(preds).flatten()
    y_true = 10 ** np.vstack(trues).flatten()
    topo_types = np.array(topo_types)

    results = {}
    unique_topos = np.unique(topo_types)

    for topo_idx in unique_topos:
        mask = (topo_types == topo_idx)
        t_pred = y_pred[mask]
        t_true = y_true[mask]

        if len(t_true) == 0:
            continue

        topo_name = TOPO_MAP_REV.get(int(topo_idx), f"Unknown-{topo_idx}")
        results[topo_name] = {
            'mape': np.mean(np.abs((t_pred - t_true) / t_true)) * 100,
            'rmse': np.sqrt(mean_squared_error(t_true, t_pred)),
            'r2':   r2_score(t_true, t_pred)
        }

    return results


def compute_per_topo_metrics_dual(model, dataloader, device) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    从DualRegressionModel计算每个topology的chiN和kstar指标。

    Args:
        model: DualRegressionModel (输出 out_chiN, out_kstar)
        dataloader: 测试集dataloader
        device: 计算设备

    Returns:
        (chiN_by_topo, kstar_by_topo): 两个字典，格式为 {topo_name: {'mape':, 'rmse':, 'r2':}}
    """
    model.eval()
    preds_chiN, trues_chiN, topo_types_chiN = [], [], []
    preds_kstar, trues_kstar, topo_types_kstar = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out_chiN, out_kstar = model(batch)

            preds_chiN.append(out_chiN.cpu().numpy().reshape(-1, 1))
            trues_chiN.append(batch.chiN_star.cpu().numpy().reshape(-1, 1))

            preds_kstar.append(out_kstar.cpu().numpy().reshape(-1, 1))
            trues_kstar.append(batch.k_star.cpu().numpy().reshape(-1, 1))

            if hasattr(batch, 'topo_type'):
                topo_types_chiN.extend(batch.topo_type.cpu().numpy().flatten())
                topo_types_kstar.extend(batch.topo_type.cpu().numpy().flatten())

    y_pred_chiN = 10 ** np.vstack(preds_chiN).flatten()
    y_true_chiN = 10 ** np.vstack(trues_chiN).flatten()
    topo_arr_chiN = np.array(topo_types_chiN)

    y_pred_kstar = 10 ** np.vstack(preds_kstar).flatten()
    y_true_kstar = 10 ** np.vstack(trues_kstar).flatten()
    topo_arr_kstar = np.array(topo_types_kstar)

    chiN_results = _aggregate_by_topo(y_pred_chiN, y_true_chiN, topo_arr_chiN)
    kstar_results = _aggregate_by_topo(y_pred_kstar, y_true_kstar, topo_arr_kstar)

    return chiN_results, kstar_results


def _aggregate_by_topo(y_pred, y_true, topo_types) -> Dict[str, Dict[str, float]]:
    """按topology聚合预测结果"""
    results = {}
    unique_topos = np.unique(topo_types)

    for topo_idx in unique_topos:
        mask = (topo_types == topo_idx)
        t_pred = y_pred[mask]
        t_true = y_true[mask]

        if len(t_true) == 0:
            continue

        topo_name = TOPO_MAP_REV.get(int(topo_idx), f"Unknown-{topo_idx}")
        results[topo_name] = {
            'mape': np.mean(np.abs((t_pred - t_true) / t_true)) * 100,
            'rmse': np.sqrt(mean_squared_error(t_true, t_pred)),
            'r2':   r2_score(t_true, t_pred)
        }

    return results


# ============ V2 核心：基于 Overall 基线打分 ============
def compute_overall_score_v2(chiN_metrics: Dict[str, float],
                             kstar_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    V2: 计算两个任务的综合得分，使用 Overall 基线。

    Args:
        chiN_metrics: {'mape':, 'rmse':, 'r2':} - chiN 整体指标
        kstar_metrics: {'mape':, 'rmse':, 'r2':} - kstar 整体指标

    Returns:
        {'chiN': float, 'kstar': float, 'composite': float}
    """
    scores = {}

    # chiN 得分
    if chiN_metrics:
        base_chiN = get_baseline_overall('chiN')
        scores['chiN'] = compute_score(
            chiN_metrics['mape'], chiN_metrics['rmse'], chiN_metrics['r2'],
            base_chiN['mape'], base_chiN['rmse'], base_chiN['r2']
        )

    # kstar 得分
    if kstar_metrics:
        base_kstar = get_baseline_overall('kstar')
        scores['kstar'] = compute_score(
            kstar_metrics['mape'], kstar_metrics['rmse'], kstar_metrics['r2'],
            base_kstar['mape'], base_kstar['rmse'], base_kstar['r2']
        )

    # 综合得分（几何平均）
    if 'chiN' in scores and 'kstar' in scores:
        scores['composite'] = float(np.sqrt(scores['chiN'] * scores['kstar']))

    return scores


def compute_overall_score_v2_from_dual(model, dataloader, device) -> Dict[str, float]:
    """
    V2: 从 DualRegressionModel 直接计算整体得分。

    Args:
        model: DualRegressionModel
        dataloader: 测试集dataloader
        device: 计算设备

    Returns:
        {'chiN': float, 'kstar': float, 'composite': float}
    """
    model.eval()
    preds_chiN, trues_chiN = [], []
    preds_kstar, trues_kstar = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out_chiN, out_kstar = model(batch)

            preds_chiN.append(out_chiN.cpu().numpy().reshape(-1, 1))
            trues_chiN.append(batch.chiN_star.cpu().numpy().reshape(-1, 1))

            preds_kstar.append(out_kstar.cpu().numpy().reshape(-1, 1))
            trues_kstar.append(batch.k_star.cpu().numpy().reshape(-1, 1))

    y_pred_chiN = 10 ** np.vstack(preds_chiN).flatten()
    y_true_chiN = 10 ** np.vstack(trues_chiN).flatten()
    y_pred_kstar = 10 ** np.vstack(preds_kstar).flatten()
    y_true_kstar = 10 ** np.vstack(trues_kstar).flatten()

    chiN_metrics = compute_overall_metrics(y_pred_chiN, y_true_chiN)
    kstar_metrics = compute_overall_metrics(y_pred_kstar, y_true_kstar)

    return compute_overall_score_v2(chiN_metrics, kstar_metrics)


# ============ 主程序测试 ============
if __name__ == "__main__":
    print("="*60)
    print("V2 统一打分脚本测试")
    print("="*60)

    # 测试：查看基线数据
    print("\n--- V2 基线数据预览（Overall 行）---")
    for task in ['chiN', 'kstar']:
        base = get_baseline_overall(task)
        print(f"\n{task} baseline (Overall):")
        print(f"  MAPE: {base['mape']:.4f}%")
        print(f"  RMSE: {base['rmse']:.4f}")
        print(f"  R2:   {base['r2']:.4f}")

    # 模拟测试
    print("\n--- 模拟打分测试 ---")
    test_chiN = {'mape': 5.0, 'rmse': 25.0, 'r2': 0.98}
    test_kstar = {'mape': 3.0, 'rmse': 0.5, 'r2': 0.99}
    scores = compute_overall_score_v2(test_chiN, test_kstar)
    print(f"\n模拟指标:")
    print(f"  chiN:  MAPE={test_chiN['mape']:.2f}%, RMSE={test_chiN['rmse']:.2f}, R2={test_chiN['r2']:.4f}")
    print(f"  kstar: MAPE={test_kstar['mape']:.2f}%, RMSE={test_kstar['rmse']:.2f}, R2={test_kstar['r2']:.4f}")
    print(f"\nV2 得分:")
    print(f"  chiN:     {scores['chiN']:.4f}")
    print(f"  kstar:    {scores['kstar']:.4f}")
    print(f"  composite: {scores['composite']:.4f}")
    print(f"\n说明：Score < 1 优于基线，> 1 差于基线")

    print("\n" + "="*60)
    print("主要接口：")
    print("  - compute_overall_metrics(y_pred, y_true): 计算整体指标")
    print("  - compute_overall_score_v2(chiN_metrics, kstar_metrics): V2打分")
    print("  - compute_overall_score_v2_from_dual(model, loader, device): V2直接打分")
    print("="*60)
