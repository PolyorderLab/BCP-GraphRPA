"""
迁移学习训练脚本 (TLtrain.py) - 修改版
=====================================
阶段一: 预训练 (Pre-training)
    - 使用 train_nnumber_1w 数据集
    - 只在重构任务 (Reconstruction) 上进行自监督预训练
    - 保存 best_pretrained_model.pth

阶段二: 微调 (Fine-tuning)
    - 加载预训练 Encoder 权重
    - 支持四种迁移策略:
        A. feature_extract  - 冻结全部 Encoder，只训练 Head
        B. progressive      - 渐进式解冻 Encoder
        C. diff_lr          - Encoder 极小 lr / Head 正常 lr
        D. full             - 全量微调
    - 目标任务: chiN 回归 + k* 回归 (dual_regression)
    - 保存 best_finetuned_{strategy}_model.pth
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# ================= 0. 路径配置 =================
PROJECT_ROOT = "/home/zxy/project/TAPD"
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from src.data_utils import load_and_process_dataset
from src.model_utils import get_model, GNNEncoder, MultiTaskModel
from src.train_utils import get_splits
from src.transfer_utils import (
    freeze_encoder,
    freeze_encoder_layers,
    get_fine_tune_optimizer,
    load_pretrained_encoder,
)
from src.viz_utils import (
    save_loss_curve, 
    evaluate_and_plot_dual
)

# ================= 1. 超参配置 =================
# --- 数据 ---
DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_nnumber_1w"

# --- 输出 ---
RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/TransferLearning"
PRETRAIN_CKPT = os.path.join(RESULT_ROOT, "best_pretrained_model.pth")

# --- 模型结构 ---
IN_FEATS = 3
EDGE_DIM = 2
HIDDEN_DIM = 64
NUM_ENC_LAYERS = 3
NUM_CLASSES = 5
MAX_NODES = 32

# --- 预训练 ---
PRETRAIN_EPOCHS = 200
PRETRAIN_LR = 1e-3
PRETRAIN_BATCH_SIZE = 64

# --- 微调 ---
FINETUNE_EPOCHS = 100
FINETUNE_BATCH_SIZE = 64
ENCODER_LR = 1e-4       # 差分学习率: Encoder
HEAD_LR = 1e-3          # 差分学习率: Head
PROGRESSIVE_UNFREEZE_EVERY = 20  # 每隔多少 epoch 解冻一层 Encoder

# --- 控制开关 ---
# 阶段一: 若已有 checkpoint 可跳过预训练
SKIP_PRETRAIN = False
# 阶段二: 要运行的微调策略 (可选: 'feature_extract', 'progressive', 'diff_lr', 'full')
STRATEGIES_TO_RUN = ['feature_extract', 'progressive', 'diff_lr', 'full']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= 2. 工具函数 =================

def build_encoder():
    return GNNEncoder(
        in_feats=IN_FEATS,
        edge_dim=EDGE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_ENC_LAYERS
    )


def compute_recon_loss(out, batch):
    """计算纯重构损失 (Adj + Features + Edge Attributes)."""
    criterion_mse = nn.MSELoss()
    out_A, out_F, out_E = out

    target_A = batch.dense_adj.view(-1, MAX_NODES, MAX_NODES)
    target_F = batch.dense_F.view(-1, MAX_NODES, 3)
    target_E = batch.dense_edge_attr.view(-1, MAX_NODES, MAX_NODES, 2)

    # binary_cross_entropy 在 autocast 下不安全，需要手动关闭
    device_type = target_A.device.type
    with torch.amp.autocast(device_type, enabled=False):
        loss_A = nn.functional.binary_cross_entropy(out_A.float(), target_A.float())
    loss_F     = criterion_mse(out_F, target_F)
    loss_E     = criterion_mse(out_E, target_E)
    
    return loss_A + loss_F + loss_E


def compute_dual_loss(out, batch):
    """计算微调阶段的双回归损失 (chiN + k*)."""
    criterion = nn.MSELoss()
    out_chiN, out_kstar = out
    return criterion(out_chiN, batch.chiN_star) + criterion(out_kstar, batch.k_star)


# ================= 3. 阶段一: 预训练 =================

def pretrain(train_loader, val_loader):
    """在重构任务上进行自监督预训练，保存最优 checkpoint."""
    print("\n" + "="*60)
    print("🚀 阶段一: 开始预训练 (Reconstruction Only)")
    print("="*60)

    # 加载重构专用模型
    model = get_model(
        task_type='reconstruction',
        in_feats=IN_FEATS,
        edge_dim=EDGE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_enc_layers=NUM_ENC_LAYERS,
        max_nodes=MAX_NODES,
        device=DEVICE
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')

    best_val_loss = float('inf')
    os.makedirs(RESULT_ROOT, exist_ok=True)

    train_hist, val_hist = [], []

    for epoch in tqdm(range(1, PRETRAIN_EPOCHS + 1), desc="预训练[重构]"):
        # --- Train ---
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(DEVICE.type):
                out = model(batch)
                loss = compute_recon_loss(out, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item() * batch.num_graphs
        train_loss = total_train_loss / len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                with torch.amp.autocast(DEVICE.type):
                    out = model(batch)
                    loss = compute_recon_loss(out, batch)
                total_val_loss += loss.item() * batch.num_graphs
        val_loss = total_val_loss / len(val_loader.dataset)

        train_hist.append(train_loss)
        val_hist.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), PRETRAIN_CKPT)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Best Val: {best_val_loss:.6f}")

    # --- 后处理: 绘图 ---
    print("\n📊 正在保存预训练损失曲线...")
    save_loss_curve(train_hist, val_hist, os.path.join(RESULT_ROOT, "pretrain_loss_curve.png"))
    
    print(f"✅ 预训练完成! 最优重构 Val Loss: {best_val_loss:.6f}")
    print(f"   Checkpoint 已保存至: {PRETRAIN_CKPT}")
    return model


# ================= 4. 阶段二: 微调 =================

def run_finetune(strategy, train_loader, val_loader):
    """
    执行一种迁移策略的微调实验。

    Args:
        strategy: 'feature_extract' | 'progressive' | 'diff_lr' | 'full'
    """
    print(f"\n{'='*60}")
    print(f"🔧 开始微调 | 策略: [{strategy.upper()}]")
    print(f"{'='*60}")

    save_dir = os.path.join(RESULT_ROOT, f"finetune_{strategy}")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "best_finetuned_model.pth")

    # --- 构造双回归模型 (chiN + k*) ---
    from src.model_utils import DualRegressionModel
    encoder = build_encoder()
    model = DualRegressionModel(encoder, HIDDEN_DIM).to(DEVICE)

    # --- 加载预训练 Encoder 权重 ---
    # 注意: load_pretrained_encoder 默认按 'encoder.' 前缀加载。 
    # 预训练模型(ReconModelWrapper)和微调模型均有 encoder 成员，键名匹配。
    model = load_pretrained_encoder(model, PRETRAIN_CKPT, DEVICE, load_heads=False)
    print(f"   ✅ 已加载预训练 Encoder 权重: {PRETRAIN_CKPT}")

    # --- 根据策略配置 Optimizer ---
    if strategy == 'feature_extract':
        # 冻结全部 Encoder，只训练 Head
        freeze_encoder(model, freeze=True)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR
        )
        print("   策略: 冻结全部 Encoder，仅训练 Task Head")

    elif strategy == 'progressive':
        # 初始冻结全部 Encoder，之后逐层解冻
        freeze_encoder(model, freeze=True)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR
        )
        print(f"   策略: 渐进解冻 (每 {PROGRESSIVE_UNFREEZE_EVERY} epoch 解冻一层)")

    elif strategy == 'diff_lr':
        # 差分学习率: Encoder 用小 lr，Head 用大 lr
        freeze_encoder(model, freeze=False)
        optimizer = get_fine_tune_optimizer(model, encoder_lr=ENCODER_LR, head_lr=HEAD_LR)
        print(f"   策略: 差分学习率 (Encoder lr={ENCODER_LR}, Head lr={HEAD_LR})")

    elif strategy == 'full':
        # 全量微调
        freeze_encoder(model, freeze=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=HEAD_LR)
        print("   策略: 全量微调 (所有参数使用统一 lr)")

    else:
        raise ValueError(f"未知迁移策略: {strategy}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')

    best_val_loss = float('inf')
    # 渐进解冻：记录当前已解冻到第几层
    current_frozen_layers = NUM_ENC_LAYERS if strategy == 'progressive' else 0

    train_hist, val_hist = [], []

    for epoch in tqdm(range(1, FINETUNE_EPOCHS + 1), desc=f"微调[{strategy}]"):

        # ---- 渐进解冻逻辑 ----
        if strategy == 'progressive':
            # 每隔 PROGRESSIVE_UNFREEZE_EVERY 个 epoch，解冻一层（从最后一层开始）
            layers_to_unfreeze = epoch // PROGRESSIVE_UNFREEZE_EVERY
            new_frozen = max(NUM_ENC_LAYERS - layers_to_unfreeze, 0)
            if new_frozen != current_frozen_layers:
                current_frozen_layers = new_frozen
                freeze_encoder_layers(model, current_frozen_layers)
                # 重建 optimizer 以纳入新激活的参数
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR
                )
                print(f"\n  ↗ Epoch {epoch}: 解冻至 Encoder 层 {current_frozen_layers}~{NUM_ENC_LAYERS - 1}")

        # --- Train ---
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(DEVICE.type):
                out = model(batch)
                loss = compute_dual_loss(out, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item() * batch.num_graphs
        train_loss = total_train_loss / len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                with torch.amp.autocast(DEVICE.type):
                    out = model(batch)
                    loss = compute_dual_loss(out, batch)
                total_val_loss += loss.item() * batch.num_graphs
        val_loss = total_val_loss / len(val_loader.dataset)

        train_hist.append(train_loss)
        val_hist.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best Val: {best_val_loss:.4f}")

    # --- 后处理: 绘图与评估 ---
    print(f"📊 正在生成微调可视化结果 [{strategy}]...")
    save_loss_curve(train_hist, val_hist, os.path.join(save_dir, "finetune_loss_curve.png"))
    
    # 评估最佳模型
    model.load_state_dict(torch.load(ckpt_path))
    ft_results = evaluate_and_plot_dual(model, val_loader, DEVICE, save_dir)

    print(f"\n✅ 微调完成 [{strategy}]! 最优 Val Loss: {best_val_loss:.4f}")
    print(f"   - chiN MAPE: {ft_results.get('chiN_mape', 0):.2f}%")
    print(f"   - k* MAPE:   {ft_results.get('kstar_mape', 0):.2f}%")
    print(f"   Checkpoint 已保存至: {ckpt_path}")
    return model, best_val_loss


# ================= 5. 主流程 =================

def main():
    print(f"{'='*60}")
    print(f"🧬 迁移学习 experiment 启动 | Device: {DEVICE}")
    print(f"{'='*60}")
    print(f"  数据目录:  {DATA_DIR}")
    print(f"  输出目录:  {RESULT_ROOT}")

    # ---- 加载数据集 ----
    # 预训练需要 mtl 类型（包含 dense_adj 等重建所需字段）
    print("\n📥 加载预训练数据集 (task_type=mtl)...")
    pretrain_dataset = load_and_process_dataset(DATA_DIR, task_type='mtl')
    tr_data, val_data, _ = get_splits(pretrain_dataset, DATA_DIR)

    pretrain_train_loader = DataLoader(
        tr_data, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
    )
    pretrain_val_loader = DataLoader(
        val_data, batch_size=PRETRAIN_BATCH_SIZE, num_workers=8, pin_memory=True
    )

    # ---- 阶段一: 预训练 ----
    if SKIP_PRETRAIN and os.path.exists(PRETRAIN_CKPT):
        print(f"\n⏭️  跳过预训练，直接加载: {PRETRAIN_CKPT}")
    else:
        pretrain(pretrain_train_loader, pretrain_val_loader)

    # ---- 加载微调数据集 ----
    # 微调只需要 dual_regression 类型（chiN + k*，无需 dense 矩阵）
    print("\n📥 加载微调数据集 (task_type=dual_regression)...")
    finetune_dataset = load_and_process_dataset(DATA_DIR, task_type='dual_regression')
    ft_tr_data, ft_val_data, _ = get_splits(finetune_dataset, DATA_DIR)

    ft_train_loader = DataLoader(
        ft_tr_data, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
    )
    ft_val_loader = DataLoader(
        ft_val_data, batch_size=FINETUNE_BATCH_SIZE, num_workers=8, pin_memory=True
    )

    # ---- 阶段二: 多策略微调实验 ----
    results = {}
    for strategy in STRATEGIES_TO_RUN:
        _, best_val = run_finetune(strategy, ft_train_loader, ft_val_loader)
        results[strategy] = best_val

    # ---- 汇总结果 ----
    print(f"\n{'='*60}")
    print("📊 微调结果汇总 (Val Loss):")
    for strategy, val_loss in sorted(results.items(), key=lambda x: x[1]):
        print(f"  [{strategy:<18}]  Best Val Loss: {val_loss:.6f}")
    print(f"{'='*60}")
    print("✅ 所有实验完成!")


if __name__ == "__main__":
    main()
