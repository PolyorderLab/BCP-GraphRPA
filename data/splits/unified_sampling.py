#!/usr/bin/env python3
"""
Unified Stratified Dataset Sampling Script

数据集划分流程（完全按照描述实现）：

  Step 1: 全量 12w 样本 → 分成 2w test pool + 10w train pool
           - 每个嵌段数（2/3/4/5 blocks）各贡献 5000 → test pool（2w）
           - 每个嵌段数（2/3/4/5 blocks）各贡献 25000 → train pool（10w）
           - 随机采样，seed=42

  Step 2: 从 train pool（10w）输出 train 集
           - 目标规模：1k, 2k, 3k, 4k, 5k, 6k, 7k, 8k, 9k, 10k,
                     20k, 30k, 40k, 50k, 60k, 70k, 80k, 90k, 100k
           - 每种规模下，四个嵌段数样本量均匀分配
           - 例如 10k → 每个嵌段数 2500 个；100k → 每个嵌段数 25000 个

  Step 3: 从 test pool（2w）输出 test 集
           - 最终 test 集：2k 个样本，四个嵌段数各 500 个

输入: data/datasets/raw_2to5/
       （子文件夹 2blocks/, 3blocks/, 4blocks/, 5blocks/，每个 30000 个 .mat+.yml）

输出: data/datasets/
       - nqc_train_pool/    （10w 训练池，扁平化）
       - nqc_test_pool/     （2w 测试池，扁平化）
       - nqc_train_1k/  ~  nqc_train_100k/  （各个规模训练集）
       - nqc_test_2k/       （2k 测试集）
"""

import os
import sys
import random
import shutil
import yaml
import pickle
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse

# ============================================================
# 配置
# ============================================================
SOURCE_DIR = str(Path(__file__).resolve().parents[1] / "datasets/raw_2to5")
BASE_DEST  = str(Path(__file__).resolve().parents[1] / "datasets")
CACHE_FILE = os.path.join(BASE_DEST, ".chiNlimited_samples_cache.pkl")

TRAIN_POOL_DIR  = os.path.join(BASE_DEST, "nqc_train_pool")
TEST_POOL_DIR   = os.path.join(BASE_DEST, "nqc_test_pool")

# 每个嵌段数的目标 test pool 规模
N_BLOCKS_VALUES = [2, 3, 4, 5]
SAMPLES_PER_BLOCK_TEST_POOL = 5000   # 2w test pool = 4 × 5000
SAMPLES_PER_BLOCK_TRAIN_POOL = 25000 # 10w train pool = 4 × 25000

# Step 2: train 集规模列表
TRAIN_TARGET_SIZES = [
    1000, 2000, 3000, 4000, 5000,
    6000, 7000, 8000, 9000, 10000,
    20000, 30000, 40000, 50000,
    60000, 70000, 80000, 90000, 100000,
]

# Step 3: 最终 test 集规模（2k）
FINAL_TEST_SIZE = 2000

SEED = 42


# ============================================================
# 数据结构
# ============================================================
class SampleInfo:
    __slots__ = ('mat_path', 'yml_path', 'n_blocks', 'topology',
                 'k_star', 'chiN_star', 'fA')

    def __init__(self, mat_path, yml_path, n_blocks, topology,
                 k_star, chiN_star, fA):
        self.mat_path   = mat_path
        self.yml_path  = yml_path
        self.n_blocks  = n_blocks
        self.topology  = topology
        self.k_star    = k_star
        self.chiN_star = chiN_star
        self.fA        = fA


# ============================================================
# 辅助函数
# ============================================================
def extract_block_count(topology: str) -> int:
    """从 topology 字符串提取嵌段数，例如 'Random_2blocks' → 2"""
    if 'blocks' in topology:
        parts = topology.replace('Random_', '').replace('blocks', '')
        try:
            return int(parts)
        except ValueError:
            return 0
    return 0


def extract_fA(data: dict) -> float:
    """从 yml dict 提取 A 嵌段总体积分数"""
    try:
        blocks = data.get('system', {}).get('components', [{}])[0].get('blocks', [])
        fA = 0.0
        for block in blocks:
            if 'segment' in block and 'A' in str(block['segment']):
                fA += block.get('length', 0.0)
        return fA
    except Exception:
        return 0.0


def process_single_file(yml_path: Path) -> tuple:
    """处理单个 yml 文件，返回 (n_blocks, SampleInfo) 或 None"""
    mat_path = yml_path.with_suffix('.mat')
    if not mat_path.exists():
        return None

    try:
        with open(yml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        system   = data.get('system', {})
        topology = system.get('topology', '')
        n_blocks = extract_block_count(topology)

        if n_blocks == 0:
            return None

        fA       = extract_fA(data)
        k_star   = system.get('k_star', 0.0)
        chiN_star = system.get('χN_star', 0.0)

        return (n_blocks, SampleInfo(
            str(mat_path), str(yml_path), n_blocks,
            topology, k_star, chiN_star, fA
        ))
    except Exception:
        return None


def scan_dataset(source_dir: str, n_workers: int = 16) -> dict:
    """并行扫描数据集，返回 {n_blocks: [SampleInfo, ...]}"""
    source_path = Path(source_dir)
    samples_by_nblocks = defaultdict(list)
    lock = threading.Lock()
    processed = 0

    yml_files = list(source_path.glob("**/*.yml"))
    total_files = len(yml_files)
    print(f"  扫描目录: {source_dir}")
    print(f"  找到 {total_files} 个 YAML 文件")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_file, y): y for y in yml_files}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                n_blocks, sample = result
                with lock:
                    samples_by_nblocks[n_blocks].append(sample)
            processed += 1
            if processed % 10000 == 0 or processed == total_files:
                sys.stdout.write(f"\r    进度: {processed}/{total_files}  ")
                sys.stdout.flush()

    print()
    return dict(samples_by_nblocks)


def stratified_uniform_sample(
    samples_by_nblocks: dict,
    total_target: int,
    seed: int = 42
) -> list:
    """
    均匀分层采样：每个嵌段数分配相同数量的样本。
    若不足，从各嵌段数中均匀补齐。
    """
    random.seed(seed)
    n_blocks_list = sorted(samples_by_nblocks.keys())
    n_categories  = len(n_blocks_list)

    base_per_cat  = total_target // n_categories
    remainder      = total_target %  n_categories

    selected = []
    for i, nb in enumerate(n_blocks_list):
        n_take = base_per_cat + (1 if i < remainder else 0)
        n_take = min(n_take, len(samples_by_nblocks[nb]))
        chosen = random.sample(samples_by_nblocks[nb], n_take)
        selected.extend(chosen)

    # 若严格模式下某些类样本不足，进一步均匀从剩余中补
    if len(selected) < total_target:
        deficit     = total_target - len(selected)
        all_ids     = set(range(len(s)))
        chosen_ids  = {id(s) for s in selected}
        remaining   = [s for s in samples_by_nblocks[nb]
                       for nb in n_blocks_list
                       if id(s) not in chosen_ids]
        if remaining:
            selected.extend(random.sample(remaining, min(deficit, len(remaining))))

    if len(selected) > total_target:
        selected = random.sample(selected, total_target)

    return selected


def stratified_uniform_sample_from_pool(
    pool_by_nblocks: dict,
    total_target: int,
    seed: int = 42
) -> list:
    """从 pool（字典结构）均匀分层采样，每个嵌段数等量。"""
    random.seed(seed)
    n_blocks_list = sorted(pool_by_nblocks.keys())
    n_categories  = len(n_blocks_list)

    base_per_cat = total_target // n_categories
    remainder    = total_target %  n_categories

    selected = []
    for i, nb in enumerate(n_blocks_list):
        n_take = base_per_cat + (1 if i < remainder else 0)
        pool   = pool_by_nblocks[nb]
        n_take = min(n_take, len(pool))
        selected.extend(random.sample(pool, n_take))

    if len(selected) < total_target:
        deficit   = total_target - len(selected)
        chosen_ids = {id(s) for s in selected}
        remaining  = [s for nb in n_blocks_list
                      for s in pool_by_nblocks[nb]
                      if id(s) not in chosen_ids]
        if remaining:
            selected.extend(random.sample(remaining, min(deficit, len(remaining))))

    return selected


def copy_samples(samples: list, dest_dir: str) -> None:
    """将样本列表复制到目标目录（扁平化，不保留子文件夹结构）。"""
    os.makedirs(dest_dir, exist_ok=True)
    for i, sample in enumerate(samples):
        mat_src = sample.mat_path
        yml_src = sample.yml_path

        # 新文件名：{nblocks}_{原文件名}
        orig_name = Path(mat_src).stem
        new_name  = f"{sample.n_blocks}blocks_{orig_name}"
        shutil.copy2(mat_src, os.path.join(dest_dir, f"{new_name}.mat"))
        shutil.copy2(yml_src, os.path.join(dest_dir, f"{new_name}.yml"))

        if (i + 1) % 5000 == 0 or (i + 1) == len(samples):
            print(f"    复制进度: {i + 1}/{len(samples)}")


def count_by_blocks(directory: str) -> dict:
    """统计目录下 .yml 文件，按嵌段数分组。"""
    counts = defaultdict(int)
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.yml'):
                # 文件名格式: {n}blocks_{orig_name}.yml
                parts = f.split('_')
                if parts:
                    block_str = parts[0].replace('blocks', '')
                    try:
                        nb = int(block_str)
                        counts[nb] += 1
                    except ValueError:
                        pass
    return dict(counts)


def print_block_distribution(title: str, samples_or_counts, is_counts=False):
    """打印各嵌段数分布。"""
    if is_counts:
        counts = samples_or_counts
    else:
        counts = defaultdict(int)
        for s in samples_or_counts:
            counts[s.n_blocks] += 1
        counts = dict(counts)

    total = sum(counts.values())
    print(f"\n{title}")
    print(f"  总计: {total}")
    for nb in sorted(counts.keys()):
        print(f"    {nb}-blocks: {counts[nb]}")


# ============================================================
# Step 1: 划分全量数据 → test pool (2w) + train pool (10w)
# ============================================================
def step1_split_pools(samples_by_nblocks: dict, seed: int = 42):
    """将全量数据均分为 test pool (2w) 和 train pool (10w)，同时保存 test pool 目录。"""
    print("\n" + "=" * 60)
    print("Step 1: 划分 Test Pool (2w) & Train Pool (10w)")
    print("=" * 60)

    random.seed(seed)

    test_pool_samples  = []
    train_pool_samples = []

    for nb in sorted(samples_by_nblocks.keys()):
        pool = sorted(samples_by_nblocks[nb], key=lambda sample: sample.yml_path)
        if len(pool) < SAMPLES_PER_BLOCK_TEST_POOL + SAMPLES_PER_BLOCK_TRAIN_POOL:
            raise ValueError(f'Insufficient samples for {nb} blocks')
        random.shuffle(pool)

        test_pool  = pool[:SAMPLES_PER_BLOCK_TEST_POOL]
        train_pool = pool[SAMPLES_PER_BLOCK_TEST_POOL:SAMPLES_PER_BLOCK_TEST_POOL + SAMPLES_PER_BLOCK_TRAIN_POOL]

        test_pool_samples.extend(test_pool)
        train_pool_samples.extend(train_pool)

        print(f"  {nb}-blocks: test_pool={len(test_pool)}, train_pool={len(train_pool)}")

    print(f"\n  汇总: test_pool={len(test_pool_samples)}, train_pool={len(train_pool_samples)}")

    # 同时将 test pool 写入磁盘
    print(f"\n  📁 保存 test pool 到 {TEST_POOL_DIR}")
    copy_samples(test_pool_samples, TEST_POOL_DIR)
    copy_samples(train_pool_samples, TRAIN_POOL_DIR)

    return test_pool_samples, train_pool_samples


# ============================================================
# Step 2: 从 train pool 采样不同规模的 train 集
# ============================================================
def step2_sample_train_sets(train_pool_samples: list, seed: int = 42):
    """从 train pool 输出 1k~100k 各规模训练集，每个嵌段数均匀。"""
    print("\n" + "=" * 60)
    print("Step 2: 从 Train Pool 采样各规模训练集 (1k~100k)")
    print("=" * 60)

    # 按嵌段数分组
    train_pool_by_nb = defaultdict(list)
    for s in train_pool_samples:
        train_pool_by_nb[s.n_blocks].append(s)

    for target_size in TRAIN_TARGET_SIZES:
        dest_dir = os.path.join(BASE_DEST, f"nqc_train_{target_size // 1000}k"
                                if target_size % 1000 == 0 and target_size >= 1000
                                else f"nqc_train_{target_size}")

        # 检查是否已存在足够数量的样本
        if os.path.exists(dest_dir):
            existing = len([f for f in os.listdir(dest_dir) if f.endswith('.yml')])
            if existing >= target_size * 0.98:
                print(f"  [跳过] {os.path.basename(dest_dir)} 已存在 ({existing} 样本)")
                continue

        print(f"\n  ▶ 采样 {target_size:,} → {os.path.basename(dest_dir)}")
        selected = stratified_uniform_sample_from_pool(
            train_pool_by_nb, target_size, seed=seed
        )
        print(f"    实际采样: {len(selected)}")
        copy_samples(selected, dest_dir)
        print(f"    完成: {dest_dir}")

    print("\n  ✅ Step 2 完成")


# ============================================================
# Step 3: 从 test pool 采样最终 2k test 集
# ============================================================
def step3_sample_test_set(test_pool_samples: list, seed: int = 42):
    """从保存的 test pool (2w) 目录中采样最终 2k test 集，每个嵌段数各 500 个。"""
    print("\n" + "=" * 60)
    print("Step 3: 从 Test Pool (2w) 采样最终 Test 集 (2k)")
    print("=" * 60)

    dest_dir = os.path.join(BASE_DEST, "nqc_test_2k")

    if os.path.exists(dest_dir):
        existing = len([f for f in os.listdir(dest_dir) if f.endswith('.yml')])
        if existing >= FINAL_TEST_SIZE * 0.98:
            print(f"  [跳过] nqc_test_2k 已存在 ({existing} 样本)")
            return

    # 从保存到磁盘的 test pool 重新扫描，按嵌段数分组后再采样
    print(f"  📂 重新扫描 test pool 目录: {TEST_POOL_DIR}")
    test_pool_by_nb = defaultdict(list)

    for root, dirs, files in os.walk(TEST_POOL_DIR):
        for fname in files:
            if not fname.endswith('.yml'):
                continue
            # 文件名格式: {nblocks}_{orig_name}.yml
            parts = fname.split('_')
            if not parts:
                continue
            block_str = parts[0].replace('blocks', '')
            try:
                nb = int(block_str)
            except ValueError:
                continue
            yml_path = os.path.join(root, fname)
            mat_path = yml_path.replace('.yml', '.mat')
            try:
                with open(yml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                system   = data.get('system', {})
                topology = system.get('topology', '')
                fA      = extract_fA(data)
                k_star  = system.get('k_star', 0.0)
                chiN_star = system.get('χN_star', 0.0)
                sample = SampleInfo(
                    mat_path, yml_path, nb,
                    topology, k_star, chiN_star, fA
                )
                test_pool_by_nb[nb].append(sample)
            except Exception:
                pass

    for nb in sorted(test_pool_by_nb):
        print(f"    {nb}-blocks: {len(test_pool_by_nb[nb])} 个可用样本")

    selected = stratified_uniform_sample_from_pool(
        test_pool_by_nb, FINAL_TEST_SIZE, seed=seed
    )
    print(f"  实际采样: {len(selected)}")
    copy_samples(selected, dest_dir)
    print(f"  ✅ 完成: {dest_dir}")


# ============================================================
# 主流程
# ============================================================
def main():
    global SOURCE_DIR, BASE_DEST, CACHE_FILE, TRAIN_POOL_DIR, TEST_POOL_DIR, SEED
    global SAMPLES_PER_BLOCK_TRAIN_POOL, SAMPLES_PER_BLOCK_TEST_POOL
    parser = argparse.ArgumentParser(description='Select development and candidate pools from raw paired samples')
    parser.add_argument('--source-dir', default=SOURCE_DIR)
    parser.add_argument('--output-dir', default=BASE_DEST)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-per-block', type=int, default=25000)
    parser.add_argument('--pool-per-block', type=int, default=5000)
    parser.add_argument('--legacy-subsets', action='store_true', help='Also run historical noncanonical training-size sampling')
    args = parser.parse_args()
    SOURCE_DIR, BASE_DEST, SEED = args.source_dir, args.output_dir, args.seed
    SAMPLES_PER_BLOCK_TRAIN_POOL = args.train_per_block
    SAMPLES_PER_BLOCK_TEST_POOL = args.pool_per_block
    if not Path(SOURCE_DIR).is_dir() or min(args.train_per_block, args.pool_per_block) < 1:
        parser.error('Source directory must exist and counts must be positive')
    TRAIN_POOL_DIR = os.path.join(BASE_DEST, 'n_train_100k' if args.train_per_block == 25000 else 'development')
    TEST_POOL_DIR = os.path.join(BASE_DEST, 'n_test_pool')
    if any(Path(p).exists() for p in (TRAIN_POOL_DIR, TEST_POOL_DIR)):
        parser.error('Output pool already exists; choose another output directory')
    Path(BASE_DEST).mkdir(parents=True, exist_ok=True)
    CACHE_FILE = os.path.join(BASE_DEST, '.chiNlimited_samples_cache.pkl')
    # Scan the requested source on each run; do not reuse a cache from another source.

    print("=" * 60)
    print("Unified Stratified Dataset Sampling")
    print("来源: train_nqc_chiNlimit/ → chiNlimited/")
    print("=" * 60)

    # 0. 扫描或加载缓存
    samples_by_nblocks = scan_dataset(SOURCE_DIR, n_workers=16)

    for nb in sorted(samples_by_nblocks.keys()):
        print(f"    {nb}-blocks: {len(samples_by_nblocks[nb])}")

    # Step 1
    test_pool, train_pool = step1_split_pools(samples_by_nblocks, seed=SEED)

    # Step 2
    if args.legacy_subsets:
        step2_sample_train_sets(train_pool, seed=SEED)

    # Step 3
    if args.legacy_subsets:
        step3_sample_test_set(test_pool, seed=SEED)

    # ============================================================
    # 最终报告
    # ============================================================
    print("\n" + "=" * 60)
    print("📋 最终报告：各数据集嵌段数样本量")
    print("=" * 60)

    print("\n▶ Train Pool (10w):")
    counts = defaultdict(int)
    for s in train_pool:
        counts[s.n_blocks] += 1
    for nb in sorted(counts):
        print(f"    {nb}-blocks: {counts[nb]}")
    print(f"    总计: {len(train_pool)}")

    print("\n▶ Test Pool (2w):")
    counts = defaultdict(int)
    for s in test_pool:
        counts[s.n_blocks] += 1
    for nb in sorted(counts):
        print(f"    {nb}-blocks: {counts[nb]}")
    print(f"    总计: {len(test_pool)}")

    for target_size in TRAIN_TARGET_SIZES:
        name = f"nqc_train_{target_size // 1000}k" if target_size % 1000 == 0 and target_size >= 1000 else f"nqc_train_{target_size}"
        d = os.path.join(BASE_DEST, name)
        if os.path.exists(d):
            c = count_by_blocks(d)
            total = sum(c.values())
            print(f"\n▶ {name} ({total:,}):")
            for nb in sorted(c):
                print(f"    {nb}-blocks: {c[nb]}")

    d_test2k = os.path.join(BASE_DEST, "nqc_test_2k")
    if os.path.exists(d_test2k):
        c = count_by_blocks(d_test2k)
        total = sum(c.values())
        print(f"\n▶ nqc_test_2k ({total:,}):")
        for nb in sorted(c):
            print(f"    {nb}-blocks: {c[nb]}")

    print("\n" + "=" * 60)
    print("✅ 全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
