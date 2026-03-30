import os
import glob
import yaml
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ================= 配置路径 =================
# 你的原始数据大文件夹 (源数据将保持原样，不会被删除)
SOURCE_DIR = "/home/zxy/project/TAPD/TrainData/data/train_topo"
SAVE_DIR = "/home/zxy/project/TAPD/TrainData/data/train_topoclassify"

# 5 种可能的拓扑结构
TOPO_TYPES = ["Linear", "Comb", "Branch", "Star", "Dendrimer"]
# ============================================

def setup_directories():
    """提前创建好所有的子文件夹，避免多线程冲突"""
    for topo in TOPO_TYPES:
        folder_path = os.path.join(SAVE_DIR, topo)
        os.makedirs(folder_path, exist_ok=True)

def process_single_sample(yml_path):
    """
    处理单个样本的核心逻辑：
    1. 读取 yml 获取拓扑结构
    2. 复制 yml 文件
    3. 寻找同名的 mat 文件并复制
    """
    try:
        # 1. 快速读取 YAML 获取拓扑类型
        with open(yml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        topo_type = config['system']['topology']
        
        # 目标路径必须是 SAVE_DIR
        target_dir = os.path.join(SAVE_DIR, topo_type)
        
        # 2. 准备源路径和目标路径
        base_name = os.path.basename(yml_path).replace('.yml', '')
        mat_path = os.path.join(SOURCE_DIR, f"{base_name}.mat")
        
        target_yml = os.path.join(target_dir, f"{base_name}.yml")
        target_mat = os.path.join(target_dir, f"{base_name}.mat")
        
        # 3. 复制文件 (使用 shutil.copy2 完美保留源文件和元数据)
        shutil.copy2(yml_path, target_yml)
        
        if os.path.exists(mat_path):
            shutil.copy2(mat_path, target_mat)
            
        return True
    except Exception as e:
        return f"Error processing {yml_path}: {e}"

def main():
    print(f"🔍 正在扫描目录: {SOURCE_DIR}")
    
    # 获取所有的 .yml 文件
    all_yml_files = glob.glob(os.path.join(SOURCE_DIR, "*.yml"))
    total_files = len(all_yml_files)
    
    if total_files == 0:
        print("❌ 未在根目录下找到任何 .yml 文件！请检查源路径。")
        return
        
    print(f"✅ 共发现 {total_files} 个样本。正在建立子文件夹...")
    setup_directories()
    
    print(f"🚀 开始多线程极速复制 (动用了 32 个线程) ...\n")
    
    success_count = 0
    errors = []
    
    # 使用 executor.map，它会边分配任务边走进度条
    with ThreadPoolExecutor(max_workers=32) as executor:
        for result in tqdm(executor.map(process_single_sample, all_yml_files), total=total_files, desc="Copying files"):
            if result is True:
                success_count += 1
            else:
                errors.append(result)

    # ================= 打印最终报告与各文件夹统计 =================
    print("\n" + "="*40)
    print(f"🎉 复制分拣彻底完成！")
    print(f"🟢 成功处理: {success_count} / {total_files} 个样本。")
    print(f"📁 数据已成功复制至: {SAVE_DIR}\n")
    
    print("📊 各拓扑结构分类统计 (样本数 = yml文件数):")
    for topo in TOPO_TYPES:
        topo_folder = os.path.join(SAVE_DIR, topo)
        # 清点每个子文件夹下的 yml 文件数量
        topo_count = len(glob.glob(os.path.join(topo_folder, "*.yml")))
        print(f"   - {topo.ljust(10)}: {topo_count} 样本")
    
    print("="*40)
    
    if errors:
        print(f"\n⚠️ 出现了 {len(errors)} 个小错误 (下面展示前10个):")
        for err in errors[:10]:
            print(err)

if __name__ == "__main__":
    main()