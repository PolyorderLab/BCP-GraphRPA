import os
import random
import shutil
import glob

def create_mini_dataset(source_dir, dest_dir, num_samples=100):
    """
    从源目录随机抽取指定数量的成对 (.mat 和 .yml) 样本，复制到目标目录。
    """
    print(f"🔍 正在扫描数据目录: {source_dir}")
    
    # 1. 找到所有的 .mat 文件
    mat_files = glob.glob(os.path.join(source_dir, "*.mat"))
    
    if not mat_files:
        print("❌ 源目录中没有找到任何 .mat 文件，请检查路径！")
        return
    
    total_files = len(mat_files)
    print(f"✅ 共找到 {total_files} 个样本。")
    
    if total_files < num_samples:
        print(f"⚠️ 警告: 源文件数量 ({total_files}) 小于请求的抽取数量 ({num_samples})。将复制所有文件。")
        num_samples = total_files

    # 2. 随机打乱并抽取
    sampled_mat_files = random.sample(mat_files, num_samples)
    
    # 3. 创建目标文件夹
    os.makedirs(dest_dir, exist_ok=True)
    print(f"📁 目标文件夹已准备: {dest_dir}")
    print(f"🚀 开始抽取 {num_samples} 个样本...\n")

    # 4. 开始成对复制
    success_count = 0
    for mat_path in sampled_mat_files:
        base_name = mat_path.replace('.mat', '')
        yml_path = base_name + '.yml'
        
        # 提取纯文件名（如 sample_00012）用于打印和拼接
        file_prefix = os.path.basename(base_name)
        
        # 确保对应的 yml 文件确实存在
        if os.path.exists(yml_path):
            dest_mat = os.path.join(dest_dir, file_prefix + '.mat')
            dest_yml = os.path.join(dest_dir, file_prefix + '.yml')
            
            # 使用 copy2 可以保留文件的原始元数据（如创建时间）
            shutil.copy2(mat_path, dest_mat)
            shutil.copy2(yml_path, dest_yml)
            success_count += 1
        else:
            print(f"⚠️ 找不到 {file_prefix}.mat 对应的 .yml 文件，已跳过。")

    print(f"\n🎉 抽取完成！成功将 {success_count} 对样本复制到: {dest_dir}")

if __name__ == "__main__":
    # ================= 配置路径 =================
    # 源数据文件夹（你刚才生成的 10 万个文件所在的地方）
    SOURCE_DIR = "/home/zxy/project/TAPD/TrainData/data/train_topo"
    
    # 目标测试文件夹（新建的用于小样本测试的文件夹）
    DEST_DIR = "/home/zxy/project/TAPD/TrainData/data/train_topo_100"
    
    # 需要抽取的样本数量
    SAMPLE_SIZE = 100
    # ============================================
    
    create_mini_dataset(SOURCE_DIR, DEST_DIR, SAMPLE_SIZE)