import os
import glob
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 定义数据目录和输出目录
directory = '/home/zxy/project/TAPD/TrainData/data/train_nnumber_1w'
# 当前脚本所在目录是 TrainData/scr
# ../distribution 会指向 TrainData/distribution
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../distribution')
# 【修复报错】: 自动创建存储图片的文件夹，如果已存在则跳过
os.makedirs(output_dir, exist_ok=True)

# 存储提取出来的数据
data_list = []

print(f"开始扫描目录: {directory} ...")

# 2. 遍历目录下的所有 .yml 文件 (包含 .yaml)
file_pattern = os.path.join(directory, '*.yml')
files = glob.glob(file_pattern)
files.extend(glob.glob(os.path.join(directory, '*.yaml')))

print(f"共找到 {len(files)} 个 YAML 文件，正在解析...")

# 3. 读取并提取数据
for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            
            if content and 'system' in content:
                sys_data = content['system']
                
                topology = sys_data.get('topology')
                k_star = sys_data.get('k_star')
                chiN_star = sys_data.get('χN_star')
                
                if topology is not None and k_star is not None and chiN_star is not None:
                    import math
                    data_list.append({
                        'n_number': topology,
                        'k_star': float(k_star),
                        'χN_star': float(chiN_star),
                        'lg_χN_star': math.log10(float(chiN_star)) if float(chiN_star) > 0 else 0
                    })
    except Exception as e:
        print(f"读取文件出错 {filepath}: {e}")

# 4. 转换为 Pandas DataFrame 以便绘图
df = pd.DataFrame(data_list)

if df.empty:
    print("未能从文件中提取到有效数据，请检查路径和文件格式。")
else:
    print(f"数据解析完成！成功提取了 {len(df)} 条有效数据。正在分别绘制两张小提琴图...")
    
    sns.set_theme(style="whitegrid")
    
    # ---------------------------------------------------------
    # 5. 绘制并保存 k_star 的小提琴图
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    # 【修复警告】: 加入 hue='Topology' 和 legend=False
    sns.violinplot(
        data=df, 
        x='n_number', 
        y='k_star', 
        hue='n_number',  
        legend=False,    
        inner="quartile", 
        palette="Set2"
    )
    
    # 【修复警告】: 在包含 LaTeX 语法的字符串前面加上 r，变为 Raw String
    plt.title(r'Distribution of $k^*$ by Topology', fontsize=14)
    plt.xlabel('Topology', fontsize=12)
    plt.ylabel(r'$k^*$', fontsize=12)
    plt.xticks(rotation=45) 
    plt.tight_layout()
    
    k_star_path = os.path.join(output_dir, 'n_violin_k_star.png')
    plt.savefig(k_star_path, dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"$k^*$ 绘图完成！图片已保存为: {k_star_path}")
    
    # ---------------------------------------------------------
    # 6. 绘制并保存 χN_star 的小提琴图
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    # 【修复警告】: 加入 hue='Topology' 和 legend=False
    sns.violinplot(
        data=df, 
        x='n_number', 
        y='lg_χN_star', 
        hue='n_number',  
        legend=False,    
        inner="quartile", 
        palette="Set3"
    )
    
    # 【修复警告】: 字符串前加 r
    plt.title(r'Distribution of $\log_{10}(\chi N^*)$ by Topology', fontsize=14)
    plt.xlabel('Topology', fontsize=12)
    plt.ylabel(r'$\log_{10}(\chi N^*)$', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    chiN_star_path = os.path.join(output_dir, 'n_violin_lgchiN_star.png')
    plt.savefig(chiN_star_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"$\chi N^*$ 绘图完成！图片已保存为: {chiN_star_path}")

print("所有任务执行完毕！")