# convert_to_excel.py
import sys
from pathlib import Path

import pandas as pd

def main():
    # 项目根目录下的 CSV 文件路径
    csv_path = Path(__file__).resolve().parent / "k_ranking_summary.csv"
    xlsx_path = Path(__file__).resolve().parent / "k_ranking_summary.xlsx"

    if not csv_path.is_file():
        print(f"[ERROR] CSV 文件未找到: {csv_path}")
        sys.exit(1)

    try:
        # 读取 CSV（自动检测分隔符、编码等）
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] 读取 CSV 失败: {e}")
        sys.exit(1)

    try:
        # 写入 Excel，使用 openpyxl 引擎（pandas 默认已包含）
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        print(f"[OK] 已生成 Excel 文件: {xlsx_path}")
    except Exception as e:
        print(f"[ERROR] 写入 Excel 失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
