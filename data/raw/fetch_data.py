"""
Script tải dữ liệu thô (raw data) Online Retail từ UCI Machine Learning Repository.
Dữ liệu tải về được lưu trực tiếp tại thư mục data/raw/ dưới định dạng CSV.
"""

import sys
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Đảm bảo mã hóa UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thư mục hiện tại và đường dẫn file lưu dữ liệu
CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = CURRENT_DIR / "online_retail_raw.csv"


def fetch_raw_data() -> pd.DataFrame:
    """
    Tải tập dữ liệu Online Retail (id=352) từ UCI ML Repository và lưu vào data/raw/online_retail_raw.csv.
    """
    print("[*] Đang tải dataset Online Retail (id=352) từ UCI Machine Learning Repository...")
    
    # fetch dataset 
    online_retail = fetch_ucirepo(id=352) 

    # metadata & variable information
    print("\n" + "=" * 50)
    print("METADATA:")
    print("=" * 50)
    print(online_retail.metadata) 
    
    print("\n" + "=" * 50)
    print("VARIABLE INFORMATION:")
    print("=" * 50)
    print(online_retail.variables)
    print("=" * 50 + "\n")

    # data (as pandas dataframes) 
    X = online_retail.data.features 
    y = online_retail.data.targets

    # Lấy toàn bộ các cột (bao gồm các cột định danh trong original/ids)
    if hasattr(online_retail.data, "original") and online_retail.data.original is not None:
        df_raw = online_retail.data.original.copy()
    else:
        dfs = []
        if hasattr(online_retail.data, "ids") and online_retail.data.ids is not None:
            dfs.append(online_retail.data.ids)
        if X is not None:
            dfs.append(X)
        if y is not None:
            dfs.append(y)
        df_raw = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    # Lưu trực tiếp vào thư mục data/raw/
    df_raw.to_csv(OUTPUT_FILE, index=False)
    print(f"[+] Đã lưu dữ liệu thô vào: {OUTPUT_FILE}")
    print(f"[+] Kích thước: {df_raw.shape[0]:,} dòng, {df_raw.shape[1]} cột")
    print(f"[+] Danh sách cột: {list(df_raw.columns)}\n")

    return df_raw


if __name__ == "__main__":
    fetch_raw_data()
