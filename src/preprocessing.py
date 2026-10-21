"""
Module tiền xử lý dữ liệu (Preprocessing) cho MarketLens.
Hỗ trợ tải dữ liệu thô (raw data) từ UCI ML Repository và các hàm làm sạch dữ liệu.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Đảm bảo mã hóa UTF-8 khi in log trên môi trường Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Định nghĩa đường dẫn mặc định
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "online_retail_raw.csv"


def fetch_raw_data(
    dataset_id: int = 352,
    save_path: Optional[str | Path] = None,
    print_info: bool = True
) -> pd.DataFrame:
    """
    Tải tập dữ liệu Online Retail từ UCI Machine Learning Repository và lưu vào thư mục data/raw/.

    Parameters:
        dataset_id (int): ID của dataset trên UCI (mặc định 352: Online Retail).
        save_path (Optional[str | Path]): Đường dẫn file CSV lưu raw data. Mặc định là 'data/raw/online_retail_raw.csv'.
        print_info (bool): In thông tin metadata và biến ra console.

    Returns:
        pd.DataFrame: DataFrame chứa toàn bộ dữ liệu thô.
    """
    print(f"[*] Đang tải dataset (id={dataset_id}) từ UCI Machine Learning Repository...")
    
    # fetch dataset 
    online_retail = fetch_ucirepo(id=dataset_id) 

    # metadata & variable information
    if print_info:
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

    # Lấy toàn bộ các cột (bao gồm IDs như InvoiceNo, StockCode nếu có)
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

    # Lưu dữ liệu thô
    target_path = Path(save_path) if save_path else RAW_DATA_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_raw.to_csv(target_path, index=False)
    print(f"[+] Đã lưu dữ liệu thô vào: {target_path}")
    print(f"[+] Kích thước dữ liệu: {df_raw.shape[0]:,} dòng, {df_raw.shape[1]} cột")
    print(f"[+] Danh sách cột: {list(df_raw.columns)}\n")

    return df_raw


def load_raw_data(file_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Đọc dữ liệu thô đã được tải về từ đĩa. Nếu chưa có, sẽ tự động gọi fetch_raw_data().

    Parameters:
        file_path (Optional[str | Path]): Đường dẫn tới file csv raw data.

    Returns:
        pd.DataFrame: DataFrame dữ liệu thô.
    """
    target_path = Path(file_path) if file_path else RAW_DATA_FILE
    if not target_path.exists():
        print(f"[!] File {target_path} chưa tồn tại. Đang tiến hành tải mới...")
        return fetch_raw_data(save_path=target_path)
    
    print(f"[*] Đang đọc raw data từ: {target_path}")
    return pd.read_csv(target_path)


if __name__ == "__main__":
    # Chạy trực tiếp để tải dữ liệu raw về data/raw/
    fetch_raw_data()
