from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "warehouse_sales_actuals.csv"


def warehouse_from_filename(filename):
    if "東京" in filename or "札幌" in filename or "仙台" in filename:
        return "東日本物流"
    if "大阪" in filename or "名古屋" in filename or "市場" in filename:
        return "大阪物流"
    if "福岡" in filename or "沖縄" in filename:
        return "九州物流"
    return None


def convert_sales_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    print("===================================")
    print("処理:", path.name)
    print("列:", df.columns.tolist())

    warehouse = warehouse_from_filename(path.name)
    if warehouse is None:
        print(f"スキップ: 倉庫名を判定できません: {path.name}")
        return pd.DataFrame(columns=["倉庫名", "品番", "月", "数量"])

    month_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 6]

    df_long = df.melt(
        id_vars=["品番"],
        value_vars=month_cols,
        var_name="月",
        value_name="数量",
    )

    df_long["倉庫名"] = warehouse
    df_long["品番"] = df_long["品番"].astype(str).str.strip()
    df_long["数量"] = pd.to_numeric(df_long["数量"], errors="coerce").fillna(0)
    df_long["月"] = pd.to_datetime(df_long["月"], format="%Y%m", errors="coerce")

    df_long = df_long.dropna(subset=["月"])

    return df_long[["倉庫名", "品番", "月", "数量"]]


def main():
    all_dfs = []

    for file in DATA_DIR.glob("*.csv"):
        if file.name == "warehouse_sales_actuals.csv":
            continue

        df = convert_sales_csv(file)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("変換できる販売実績CSVがありません。ファイル名を確認してください。")

    merged = pd.concat(all_dfs, ignore_index=True)

    merged = (
        merged.groupby(["倉庫名", "品番", "月"], as_index=False)["数量"]
        .sum()
        .sort_values(["倉庫名", "品番", "月"])
    )

    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("===================================")
    print("完成:", OUTPUT_PATH)
    print(merged.head())
    print("件数:", merged.shape)


if __name__ == "__main__":
    main()