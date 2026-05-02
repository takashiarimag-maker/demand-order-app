import pandas as pd


def load_sales_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    required_cols = ["倉庫名", "品番", "月", "数量"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"必要列がありません: {missing}")

    df["倉庫名"] = df["倉庫名"].astype(str)
    df["品番"] = df["品番"].astype(str)
    df["月"] = pd.to_datetime(df["月"])
    df["数量"] = pd.to_numeric(df["数量"], errors="coerce").fillna(0)

    return df.sort_values(["倉庫名", "品番", "月"]).reset_index(drop=True)


def make_training_data(df):
    df = df.copy()
    df = df.sort_values(["倉庫名", "品番", "月"]).reset_index(drop=True)

    group_cols = ["倉庫名", "品番"]

    # 過去1〜3か月の数量
    df["lag1"] = df.groupby(group_cols)["数量"].shift(1)
    df["lag2"] = df.groupby(group_cols)["数量"].shift(2)
    df["lag3"] = df.groupby(group_cols)["数量"].shift(3)

    # 過去3か月平均
    df["rolling_mean_3"] = (
        df.groupby(group_cols)["数量"]
        .transform(lambda s: s.shift(1).rolling(window=3).mean())
    )

    # 月・年
    df["month_num"] = df["月"].dt.month
    df["year"] = df["月"].dt.year

    # 1か月先の数量を予測対象にする
    df["target_t_plus_1"] = df.groupby(group_cols)["数量"].shift(-1)

    train_df = df.dropna(
        subset=[
            "lag1",
            "lag2",
            "lag3",
            "rolling_mean_3",
            "target_t_plus_1",
        ]
    ).copy()

    return train_df.reset_index(drop=True)