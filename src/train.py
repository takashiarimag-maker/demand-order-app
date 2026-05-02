from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import load_sales_data, make_training_data


def train_and_save_model():
    df = load_sales_data("data/warehouse_sales_actuals.csv")
    train_df = make_training_data(df)

    print("=== 学習用データ確認 ===")
    print(train_df.head())
    print(train_df.columns)
    print(train_df.shape)

    feature_cols = [
        "倉庫名",
        "品番",
        "month_num",
        "year",
        "lag1",
        "lag2",
        "lag3",
        "rolling_mean_3",
    ]
    target_col = "target_t_plus_1"

    latest_month = train_df["月"].max()
    split_month = latest_month - pd.DateOffset(months=6)

    train_part = train_df[train_df["月"] < split_month].copy()
    test_part = train_df[train_df["月"] >= split_month].copy()

    print("=== 分割確認 ===")
    print("train:", train_part.shape)
    print("test :", test_part.shape)
    print("split_month:", split_month)

    X_train = train_part[feature_cols]
    y_train = train_part[target_col]
    X_test = test_part[feature_cols]
    y_test = test_part[target_col]

    categorical_cols = ["倉庫名", "品番"]
    numeric_cols = ["month_num", "year", "lag1", "lag2", "lag3", "rolling_mean_3"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    print("=== 評価結果 ===")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    Path("models").mkdir(exist_ok=True)
    model_path = "models/rf_demand_1m.pkl"
    joblib.dump(pipeline, model_path)
    print(f"モデル保存完了: {model_path}")


if __name__ == "__main__":
    train_and_save_model()