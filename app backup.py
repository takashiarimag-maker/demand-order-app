import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import load_sales_data


MODEL_PATH = "models/rf_demand_1m.pkl"
SALES_PATH = "data/warehouse_sales_actuals.csv"
DEFAULT_INCOMING_PATH = "data/発注残数量整形_20260428.csv"

VALID_WAREHOUSES = ["東日本物流", "大阪物流", "九州物流"]

WAREHOUSE_ALIASES = {
    "東日本物流": "東日本物流",
    "関東物流": "東日本物流",
    "東京": "東日本物流",
    "札幌": "東日本物流",
    "仙台": "東日本物流",
    "大阪物流": "大阪物流",
    "大阪": "大阪物流",
    "名古屋": "大阪物流",
    "市場開拓": "大阪物流",
    "九州物流": "九州物流",
    "福岡": "九州物流",
    "沖縄": "九州物流",
}


def normalize_warehouse_name(name):
    return WAREHOUSE_ALIASES.get(str(name).strip(), str(name).strip())


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_sales():
    df = load_sales_data(SALES_PATH)
    df["倉庫名"] = df["倉庫名"].astype(str)
    df["品番"] = df["品番"].astype(str)
    return df


def read_csv_flexible(file_or_path):
    for enc in ["utf-8-sig", "cp932", "utf-8"]:
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSVを読み込めませんでした。文字コードか形式を確認してください。")


def prepare_stock_df(df):
    df.columns = [str(c).strip() for c in df.columns]

    if "引当可能数" in df.columns:
        df = df.rename(columns={"引当可能数": "現在庫"})

    required = ["倉庫名", "品番", "品名", "現在庫"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"現在庫CSVに必要な列がありません: {missing}")

    df = df[required].copy()
    df["倉庫名"] = df["倉庫名"].map(normalize_warehouse_name)
    df["品番"] = df["品番"].astype(str).str.strip()
    df["品名"] = df["品名"].astype(str)
    df["現在庫"] = pd.to_numeric(df["現在庫"], errors="coerce").fillna(0)

    df = df[df["倉庫名"].isin(VALID_WAREHOUSES)].copy()

    # 品名にPBを含むものだけ対象
    df = df[df["品名"].str.contains("PB", na=False)].copy()

    return df.groupby(["倉庫名", "品番", "品名"], as_index=False)["現在庫"].sum()


def prepare_incoming_df(df, coverage_days):
    df.columns = [str(c).strip() for c in df.columns]

    # 入庫倉庫 → 倉庫名 に変換
    if "倉庫名" not in df.columns and "入庫倉庫" in df.columns:
        df = df.rename(columns={"入庫倉庫": "倉庫名"})

    required = ["倉庫名", "品番"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"発注残CSVに必要な列がありません: {missing}")

    date_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
    if not date_cols:
        raise ValueError("発注残CSVに入荷日列（例：20260420）がありません。")

    df = df.copy()
    df["倉庫名"] = df["倉庫名"].map(normalize_warehouse_name)
    df["品番"] = df["品番"].astype(str).str.strip()

    long_df = df.melt(
        id_vars=["倉庫名", "品番"],
        value_vars=date_cols,
        var_name="入荷日",
        value_name="入荷予定数",
    )

    long_df["入荷日"] = pd.to_datetime(long_df["入荷日"], format="%Y%m%d", errors="coerce")
    long_df["入荷予定数"] = pd.to_numeric(long_df["入荷予定数"], errors="coerce").fillna(0)

    today = pd.Timestamp.today().normalize()
    end_date = today + pd.Timedelta(days=int(coverage_days))

    long_df = long_df[
        (long_df["入荷日"].notna())
        & (long_df["入荷日"] >= today)
        & (long_df["入荷日"] <= end_date)
    ].copy()

    result = (
        long_df.groupby(["倉庫名", "品番"], as_index=False)["入荷予定数"]
        .sum()
        .rename(columns={"入荷予定数": "有効発注残数"})
    )

    return result


def make_next_feature_row(warehouse, item, next_month, lag1, lag2, lag3):
    return pd.DataFrame(
        [{
            "倉庫名": str(warehouse),
            "品番": str(item),
            "month_num": next_month.month,
            "year": next_month.year,
            "lag1": lag1,
            "lag2": lag2,
            "lag3": lag3,
            "rolling_mean_3": (lag1 + lag2 + lag3) / 3.0,
        }]
    )


def recursive_forecast(model, warehouse, item, history_df, steps):
    hist = history_df[
        (history_df["倉庫名"].astype(str) == str(warehouse))
        & (history_df["品番"].astype(str) == str(item))
    ].sort_values("月")

    if len(hist) < 3:
        raise ValueError("過去3か月分の販売実績が足りません。")

    last_date = hist["月"].max()
    last_values = hist["数量"].tail(3).tolist()
    lag3, lag2, lag1 = last_values[0], last_values[1], last_values[2]

    months = []
    preds = []
    current_date = last_date

    for _ in range(steps):
        next_month = current_date + pd.DateOffset(months=1)

        X_pred = make_next_feature_row(
            warehouse=warehouse,
            item=item,
            next_month=next_month,
            lag1=lag1,
            lag2=lag2,
            lag3=lag3,
        )

        pred = max(0, float(model.predict(X_pred)[0]))

        months.append(next_month)
        preds.append(pred)

        lag3, lag2, lag1 = lag2, lag1, pred
        current_date = next_month

    return months, preds


def calc_required_and_order(preds, current_stock, incoming_stock, shortage_rate_pct, coverage_days):
    forecast_monthly_avg = sum(preds) / len(preds)
    forecast_daily = forecast_monthly_avg / 30.0

    base_required = forecast_daily * coverage_days
    required_stock = base_required * (1 + shortage_rate_pct / 100.0)

    available_stock = current_stock + incoming_stock
    order_qty = max(0, required_stock - available_stock)

    return forecast_monthly_avg, forecast_daily, required_stock, available_stock, order_qty


def filter_items(items, item_type):
    items = [str(i) for i in items if pd.notna(i)]

    if item_type == "かぶせ":
        return sorted([i for i in items if len(i) >= 2 and i[-2] == "W"])
    if item_type == "平":
        return sorted([i for i in items if len(i) >= 2 and i[-2] == "S"])
    return sorted(items)


def run_forecast_rows(target_df, model, sales_df, shortage_rate_pct, coverage_days):
    results = []
    steps = max(1, math.ceil(coverage_days / 30))

    for _, row in target_df.iterrows():
        warehouse = row["倉庫名"]
        item = row["品番"]
        item_name = row["品名"]
        current_stock = float(row.get("現在庫", 0))
        incoming_stock = float(row.get("有効発注残数", 0))

        try:
            months, preds = recursive_forecast(
                model=model,
                warehouse=warehouse,
                item=item,
                history_df=sales_df,
                steps=steps,
            )

            forecast_monthly_avg, forecast_daily, required_stock, available_stock, order_qty = calc_required_and_order(
                preds=preds,
                current_stock=current_stock,
                incoming_stock=incoming_stock,
                shortage_rate_pct=shortage_rate_pct,
                coverage_days=coverage_days,
            )

            result = {
                "倉庫名": warehouse,
                "品番": item,
                "品名": item_name,
                "現在庫": round(current_stock, 2),
                "有効発注残数": round(incoming_stock, 2),
                "使用可能在庫": round(available_stock, 2),
                "予測月間需要": round(forecast_monthly_avg, 2),
                "予測日販": round(forecast_daily, 2),
                "必要在庫数量": round(required_stock, 2),
                "推奨発注数量": round(order_qty, 2),
                "エラー": "",
            }

            for i, pred in enumerate(preds, start=1):
                result[f"{i}か月後予測"] = round(pred, 2)

            results.append(result)

        except Exception as e:
            results.append({
                "倉庫名": warehouse,
                "品番": item,
                "品名": item_name,
                "現在庫": round(current_stock, 2),
                "有効発注残数": round(incoming_stock, 2),
                "使用可能在庫": round(current_stock + incoming_stock, 2),
                "予測月間需要": None,
                "予測日販": None,
                "必要在庫数量": None,
                "推奨発注数量": None,
                "エラー": str(e),
            })

    return pd.DataFrame(results)


st.set_page_config(page_title="需要予測ベース発注支援アプリ", layout="wide")
st.title("需要予測ベース発注支援アプリ")

model = load_model()
sales_df = load_sales()

st.sidebar.header("算出条件")

shortage_rate_pct = st.sidebar.number_input(
    "欠品対策率（％）",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
)

lead_time_days = st.sidebar.number_input(
    "納入リードタイム（日）",
    min_value=1,
    max_value=365,
    value=60,
    step=1,
)

order_interval_days = st.sidebar.number_input(
    "発注間隔（日）",
    min_value=1,
    max_value=365,
    value=30,
    step=1,
)

extra_days = st.sidebar.number_input(
    "余剰確保日数（日）",
    min_value=0,
    max_value=365,
    value=30,
    step=1,
)

coverage_days = int(lead_time_days + order_interval_days + extra_days)

st.sidebar.caption(f"必要在庫の対象期間：{coverage_days} 日")

st.header("CSVアップロード")

stock_file = st.file_uploader("現在庫CSVをアップロード", type=["csv"])
incoming_file = st.file_uploader("発注残数量CSVをアップロード（未指定なら data 内の整形済みCSVを使用）", type=["csv"])

if stock_file is None:
    st.info("現在庫CSVをアップロードしてください。")
    st.stop()

try:
    stock_raw = read_csv_flexible(stock_file)
    stock_df = prepare_stock_df(stock_raw)

    incoming_df = None

    if incoming_file is not None:
        incoming_raw = read_csv_flexible(incoming_file)
        incoming_df = prepare_incoming_df(incoming_raw, lead_time_days)
    elif Path(DEFAULT_INCOMING_PATH).exists():
        incoming_raw = read_csv_flexible(DEFAULT_INCOMING_PATH)
        incoming_df = prepare_incoming_df(incoming_raw, lead_time_days)

    if incoming_df is not None:
        stock_df = stock_df.merge(
            incoming_df,
            on=["倉庫名", "品番"],
            how="left",
        )
        stock_df["有効発注残数"] = stock_df["有効発注残数"].fillna(0)
    else:
        stock_df["有効発注残数"] = 0

except Exception as e:
    st.error(f"CSV処理中にエラーが発生しました: {e}")
    st.stop()

st.subheader("取込済み対象データ")
st.dataframe(stock_df.head(20), use_container_width=True)

st.markdown("---")
st.header("単品予測")

col1, col2, col3 = st.columns(3)

with col1:
    warehouses = sorted(stock_df["倉庫名"].unique().tolist())
    selected_warehouse = st.selectbox("倉庫", warehouses)

with col2:
    item_type = st.radio("品番フィルター", ["全件", "かぶせ", "平"], horizontal=True)

items = stock_df[stock_df["倉庫名"] == selected_warehouse]["品番"].unique().tolist()
items = filter_items(items, item_type)

with col3:
    selected_item = st.selectbox("品番", items) if items else None

if selected_item:
    single_df = stock_df[
        (stock_df["倉庫名"] == selected_warehouse)
        & (stock_df["品番"] == selected_item)
    ].copy()

    item_name = single_df["品名"].iloc[0]
    current_stock = single_df["現在庫"].sum()
    incoming_stock = single_df["有効発注残数"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("品名", item_name)
    c2.metric("現在庫", f"{current_stock:.2f}")
    c3.metric("有効発注残数", f"{incoming_stock:.2f}")

    if st.button("単品予測を実行"):
        result_df = run_forecast_rows(
            single_df,
            model,
            sales_df,
            shortage_rate_pct,
            coverage_days,
        )

        st.subheader("単品予測結果")
        st.dataframe(result_df, use_container_width=True)

st.markdown("---")
st.header("一括予測")

bulk_type = st.radio(
    "一括予測の品番フィルター",
    ["全件", "かぶせ", "平"],
    horizontal=True,
    key="bulk_type",
)

bulk_df = stock_df.copy()

if bulk_type == "かぶせ":
    bulk_df = bulk_df[
        bulk_df["品番"].astype(str).str.len().ge(2)
        & (bulk_df["品番"].astype(str).str[-2] == "W")
    ].copy()
elif bulk_type == "平":
    bulk_df = bulk_df[
        bulk_df["品番"].astype(str).str.len().ge(2)
        & (bulk_df["品番"].astype(str).str[-2] == "S")
    ].copy()

if st.button("一括予測を実行"):
    result_df = run_forecast_rows(
        bulk_df,
        model,
        sales_df,
        shortage_rate_pct,
        coverage_days,
    )

    st.subheader("一括予測結果")
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="予測結果CSVをダウンロード",
        data=csv,
        file_name="forecast_order_result.csv",
        mime="text/csv",
    )