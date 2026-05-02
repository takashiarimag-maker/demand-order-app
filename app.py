import math
import joblib
import pandas as pd
import streamlit as st
st.markdown("""
<style>
.main {
    background-color: #F5F7FA;
}
div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.features import load_sales_data


MODEL_PATH = "models/rf_demand_1m.pkl"
SALES_PATH = "data/warehouse_sales_actuals.csv"

VALID_WAREHOUSES = ["東日本物流", "大阪物流", "九州物流"]

WAREHOUSE_ALIASES = {
    "東日本物流": "東日本物流",
    "関東物流": "東日本物流",
    "札幌": "東日本物流",
    "仙台": "東日本物流",
    "東京": "東日本物流",
    "大阪物流": "大阪物流",
    "大阪": "大阪物流",
    "名古屋": "大阪物流",
    "市場開拓": "大阪物流",
    "九州物流": "九州物流",
    "福岡": "九州物流",
    "沖縄": "九州物流",
}


st.set_page_config(page_title="発注支援アプリ", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "main"


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


def read_csv_flexible(file):
    for enc in ["utf-8-sig", "cp932", "utf-8"]:
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSVを読み込めませんでした。")


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
    df = df[df["品名"].str.contains("PB", na=False)].copy()

    return df.groupby(["倉庫名", "品番", "品名"], as_index=False)["現在庫"].sum()


def prepare_incoming_long_df(df):
    df.columns = [str(c).strip() for c in df.columns]

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

    long_df = long_df[
        long_df["倉庫名"].isin(VALID_WAREHOUSES)
        & long_df["入荷日"].notna()
        & (long_df["入荷予定数"] > 0)
    ].copy()

    return long_df


def summarize_incoming_between(incoming_long_df, start_date, end_date):
    if incoming_long_df is None or incoming_long_df.empty:
        return pd.DataFrame(columns=["倉庫名", "品番", "有効発注残数"])

    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    target = incoming_long_df[
        (incoming_long_df["入荷日"] >= start_date)
        & (incoming_long_df["入荷日"] <= end_date)
    ].copy()

    return (
        target.groupby(["倉庫名", "品番"], as_index=False)["入荷予定数"]
        .sum()
        .rename(columns={"入荷予定数": "有効発注残数"})
    )


def prepare_target_items_df(df):
    df.columns = [str(c).strip() for c in df.columns]

    if "品番" not in df.columns:
        raise ValueError("予測対象品番CSVには '品番' 列が必要です。")

    return (
        df["品番"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )


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

        x_pred = make_next_feature_row(
            warehouse=warehouse,
            item=item,
            next_month=next_month,
            lag1=lag1,
            lag2=lag2,
            lag3=lag3,
        )

        pred = max(0, float(model.predict(x_pred)[0]))

        months.append(next_month)
        preds.append(pred)

        lag3, lag2, lag1 = lag2, lag1, pred
        current_date = next_month

    return months, preds


def calc_current_order(preds, current_stock, incoming_stock, coverage_days):
    forecast_monthly = sum(preds) / len(preds)
    forecast_daily = forecast_monthly / 30.0
    required_stock = forecast_daily * coverage_days
    order_qty = max(0, required_stock - current_stock - incoming_stock)

    return forecast_monthly, forecast_daily, required_stock, order_qty


def run_forecast_rows(
    target_df,
    model,
    sales_df,
    coverage_days,
    progress_bar=None,
    progress_text=None,
):
    results = []
    steps = max(1, math.ceil(coverage_days / 30))
    total = len(target_df)

    if total == 0:
        return pd.DataFrame(results)

    for idx, (_, row) in enumerate(target_df.iterrows(), start=1):
        warehouse = row["倉庫名"]
        item = row["品番"]
        item_name = row["品名"]
        current_stock = float(row.get("現在庫", 0))
        incoming_stock = float(row.get("有効発注残数", 0))

        try:
            _, preds = recursive_forecast(
                model=model,
                warehouse=warehouse,
                item=item,
                history_df=sales_df,
                steps=steps,
            )

            forecast_monthly, forecast_daily, required_stock, order_qty = calc_current_order(
                preds=preds,
                current_stock=current_stock,
                incoming_stock=incoming_stock,
                coverage_days=coverage_days,
            )

            result = {
                "倉庫名": warehouse,
                "品番": item,
                "品名": item_name,
                "現在庫": round(current_stock, 2),
                "有効発注残数": round(incoming_stock, 2),
                "予測日販": round(forecast_daily, 2),
                "予測月販": round(forecast_monthly, 2),
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
                "予測日販": None,
                "予測月販": None,
                "必要在庫数量": None,
                "推奨発注数量": None,
                "エラー": str(e),
            })

        if progress_bar is not None:
            progress_bar.progress(idx / total)

        if progress_text is not None:
            percent = int((idx / total) * 100)
            progress_text.write(f"計算進捗：{percent}%")

    if progress_text is not None:
        progress_text.success("計算完了：100%")

    return pd.DataFrame(results)


def run_future_order_rows(
    target_df,
    model,
    sales_df,
    incoming_long_df,
    today_date,
    order_date,
    lead_time_days,
    order_interval_days,
    extra_days,
    progress_bar=None,
    progress_text=None,
):
    results = []

    today = pd.Timestamp(today_date).normalize()
    order_date = pd.Timestamp(order_date).normalize()

    days_until_order = max(0, (order_date - today).days)
    coverage_days = int(lead_time_days + order_interval_days + extra_days)

    total_days_needed = days_until_order + coverage_days
    steps = max(1, math.ceil(total_days_needed / 30))
    total = len(target_df)

    for idx, (_, row) in enumerate(target_df.iterrows(), start=1):
        warehouse = row["倉庫名"]
        item = row["品番"]
        item_name = row["品名"]
        current_stock = float(row.get("現在庫", 0))

        try:
            _, preds = recursive_forecast(
                model=model,
                warehouse=warehouse,
                item=item,
                history_df=sales_df,
                steps=steps,
            )

            forecast_monthly = sum(preds) / len(preds)
            forecast_daily = forecast_monthly / 30.0

            demand_until_order = forecast_daily * days_until_order

            incoming_until_order_df = summarize_incoming_between(
                incoming_long_df,
                today,
                order_date,
            )
            incoming_until_order = incoming_until_order_df[
                (incoming_until_order_df["倉庫名"] == warehouse)
                & (incoming_until_order_df["品番"] == item)
            ]["有効発注残数"].sum()

            projected_stock_on_order_date = current_stock + incoming_until_order - demand_until_order

            incoming_after_order_df = summarize_incoming_between(
                incoming_long_df,
                order_date,
                order_date + pd.Timedelta(days=int(lead_time_days)),
            )
            incoming_after_order = incoming_after_order_df[
                (incoming_after_order_df["倉庫名"] == warehouse)
                & (incoming_after_order_df["品番"] == item)
            ]["有効発注残数"].sum()

            required_stock_at_order = forecast_daily * coverage_days

            future_order_qty = max(
                0,
                required_stock_at_order - projected_stock_on_order_date - incoming_after_order
            )

            results.append({
                "倉庫名": warehouse,
                "品番": item,
                "品名": item_name,
                "現在庫": round(current_stock, 2),
                "今日から発注予定日までの予測需要": round(demand_until_order, 2),
                "発注予定日までの入荷予定数": round(incoming_until_order, 2),
                "発注予定日時点の予測在庫": round(projected_stock_on_order_date, 2),
                "発注予定日後LT内入荷予定数": round(incoming_after_order, 2),
                "予測日販": round(forecast_daily, 2),
                "予測月販": round(forecast_monthly, 2),
                "次回必要在庫数量": round(required_stock_at_order, 2),
                "次回発注予定数量": round(future_order_qty, 2),
                "エラー": "",
            })

        except Exception as e:
            results.append({
                "倉庫名": warehouse,
                "品番": item,
                "品名": item_name,
                "現在庫": round(current_stock, 2),
                "今日から発注予定日までの予測需要": None,
                "発注予定日までの入荷予定数": None,
                "発注予定日時点の予測在庫": None,
                "発注予定日後LT内入荷予定数": None,
                "予測日販": None,
                "予測月販": None,
                "次回必要在庫数量": None,
                "次回発注予定数量": None,
                "エラー": str(e),
            })

        if progress_bar is not None:
            progress_bar.progress(idx / total)

        if progress_text is not None:
            percent = int((idx / total) * 100)
            progress_text.write(f"次回数量算出進捗：{percent}%")

    if progress_text is not None:
        progress_text.success("次回数量算出完了：100%")

    return pd.DataFrame(results)


def filter_items(items, item_type):
    items = [str(i) for i in items if pd.notna(i)]
    if item_type == "かぶせ":
        return sorted([i for i in items if len(i) >= 2 and i[-2] == "W"])
    if item_type == "平":
        return sorted([i for i in items if len(i) >= 2 and i[-2] == "S"])
    return sorted(items)


def show_abc_analysis(sales_df):
    st.title("📊 ABC分析")
    st.caption("品番単位で全倉庫の販売実績を合算し、累積構成比80%までをA、90%までをB、残りをCに分類します。")

    # ★ 倉庫を無視して品番単位で集計
    abc = (
        sales_df.groupby("品番", as_index=False)["数量"]
        .sum()
        .sort_values("数量", ascending=False)
    )

    total = abc["数量"].sum()

    # 構成比・累積
    abc["構成比"] = abc["数量"] / total
    abc["累積構成比"] = abc["構成比"].cumsum()

    # ABC判定
    def rank(x):
        if x <= 0.8:
            return "A"
        elif x <= 0.9:
            return "B"
        else:
            return "C"

    abc["ABCランク"] = abc["累積構成比"].apply(rank)

    # 表示
    st.dataframe(abc, use_container_width=True)

    # CSV出力
    csv = abc.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "ABC分析結果CSVをダウンロード",
        csv,
        "abc_analysis.csv",
        "text/csv"
    )

    # 上位表示（見やすさ用）
    st.subheader("上位20品番")
    st.bar_chart(abc.head(20).set_index("品番")["数量"])

    # 戻る
    if st.button("メイン画面に戻る"):
        st.session_state.page = "main"
        st.rerun()


def highlight_order(row):
    qty = row.get("推奨発注数量", 0)
    if pd.notna(qty) and qty > 0:
        return ["background-color: #ffe5e5"] * len(row)
    return [""] * len(row)
# ===== モデル評価用 =====

def make_training_dataset_for_eval(sales_df):
    df = sales_df.copy()
    df["月"] = pd.to_datetime(df["月"])
    df = df.sort_values(["倉庫名", "品番", "月"])

    df["lag1"] = df.groupby(["倉庫名", "品番"])["数量"].shift(1)
    df["lag2"] = df.groupby(["倉庫名", "品番"])["数量"].shift(2)
    df["lag3"] = df.groupby(["倉庫名", "品番"])["数量"].shift(3)
    df["rolling_mean_3"] = df[["lag1", "lag2", "lag3"]].mean(axis=1)
    df["month_num"] = df["月"].dt.month
    df["year"] = df["月"].dt.year

    df = df.dropna()

    X = df[["倉庫名","品番","month_num","year","lag1","lag2","lag3","rolling_mean_3"]]
    y = df["数量"]

    return df, X, y


def evaluate_all_items(model, sales_df, test_months=6):
    df = sales_df.copy()
    df["月"] = pd.to_datetime(df["月"])

    # ★ Aランク品だけに絞る
    abc = (
        df.groupby("品番", as_index=False)["数量"]
        .sum()
        .sort_values("数量", ascending=False)
    )

    total = abc["数量"].sum()
    abc["累積構成比"] = (abc["数量"] / total).cumsum()

    a_items = abc[abc["累積構成比"] <= 0.8]["品番"].astype(str).tolist()
    st.info(f"Aランク評価対象品番数：{len(a_items)}")

    df = df[df["品番"].astype(str).isin(a_items)].copy()
    df = df.sort_values("月")

    split_date = df["月"].max() - pd.DateOffset(months=test_months)

    train_sales_df = df[df["月"] < split_date].copy()
    test_sales_df = df[df["月"] >= split_date].copy()

    _, X_train, y_train = make_training_dataset_for_eval(train_sales_df)

    eval_model = clone(model)
    eval_model.fit(X_train, y_train)

    results = []

    pairs = test_sales_df[["倉庫名", "品番"]].drop_duplicates().values.tolist()

    for warehouse, item in pairs:
        test_target = test_sales_df[
            (test_sales_df["倉庫名"].astype(str) == str(warehouse)) &
            (test_sales_df["品番"].astype(str) == str(item))
        ].sort_values("月")

        if len(test_target) == 0:
            continue

        try:
            _, preds = recursive_forecast(
                eval_model,
                warehouse,
                item,
                train_sales_df,
                len(test_target),
            )

            y_true = test_target["数量"].values
            y_pred = preds[:len(y_true)]

            for m, a, p in zip(test_target["月"], y_true, y_pred):
                results.append({
                    "倉庫名": warehouse,
                    "品番": item,
                    "月": m,
                    "実績": a,
                    "予測": round(p, 2),
                    "誤差": round(a - p, 2),
                    "絶対誤差": round(abs(a - p), 2),
                    "エラー": "",
                })

        except Exception as e:
            results.append({
                "倉庫名": warehouse,
                "品番": item,
                "月": None,
                "実績": None,
                "予測": None,
                "誤差": None,
                "絶対誤差": None,
                "エラー": str(e),
            })

    df_res = pd.DataFrame(results)
    df_valid = df_res.dropna(subset=["実績", "予測"])

    if df_valid.empty:
        raise ValueError("評価できるAランク品がありませんでした。")

    mae = mean_absolute_error(df_valid["実績"], df_valid["予測"])
    rmse = mean_squared_error(df_valid["実績"], df_valid["予測"], squared=False)

    return df_res, mae, rmse


# =========================
# UI
# =========================

st.title("📦 需要予測ベース発注支援アプリ")
st.caption("過去販売実績・現在庫・発注残をもとに、倉庫別の必要在庫数量と推奨発注数量を算出します。")

model = load_model()
sales_df = load_sales()

st.sidebar.header("⚙️ 算出条件")

lead_time_days = st.sidebar.number_input("納入リードタイム（日）", 1, 365, 60, 1)
order_interval_days = st.sidebar.number_input("発注間隔（日）", 1, 365, 30, 1)
extra_days = st.sidebar.number_input("余剰確保日数（日）", 0, 365, 30, 1)

coverage_days = int(lead_time_days + order_interval_days + extra_days)

st.sidebar.metric("必要在庫対象期間", f"{coverage_days} 日")

st.sidebar.markdown("---")
st.sidebar.header("📅 次回発注予定数量算出")
future_today = st.sidebar.date_input("今日の日付")
future_order_date = st.sidebar.date_input("発注予定日")
future_lead_time = st.sidebar.number_input("次回 納入リードタイム（日）", 1, 365, int(lead_time_days), 1)
future_interval = st.sidebar.number_input("次回 発注間隔（日）", 1, 365, int(order_interval_days), 1)
future_extra = st.sidebar.number_input("次回 余剰確保日数（日）", 0, 365, int(extra_days), 1)

st.sidebar.markdown("---")
if st.sidebar.button("📊 ABC分析"):
    st.session_state.page = "abc"
    st.rerun()

# CSV Upload
st.sidebar.markdown("---")
st.sidebar.header("📁 CSVアップロード")
stock_file = st.sidebar.file_uploader("現在庫CSV", type=["csv"])
incoming_file = st.sidebar.file_uploader("発注残数量CSV", type=["csv"])
target_item_file = st.sidebar.file_uploader("予測対象品番CSV（任意）", type=["csv"])

if st.session_state.page == "abc":
    show_abc_analysis(sales_df)
    st.stop()

if stock_file is None:
    st.info("サイドバーから現在庫CSVをアップロードしてください。")
    st.stop()

try:
    stock_df = prepare_stock_df(read_csv_flexible(stock_file))

    incoming_long_df = pd.DataFrame(columns=["倉庫名", "品番", "入荷日", "入荷予定数"])
    if incoming_file is not None:
        incoming_long_df = prepare_incoming_long_df(read_csv_flexible(incoming_file))

    current_incoming_df = summarize_incoming_between(
        incoming_long_df,
        pd.Timestamp.today().normalize(),
        pd.Timestamp.today().normalize() + pd.Timedelta(days=int(lead_time_days)),
    )

    stock_df = stock_df.merge(
        current_incoming_df,
        on=["倉庫名", "品番"],
        how="left",
    )
    stock_df["有効発注残数"] = stock_df["有効発注残数"].fillna(0)

    if target_item_file is not None:
        target_items = prepare_target_items_df(read_csv_flexible(target_item_file))
        stock_df = stock_df[stock_df["品番"].astype(str).isin(target_items)].copy()

        if stock_df.empty:
            st.error("予測対象品番CSVに一致する品番が現在庫CSVにありません。")
            st.stop()

except Exception as e:
    st.error(f"CSV処理中にエラーが発生しました: {e}")
    st.stop()

if st.sidebar.button("次回数量算出"):
    st.session_state.page = "future"

st.success("CSVを読み込みました。")

if st.session_state.page == "future":
    st.header("📅 次回発注予定数量 算出結果")

    progress_bar = st.sidebar.progress(0)
    progress_text = st.sidebar.empty()

    result_df = run_future_order_rows(
        stock_df,
        model,
        sales_df,
        incoming_long_df,
        future_today,
        future_order_date,
        future_lead_time,
        future_interval,
        future_extra,
        progress_bar,
        progress_text,
    )

    need_df = result_df[
        pd.to_numeric(result_df["次回発注予定数量"], errors="coerce").fillna(0) > 0
    ].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("次回発注必要品番数", f"{len(need_df):,}")
    c2.metric("次回発注予定数量合計", f"{pd.to_numeric(result_df['次回発注予定数量'], errors='coerce').fillna(0).sum():,.0f}")
    c3.metric("予測不可件数", f"{(result_df['エラー'] != '').sum():,}")

    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("次回発注予定数量CSVをダウンロード", csv, "future_order_plan.csv", "text/csv")

    if not need_df.empty:
        st.markdown("### 🏭 倉庫別 次回発注必要CSV")
        for wh in sorted(need_df["倉庫名"].dropna().unique()):
            wh_df = need_df[need_df["倉庫名"] == wh].copy()
            wh_csv = wh_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                f"{wh} の次回発注必要CSVをダウンロード",
                wh_csv,
                f"future_order_required_{wh}.csv",
                "text/csv",
            )

    if st.button("メイン画面に戻る"):
        st.session_state.page = "main"
        st.rerun()

    st.stop()

# Main dashboard
total_items = stock_df["品番"].nunique()
total_stock = stock_df["現在庫"].sum()
total_incoming = stock_df["有効発注残数"].sum()

m1, m2, m3 = st.columns(3)
m1.metric("対象品番数", f"{total_items:,}")
m2.metric("現在庫合計", f"{total_stock:,.0f}")
m3.metric("有効発注残数合計", f"{total_incoming:,.0f}")

with st.expander("取込済みデータを確認"):
    st.dataframe(stock_df.head(50), use_container_width=True)

st.markdown("---")
st.header("② 単品予測")

left, right = st.columns([1, 2])

with left:
    selected_warehouse = st.selectbox("倉庫", sorted(stock_df["倉庫名"].unique().tolist()))
    item_type = st.radio("品番フィルター", ["全件", "かぶせ", "平"], horizontal=True)

    candidate_items = stock_df[stock_df["倉庫名"] == selected_warehouse]["品番"].unique().tolist()
    candidate_items = filter_items(candidate_items, item_type)

    search_word = st.text_input("品番検索", "")
    if search_word:
        candidate_items = [x for x in candidate_items if search_word.upper() in x.upper()]

    selected_item = st.selectbox("品番", candidate_items) if candidate_items else None

with right:
    if selected_item:
        single_df = stock_df[
            (stock_df["倉庫名"] == selected_warehouse)
            & (stock_df["品番"] == selected_item)
        ].copy()

        item_name = single_df["品名"].iloc[0]
        current_stock = single_df["現在庫"].sum()
        incoming_stock = single_df["有効発注残数"].sum()

        k1, k2, k3 = st.columns(3)
        k1.metric("品名", item_name)
        k2.metric("現在庫", f"{current_stock:,.0f}")
        k3.metric("有効発注残数", f"{incoming_stock:,.0f}")

        if st.button("単品予測を実行", type="primary"):
            result_df = run_forecast_rows(
                single_df,
                model,
                sales_df,
                coverage_days,
                st.sidebar.progress(0),
                st.sidebar.empty(),
            )

            row = result_df.iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("予測日販", "-" if pd.isna(row["予測日販"]) else f"{row['予測日販']:,.2f}")
            c2.metric("予測月販", "-" if pd.isna(row["予測月販"]) else f"{row['予測月販']:,.0f}")
            c3.metric("必要在庫数量", "-" if pd.isna(row["必要在庫数量"]) else f"{row['必要在庫数量']:,.0f}")
            c4.metric("推奨発注数量", "-" if pd.isna(row["推奨発注数量"]) else f"{row['推奨発注数量']:,.0f}")

            if pd.notna(row["推奨発注数量"]) and row["推奨発注数量"] > 0:
                st.error("⚠️ 発注が必要です。")
            elif row["エラー"]:
                st.warning(f"予測不可: {row['エラー']}")
            else:
                st.success("✅ 現時点では発注不要です。")

            st.dataframe(result_df, use_container_width=True)

st.markdown("---")
st.header("③ 一括予測")



bulk_type = st.radio("一括予測フィルター", ["全件", "かぶせ", "平"], horizontal=True)
bulk_search = st.text_input("一括 品番検索", "")

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

if bulk_search:
    bulk_df = bulk_df[
        bulk_df["品番"].astype(str).str.contains(bulk_search, case=False, na=False)
    ].copy()

st.write(f"一括予測対象：**{len(bulk_df):,} 行**")

if st.button("一括予測を実行", type="primary"):
    progress_bar = st.sidebar.progress(0)
    progress_text = st.sidebar.empty()

    result_df = run_forecast_rows(
        bulk_df,
        model,
        sales_df,
        coverage_days,
        progress_bar,
        progress_text,
    )

    order_needed = result_df[
        pd.to_numeric(result_df["推奨発注数量"], errors="coerce").fillna(0) > 0
    ].copy()

    r1, r2, r3 = st.columns(3)
    r1.metric("発注必要品番数", f"{len(order_needed):,}")
    r2.metric(
        "推奨発注数量合計",
        f"{pd.to_numeric(result_df['推奨発注数量'], errors='coerce').fillna(0).sum():,.0f}"
    )
    r3.metric("予測不可件数", f"{(result_df['エラー'] != '').sum():,}")

    st.dataframe(result_df, use_container_width=True)

    all_csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "全予測結果CSVをダウンロード",
        all_csv,
        "forecast_order_result_all.csv",
        "text/csv",
    )

    st.markdown("### 🏭 倉庫別 発注必要CSV")

    if order_needed.empty:
        st.info("発注が必要な品番はありません。")
    else:
        for wh in sorted(order_needed["倉庫名"].dropna().unique()):
            wh_df = order_needed[order_needed["倉庫名"] == wh].copy()
            wh_csv = wh_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                f"{wh} の発注必要CSVをダウンロード",
                wh_csv,
                f"order_required_{wh}.csv",
                "text/csv",
            )



# =========================
# ④ モデル評価
# =========================

st.markdown("---")
st.header("④ モデル評価")

st.caption(
    "ABC分析で販売数量の累積構成比80%以内に入るAランク品だけを対象に、直近期間をテストデータとして予測精度を評価します。"
)

eval_months = st.number_input(
    "評価に使う直近月数",
    min_value=1,
    max_value=12,
    value=1,
    step=1,
)

st.info("まずは評価月数を1か月にして動作確認するのがおすすめです。")

if st.button("Aランク品だけモデル評価を実行", type="primary"):
    try:
        progress_area = st.empty()

        with st.spinner("モデル評価中です。全品番を処理しているため時間がかかる場合があります..."):
            progress_area.info("評価処理を開始しました。しばらくお待ちください。")

            df_eval, mae, rmse = evaluate_all_items(
                model=model,
                sales_df=sales_df,
                test_months=int(eval_months),
            )

        progress_area.success("モデル評価が完了しました。")

        e1, e2 = st.columns(2)
        e1.metric("MAE", f"{mae:,.2f}")
        e2.metric("RMSE", f"{rmse:,.2f}")

        st.dataframe(df_eval, use_container_width=True)

        eval_csv = df_eval.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "モデル評価結果CSVをダウンロード",
            eval_csv,
            "model_evaluation_result.csv",
            "text/csv",
        )

    except Exception as e:
        st.error(f"モデル評価中にエラーが発生しました: {e}")