from src.features import load_sales_data

df = load_sales_data("data/販売統計別売上数量実績_20260330125326.csv")

print(df.head(10))
print(df.columns)
print(df.shape)