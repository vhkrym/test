from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "purchase_orders.csv"
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, encoding="utf-8")

# 保留原始数据质量指标
raw_rows = len(df)
duplicate_extra_rows = int(df["po_id"].duplicated().sum())
duplicate_ids = int(
    df.loc[df["po_id"].duplicated(keep=False), "po_id"].nunique()
)

# 已核查重复编号的字段内容一致，分析时每个采购单只保留一条
df = df.drop_duplicates(subset=["po_id"], keep="first").copy()

print("原始记录数:", raw_rows)
print("重复编号涉及的额外记录数:", duplicate_extra_rows)
print("重复编号数量:", duplicate_ids)
print("去重后采购单数:", len(df))

required = {
    "po_id", "created_at", "supplier_synced_at",
    "purchasing_dept", "material_category",
    "avail_qty_snapshot", "quote_amount",
    "fulfillment_status",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV 缺少字段: {sorted(missing)}")

# 统一时间和数值类型
for col in ["created_at", "supplier_synced_at"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

for col in ["avail_qty_snapshot", "quote_amount", "fulfillment_status"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["sync_age_hours"] = (
    df["created_at"] - df["supplier_synced_at"]
).dt.total_seconds() / 3600
# 仅对已回传履约结果的订单计算拒单率
known = df[df["fulfillment_status"].isin([0, 1])].copy()
known["rejected"] = known["fulfillment_status"].eq(0).astype(int)



print("=" * 55)
print("1. 数据质量")
print("=" * 55)
print("总订单数:", len(df))
print("原始数据中的重复编号数:", duplicate_ids)
# 查看重复编号具体情况，不直接删除
duplicate_rows = df[df["po_id"].duplicated(keep=False)]

print("\n重复编号涉及的记录数:",
      len(duplicate_rows))
print("重复编号涉及的不同采购单数:",
      duplicate_rows["po_id"].nunique())

if not duplicate_rows.empty:
    print("\n重复编号样例:")
    print(
        duplicate_rows.sort_values("po_id")
        .head(20)
        .to_string(index=False)
    )
print("各字段缺失数:")
print(df.isna().sum().to_string())

print("\n履约状态分布（包含未回传）:")
print(df["fulfillment_status"].value_counts(dropna=False).sort_index())

print("\n" + "=" * 55)
print("2. 总体业务指标")
print("=" * 55)
rejected = int(known["rejected"].sum())
accepted = int((known["rejected"] == 0).sum())
unknown = int(df["fulfillment_status"].isna().sum())

print(f"总订单数: {len(df)}")
print(f"接单数: {accepted}")
print(f"拒单数: {rejected}")
print(f"未回传数: {unknown}")
print(f"已知结果订单拒单率: {rejected / len(known):.2%}"
      if len(known) else "已知结果订单拒单率: 无可用数据")

def grouped_rate(data, column):
    temp = data.groupby(column, dropna=False, observed=False).agg(
        orders=("rejected", "size"),
        rejections=("rejected", "sum"),
        rejection_rate=("rejected", "mean"),
    )
    return temp.sort_values(
        ["rejection_rate", "orders"], ascending=[False, False]
    )

# 按品类、部门查看拒单率
for col in ["material_category", "purchasing_dept"]:
    print(f"\n按 {col} 分组的拒单率:")
    print(grouped_rate(known, col).to_string())

# 库存分组：分位数用于探索数据，不代表最终预警阈值
stock = known["avail_qty_snapshot"]
if stock.notna().sum() and stock.nunique() > 1:
    known["stock_group"] = pd.qcut(
        stock, q=4, duplicates="drop"
    )
    print("\n按库存分位组的拒单率:")
    print(grouped_rate(known, "stock_group").to_string())

# 同步时效分组：保留负值，便于发现时间异常
age = known["sync_age_hours"]
if age.notna().sum() and age.nunique() > 1:
    known["sync_age_group"] = pd.qcut(
        age, q=4, duplicates="drop"
    )
    print("\n按同步时间差分位组的拒单率:")
    print(grouped_rate(known, "sync_age_group").to_string())

# 输出原始数据概览，方便复核
summary = {
    "total_orders": int(len(df)),
    "accepted_orders": accepted,
    "rejected_orders": rejected,
    "unknown_orders": unknown,
    "known_result_orders": int(len(known)),
    "rejection_rate_among_known": (
        float(known["rejected"].mean()) if len(known) else None
    ),
    "raw_rows": raw_rows,
    "duplicate_extra_rows": duplicate_extra_rows,
    "duplicate_po_ids": duplicate_ids,
    "negative_sync_age_orders": int((df["sync_age_hours"] < 0).sum()),
}

with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n已保存总体统计: {OUT / 'summary.json'}")