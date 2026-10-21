
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent
df = pd.read_csv(base / "purchase_orders.csv")

dup = df[df.duplicated("po_id", keep=False)].copy()

print("原始记录数:", len(df))
print("唯一采购单编号数:", df["po_id"].nunique())
print("重复编号涉及的记录数:", len(dup))
print("重复编号涉及的不同采购单数:", dup["po_id"].nunique())

if not dup.empty:
    print("\n重复记录明细:")
    print(
        dup.sort_values("po_id").to_string(index=False)
    )

    # 比较相同 po_id 的记录内容是否完全一致
    compare_cols = [c for c in df.columns if c != "po_id"]
    conflicts = (
        dup.groupby("po_id")[compare_cols]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    print("\n存在字段冲突的重复编号数:", int(conflicts.sum()))

    print("\n冲突编号:")
    print(conflicts[conflicts].index.tolist())
