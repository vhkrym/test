
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "purchase_orders.csv"
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, encoding="utf-8")

raw_rows = len(df)
duplicate_extra_rows = int(df["po_id"].duplicated().sum())

df = df.drop_duplicates(subset=["po_id"], keep="first").copy()

print("原始记录数:", raw_rows)
print("去重后的记录数:", len(df))
print("去掉的完全重复记录数:", duplicate_extra_rows)

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["supplier_synced_at"] = pd.to_datetime(
    df["supplier_synced_at"], errors="coerce"
)
for col in ["fulfillment_status", "avail_qty_snapshot"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 暂不擅自删除重复编号，先保留原始记录并在报告中披露
data = df[df["fulfillment_status"].isin([0, 1])].copy()
data["rejected"] = (data["fulfillment_status"] == 0).astype(int)
data["sync_age_hours"] = (
    data["created_at"] - data["supplier_synced_at"]
).dt.total_seconds() / 3600

data = data.dropna(
    subset=["created_at", "sync_age_hours",
            "avail_qty_snapshot", "rejected"]
).sort_values("created_at").reset_index(drop=True)

if len(data) < 20:
    raise ValueError("有效订单太少，无法进行时间切分回测")

# 先用较早的 70% 订单选规则，较新的 30% 用于评估
split = int(len(data) * 0.7)
train = data.iloc[:split].copy()
test = data.iloc[split:].copy()

def predict(frame, rule, threshold=None):
    if rule == "sync":
        return frame["sync_age_hours"] > threshold
    if rule == "stock":
        return frame["avail_qty_snapshot"] <= threshold
    if rule == "combined":
        sync_threshold, stock_threshold = threshold
        return (
            (frame["sync_age_hours"] > sync_threshold)
            | (frame["avail_qty_snapshot"] <= stock_threshold)
        )
    raise ValueError(rule)

def evaluate(frame, alert):
    y = frame["rejected"].astype(bool)
    alert = pd.Series(alert, index=frame.index).astype(bool)

    tp = int((alert & y).sum())
    fp = int((alert & ~y).sum())
    fn = int((~alert & y).sum())
    total_alerts = tp + fp
    total_rejections = int(y.sum())

    # 理想化假设：每个命中的拒单均可被人工介入避免
    gross_avoided_loss = 75 * tp
    alert_cost = 10 * total_alerts
    net_benefit = gross_avoided_loss - alert_cost

    return {
        "orders": int(len(frame)),
        "rejections": total_rejections,
        "alerts": total_alerts,
        "true_positive": tp,
        "false_positive": fp,
        "missed_rejections": fn,
        "precision": round(tp / total_alerts, 4)
            if total_alerts else 0,
        "recall": round(tp / total_rejections, 4)
            if total_rejections else 0,
        "gross_avoided_loss_yuan": gross_avoided_loss,
        "alert_cost_yuan": alert_cost,
        "theoretical_net_benefit_yuan": net_benefit,
    }

# 候选阈值仅用于探索，最终阈值由训练集决定
rules = []
for threshold in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    rules.append(("sync", threshold))

for threshold in [5, 8, 10, 15, 20, 25]:
    rules.append(("stock", threshold))

for sync_threshold in [0.1, 0.2, 0.3, 0.4]:
    for stock_threshold in [5, 8, 10, 15]:
        rules.append(("combined", (sync_threshold, stock_threshold)))

candidates = []
for rule, threshold in rules:
    result = evaluate(train, predict(train, rule, threshold))
    candidates.append({
        "rule": rule,
        "threshold": threshold,
        **result,
    })

# 以训练集理论净收益选择规则；同收益时优先较少预警
best = sorted(
    candidates,
    key=lambda x: (
        x["theoretical_net_benefit_yuan"],
        -x["alerts"],
    ),
    reverse=True,
)[0]

test_result = evaluate(
    test, predict(test, best["rule"], best["threshold"])
)

# 测试集基线：不预警、不发生预警成本
baseline = {
    "orders": int(len(test)),
    "rejections": int(test["rejected"].sum()),
    "alerts": 0,
    "true_positive": 0,
    "false_positive": 0,
    "missed_rejections": int(test["rejected"].sum()),
    "precision": 0,
    "recall": 0,
    "gross_avoided_loss_yuan": 0,
    "alert_cost_yuan": 0,
    "theoretical_net_benefit_yuan": 0,
}

result = {
    "method": "按创建时间排序，前70%选规则，后30%评估",
    "warning": (
        "假设每个命中的拒单均能避免；净收益为理论估算。"
        "重复po_id尚未去重，且阈值搜索可能存在过拟合。"
    ),
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "best_rule": best,
    "test_result": test_result,
    "test_baseline_no_alert": baseline,
    "candidate_rules_train": candidates,
}

with open(OUT / "backtest.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=str)

print("训练集订单数:", len(train))
print("测试集订单数:", len(test))
print("\n训练集选择的规则:")
print(json.dumps(best, ensure_ascii=False, indent=2, default=str))
print("\n较新订单上的测试结果:")
print(json.dumps(test_result, ensure_ascii=False, indent=2))
print("\n不预警基线:")
print(json.dumps(baseline, ensure_ascii=False, indent=2))
print("\n结果已保存:", OUT / "backtest.json")
