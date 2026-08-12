"""SSQ 双色球预测项目 —— 方法论实证检验 (纯标准库)
目的：用真实历史数据客观检验"机器学习能否预测开奖号码"这一根本假设。
不依赖任何第三方库，结论可复现。
"""
import csv, math, random
from collections import Counter

random.seed(42)
PATH = "ml/data/1.csv"

# ---------- 1. 加载数据 ----------
rows = []
with open(PATH, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        try:
            reds = [int(row[f"Red{i}"]) for i in range(1, 7)]
            blue = int(row["Blue1"])
            rows.append((reds, blue))
        except Exception:
            pass
n = len(rows)
print(f"载入开奖期数: {n}")

# ---------- 2. 红球位置是否只是"排序后的产物" ----------
sorted_ok = all(reds == sorted(reds) for reds, _ in rows)
print(f"[检查] Red1..Red6 恒为升序排列(说明位置无真实含义): {sorted_ok}")

# ---------- 3. 红/蓝球频率均匀性检验 (卡方) ----------
cnt = Counter()
for reds, _ in rows:
    for x in reds:
        cnt[x] += 1
total_red = 6 * n
exp_red = total_red / 33.0
chi2_red = sum((cnt[x] - exp_red) ** 2 / exp_red for x in range(1, 34))
print(f"[卡方] 红球33个号码频率 vs 均匀: chi2={chi2_red:.2f} (df=32, 0.05临界≈46.19)")

bc = Counter(b for _, b in rows)
exp_blue = n / 16.0
chi2_blue = sum((bc[x] - exp_blue) ** 2 / exp_blue for x in range(1, 17))
print(f"[卡方] 蓝球16个号码频率 vs 均匀: chi2={chi2_blue:.2f} (df=15, 0.05临界≈25.0)")

# ---------- 4. 各位置分布(揭示"位置"是被排序造出来的伪变量) ----------
print("[分布] 红球各位置(已排序)均值/标准差/范围:")
for p in range(6):
    vals = [rows[i][0][p] for i in range(n)]
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / n)
    print(f"  pos{p+1}: mean={m:.1f} std={sd:.1f} min={min(vals)} max={max(vals)}")

# ---------- 5. 和值序列的自相关(检验是否存在时间可预测性) ----------
sums = [sum(r) for r, _ in rows]
m = sum(sums) / n
def autocorr(lag):
    num = sum((sums[i] - m) * (sums[i - lag] - m) for i in range(lag, n))
    den = sum((s - m) ** 2 for s in sums)
    return num / den
print(f"[自相关] 和值 lag1={autocorr(1):.4f}  lag5={autocorr(5):.4f}  lag20={autocorr(20):.4f} (≈0表示无时间依赖)")

# ---------- 6. 基线预测: 历史最常出现的6个号 vs 随机6号 ----------
def overlap(a, b):
    return len(set(a) & set(b))

warm = 300
ov_mf, ov_rf = [], []
W = 100
for i in range(warm, n):
    # 全局最频繁
    c = Counter()
    for reds, _ in rows[:i]:
        for x in reds:
            c[x] += 1
    top6 = [x for x, _ in c.most_common(6)]
    ov_mf.append(overlap(top6, rows[i][0]))
    # 近100期最频繁
    c2 = Counter()
    for reds, _ in rows[max(0, i - W):i]:
        for x in reds:
            c2[x] += 1
    top6r = [x for x, _ in c2.most_common(6)]
    ov_rf.append(overlap(top6r, rows[i][0]))

exp_random = 6 * 6 / 33.0  # 随机选6个号与中奖6号的期望重叠(超几何)
print(f"[基线] 全局最频繁6号 平均重叠={sum(ov_mf)/len(ov_mf):.3f}")
print(f"[基线] 近100期最频繁6号 平均重叠={sum(ov_rf)/len(ov_rf):.3f}")
print(f"[理论] 随机6号 期望重叠={exp_random:.3f}  (三者无显著差异 → 历史频率不含预测信号)")

# ---------- 7. 中头奖的理论概率 ----------
C33_6 = math.comb(33, 6)
print(f"[概率] 红球全中=1/{C33_6:,}  加蓝球=1/{C33_6*16:,}")

# ---------- 8. 生成5组"符合历史统计特征"的高似然号码(非中奖预测!) ----------
# 按经验边际分布抽样 + 和值约束，得到"看起来最像历史开奖"的组合
red_weights = [cnt.get(x, 1) for x in range(1, 34)]
blue_weights = [bc.get(x, 1) for x in range(1, 17)]

def sample_red_set():
    for _ in range(1000):
        s = sorted(random.choices(range(1, 34), weights=red_weights, k=6))
        if len(set(s)) == 6 and 80 <= sum(s) <= 130:
            return s
    return sorted(random.sample(range(1, 34), 6))

print("\n[高似然号码·5组] 基于历史边际分布抽样(仅代表'统计上像历史',不预测中奖):")
for k in range(5):
    reds = sample_red_set()
    blue = random.choices(range(1, 17), weights=blue_weights, k=1)[0]
    print(f"  组{k+1}: 红 {reds}  蓝 {blue}  和值={sum(reds)}")
