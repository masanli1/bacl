"""模式性冗余的学习：把"哪些动作可以删"沉淀成规则

流程：
  1. 标准方法给出序列（能到，但可能有冗余）
  2. 逐动作试删，记录 每个动作"可删/不可删"以及它的特征
  3. 从这些记录里学出规则：什么样的动作大概率可删
  4. 用规则直接删（一次验证），而不是逐个试删
     → 比"逐个试"快得多，也比标准方法更短
"""
from __future__ import annotations
import json
import numpy as np
import garage_env as GE
import garage_combo as GC

DT, ARC = 0.05, 1.0
GE.GOAL = np.array([14.1, 15.0, np.pi / 2])
GC.GOAL = GE.GOAL
from garage_combo import ComboEnv
from combo_search import beam
from guided_learn import simulate


def feat(seq, i, dist_at):
    """动作 i 的特征（离散化，便于统计）"""
    d, s, L = ComboEnv.decode(seq[i])
    return (
        "fwd" if d > 0 else "rev",
        "L" if s > 0 else ("0" if s == 0 else "R"),
        "short" if L <= 0.6 else ("mid" if L <= 1.5 else "long"),
        "last" if i == len(seq) - 1 else ("tail" if i >= len(seq) - 3 else "body"),
        "near" if dist_at[i] < 2.0 else "far",
    )


def collect(seqs_starts, rng):
    """逐动作试删，返回 [(特征, 可删?)]"""
    data = []
    for st, seq in seqs_starts:
        env = ComboEnv(arc_len=ARC, max_steps=len(seq) + 2, dt=DT, seed=0)
        env.reset(start=np.array(st, float)); env.margin = 0.10
        dist_at = []
        for a in seq:
            dist_at.append(float(np.linalg.norm(env.S[:2] - GE.GOAL[:2])))
            env.step(a)
        for i in range(len(seq)):
            trial = seq[:i] + seq[i+1:]
            if len(trial) < 2:
                continue
            ok, coll, _ = simulate(st, trial)
            data.append((feat(seq, i, dist_at), bool(ok)))
    return data


NAMES = ("方向", "转向", "长度", "位置", "离目标")


def learn_rules(data, min_n=10, thresh=0.45):
    """按单个特征统计可删率（数据少时比组合特征稳）"""
    cnt = {}
    for f, ok in data:
        for k, val in enumerate(f):
            key = (NAMES[k], val)
            c = cnt.setdefault(key, [0, 0])
            c[0] += 1; c[1] += int(ok)
    rules = {k: (w / n, n) for k, (n, w) in cnt.items() if n >= min_n and w / n >= thresh}
    return rules


def apply_rules(seq, dist_at, rules, need=1):
    """命中 ≥need 条"可删"规则的动作先删掉；删完再验证一次"""
    keep = []
    for i, a in enumerate(seq):
        f = feat(seq, i, dist_at)
        hits = sum(1 for k, val in enumerate(f) if (NAMES[k], val) in rules)
        if hits >= need and len(seq) > 3 and i != 0:
            continue
        keep.append(a)
    return keep if len(keep) >= 2 else list(seq)


def main(n_seq=20, n_test=10):
    rng = np.random.default_rng(0)
    def rand_start():
        return [float(rng.uniform(4, 12)), float(rng.uniform(6, 9.5)),
                float(rng.uniform(-0.2, 0.2)), 0.0]

    print("① 标准方法生成序列 …")
    train, test = [], []
    for _ in range(n_seq):
        st = rand_start(); seq, S, ok = beam(st, depth=14, width=40, max_steps=16)
        if ok: train.append((st, seq))
    for _ in range(n_test):
        st = rand_start(); seq, S, ok = beam(st, depth=14, width=40, max_steps=16)
        if ok: test.append((st, seq))
    print("   训练 %d 条 / 测试 %d 条" % (len(train), len(test)))

    print("② 逐动作试删，收集可删样本 …")
    data = collect(train, rng)
    print("   样本 %d 条，可删 %d 条" % (len(data), sum(1 for _, ok in data if ok)))

    print("③ 学出规则 …")
    rules = learn_rules(data)
    for f, (p, n) in sorted(rules.items(), key=lambda x: -x[1][0])[:8]:
        print("   %s = %-6s → 可删率 %.0f%%（%d 次）" % (f[0], f[1], p * 100, n))

    print("④ 在测试集上：标准 vs 规则删 vs 逐个试删 …")
    res = {"std": [0, 0, 0], "rule": [0, 0, 0], "trial": [0, 0, 0]}   # 成功数, 动作数, 仿真次数
    for st, seq in test:
        ok0, _, _ = simulate(st, seq)
        if ok0:
            res["std"][0] += 1; res["std"][1] += len(seq); res["std"][2] += 1
        env = ComboEnv(arc_len=ARC, max_steps=len(seq) + 2, dt=DT, seed=0)
        env.reset(start=np.array(st, float)); env.margin = 0.10
        dist_at = []
        for a in seq:
            dist_at.append(float(np.linalg.norm(env.S[:2] - GE.GOAL[:2]))); env.step(a)
        cand = apply_rules(seq, dist_at, rules)
        ok1, _, _ = simulate(st, cand)
        res["rule"][2] += 1
        if ok1:
            res["rule"][0] += 1; res["rule"][1] += len(cand)
        else:
            res["rule"][0] += 1; res["rule"][1] += len(seq)
        # 逐个试删（作为对照，统计仿真次数）
        cur = list(seq); trials = 0
        improved = True
        while improved:
            improved = False
            for i in range(1, len(cur)):
                trial = cur[:i] + cur[i+1:]; trials += 1
                ok2, _, _ = simulate(st, trial)
                if ok2:
                    cur = trial; improved = True; break
        res["trial"][2] += trials
        ok3, _, _ = simulate(st, cur)
        if ok3:
            res["trial"][0] += 1; res["trial"][1] += len(cur)

    n = max(len(test), 1)
    print("\n=== 结果（%d 条测试序列）===" % len(test))
    for name, label in (("std", "标准方法"), ("rule", "规则删"), ("trial", "逐个试删")):
        ok, acts, sims = res[name]
        print("%-8s 成功 %d/%d，平均动作 %.1f，验证/试删次数 %d"
              % (label, ok, len(test), acts / max(ok, 1), sims))
    json.dump({"rules": {f"{k[0]}={k[1]}": [round(v[0],2), v[1]] for k, v in rules.items()},
               "std": res["std"], "rule": res["rule"], "trial": res["trial"]},
              open("pattern_stats.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
