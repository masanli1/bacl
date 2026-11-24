"""有标准方法引导的学习：学得比标准更好

标准方法（老师）：束搜索给出的动作序列 —— 能用，但常有冗余动作
学习过程：
  1. 用标准方法在若干起点上示范 → 把 (状态, 动作) 灌进 Q 表（模仿）
  2. 自己跑，边跑边学（强化）
  3. **精简**：成功后试着去掉其中若干动作，还成功就保留
     → 学到的序列比标准更短，这就是"剔除缺点"

评估：留出起点上，标准方法 vs 学到的策略：成功率、平均动作数。
"""
from __future__ import annotations
import json
import numpy as np
import garage_env as GE
import garage_combo as GC

DT, ARC, MAX_STEPS = 0.05, 1.0, 24
GE.GOAL = np.array([14.1, 15.0, np.pi / 2])
GC.GOAL = GE.GOAL
from garage_combo import ComboEnv
from combo_search import beam
from skills import pose_dist

BX, BY, BTH = 0.8, 0.8, np.deg2rad(15)


def key(S):
    return (int(np.round(S[0]/BX)), int(np.round(S[1]/BY)),
            int(np.round(GE.wrap_pi(S[2])/BTH)))


def simulate(start, seq, dt=DT, margin=0.10):
    """只用序列跑一遍，返回 (成功, 碰撞, 动作数)"""
    env = ComboEnv(arc_len=ARC, max_steps=len(seq)+2, dt=dt, seed=0)
    env.reset(start=np.array(start, float)); env.margin = margin
    for a in seq:
        _, r, done, info = env.step(a)
        if info.get("collision"):
            return False, True, len(seq)
        if info.get("success"):
            return True, False, len(seq)
        if done:
            break
    ok = (np.linalg.norm(env.S[:2]-GE.GOAL[:2]) < 0.35 and
          abs(GE.wrap_pi(env.S[2]-GE.GOAL[2])) < np.deg2rad(12))
    return ok, False, len(seq)


def standard_plan(start, width=40, depth=14):
    """标准方法：束搜索"""
    seq, S, ok = beam(start, depth=depth, width=width, max_steps=depth+2)
    return seq if ok else None


def prune_sequence(start, seq):
    """精简：逐个尝试删掉动作，还能成功就删（学到的比标准更短）"""
    cur = list(seq)
    improved = True
    while improved:
        improved = False
        for i in range(len(cur)):
            if len(cur) <= 2:
                break
            trial = cur[:i] + cur[i+1:]
            ok, coll, _ = simulate(start, trial)
            if ok:
                cur = trial
                improved = True
                break
    return cur


def main(n_teacher=12, episodes=1500):
    rng = np.random.default_rng(0)
    # ---------- 标准方法 ----------
    print("① 标准方法（束搜索）在示范起点上工作 …")
    teacher_starts, demos = [], []
    for _ in range(n_teacher):
        st = [float(rng.uniform(4, 12)), float(rng.uniform(6, 9.5)),
              float(rng.uniform(-0.2, 0.2)), 0.0]
        seq = standard_plan(st)
        if seq:
            teacher_starts.append(st); demos.append((st, seq))
    print("   标准方法解出 %d/%d 条" % (len(demos), n_teacher))

    # ---------- 模仿：灌 Q 表 ----------
    Q = {}
    for st, seq in demos:
        env = ComboEnv(arc_len=ARC, max_steps=len(seq)+2, dt=DT, seed=0)
        env.reset(start=np.array(st, float)); env.margin = 0.10
        for a in seq:
            k = key(env.S)
            arr = Q.setdefault(k, np.zeros(ComboEnv.N_ACT))
            arr[a] = max(arr[a], 5.0)
            _, r, done, info = env.step(a)
            if done or info.get("collision"): break
    print("② 模仿后 Q 表覆盖 %d 个状态" % len(Q))

    # ---------- 自学 + 强化 ----------
    print("③ 自学（Q-learning 强化）…")
    env = ComboEnv(arc_len=ARC, max_steps=MAX_STEPS, dt=DT, seed=0)
    for ep in range(episodes):
        eps = max(0.05, 0.4*(1-ep/episodes))
        env.reset()
        for _ in range(MAX_STEPS):
            k = key(env.S); qa = Q.setdefault(k, np.zeros(ComboEnv.N_ACT))
            a = int(rng.integers(0, ComboEnv.N_ACT)) if rng.random() < eps else int(np.argmax(qa))
            d0 = np.linalg.norm(env.S[:2]-GE.GOAL[:2])
            _, r, done, info = env.step(a)
            d1 = np.linalg.norm(env.S[:2]-GE.GOAL[:2])
            rew = (d0-d1)*8.0 - 0.05
            if info.get("collision"): rew -= 10.0
            if info.get("success"): rew += 25.0
            k2 = key(env.S); q2 = Q.setdefault(k2, np.zeros(ComboEnv.N_ACT))
            tgt = rew if (done or info.get("collision")) else rew + 0.97*float(np.max(q2))
            qa[a] += 0.3*(tgt - qa[a])
            if done or info.get("collision"): break

    # ---------- 评估：标准 vs 学到 ----------
    print("④ 留出起点：标准 vs 学到 …")
    starts = [[float(rng.uniform(4, 12)), float(rng.uniform(6, 9.5)),
               float(rng.uniform(-0.2, 0.2)), 0.0] for _ in range(40)]
    std_ok = std_steps = 0; std_n = 0
    lrn_ok = lrn_steps = 0; lrn_n = 0
    for st in starts:
        seq = standard_plan(st)
        std_n += 1
        if seq:
            ok, _, _ = simulate(st, seq)
            if ok:
                std_ok += 1; std_steps += len(seq); std_n and None
        # 学到：跑策略
        env = ComboEnv(arc_len=ARC, max_steps=MAX_STEPS, dt=DT, seed=0)
        env.reset(start=np.array(st, float)); env.margin = 0.10
        acts = []
        for _ in range(MAX_STEPS):
            a = int(np.argmax(Q.get(key(env.S), np.zeros(ComboEnv.N_ACT))))
            _, r, done, info = env.step(a); acts.append(a)
            if info.get("collision") or done: break
        ok = (np.linalg.norm(env.S[:2]-GE.GOAL[:2]) < 0.35 and
              abs(GE.wrap_pi(env.S[2]-GE.GOAL[2])) < np.deg2rad(12))
        if ok:
            pruned = prune_sequence(st, acts)
            ok2, _, _ = simulate(st, pruned)
            lrn_ok += 1; lrn_steps += len(pruned) if ok2 else len(acts); lrn_n += 1

    print("\n=== 结果（40 个留出起点）===")
    print("标准方法：成功 %d/%d，平均动作 %s"
          % (std_ok, std_n, "%.1f" % (std_steps/max(std_ok,1))))
    print("学到策略：成功 %d/%d，平均动作 %s"
          % (lrn_ok, lrn_n, "%.1f" % (lrn_steps/max(lrn_ok,1))))
    json.dump({"standard_ok": std_ok, "standard_steps": std_steps/max(std_ok,1),
               "learned_ok": lrn_ok, "learned_steps": lrn_steps/max(lrn_ok,1),
               "n": std_n}, open("guided_stats.json","w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
