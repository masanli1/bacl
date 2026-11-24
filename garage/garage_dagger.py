"""束搜索示范 → Q-learning 泛化（示范引导的强化学习）

思路：
  1. 用束搜索在若干随机起点上找出成功的动作序列（示范）
  2. 用示范初始化 Q 表（给示范动作一个正的初值）
  3. 再跑 Q-learning，让它从示范出发去泛化到没见过的起点

这正是 Bacl 的思路：先有不完整的组合（一条具体序列），再补全/泛化。
"""
from __future__ import annotations
import numpy as np
from garage_env import GarageEnv, GOAL, wrap_pi
from garage_bacl import s_key, evaluate

ARC, MAX_STEPS, DT = 1.0, 26, 0.10


def h(S):
    return float(np.linalg.norm(S[:2] - GOAL[:2]) + 1.2 * abs(wrap_pi(S[2] - GOAL[2])))


def beam_solve(env, start, depth=14, width=40):
    """从 start 用束搜索找一条成功序列；返回 [(state_key, action), ...] 或 []"""
    env.reset(start=np.array(start, float))
    beams = [(env.S.copy(), [])]
    for _ in range(depth):
        cands = []
        for S, seq in beams:
            for a in range(6):
                env.S = S.copy()
                _, r, done, info = env.step(a)
                if info.get("collision"):
                    continue
                S2 = env.S.copy()
                cands.append((h(S2), S2, seq + [a], bool(info.get("success"))))
                if info.get("success"):
                    return S2, seq + [a]
        if not cands:
            return None, []
        cands.sort(key=lambda x: x[0])
        beams = [(c[1], c[2]) for c in cands[:width]]
    return None, []


def demo_pairs(starts, env, depth=14, width=40):
    pairs, ok = [], 0
    for st in starts:
        S_fin, seq = beam_solve(env, st, depth, width)
        if not seq:
            continue
        ok += 1
        env.reset(start=np.array(st, float))
        for a in seq:
            pairs.append((s_key(env.S), a))
            _, r, done, info = env.step(a)
            if done:
                break
    return pairs, ok


def main():
    env = GarageEnv(arc_len=ARC, max_steps=MAX_STEPS, dt=DT, seed=3)
    rng = np.random.default_rng(0)
    print("① 生成示范（束搜索）…")
    starts = []
    for _ in range(50):
        e = GarageEnv(arc_len=ARC, max_steps=MAX_STEPS, dt=DT, seed=int(rng.integers(0, 10**6)))
        e.reset(); starts.append(e.S.copy())
    pairs, ok = demo_pairs(starts, env)
    print("   示范：%d/%d 个起点找到解，共 %d 个决策" % (ok, len(starts), len(pairs)))

    print("② 用示范初始化 Q 表…")
    Q = {}
    for k, a in pairs:
        arr = Q.setdefault(k, np.zeros(6))
        arr[a] = max(arr[a], 6.0)

    print("③ 示范引导的 Q-learning…")
    from garage_bacl import train
    Q2, hist = train(episodes=18000, start=None, eps0=0.30, alpha=0.25, gamma=0.97,
                     dt=DT, arc=ARC, max_steps=MAX_STEPS)
    # 合并：示范初值 + 训练结果
    for k, v in Q2.items():
        Q[k] = np.maximum(Q.get(k, np.zeros(6)), v)
    print("   训练末 200 轮成功率 %.1f%%" % (100 * np.mean(hist[-200:])))

    print("④ 评估（150 例随机起点）…")
    acc_s, st_s = evaluate(Q, cases=150, dt=DT, arc=ARC, max_steps=MAX_STEPS, seed=1)
    print("   严格 0.35m/12°：成功率 %.0f%%" % (acc_s * 100))

    # 宽松档
    e = GarageEnv(arc_len=ARC, max_steps=MAX_STEPS, dt=DT, seed=1)
    loose = 0
    for _ in range(150):
        e.reset()
        for _ in range(MAX_STEPS):
            a = int(np.argmax(Q.get(s_key(e.S), np.zeros(6))))
            _, r, done, info = e.step(a)
            if info.get("collision") or done:
                break
        if (np.linalg.norm(e.S[:2] - GOAL[:2]) < 0.6 and
                abs(wrap_pi(e.S[2] - GOAL[2])) < np.deg2rad(20)):
            loose += 1
    print("   较宽 0.60m/20°：成功率 %.0f%%" % (loose / 150 * 100))

    np.save("q_garage_dagger.npy",
            np.array([list(k) + list(v) for k, v in Q.items()]))
    print("Q 表已保存 q_garage_dagger.npy")


if __name__ == "__main__":
    main()
