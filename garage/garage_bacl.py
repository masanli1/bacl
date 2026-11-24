"""用基础动作组合学习做倒车入库

状态 s：(x, y, theta) 离散化
动作 a：6 个基本动作（前进/后退 × 左打/直行/右打）
奖励 r：靠近车位 ×8 + 入库 +30 − 碰撞 −20 − 每步 0.05
学习：Q-learning
"""
from __future__ import annotations
import numpy as np
from garage_env import GarageEnv, GOAL, wrap_pi

BX, BY, BTH = 0.6, 0.6, np.deg2rad(10)


def s_key(S):
    return (int(np.round(S[0] / BX)), int(np.round(S[1] / BY)),
            int(np.round(wrap_pi(S[2]) / BTH)))


def train(episodes=1500, dt=0.10, arc=1.0, max_steps=26, seed=0,
          start=None, alpha=0.30, gamma=0.97, eps0=0.9):
    env = GarageEnv(arc_len=arc, max_steps=max_steps, dt=dt, seed=seed)
    rng = np.random.default_rng(seed)
    Q = {}
    hist = []
    for ep in range(episodes):
        eps = max(0.05, eps0 * (1 - ep / episodes))
        if start is None:
            obs = env.reset()
        else:
            obs = env.reset(start=np.array(start, float))
        S = env.S.copy()
        ok = False
        for _ in range(max_steps):
            k = s_key(S)
            qa = Q.setdefault(k, np.zeros(6))
            a = int(rng.integers(0, 6)) if rng.random() < eps else int(np.argmax(qa))
            obs, r, done, info = env.step(a)
            S2 = env.S.copy()
            k2 = s_key(S2)
            q2 = Q.setdefault(k2, np.zeros(6))
            target = r if done else r + gamma * np.max(q2)
            qa[a] += alpha * (target - qa[a])
            S = S2
            if info.get("success"):
                ok = True
                break
            if info.get("collision"):
                break
        hist.append(1 if ok else 0)
    return Q, hist


def evaluate(Q, cases=100, dt=0.10, arc=1.0, max_steps=26, seed=1):
    env = GarageEnv(arc_len=arc, max_steps=max_steps, dt=dt, seed=seed)
    ok = 0
    steps = []
    for _ in range(cases):
        env.reset()
        n = 0
        for _ in range(max_steps):
            a = int(np.argmax(Q.get(s_key(env.S), np.zeros(6))))
            _, r, done, info = env.step(a)
            n += 1
            if info.get("success"):
                ok += 1
                break
            if info.get("collision"):
                break
        steps.append(n)
    return ok / cases, float(np.mean(steps))


if __name__ == "__main__":
    print("随机起点训练 12000 轮（弧长 1.0 m，26 步上限）…")
    start = [6.0, 7.5, 0.0, 0.0]
    Q, hist = train(episodes=12000, start=None)
    print("训练后最近 200 轮成功率 %.1f%%" % (100 * np.mean(hist[-200:])))
    acc, st = evaluate(Q, cases=100)
    print("评估（随机起点 100 例）：成功率 %.0f%%，平均 %.1f 个动作" % (acc * 100, st))
    np.save("q_garage.npy", np.array([list(k) + list(v) for k, v in Q.items()]))
