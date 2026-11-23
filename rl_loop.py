"""把"不断截肢求最优"显式写成强化学习

状态 s：当前解的特征（节点数、长度比、是否碰撞、关节最大增量）
动作 a：删节点 / 投影到直线 / 重解连续 IK / 短路 / 停止
奖励 r：长度减少量 − 碰撞罚 − 关节突变罚
学习：Q-learning 学"什么状态该用哪种修复动作"

与 PPO 的区别：这里学的不是"怎么走"，而是"怎么改自己的解"——
每一步动作都把状态（当前解）变成一个更好的新状态。
"""
from __future__ import annotations

import json
import numpy as np
from arm_env import ArmAvoidEnv, fk
import structured_search2 as S2
from chain_path import chain_nodes, sample_traj, max_jump
from ee_path import ee_line_segment, ee_length

ACTIONS = ["删节点", "投影到直线", "重解连续IK", "短路", "停止"]


def state_key(nodes, length, straight, collide, jump):
    nb = min(len(nodes), 12)
    ratio = length / max(straight, 1e-9)
    rb = 0 if ratio < 1.05 else (1 if ratio < 1.5 else 2)
    cb = 1 if collide else 0
    jb = 1 if jump > 0.2 else 0
    return (nb, rb, cb, jb)


def eval_solution(nodes, obstacles):
    """长度、是否碰撞、最大关节增量"""
    qs = sample_traj(nodes, n_per_seg=6)
    P = np.array([fk(q)[1] for q in qs])
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    collide = False
    for q in qs:
        pts = fk(q)[0]
        for ob in obstacles:
            c, r = ob[:3], ob[3]
            for i in range(len(pts)-1):
                a, b = pts[i], pts[i+1]
                ab = b - a; den = float(ab @ ab) + 1e-12
                t = np.clip(float((c - a) @ ab) / den, 0, 1)
                if np.linalg.norm(a + t*ab - c) < r:
                    collide = True; break
            if collide: break
        if collide: break
    return L, collide, max_jump(qs), qs


def apply_action(a, nodes, obstacles, straight, p0, p1):
    """执行一个修复动作，返回新节点（或 None 表示无效）。"""
    nodes = [np.asarray(q, float).copy() for q in nodes]
    if a == 0:                                    # 删节点
        if len(nodes) <= 3: return None
        best = None
        for i in range(1, len(nodes)-1):
            t = nodes[:i] + nodes[i+1:]
            L, col, jmp, _ = eval_solution(t, obstacles)
            if best is None or L < best[0]:
                best = (L, t)
        return best[1] if best else None
    if a == 1:                                    # 投影到直线
        d = p1 - p0; L2 = float(d @ d) + 1e-12
        out = nodes[:]
        moved = 0
        for i in range(1, len(nodes)-1):
            t = float(np.clip((fk(out[i])[1] - p0) @ d / L2, 0.02, 0.98))
            q, err = S2.ik_solve(p0 + t*d, out[i])
            if err < 0.04:
                out[i] = q; moved += 1
        return out if moved else None
    if a == 2:                                    # 重解连续 IK
        return chain_nodes(p0, p1, nodes[0], K=max(4, len(nodes)))
    if a == 3:                                    # 短路
        if len(nodes) <= 3: return None
        best = None
        for i in range(len(nodes)-2):
            for j in range(i+2, min(len(nodes), i+4)):   # 只试相邻两段，避免组合爆炸
                cand = nodes[:i+1] + nodes[j:]
                L, col, jmp, _ = eval_solution(cand, obstacles)
                if best is None or L < best[0]:
                    best = (L, cand)
        return best[1] if best else None
    return None


def train(env, episodes=30, alpha=0.35, gamma=0.9, eps0=0.5, max_steps=6, verbose=True):
    rng = np.random.default_rng(0)
    q_goal, _ = S2.ik_solve(env.target, env.q)
    p0, p1 = fk(env.q)[1], fk(q_goal)[1]
    straight = float(np.linalg.norm(p1 - p0))

    Q = {}
    curves = []
    for ep in range(episodes):
        eps = max(0.05, eps0 * (1 - ep / episodes))
        # 初始状态：粗糙解（节点散布、长度长），让 RL 有改善余地
        import shorten2
        rr = shorten2.solve(env, K=6, max_dev=0.9, seed=ep)
        nodes = rr["nodes"] if rr is not None else chain_nodes(p0, p1, env.q, K=9)
        L, col, jmp, _ = eval_solution(nodes, env.obstacles)
        path_len = [L]; steps_used = 0; reached = False
        for step in range(max_steps):
            s = state_key(nodes, L, straight, col, jmp)
            qs_ = Q.setdefault(s, np.zeros(len(ACTIONS)))
            if rng.random() < eps:
                a = int(rng.integers(0, len(ACTIONS)-1))
            else:
                a = int(np.argmax(qs_[:-1])) if np.any(qs_[:-1]) else int(rng.integers(0, 4))
            new_nodes = apply_action(a, nodes, env.obstacles, straight, p0, p1)
            if new_nodes is None:
                qs_[a] += alpha * (0.0 - qs_[a])          # 无效动作给 0 奖励
                continue
            L2, col2, jmp2, _ = eval_solution(new_nodes, env.obstacles)
            r = (L - L2) * 20.0
            if col2 and not col: r -= 2.0
            if jmp2 > 0.2 and jmp <= 0.2: r -= 1.0
            if L2 >= L - 1e-4: r -= 0.2                   # 没变短给小额惩罚
            s2 = state_key(new_nodes, L2, straight, col2, jmp2)
            q2 = Q.setdefault(s2, np.zeros(len(ACTIONS)))
            qs_[a] += alpha * (r + gamma * np.max(q2[:-1]) - qs_[a])
            if L2 < L - 1e-4:
                nodes, L, col, jmp = new_nodes, L2, col2, jmp2
                steps_used += 1
            if L <= straight * 1.002 and not col:
                reached = True
            path_len.append(L)
        curves.append((path_len, steps_used, reached))
    return Q, curves, straight, (nodes, L, col, jmp)


if __name__ == "__main__":
    env = ArmAvoidEnv(seed=5); env.reset()
    Q, curves, straight, final = train(env, episodes=30)
    nodes, L, col, jmp = final
    print("最优长度 %.3f m（直线 %.3f m，%.2f 倍），碰撞=%s，最大关节增量=%.4f"
          % (L, straight, L/straight, col, jmp))
    print("\n学到的策略（状态 → 最该用的动作）：")
    for s in sorted(Q.keys())[:8]:
        act = ACTIONS[int(np.argmax(Q[s][:-1]))]
        print("  状态(节点%d, 长度档%d, 碰撞%d, 突变%d) → %s（Q=%.2f）"
              % (s[0], s[1], s[2], s[3], act, Q[s].max()))
