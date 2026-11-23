"""基础动作组合学习（Basic Action Combination Learning）

核心：从不完整的动作组合出发，通过"加动作 / 删节点 / 重解 / 投影 / 短路"，
      把残缺的组合补全、重组，学出更多可行且更短的动作方式。

状态 s：当前动作组合的特征（节点数、长度比、是否碰撞、关节最大增量）
动作 a：① 加绕行动作 ② 删节点 ③ 重解连续IK ④ 投影到直线 ⑤ 短路
奖励 r：长度减少 ×20 + 解决碰撞 +8 − 关节突变罚 − 空动作罚
学习：Q-learning —— 学"什么状态该补哪种动作"
"""
from __future__ import annotations

import numpy as np
from arm_env import fk
import structured_search2 as S2
import structured_search3 as S3
from chain_path import chain_nodes, sample_traj, max_jump

ACTIONS = ["加绕行动作", "删节点", "重解连续IK", "投影到直线", "短路"]


def collide(qs, obstacles):
    for q in qs:
        pts = fk(q)[0]
        for ob in obstacles:
            c, r = ob[:3], ob[3]
            for i in range(len(pts)-1):
                a, b = pts[i], pts[i+1]
                ab = b - a; den = float(ab @ ab) + 1e-12
                t = np.clip(float((c - a) @ ab) / den, 0, 1)
                if np.linalg.norm(a + t*ab - c) < r:
                    return True
    return False


def evaluate(nodes, obstacles):
    qs = sample_traj(nodes, n_per_seg=6)
    P = np.array([fk(q)[1] for q in qs])
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return L, collide(qs, obstacles), max_jump(qs)


def act_add_detour(nodes, obstacles, rng):
    """加动作：给整条路径鼓一个包（远离最近障碍），链式求解保证连续。

    这是"补全不完整动作组合"的核心动作：不解局部，而是整体换个走法。
    """
    p_start, p_end = fk(nodes[0])[1], fk(nodes[-1])[1]
    d = p_end - p_start
    L2 = float(d @ d) + 1e-12
    # 找离整条直线最近的障碍
    best = None
    for ob in obstacles:
        c, r = ob[:3], ob[3]
        t = float(np.clip((c - p_start) @ d / L2, 0, 1))
        foot = p_start + t * d
        dist = float(np.linalg.norm(foot - c)) - r
        if best is None or dist < best[0]:
            n = foot - c
            nn = np.linalg.norm(n)
            n = n / nn if nn > 1e-6 else np.array([0, 0, 1.0])
            best = (dist, n)
    if best is None:
        return None
    n_dir = best[1]
    for amp in (0.10, 0.20, 0.30, 0.45, 0.60, 0.80):
        cand = chain_nodes(p_start, p_end, nodes[0], K=max(6, len(nodes)),
                           detour=(amp, n_dir))
        if cand is None:
            continue
        if not evaluate(cand, obstacles)[1]:
            return cand
    return None


def act_del_node(nodes, obstacles):
    if len(nodes) <= 3: return None
    best = None
    for i in range(1, len(nodes)-1):
        cand = [np.asarray(q) for q in list(nodes[:i]) + list(nodes[i+1:])]
        L, col, jmp = evaluate(cand, obstacles)
        if best is None or (col, L) < (best[0], best[1]):
            best = (col, L, cand)
    return best[2] if best else None


def act_reik(nodes, obstacles, p0, p1):
    return chain_nodes(p0, p1, nodes[0], K=max(4, len(nodes)))


def act_project(nodes, obstacles, p0, p1):
    d = p1 - p0; L2 = float(d @ d) + 1e-12
    out = [q.copy() for q in nodes]
    moved = 0
    for i in range(1, len(out)-1):
        t = float(np.clip((fk(out[i])[1] - p0) @ d / L2, 0.02, 0.98))
        q, err = S2.ik_solve(p0 + t*d, out[i])
        if err < 0.05:
            out[i] = q; moved += 1
    return out if moved else None


def act_shortcut(nodes, obstacles):
    if len(nodes) <= 3: return None
    best = None
    for i in range(len(nodes)-2):
        for j in range(i+2, min(len(nodes), i+4)):
            cand = [np.asarray(q) for q in list(nodes[:i+1]) + list(nodes[j:])]
            L, col, jmp = evaluate(cand, obstacles)
            if best is None or (col, L) < (best[0], best[1]):
                best = (col, L, cand)
    return best[2] if best else None


def state_key(nodes, L, straight, col, jmp):
    ratio = L / max(straight, 1e-9)
    rb = 0 if ratio < 1.05 else (1 if ratio < 1.5 else 2)
    return (min(len(nodes), 20), rb, int(col), int(jmp > 0.2))


def learn(env, episodes=40, alpha=0.4, gamma=0.9, eps0=0.6, max_steps=6, seed=0):
    rng = np.random.default_rng(seed)
    q_goal, _ = S2.ik_solve(env.target, env.q)
    p0, p1 = fk(env.q)[1], fk(q_goal)[1]
    straight = float(np.linalg.norm(p1 - p0))
    Q, log = {}, []
    for ep in range(episodes):
        eps = max(0.05, eps0 * (1 - ep / episodes))
        # 起点：不完整的组合（直线节点，可能整条撞）
        nodes = chain_nodes(p0, p1, env.q, K=6)
        L, col, jmp = evaluate(nodes, env.obstacles)
        trace = [(L, col)]
        for _ in range(max_steps):
            s = state_key(nodes, L, straight, col, jmp)
            qa = Q.setdefault(s, np.zeros(len(ACTIONS)))
            if rng.random() < eps:
                a = int(rng.integers(0, len(ACTIONS)))
            else:
                a = int(np.argmax(qa)) if np.any(qa) else int(rng.integers(0, len(ACTIONS)))
            if a == 0: new = act_add_detour(nodes, env.obstacles, rng)
            elif a == 1: new = act_del_node(nodes, env.obstacles)
            elif a == 2: new = act_reik(nodes, env.obstacles, p0, p1)
            elif a == 3: new = act_project(nodes, env.obstacles, p0, p1)
            else: new = act_shortcut(nodes, env.obstacles)
            if new is None:
                qa[a] += alpha * (0.0 - qa[a]); continue
            L2, col2, jmp2 = evaluate(new, env.obstacles)
            r = (L - L2) * 20.0
            if col and not col2: r += 8.0                 # 解决碰撞是大功
            if col2 and not col: r -= 3.0
            if jmp2 > 0.2 and jmp <= 0.2: r -= 1.0
            if L2 >= L - 1e-4 and not (col and not col2): r -= 0.2
            s2 = state_key(new, L2, straight, col2, jmp2)
            q2 = Q.setdefault(s2, np.zeros(len(ACTIONS)))
            qa[a] += alpha * (r + gamma * np.max(q2) - qa[a])
            if (col2, L2) < (col, L):
                nodes, L, col, jmp = new, L2, col2, jmp2
            trace.append((L, col))
        log.append(trace)
    return Q, log, straight, (nodes, L, col, jmp)
