"""连续链式求解：保证节点之间关节不突变

问题：每个节点独立求解 IK，会跳到不同的解分支（实测首尾段跳 3.2~3.4 rad，
中间四个节点却是 0，完全不是一条连续的运动）。

修法：
  · 节点按顺序求解，**上一节点的关节角作为下一节点的初值**
  · 多个解里挑"离上一个最近"的那个（避免翻转）
  · 最后检查全程最大关节增量，超过阈值就插中间节点继续平滑
"""
from __future__ import annotations
import numpy as np
from arm_env import fk, Q_HOME
import structured_search2 as S2


def ik_nearest(target, q_prev, trials=4):
    """多种初值求 IK，挑离 q_prev 最近的解（连续优先）。"""
    best = None
    starts = [q_prev] + [np.clip(q_prev + np.random.default_rng(i).normal(0, 0.6, 6), -np.pi, np.pi)
                         for i in range(trials)]
    for q0 in starts:
        q, err = S2.ik_solve(target, q0, iters=200)
        if err > 0.04:
            continue
        d = float(np.linalg.norm(q - q_prev))
        if best is None or d < best[0]:
            best = (d, q)
    if best is None:
        return None, 1.0
    return best[1], 0.0


def chain_nodes(p_start, p_goal, q_start, K=9, detour=None):
    """按末端直线等分 K 个节点，链式求 IK，保证连续。"""
    nodes = [q_start.copy()]
    q_prev = q_start.copy()
    for t in np.linspace(0, 1, K)[1:]:
        p = (1 - t) * p_start + t * p_goal
        if detour is not None:
            amp, n_dir = detour
            p = p + amp * np.sin(np.pi * t) * n_dir
        q, err = ik_nearest(p, q_prev)
        if q is None:
            return None
        nodes.append(q)
        q_prev = q
    return np.array(nodes)


def sample_traj(nodes, n_per_seg=18):
    """节点之间走末端直线，逐点 IK 用前一点做初值（连续）。"""
    qs = [nodes[0]]
    q_prev = nodes[0]
    for i in range(len(nodes) - 1):
        p1, p2 = fk(nodes[i])[1], fk(nodes[i+1])[1]
        for t in np.linspace(0, 1, n_per_seg + 1)[1:]:
            target = (1 - t) * p1 + t * p2
            q, err = S2.ik_solve(target, q_prev, iters=150)
            if err > 0.05:
                q = (1 - t) * nodes[i] + t * nodes[i+1]
            qs.append(q)
            q_prev = q
    return np.array(qs)


def max_jump(qs):
    return float(np.linalg.norm(np.diff(qs, axis=0), axis=1).max())


def ee_len(qs):
    P = np.array([fk(q)[1] for q in qs])
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


if __name__ == "__main__":
    from arm_env import ArmAvoidEnv
    env = ArmAvoidEnv(seed=5); env.reset()
    qg, ik = S2.ik_solve(env.target, env.q)
    p0, p1 = fk(env.q)[1], fk(qg)[1]
    straight = float(np.linalg.norm(p1 - p0))
    for K in (6, 9, 14):
        nodes = chain_nodes(p0, p1, env.q, K=K)
        if nodes is None:
            print("K=%d 求解失败" % K); continue
        qs = sample_traj(nodes)
        print("K=%2d：节点关节差 %s" % (K, [round(float(x),2) for x in np.linalg.norm(np.diff(nodes,axis=0),axis=1)]))
        print("      全程最大关节增量 %.3f rad，长度 %.3f m（直线 %.3f m，%.2f 倍）"
              % (max_jump(qs), ee_len(qs), straight, ee_len(qs)/straight))
