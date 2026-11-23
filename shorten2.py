"""更短的路径求解：节点限定在起点—终点连线附近

上一版的问题：中间节点从整个关节空间随机取，末端位置散得很开，
结果路径长 4.000 m，而起点到终点的直线只有 0.751 m（5.3 倍）。

修法：候选节点必须落在连线附近的"管道"里（默认半径 0.35 m），
      这样 DP 自然给出接近直线的路，再由动作/截肢处理障碍。
"""
from __future__ import annotations

import numpy as np
from arm_env import fk, Q_HOME
import structured_search3 as S3
import structured_search2 as S2
from shorten import dense_len, try_connect


def candidates_near_line(env, ee0, ee1, n=120, max_dev=0.35, seed=0):
    """只保留末端离直线较近、且碰撞自由的关节向量。"""
    rng = np.random.default_rng(seed)
    d = ee1 - ee0
    L2 = float(d @ d) + 1e-12
    out = []
    for _ in range(n * 25):
        q = np.clip(Q_HOME + rng.normal(0, 1.2, 6), -np.pi, np.pi)
        pts, ee = fk(q)
        t = float(np.clip((ee - ee0) @ d / L2, 0.0, 1.0))
        dev = float(np.linalg.norm(ee - (ee0 + t * d)))
        if dev > max_dev:
            continue
        ok = True
        for ob in env.obstacles:
            c, r = ob[:3], ob[3]
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                ab = b - a
                den = float(ab @ ab) + 1e-12
                tt = np.clip(float((c - a) @ ab) / den, 0, 1)
                if np.linalg.norm(a + tt * ab - c) < r:
                    ok = False; break
            if not ok: break
        if ok:
            out.append(q)
        if len(out) >= n:
            break
    return np.array(out, float)


def solve(env, K=6, max_dev=0.35, seed=0, verbose=False):
    ee0 = fk(env.q)[1]
    q_goal, ik = S2.ik_solve(env.target, env.q)
    if ik >= 0.05:
        rng = np.random.default_rng(7)
        for _ in range(8):
            qt, e = S2.ik_solve(env.target, np.clip(Q_HOME + rng.normal(0, 1.1, 6), -np.pi, np.pi))
            if e < ik: q_goal, ik = qt, e
    if ik >= 0.05:
        return None
    ee1 = fk(q_goal)[1]
    cand = candidates_near_line(env, ee0, ee1, seed=seed)
    if len(cand) < 10:
        cand = S2.free_candidates(env, seed=seed)     # 退化：用普通候选

    layers = [np.array([env.q])] + [cand] * (K - 1) + [np.array([q_goal])]
    cost = np.zeros(1); parents = []
    for li in range(1, len(layers)):
        prev, cur = layers[li-1], layers[li]
        cc = np.full(len(cur), np.inf); par = np.full(len(cur), -1)
        ee_p = np.array([fk(q)[1] for q in prev]); ee_c = np.array([fk(q)[1] for q in cur])
        for j in range(len(cur)):
            for i in range(len(prev)):
                if not np.isfinite(cost[i]): continue
                c = cost[i] + float(np.linalg.norm(ee_c[j] - ee_p[i]))
                if not S2.seg_clear(prev[i], cur[j], env.obstacles): c += 50.0
                if c < cc[j]: cc[j] = c; par[j] = i
        if li < len(layers) - 1:
            keep = min(len(cur), max(24, len(cur)*3//4))
            order = np.argsort(cc)[:keep]
            layers[li] = cur[order]; cc = cc[order]; par = par[order]
        parents.append(par); cost = cc
    idx = int(np.argmin(cost)); path=[layers[-1][idx]]
    for li in range(len(layers)-1, 0, -1):
        idx = parents[li-1][idx]
        if idx < 0: break
        path.append(layers[li-1][idx])
    nodes = np.array(path[::-1])

    rng = np.random.default_rng(seed+1)
    segs, added, fails = [], 0, 0
    for i in range(len(nodes)-1):
        ok, seg, a = try_connect(nodes[i], nodes[i+1], env.obstacles, rng)
        added += a
        if not ok: fails += 1
        segs.append(seg)
    L = sum(dense_len(s) for s in segs)
    if verbose:
        print("  管道解: 节点 %d, 长度 %.3f m, 动作 %d, 失败段 %d" % (len(nodes), L, added, fails))
    return {"nodes": nodes, "segs": segs, "length": L, "actions": added, "fails": fails}


if __name__ == "__main__":
    from arm_env import ArmAvoidEnv
    env = ArmAvoidEnv(seed=5); env.reset()
    r = solve(env, verbose=True)
    ee0 = fk(env.q)[1]; ee1 = fk(r["nodes"][-1])[1]
    straight = float(np.linalg.norm(ee1 - ee0))
    print("  直线 %.3f m → 解 %.3f m（%.2f 倍）" % (straight, r["length"], r["length"]/straight))
