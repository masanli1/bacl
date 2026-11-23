"""三段式求解 v2：DP 出初解 → 截肢重连（切段 → 重接 → 直到全通）

第一阶段（DP）：固定起点/终点、固定层数 K，在每层候选里选关节向量，得到一条初解。
第二阶段（截肢）：逐段检查；有碰撞的段落把两端"截断"，在候选集里重新选一个
                  中间节点接上去，直到整条路径无碰撞。这就是"最优路线运算"。
"""
from __future__ import annotations

import time
import numpy as np
from arm_env import ArmAvoidEnv, fk, Q_HOME

N_CAND = 96
K_LAYERS = 6
SEG_SAMPLES = 6
MAX_REPAIR = 20

_FK_CACHE = {}


def _fk(q):
    k = np.round(q, 6).tobytes()
    r = _FK_CACHE.get(k)
    if r is None:
        r = fk(np.round(q, 6))
        _FK_CACHE[k] = r
    return r


def seg_clear(q1, q2, obstacles, samples=SEG_SAMPLES):
    """该段是否无碰撞。"""
    for t in np.linspace(0, 1, samples):
        q = q1 * (1 - t) + q2 * t
        pts, _ = _fk(q)
        for ob in obstacles:
            c, r = ob[:3], ob[3]
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                ab = b - a
                den = float(ab @ ab) + 1e-12
                tt = np.clip(float((c - a) @ ab) / den, 0, 1)
                if np.linalg.norm(a + tt * ab - c) < r:
                    return False
    return True


def ik_solve(target, q0=None, iters=400, damp=0.05):
    q = Q_HOME.copy() if q0 is None else q0.copy()
    for _ in range(iters):
        _, ee = _fk(q)
        err = np.asarray(target) - ee
        if np.linalg.norm(err) < 1e-4:
            break
        J = np.zeros((3, 6)); eps = 1e-4
        for i in range(6):
            dq = q.copy(); dq[i] += eps
            J[:, i] = (_fk(dq)[1] - ee) / eps
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + damp * np.eye(3), err), -np.pi, np.pi)
    return q, float(np.linalg.norm(np.asarray(target) - _fk(q)[1]))


def free_candidates(env, n=N_CAND, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n * 15):
        q = np.clip(Q_HOME + rng.normal(0, 1.3, 6), -np.pi, np.pi)
        pts, _ = _fk(q)
        ok = True
        for ob in env.obstacles:
            c, r = ob[:3], ob[3]
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                ab = b - a; den = float(ab @ ab) + 1e-12
                tt = np.clip(float((c - a) @ ab) / den, 0, 1)
                if np.linalg.norm(a + tt * ab - c) < r:
                    ok = False; break
            if not ok: break
        if ok: out.append(q)
        if len(out) >= n: break
    return np.array(out, float)


def _nearest(tree, q):
    return int(np.argmin([np.linalg.norm(t - q) for t in tree]))


def _path_to_root(tree, par, i):
    out = []
    while i >= 0:
        out.append(tree[i]); i = par[i]
    return out[::-1]


def rrt_connect(q1, q2, obstacles, max_iter=500, step=0.30, seed=0):
    """双向 RRT：把两个关节向量连成无碰撞通路（应付非线性导致的段内碰撞）。"""
    rng = np.random.default_rng(seed)
    A, pA = [q1], [-1]
    B, pB = [q2], [-1]

    def grow(tree, par, target):
        i = _nearest(tree, target)
        d = target - tree[i]
        n = np.linalg.norm(d)
        if n < 1e-9:
            return None
        q_new = tree[i] + d / n * min(step, n)
        if not seg_clear(tree[i], q_new, obstacles):
            return None
        tree.append(q_new); par.append(i)
        return len(tree) - 1

    for _ in range(max_iter):
        q_rand = np.clip(Q_HOME + rng.normal(0, 1.6, 6), -np.pi, np.pi)
        ia = grow(A, pA, q_rand)
        if ia is None:
            continue
        jb = _nearest(B, A[ia])
        if np.linalg.norm(B[jb] - A[ia]) < step and seg_clear(A[ia], B[jb], obstacles):
            pa_ = _path_to_root(A, pA, ia)
            pb_ = _path_to_root(B, pB, jb)
            return np.array(pa_ + pb_[::-1], float)
        q_rand = np.clip(Q_HOME + rng.normal(0, 1.6, 6), -np.pi, np.pi)
        ib = grow(B, pB, q_rand)
        if ib is None:
            continue
        ja = _nearest(A, B[ib])
        if np.linalg.norm(A[ja] - B[ib]) < step and seg_clear(B[ib], A[ja], obstacles):
            pa_ = _path_to_root(A, pA, ja)
            pb_ = _path_to_root(B, pB, ib)
            return np.array(pa_ + pb_[::-1], float)
    return None


def solve(env, K=K_LAYERS, verbose=False):
    t0 = time.time()
    q_start = env.q.copy()
    target = env.target

    # 固定位置：终点用 IK（多点重启）
    q_goal, ik_err = ik_solve(target, q_start)
    if ik_err >= 0.05:
        rng = np.random.default_rng(7)
        for _ in range(8):
            q_try, e = ik_solve(target, np.clip(Q_HOME + rng.normal(0, 1.1, 6), -np.pi, np.pi))
            if e < ik_err: q_goal, ik_err = q_try, e
            if ik_err < 0.01: break
    if ik_err >= 0.05:
        return {"ok": False, "stage": "ik", "ik_err": ik_err,
                "ms": int((time.time() - t0) * 1000), "checks": 0}

    cand = free_candidates(env, seed=int(env.rng.integers(0, 10**6)))
    if len(cand) < 8:
        return {"ok": False, "stage": "cand", "ik_err": ik_err,
                "ms": int((time.time() - t0) * 1000), "checks": 0}

    # ---------------- 第一阶段：分层 DP（带回溯）
    layers = [np.array([q_start])] + [cand] * (K - 1) + [np.array([q_goal])]
    cost = np.zeros(1)
    parents_all = []
    checks = 0
    for li in range(1, len(layers)):
        prev, cur = layers[li - 1], layers[li]
        pc = cost
        cc = np.full(len(cur), np.inf)
        par = np.full(len(cur), -1)
        ee_prev = np.array([_fk(q)[1] for q in prev])
        ee_cur = np.array([_fk(q)[1] for q in cur])
        for j in range(len(cur)):
            for i in range(len(prev)):
                checks += 1
                if not np.isfinite(pc[i]): continue
                c = pc[i] + float(np.linalg.norm(ee_cur[j] - ee_prev[i]))
                if not seg_clear(prev[i], cur[j], env.obstacles):
                    c += 50.0
                if c < cc[j]:
                    cc[j] = c; par[j] = i
        if li < len(layers) - 1:
            keep = min(len(cur), max(24, len(cur) * 3 // 4))
            order = np.argsort(cc)[:keep]
            layers[li] = cur[order]; cc = cc[order]; par = par[order]
        parents_all.append(par)
        cost = cc

    # 回溯出路径
    idx = int(np.argmin(cost))
    path = [layers[-1][idx]]
    for li in range(len(layers) - 1, 0, -1):
        idx = parents_all[li - 1][idx]
        if idx < 0: break
        path.append(layers[li - 1][idx])
    path = np.array(path[::-1])
    stage1_ok = len(path) == len(layers)

    # ---------------- 第二阶段：截肢重连
    repairs = 0
    if stage1_ok:
        while repairs < MAX_REPAIR:
            bad = None
            for i in range(len(path) - 1):
                if not seg_clear(path[i], path[i + 1], env.obstacles):
                    bad = i; break
            if bad is None: break                     # 全通
            # ① 先试双向 RRT（关节空间直接重连，最通用）
            sub = None
            for sd in range(4):
                sub = rrt_connect(path[bad], path[bad + 1], env.obstacles,
                                  seed=int(env.rng.integers(0, 10**6)))
                if sub is not None and len(sub) >= 2:
                    break
            if sub is not None and len(sub) >= 2:
                path = np.concatenate([path[:bad + 1], sub[1:-1], path[bad + 1:]], axis=0)
                repairs += 1
                checks += len(sub)
                continue

            # ② 退化：插入候选节点
            best = None
            for q in cand:
                if not seg_clear(path[bad], q, env.obstacles): continue
                if not seg_clear(q, path[bad + 1], env.obstacles): continue
                c = (np.linalg.norm(_fk(q)[1] - _fk(path[bad])[1])
                     + np.linalg.norm(_fk(path[bad + 1])[1] - _fk(q)[1]))
                checks += 2
                if best is None or c < best[0]:
                    best = (c, q)
            if best is not None:
                path = np.insert(path, bad + 1, best[1], axis=0)
            else:
                # 单点接不通 → 双点重连：取离该段中点最近的候选，两两组合试
                mid = 0.5 * (_fk(path[bad])[1] + _fk(path[bad + 1])[1])
                ee_c = np.array([_fk(q)[1] for q in cand])
                near = np.argsort(np.linalg.norm(ee_c - mid, axis=1))[:18]
                found = None
                for a_i in range(len(near)):
                    qa = cand[near[a_i]]
                    if not seg_clear(path[bad], qa, env.obstacles):
                        continue
                    for b_i in range(a_i + 1, len(near)):
                        qb = cand[near[b_i]]
                        checks += 2
                        if seg_clear(qa, qb, env.obstacles) and seg_clear(qb, path[bad + 1], env.obstacles):
                            found = (qa, qb); break
                    if found: break
                if found:
                    path = np.insert(path, bad + 1, found[0], axis=0)
                    path = np.insert(path, bad + 2, found[1], axis=0)
                else:
                    # 最后退路：关节空间二分，继续尝试
                    path = np.insert(path, bad + 1, 0.5 * (path[bad] + path[bad + 1]), axis=0)
            repairs += 1

    all_clear = all(seg_clear(path[i], path[i + 1], env.obstacles) for i in range(len(path) - 1)) if stage1_ok else False
    ok = bool(stage1_ok and ik_err < 0.05 and all_clear)
    return {"ok": ok, "stage": "done" if ok else ("repair" if stage1_ok else "dp"),
            "ik_err": ik_err, "checks": checks, "repairs": repairs,
            "path_nodes": len(path) if stage1_ok else 0,
            "ms": int((time.time() - t0) * 1000),
            "exhaustive": float(len(cand) ** K)}


def benchmark(episodes=40, seed=5):
    env = ArmAvoidEnv(seed=seed)
    rs = []
    for _ in range(episodes):
        env.reset(); rs.append(solve(env))
    ok = sum(1 for r in rs if r.get("ok"))
    from collections import Counter
    why = Counter(r.get("stage") for r in rs if not r.get("ok"))
    print(f"三段式 v2：成功 {ok}/{episodes} = {ok/episodes:.0%}")
    if why: print("  失败原因:", dict(why))
    print(f"  平均检查 {np.mean([r.get('checks',0) for r in rs]):.0f} 次  "
          f"平均截肢 {np.mean([r.get('repairs',0) for r in rs]):.1f} 轮  "
          f"平均耗时 {np.mean([r.get('ms',0) for r in rs]):.0f} ms  "
          f"穷举规模 {np.mean([r.get('exhaustive',0) for r in rs]):.2e}")
    return ok / episodes


if __name__ == "__main__":
    benchmark()
