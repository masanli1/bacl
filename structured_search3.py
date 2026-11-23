"""三段式求解 v3：节点固定，碰撞靠"加动作"解决

与 v2 的区别（这是关键）：
  v2：碰撞 → 往路径里**插入新节点**（节点数会变，违反"固定数量"）
  v3：碰撞 → 给这一段**增加新动作**（节点数不变，动作数增加）

模型：
  · 固定位置：起点 q₀、终点 IK 解 —— 不搜索
  · 固定数量：节点层数 K（例如 6）—— 永远不变
  · 不固定　：① 每层的关节向量  ② 每段上叠加的动作（高斯脉冲）

  段的轨迹：q(s) = lerp(q₁,q₂,s) + Σ aᵢ · exp(-((s-tᵢ)/w)²) · dᵢ
  动作 = (tᵢ, w, dᵢ 向量)；碰撞时只加/调动作，节点不增不减。
"""
from __future__ import annotations

import time
import numpy as np
from arm_env import ArmAvoidEnv, fk, Q_HOME
import structured_search2 as S2

N_CAND = 96
K_LAYERS = 6
SAMPLES = 12


def penetration(q, obstacles):
    pts = fk(q)[0]
    worst = 0.0
    for ob in obstacles:
        c, r = ob[:3], ob[3]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ab = b - a
            den = float(ab @ ab) + 1e-12
            t = np.clip(float((c - a) @ ab) / den, 0, 1)
            d = float(np.linalg.norm(a + t * ab - c))
            worst = max(worst, r - d)
    return worst


class ActionSegment:
    """一段轨迹 = 直线 + 若干动作（高斯脉冲）"""

    def __init__(self, q1, q2):
        self.q1, self.q2 = q1.copy(), q2.copy()
        self.acts = []          # [(t, w, vec6)]

    def q(self, s):
        v = (1 - s) * self.q1 + s * self.q2
        for (t, w, d) in self.acts:
            v = v + np.exp(-((s - t) / w) ** 2) * d
        return np.clip(v, -np.pi, np.pi)

    def clear(self, obstacles, samples=SAMPLES):
        return all(penetration(self.q(s), obstacles) <= 0.0 for s in np.linspace(0, 1, samples))

    def worst(self, obstacles, samples=SAMPLES):
        return max(penetration(self.q(s), obstacles) for s in np.linspace(0, 1, samples))


def repair_with_actions(seg: ActionSegment, obstacles, rng, tries=60):
    """碰撞时增加动作（而不是插节点），直到该段无碰撞。"""
    added = 0
    for round_ in range(6):
        if seg.clear(obstacles):
            return True, added
        best = None
        # 在现有动作基础上，加一个高斯脉冲动作去绕开
        for _ in range(tries):
            t = rng.uniform(0.25, 0.75)
            w = rng.uniform(0.10, 0.28)
            scale = rng.uniform(0.25, 1.1)
            d = rng.normal(0, 1, 6)
            d = d / (np.linalg.norm(d) + 1e-9) * scale
            seg.acts.append((t, w, d))
            wv = seg.worst(obstacles)
            if best is None or wv < best[0]:
                best = (wv, (t, w, d))
            seg.acts.pop()
        if best is None:
            return seg.clear(obstacles), added
        seg.acts.append(best[1]); added += 1
        if best[0] <= 0.0:
            return True, added
    return seg.clear(obstacles), added


def solve(env, K=K_LAYERS, seed=0, dp_avoid=True):
    t0 = time.time()
    q_start = env.q.copy()
    q_goal, ik = S2.ik_solve(env.target, q_start)
    if ik >= 0.05:
        rng = np.random.default_rng(7)
        for _ in range(8):
            qt, e = S2.ik_solve(env.target, np.clip(Q_HOME + rng.normal(0, 1.1, 6), -np.pi, np.pi))
            if e < ik: q_goal, ik = qt, e
    if ik >= 0.05:
        return {"ok": False, "stage": "ik", "ik_err": ik, "actions": 0,
                "nodes": 0, "ms": int((time.time() - t0) * 1000)}

    cand = S2.free_candidates(env, seed=int(env.rng.integers(0, 10**6)))
    layers = [np.array([q_start])] + [cand] * (K - 1) + [np.array([q_goal])]

    # ---- 第一阶段：DP 选每层关节向量（节点数固定 K）
    cost = np.zeros(1); parents = []; checks = 0
    for li in range(1, len(layers)):
        prev, cur = layers[li - 1], layers[li]
        cc = np.full(len(cur), np.inf); par = np.full(len(cur), -1)
        ee_p = np.array([fk(q)[1] for q in prev]); ee_c = np.array([fk(q)[1] for q in cur])
        for j in range(len(cur)):
            for i in range(len(prev)):
                checks += 1
                if not np.isfinite(cost[i]): continue
                c = cost[i] + float(np.linalg.norm(ee_c[j] - ee_p[i]))
                if dp_avoid and not S2.seg_clear(prev[i], cur[j], env.obstacles): c += 50.0
                if c < cc[j]: cc[j] = c; par[j] = i
        if li < len(layers) - 1:
            keep = min(len(cur), max(24, len(cur) * 3 // 4))
            order = np.argsort(cc)[:keep]
            layers[li] = cur[order]; cc = cc[order]; par = par[order]
        parents.append(par); cost = cc

    idx = int(np.argmin(cost)); path = [layers[-1][idx]]
    for li in range(len(layers) - 1, 0, -1):
        idx = parents[li - 1][idx]
        if idx < 0: break
        path.append(layers[li - 1][idx])
    path = np.array(path[::-1])

    # ---- 第二阶段：碰撞段加动作（节点不变）
    rng = np.random.default_rng(seed + 1)
    total_actions = 0; fails = 0
    segs = []
    for i in range(len(path) - 1):
        seg = ActionSegment(path[i], path[i + 1])
        ok, added = repair_with_actions(seg, env.obstacles, rng)
        total_actions += added
        if not ok: fails += 1
        segs.append(seg)

    return {"ok": fails == 0 and ik < 0.05, "stage": "done" if fails == 0 else "actions",
            "ik_err": ik, "actions": total_actions, "nodes": len(path),
            "checks": checks, "ms": int((time.time() - t0) * 1000),
            "exhaustive": float(len(cand) ** K)}


def benchmark(episodes=40, seed=5, dp_avoid=True):
    env = ArmAvoidEnv(seed=seed)
    rs = []
    for _ in range(episodes):
        env.reset(); rs.append(solve(env, dp_avoid=dp_avoid))
    ok = sum(1 for r in rs if r.get("ok"))
    from collections import Counter
    why = Counter(r.get("stage") for r in rs if not r.get("ok"))
    print(f"三段式 v3（节点固定 + 加动作，DP避碰撞={dp_avoid}）：成功 {ok}/{episodes} = {ok/episodes:.0%}")
    if why: print("  失败:", dict(why))
    print("  节点数（应恒为 K+1=%d）: %s" % (K_LAYERS + 1, sorted({r.get('nodes',0) for r in rs if r.get('ok')})))
    print(f"  平均追加动作 {np.mean([r.get('actions',0) for r in rs]):.1f} 个  "
          f"平均检查 {np.mean([r.get('checks',0) for r in rs]):.0f} 次  "
          f"平均耗时 {np.mean([r.get('ms',0) for r in rs]):.0f} ms")
    return ok / episodes


if __name__ == "__main__":
    print("--- 第一阶段就避碰撞 ---")
    benchmark(40, dp_avoid=True)
    print("\n--- 第一阶段不避碰撞，全靠第二阶段加动作 ---")
    benchmark(40, dp_avoid=False)
