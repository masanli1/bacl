"""末端空间求解（碰撞检查与轨迹用同一套插值，不再矛盾）

段的定义：p(t) = lerp(p1,p2,t) + Σ aᵢ·sin(πᵢ·t)·nᵢ
  · aᵢ 是"动作"的幅度，nᵢ 是绕开障碍的方向 —— 动作在末端空间里，长度可精确计算
  · 节点（p1、p2）不变，只改中间怎么走
"""
from __future__ import annotations
import numpy as np
from arm_env import fk
import structured_search2 as S2


def _closest_obstacle(p0, p1, obstacles):
    d = p1 - p0
    L2 = float(d @ d) + 1e-12
    best = None
    for ob in obstacles:
        c, r = ob[:3], ob[3]
        t = float(np.clip((c - p0) @ d / L2, 0, 1))
        foot = p0 + t * d
        dist = float(np.linalg.norm(foot - c))
        if best is None or dist - r < best[0]:
            n = foot - c
            nn = np.linalg.norm(n)
            n = n / nn if nn > 1e-6 else np.array([0, 0, 1.0])
            best = (dist - r, c, r, t, n)
    return best


def _clear(qs, obstacles):
    for q in qs:
        pts = fk(q)[0]
        for ob in obstacles:
            c, r = ob[:3], ob[3]
            for i in range(len(pts)-1):
                a, b = pts[i], pts[i+1]
                ab = b - a; den = float(ab @ ab) + 1e-12
                t = np.clip(float((c - a) @ ab) / den, 0, 1)
                if np.linalg.norm(a + t*ab - c) < r:
                    return False
    return True


def make_curve(q1, q2, amp=0.0, n_dir=None, phases=1, n=26):
    """末端曲线 → 关节序列（逐点 IK，从前一点出发保证连续）"""
    p1, p2 = fk(q1)[1], fk(q2)[1]
    qs, q_prev = [q1], q1
    for t in np.linspace(0, 1, n)[1:]:
        p = (1-t)*p1 + t*p2
        if amp and n_dir is not None:
            p = p + amp * np.sin(np.pi * t) * n_dir
        q, err = S2.ik_solve(p, q_prev, iters=120)
        if err > 0.06:
            return None
        qs.append(q); q_prev = q
    return np.array(qs)


def seg_solve(q1, q2, obstacles, max_amp=0.8, steps=16):
    """返回 (是否成功, 关节序列, 动作描述)"""
    qs = make_curve(q1, q2)
    if qs is not None and _clear(qs, obstacles):
        return True, qs, None
    info = _closest_obstacle(fk(q1)[1], fk(q2)[1], obstacles)
    if info is None:
        return False, None, None
    gap, c, r, t, n = info
    for k in range(1, steps + 1):
        amp = max_amp * k / steps
        qs = make_curve(q1, q2, amp=amp, n_dir=n)
        if qs is not None and _clear(qs, obstacles):
            return True, qs, {"amp": float(amp), "n": n.tolist(), "t": float(t),
                              "obstacle_r": float(r)}
    return False, None, None


def curve_length(qs):
    P = np.array([fk(q)[1] for q in qs])
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


if __name__ == "__main__":
    from arm_env import ArmAvoidEnv
    import shorten2
    for seed in (5, 6, 8):
        env = ArmAvoidEnv(seed=seed); env.reset()
        qg, ik = S2.ik_solve(env.target, env.q)
        p0, p1 = fk(env.q)[1], fk(qg)[1]
        straight = float(np.linalg.norm(p1 - p0))
        ok, qs, act = seg_solve(env.q, qg, env.obstacles)
        if ok:
            L = curve_length(qs)
            print("seed=%d 直线 %.3f m → 解 %.3f m（%.2f 倍）%s"
                  % (seed, straight, L, L/straight,
                     "" if act is None else "（绕行幅度 %.2f m）" % act["amp"]))
        else:
            print("seed=%d 无解" % seed)
