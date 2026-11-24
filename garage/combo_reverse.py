"""倒车入库的动作组合：前进接近 + 倒车弧

几何：
  倒车弧的半径 R = WB / tan(δ_max) ≈ 3.86 m
  从目标位反向推演，得到"倒车弧的起点" S0（在通道里）
  再用前进动作把车开到 S0，最后用倒车弧拼进库
两个阶段都是**动作组合**，不涉及连续控制。
"""
from __future__ import annotations
import numpy as np
from garage_combo import ComboEnv
from garage_env import GOAL, AISLE_Y, MAX_STEER, MAX_SPEED, MAX_REV, WB, wrap_pi

R_ARC = WB / np.tan(MAX_STEER)


def back_arc_starts(arc_len, steer, dt=0.02):
    S = np.array([*GOAL, 0.0], float)
    poses = []
    step = 0.02
    n = int(arc_len / step)
    # 从终点"前进"积分：这些位姿倒车（同样转向）能回到终点
    for _ in range(n):
        S[3] = steer
        S[2] = wrap_pi(S[2] + (MAX_SPEED / WB) * np.tan(S[3]) * step)
        S[0] += MAX_SPEED * np.cos(S[2]) * step
        S[1] += MAX_SPEED * np.sin(S[2]) * step
        poses.append(S.copy())
    return poses


def approach_beam(start, S_target, tol_d=0.20, tol_a=np.deg2rad(6), depth=10, width=30):
    """只用前进动作，从 start 开到 S_target 附近"""
    env = ComboEnv(arc_len=1.0, max_steps=depth + 2, dt=0.05, seed=0)
    env.reset(start=np.array(start, float)); env.margin = 0.12
    fwd_actions = [a for a in range(ComboEnv.N_ACT) if ComboEnv.decode(a)[0] > 0]
    beams = [(env.S.copy(), [])]
    best = (1e9, None, None)
    for _ in range(depth):
        cands = []
        for S, seq in beams:
            for a in fwd_actions:
                env.S = S.copy()
                _, r, done, info = env.step(a)
                if info.get("collision"):
                    continue
                S2 = env.S.copy()
                d = np.linalg.norm(S2[:2] - S_target[:2])
                ang = abs(wrap_pi(S2[2] - S_target[2]))
                c = d + 1.5 * ang + 0.02 * len(seq)
                if d < tol_d and ang < tol_a:
                    return seq + [a], S2, True
                cands.append((c, S2, seq + [a]))
                if c < best[0]:
                    best = (c, S2, seq + [a])
        if not cands:
            break
        cands.sort(key=lambda x: x[0])
        beams = [(c[1], c[2]) for c in cands[:width]]
    return best[2], best[1], False


def solve(start, depth=7, width=10):
    """前进组合 + 倒车弧"""
    best = None
    for steer in (+MAX_STEER, -MAX_STEER):
        poses = back_arc_starts(9.0, steer)
        cands = [p for i, p in enumerate(poses) if i % 10 == 0
                 and 2.5 < p[0] < 28.0 and 6.0 < p[1] < AISLE_Y - 0.2]
        for S0 in cands:
            seq, S, ok = approach_beam(start, S0, tol_d=0.30, tol_a=np.deg2rad(8),
                                       depth=depth, width=width)
            if not ok:
                continue
            dth = abs(wrap_pi(GOAL[2] - S0[2]))
            arc = float(dth * WB / np.tan(abs(steer)))   # 弧长 = 转角 × 半径
            n_rev = max(1, int(np.ceil(arc / 2.0)))
            total = len(seq) + n_rev
            if best is None or total < best[0]:
                best = (total, seq, S0, steer, arc, n_rev)
    return best


if __name__ == "__main__":
    starts = [[6.0, 7.5, 0.0, 0.0], [9.0, 8.0, 0.0, 0.0]]
    for st in starts:
        b = solve(st)
        if b is None:
            print("起点 %s：没找到组合" % st); continue
        total, seq, S0, steer, arc, n_rev = b
        print("起点 (%.1f, %.1f)：前进 %d 个动作到 (%.2f, %.2f, %.0f°)，再倒车 %.1f m（%.0f°，%d 段）"
              % (st[0], st[1], len(seq), S0[0], S0[1], np.rad2deg(S0[2]), arc, np.rad2deg(steer), n_rev))
        print("   前进: " + " → ".join(ComboEnv.name(a) for a in seq))
