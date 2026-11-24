"""组合搜索 + 倒车优先偏好（仍然只在动作组合层面）

代价 = 到目标距离 + 1.5×朝向差
     + 0.02×步数
     + 3.0×（前进且车身已进入车位区）   ← 禁止"车头扎进车位"
     + 0.3×（前进动作数）              ← 整体偏好倒车
"""
from __future__ import annotations
import numpy as np
from garage_combo import ComboEnv
from garage_env import GOAL, AISLE_Y, wrap_pi

TOL_D, TOL_A = 0.10, np.deg2rad(3)


def in_slot_zone(S):
    return S[1] > AISLE_Y - 0.6


def cost(S, seq):
    d = np.linalg.norm(S[:2] - GOAL[:2])
    ang = abs(wrap_pi(S[2] - GOAL[2]))
    c = d + 1.5 * ang + 0.02 * len(seq)
    fwd = 0
    for a in seq:
        direction, _, _ = ComboEnv.decode(a)
        if direction > 0:
            fwd += 1
    c += 0.05 * fwd
    if seq:
        last_dir = ComboEnv.decode(seq[-1])[0]
        if in_slot_zone(S) and last_dir > 0:
            c += 0.60
    return float(c)


def beam(start, depth=16, width=40, dt=0.05, max_steps=20):
    env = ComboEnv(arc_len=1.0, max_steps=max_steps, dt=dt, seed=0)
    env.reset(start=np.array(start, float))
    env.margin = 0.12
    beams = [(env.S.copy(), [])]
    best = (cost(env.S, []), env.S.copy(), [])
    for _ in range(depth):
        cands = []
        for S, seq in beams:
            for a in range(ComboEnv.N_ACT):
                env.S = S.copy()
                _, r, done, info = env.step(a)
                if info.get("collision"):
                    continue
                S2 = env.S.copy()
                seq2 = seq + [a]
                err_d = np.linalg.norm(S2[:2] - GOAL[:2])
                err_a = abs(wrap_pi(S2[2] - GOAL[2]))
                if err_d < TOL_D and err_a < TOL_A:
                    return seq2, S2, True
                cands.append((cost(S2, seq2), S2, seq2))
        if not cands:
            break
        cands.sort(key=lambda x: x[0])
        beams = [(c[1], c[2]) for c in cands[:width]]
        if cands[0][0] < best[0]:
            best = cands[0]
    return best[2], best[1], False


def replay(seq, start, dt=0.05, margin=0.06):
    env = ComboEnv(arc_len=1.0, max_steps=len(seq) + 1, dt=dt, seed=0)
    env.reset(start=np.array(start, float))
    env.margin = margin
    poses = [env.S.copy()]
    for a in seq:
        _, r, done, info = env.step(a)
        if info.get("collision"):
            return poses, True
        poses.append(env.S.copy())
    return poses, False


if __name__ == "__main__":
    starts = [[6.0, 7.5, 0.0, 0.0], [9.0, 8.0, 0.0, 0.0], [7.676, 6.181, -0.18, 0.0],
              [11.0, 6.5, 0.0, 0.0]]
    for st in starts:
        seq, S, ok = beam(st)
        poses, col = replay(seq, st)
        err_d = np.linalg.norm(poses[-1][:2] - GOAL[:2])
        err_a = np.rad2deg(abs(wrap_pi(poses[-1][2] - GOAL[2])))
        rev = sum(1 for a in seq if ComboEnv.decode(a)[0] < 0)
        print("起点 (%.1f, %.1f)：%d 个动作（倒车 %d 次），误差 %.3f m / %.1f°%s"
              % (st[0], st[1], len(seq), rev, err_d, err_a, "（复算碰撞）" if col else ""))
        print("   " + " → ".join(ComboEnv.name(a) for a in seq))
