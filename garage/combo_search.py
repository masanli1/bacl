"""在 30 个组合动作上搜索，够到目标位"""
from __future__ import annotations
import numpy as np
from garage_combo import ComboEnv
from garage_env import GOAL, wrap_pi


def h(S, nstep=0):
    d = np.linalg.norm(S[:2] - GOAL[:2])
    ang = abs(wrap_pi(S[2] - GOAL[2]))
    return float(d + 1.5 * ang + 0.02 * nstep)


def beam(start, depth=12, width=24, dt=0.05, max_steps=16, tol_d=0.10, tol_a=np.deg2rad(3)):
    env = ComboEnv(arc_len=1.0, max_steps=max_steps, dt=dt, seed=0)
    env.reset(start=np.array(start, float))
    env.margin = 0.12          # 搜索时留余量，避免粗步长/细步长不一致
    beams = [(env.S.copy(), [])]
    best = (h(env.S), env.S.copy(), [])
    for d in range(depth):
        cands = []
        for S, seq in beams:
            for a in range(ComboEnv.N_ACT):
                env.S = S.copy()
                _, r, done, info = env.step(a)
                if info.get("collision"):
                    continue
                S2 = env.S.copy()
                sc = h(S2, len(seq) + 1)
                err_d = np.linalg.norm(S2[:2] - GOAL[:2])
                err_a = abs(wrap_pi(S2[2] - GOAL[2]))
                if err_d < tol_d and err_a < tol_a:
                    return seq + [a], S2, True
                cands.append((sc, S2, seq + [a]))
        if not cands:
            break
        cands.sort(key=lambda x: x[0])
        beams = [(c[1], c[2]) for c in cands[:width]]
        if cands[0][0] < best[0]:
            best = (cands[0][0], cands[0][1], cands[0][2])
    return best[2], best[1], False


def verify(seq, start, dt=0.05):
    """用与搜索同样的步长复算（自洽），返回终点与误差"""
    env = ComboEnv(arc_len=1.0, max_steps=len(seq) + 1, dt=dt, seed=0)
    env.reset(start=np.array(start, float))
    env.margin = 0.12          # 搜索时留余量，避免粗步长/细步长不一致
    for a in seq:
        _, r, done, info = env.step(a)
        if info.get("collision"):
            return None, True
        if done:
            break
    return env.S.copy(), False


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    starts = [[6.0, 7.5, 0.0, 0.0], [9.0, 8.0, 0.0, 0.0], [11.0, 6.5, 0.0, 0.0]]
    for _ in range(3):
        starts.append([float(rng.uniform(4, 13)), float(rng.uniform(6, 10)), float(rng.uniform(-0.2, 0.2)), 0.0])
    for st in starts:
        seq, S, ok = beam(st)
        names = [ComboEnv.name(a) for a in seq]
        Sf, col = verify(seq, st)
        if Sf is not None:
            ed = np.linalg.norm(Sf[:2] - GOAL[:2]); ea = np.rad2deg(abs(wrap_pi(Sf[2] - GOAL[2])))
        else:
            ed, ea = 99, 99
        print("起点 %s → %s，%d 个动作，精细复算误差 %.3f m / %.1f°%s"
              % (st, "找到" if ok else "未找到", len(seq), ed, ea, "（碰撞）" if col else ""))
        print("   " + " → ".join(names))
