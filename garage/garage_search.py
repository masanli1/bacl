"""动作组合的束搜索：验证倒车入库是否可解，并找出动作序列"""
import numpy as np
from garage_env import GarageEnv, GOAL, wrap_pi

def h(S):
    """启发：到目标位置 + 朝向差 + 需要倒车的程度"""
    d = np.linalg.norm(S[:2] - GOAL[:2])
    ang = abs(wrap_pi(S[2] - GOAL[2]))
    return d + 1.2 * ang

def beam(start, env, depth=12, width=40, arc=1.0):
    beams = [(env.S.copy(), [])]
    best = (h(env.S), env.S.copy(), [])
    for d in range(depth):
        cands = []
        for S, seq in beams:
            for a in range(6):
                env.S = S.copy()
                _, r, done, info = env.step(a)
                if info.get("collision"):
                    continue
                S2 = env.S.copy()
                sc = h(S2)
                cands.append((sc, S2, seq + [a], bool(info.get("success"))))
                if info.get("success"):
                    return seq + [a], S2, True
        if not cands:
            break
        cands.sort(key=lambda x: x[0])
        beams = [(c[1], c[2]) for c in cands[:width]]
        if cands[0][0] < best[0]:
            best = (cands[0][0], cands[0][1], cands[0][2])
    return best[2], best[1], False

if __name__ == "__main__":
    env = GarageEnv(arc_len=1.0, max_steps=30, dt=0.10, seed=0)
    starts = [[6.0, 7.5, 0.0, 0.0], [8.0, 8.5, 0.0, 0.0], [12.0, 6.5, 0.0, 0.0]]
    for st in starts:
        env.reset(start=np.array(st, float))
        seq, S, ok = beam(None, env, depth=12, width=50, arc=1.0)
        names = [GarageEnv.MANEUVERS[a][0] for a in seq]
        print("起点 %s → %s，%d 个动作：%s" % (st, "成功" if ok else "未达", len(seq), " → ".join(names)))
        if ok:
            print("   终点 %s  误差 %.2f m / %.0f°" % (np.round(S[:3],2).tolist(),
                  np.linalg.norm(S[:2]-GOAL[:2]), np.rad2deg(abs(wrap_pi(S[2]-GOAL[2])))))
