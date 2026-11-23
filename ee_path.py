"""末端空间插值：节点之间按末端直线运动（这才是机器人真实的走法）

之前按关节空间插值，关节转得均匀但末端轨迹是弯的，所以"长度"算出来 4 m，
而起点到终点直线只有 0.75 m。改成末端直线后长度才有意义。
"""
from __future__ import annotations

import numpy as np
from arm_env import fk, Q_HOME
import structured_search2 as S2


def ee_line_segment(q1, q2, n=24):
    """从 q1 到 q2 走末端直线：末端沿直线插值，逐点解逆解。"""
    p1, p2 = fk(q1)[1], fk(q2)[1]
    qs = [q1]
    q_prev = q1
    for t in np.linspace(0, 1, n)[1:]:
        target = (1 - t) * p1 + t * p2
        q, err = S2.ik_solve(target, q_prev, iters=120)
        if err > 0.06:                     # 解不出来就退回关节插值
            qs.append((1 - t) * q1 + t * q2)
        else:
            qs.append(q)
        q_prev = qs[-1]
    return np.array(qs)


def ee_length(qs):
    P = np.array([fk(q)[1] for q in qs])
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def path_length_ee(nodes, n=24):
    total = 0.0
    for i in range(len(nodes) - 1):
        total += ee_length(ee_line_segment(nodes[i], nodes[i+1], n))
    return total


if __name__ == "__main__":
    from arm_env import ArmAvoidEnv
    import shorten2
    env = ArmAvoidEnv(seed=5); env.reset()
    r = shorten2.solve(env, verbose=True)
    nodes = r["nodes"]
    L_ee = path_length_ee(nodes)
    ee0, ee1 = fk(nodes[0])[1], fk(nodes[-1])[1]
    straight = float(np.linalg.norm(ee1 - ee0))
    print("  关节插值长度 %.3f m" % r["length"])
    print("  末端直线长度 %.3f m（直线本身 %.3f m，%.2f 倍）" % (L_ee, straight, L_ee/straight))
