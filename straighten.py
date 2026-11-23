"""把节点末端投影到起点—终点连线上，做最后的缩短"""
from __future__ import annotations
import numpy as np
from arm_env import fk
import structured_search2 as S2
from ee_path import ee_line_segment, ee_length


def seg_clear_ee(q1, q2, obstacles, n=20):
    for q in ee_line_segment(q1, q2, n):
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


def straighten(nodes, obstacles, tries=3):
    """把每个中间节点的末端投影到连线上；不撞就采纳。"""
    nodes = [np.asarray(q, float).copy() for q in nodes]
    p0, p1 = fk(nodes[0])[1], fk(nodes[-1])[1]
    d = p1 - p0
    L2 = float(d @ d) + 1e-12
    improved = 0
    for _ in range(tries):
        changed = False
        for i in range(1, len(nodes)-1):
            pe = fk(nodes[i])[1]
            t = float(np.clip((pe - p0) @ d / L2, 0.02, 0.98))
            target = p0 + t * d
            q_new, err = S2.ik_solve(target, nodes[i])
            if err > 0.05:
                continue
            trial = nodes.copy()
            trial[i] = q_new
            # 只看相邻两段是否仍无碰撞
            if (seg_clear_ee(trial[i-1], trial[i], obstacles) and
                    seg_clear_ee(trial[i], trial[i+1], obstacles)):
                nodes = trial; improved += 1; changed = True
        if not changed:
            break
    L = sum(ee_length(ee_line_segment(nodes[i], nodes[i+1])) for i in range(len(nodes)-1))
    return nodes, L, improved


if __name__ == "__main__":
    from arm_env import ArmAvoidEnv
    import shorten2, ee_path
    env = ArmAvoidEnv(seed=5); env.reset()
    r = shorten2.solve(env, verbose=False)
    print("管道解: 节点 %d, 长度 %.3f m" % (len(r["nodes"]), ee_path.path_length_ee(r["nodes"])))
    nodes2, L2, imp = straighten(r["nodes"], env.obstacles)
    ee0, ee1 = fk(nodes2[0])[1], fk(nodes2[-1])[1]
    straight = float(np.linalg.norm(ee1 - ee0))
    print("拉直后: 节点 %d, 长度 %.3f m（直线 %.3f m，%.2f 倍），改动 %d 个节点"
          % (len(nodes2), L2, straight, L2/straight, imp))
