"""导出 BACL 学习结果（被挡场景）：学习前碰撞 vs 学习后绕行 + 学到的动作"""
from __future__ import annotations
import json
import numpy as np
from arm_env import ArmAvoidEnv, fk
import structured_search2 as S2
from chain_path import chain_nodes, sample_traj, max_jump
from bacl import learn, evaluate, ACTIONS


def pack(nodes, obstacles):
    qs = sample_traj(nodes, n_per_seg=12)
    P = np.array([fk(q)[1] for q in qs])
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return {"nodes": [np.asarray(q).tolist() for q in nodes],
            "traj": [p.tolist() for p in P],
            "length": L}


def main():
    env = ArmAvoidEnv(seed=8); env.reset()
    q_goal, _ = S2.ik_solve(env.target, env.q)
    p0, p1 = fk(env.q)[1], fk(q_goal)[1]
    straight = float(np.linalg.norm(p1 - p0))

    # 学习前的残缺组合：直线节点（会撞）
    before_nodes = chain_nodes(p0, p1, env.q, K=6)
    before = pack(before_nodes, env.obstacles)
    before["collide"] = True

    # 学习
    Q, log, straight2, final = learn(env, episodes=40)
    nodes_f, L, col, jmp = final
    after = pack(nodes_f, env.obstacles)
    after["collide"] = bool(col)

    # 学到的动作：从 Q 表里取碰撞状态的最优动作
    learned = []
    for s in sorted(Q.keys()):
        a = int(np.argmax(Q[s]))
        if abs(Q[s][a]) < 1e-6: continue
        learned.append({"state": f"节点{s[0]} · 长度档{s[1]} · 碰撞{s[2]} · 突变{s[3]}",
                        "action": ACTIONS[a], "q": float(Q[s][a]),
                        "collision": bool(s[2])})

    # 播放帧（末端弧长均匀）
    qs = sample_traj(nodes_f, n_per_seg=16)
    ee = np.array([fk(q)[1] for q in qs])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ee, axis=0), axis=1))])
    idx = np.clip(np.searchsorted(cum, np.linspace(0, cum[-1], 160)), 0, len(qs)-1)
    data = {
        "scene": "被挡场景（Bacl 学习）",
        "obstacles": [{"c": np.asarray(o[:3]).tolist(), "r": float(o[3])} for o in env.obstacles],
        "start_ee": np.asarray(fk(nodes_f[0])[1]).tolist(),
        "goal_ee": np.asarray(fk(nodes_f[-1])[1]).tolist(),
        "straight": straight,
        "before": {"nodes": before["nodes"], "segs_traj": [before["traj"]],
                   "actions": [], "length": before["length"], "collide": True},
        "after": {"nodes": after["nodes"], "segs_traj": [after["traj"]],
                  "actions": [{"seg": -1, "t": 0.5, "w": 0.25,
                               "d": [0,0,0,0,0,0], "amp": None}],
                  "length": after["length"], "collide": bool(col)},
        "poses": [np.asarray(qs[i]).tolist() for i in idx],
        "ee_frames": [ee[i].tolist() for i in idx],
        "max_joint_step": max_jump(qs),
        "learned_policy": learned,
        "episodes": len(log),
    }
    json.dump(data, open("bacl_solve.json", "w"), ensure_ascii=False)
    print("bacl_solve.json: 直线 %.3f | 学习前 %.3f（碰撞）| 学习后 %.3f（%s）| %.2f 倍"
          % (straight, before["length"], after["length"],
             "无碰撞" if not col else "仍碰撞", after["length"]/straight))
    print("策略条目:", [(r["state"], r["action"]) for r in learned[:4]])


if __name__ == "__main__":
    main()
