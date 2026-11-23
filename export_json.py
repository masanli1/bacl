"""把求解结果导出成前端用的 JSON：节点、动作、轨迹、机械臂姿态"""
from __future__ import annotations
import json
import numpy as np
from arm_env import fk, ArmAvoidEnv
import structured_search2 as S2
from shorten import run, dense_len
from ee_path import ee_line_segment, ee_length
from chain_path import chain_nodes, sample_traj, max_jump
import structured_search3 as S3


def seg_full(q1, q2, n=40):
    """段轨迹：末端直线插值（机器人真实走法），返回 (EE 点, 动作列表)"""
    qs = ee_line_segment(q1, q2, n)
    pts = np.array([fk(q)[1] for q in qs])
    return pts.tolist(), []


def sample_by_arclength(segs, n=160):
    dense_q = []
    for seg in segs:
        for s in np.linspace(0, 1, 160, endpoint=False):
            dense_q.append(seg.q(s))
    dense_q.append(segs[-1].q(1.0))
    ee = np.array([fk(q)[1] for q in dense_q])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ee, axis=0), axis=1))])
    total = cum[-1]
    idx = np.clip(np.searchsorted(cum, np.linspace(0, total, n)), 0, len(dense_q) - 1)
    return dense_q, ee, idx


def build(seed=5):
    """导出：节点取自"直线附近候选 + DP + 拉直"，轨迹用末端直线插值。"""
    import shorten2, straighten
    env = ArmAvoidEnv(seed=seed)
    env.reset()
    # 旧解（独立 IK，会有突变）——只用于"截肢前"对比
    r = shorten2.solve(env, verbose=False)
    nodes0 = r["nodes"]
    # 新解：链式 IK，节点之间关节连续
    q_goal, ik = S2.ik_solve(env.target, env.q)
    p0, p1 = fk(env.q)[1], fk(q_goal)[1]
    nodes_f = chain_nodes(p0, p1, env.q, K=9)
    if nodes_f is None:
        nodes_f = nodes0

    def pack(nodes, continuous=False):
        traj = []
        if continuous:
            qs_all = sample_traj(nodes, n_per_seg=16)
            traj = [[fk(q)[1].tolist() for q in qs_all]]
        else:
            for i in range(len(nodes) - 1):
                qs = ee_line_segment(nodes[i], nodes[i+1], 40)
                traj.append([fk(q)[1].tolist() for q in qs])
        length = sum(ee_length(ee_line_segment(nodes[i], nodes[i+1]))
                     for i in range(len(nodes) - 1))
        return {
            "nodes": [np.asarray(q).tolist() for q in nodes],
            "segs_traj": traj,
            "actions": [],
            "length": float(length),
        }

    before = pack(nodes0)
    after = pack(nodes_f, continuous=True)

    all_q = list(sample_traj(nodes_f, n_per_seg=16))
    all_ee = [fk(q)[1].tolist() for q in all_q]
    ee_arr = np.array(all_ee)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ee_arr, axis=0), axis=1))])
    idx = np.clip(np.searchsorted(cum, np.linspace(0, cum[-1], 160)), 0, len(all_q)-1)
    data = {
        "obstacles": [{"c": np.asarray(ob[:3]).tolist(), "r": float(ob[3])} for ob in env.obstacles],
        "start_ee": np.asarray(fk(nodes_f[0])[1]).tolist(),
        "goal_ee": np.asarray(fk(nodes_f[-1])[1]).tolist(),
        "straight": float(np.linalg.norm(fk(nodes_f[-1])[1] - fk(nodes_f[0])[1])),
        "before": before, "after": after,
        "poses": [np.asarray(all_q[i]).tolist() for i in idx],
        "ee_frames": [ee_arr[i].tolist() for i in idx],
        "lengths": [before["length"], after["length"]],
        "actions_count": 0,
        "max_joint_step": max_jump(np.array(all_q)),
    }
    with open("solve.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("solve.json: 直线 %.3f m | 截肢前 %.3f m | 连续解 %.3f m（%.2f 倍）| 最大关节增量 %.4f rad"
          % (data["straight"], before["length"], after["length"],
             after["length"]/max(data["straight"], 1e-9), data["max_joint_step"]))


if __name__ == "__main__":
    build()
