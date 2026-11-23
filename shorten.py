"""路径截肢迭代（动作学习）

逻辑：
  1. 先用三段式解出一条可用轨迹（节点 + 动作）
  2. 反复截肢：把路径里某一段切掉，用更短的连接（必要时加动作）接回去
  3. 只要更短且无碰撞就采纳，继续下一轮
  4. 直到再也切不动 —— 这个过程就是"学习"：每一轮都在上一轮的解上改进一步

注意：这里学的不是策略参数（不是 PPO），而是**动作序列本身**——
      哪些段可以合成一段、哪些地方需要加动作、加在哪。
"""
from __future__ import annotations

import json
import numpy as np
from arm_env import ArmAvoidEnv, fk
import structured_search3 as S3
import structured_search2 as S2
from show2 import dp_nodes


def path_len(seg_list):
    return float(sum(np.linalg.norm(fk(s.q(0.0))[1] - fk(s.q(1.0))[1]) for s in seg_list))


def dense_len(seg, n=30):
    ee = np.array([fk(seg.q(t))[1] for t in np.linspace(0, 1, n)])
    return float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum())


def try_connect(q1, q2, obstacles, rng, max_tries=40):
    """把两点直接连起来；撞了就用动作修。返回 (是否成功, 段, 动作数)。"""
    seg = S3.ActionSegment(q1, q2)
    if seg.clear(obstacles):
        return True, seg, 0
    ok, added = S3.repair_with_actions(seg, obstacles, rng, tries=max_tries)
    return ok, seg, added


def shorten(nodes, segs, obstacles, rng, max_rounds=12, verbose=True, return_history=False):
    """迭代截肢：删节点 / 合并相邻段，只要能变短就采纳。"""
    nodes = [q.copy() for q in nodes]
    segs = list(segs)
    history = [( [q.copy() for q in nodes], list(segs), sum(dense_len(s) for s in segs) )]
    lengths = [sum(dense_len(s) for s in segs)]
    actions = [sum(len(s.acts) for s in segs)]

    for rnd in range(max_rounds):
        improved = False
        i = 1
        while i < len(nodes) - 1:                     # 试删中间节点
            trial_nodes = nodes[:i] + nodes[i+1:]
            a, b = nodes[i-1], nodes[i+1]
            ok, seg_new, added = try_connect(a, b, obstacles, rng)
            if ok:
                new_len = dense_len(seg_new)
                old_len = dense_len(segs[i-1]) + dense_len(segs[i])
                if new_len < old_len - 1e-3:
                    nodes = trial_nodes
                    segs = segs[:i-1] + [seg_new] + segs[i+1:]
                    improved = True
                    history.append(([q.copy() for q in nodes], list(segs),
                                    sum(dense_len(s) for s in segs)))
                    if verbose:
                        print(f"  第 {rnd+1} 轮：删节点 {i}，{old_len:.3f} → {new_len:.3f} m，加 {added} 个动作")
                    continue
            i += 1
        lengths.append(sum(dense_len(s) for s in segs))
        actions.append(sum(len(s.acts) for s in segs))
        if not improved:
            if verbose: print(f"  第 {rnd+1} 轮：无改进，收敛")
            break
    if return_history:
        return nodes, segs, lengths, actions, history
    return nodes, segs, lengths, actions


def run(seed=5, verbose=True, return_history=False):
    env = ArmAvoidEnv(seed=seed)
    env.reset()
    nodes = dp_nodes(env)
    # 第一阶段解（加动作修碰撞）
    rng = np.random.default_rng(42)
    segs = []
    for i in range(len(nodes)-1):
        ok, seg, added = try_connect(nodes[i], nodes[i+1], env.obstacles, rng)
        segs.append(seg)
    L0 = sum(dense_len(s) for s in segs)
    if verbose:
        print(f"初始：{len(nodes)} 个节点，{sum(len(s.acts) for s in segs)} 个动作，长度 {L0:.3f} m")
    nodes2, segs2, lens, acts, history = shorten(nodes, segs, env.obstacles, rng,
                                                verbose=verbose, return_history=True)
    L1 = lens[-1]
    if verbose:
        print(f"截肢后：{len(nodes2)} 个节点，{acts[-1]} 个动作，长度 {L1:.3f} m "
              f"（缩短 {(1-L1/L0)*100:.1f}%）")
    if return_history:
        return env, nodes, segs, nodes2, segs2, lens, acts, history
    return env, nodes, segs, nodes2, segs2, lens, acts


if __name__ == "__main__":
    run()
