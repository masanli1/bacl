"""对比：独立解 IK（突变）vs 链式解 IK（连续）"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from arm_env import ArmAvoidEnv, fk
import structured_search2 as S2
from chain_path import chain_nodes, sample_traj, max_jump, ee_len

font_manager.fontManager.addfont("/System/Library/Fonts/Hiragino Sans GB.ttc")
plt.rcParams["font.family"] = ["Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

env = ArmAvoidEnv(seed=5); env.reset()
qg, ik = S2.ik_solve(env.target, env.q)
p0, p1 = fk(env.q)[1], fk(qg)[1]
straight = float(np.linalg.norm(p1 - p0))

# 旧：历史上"每节点独立解 IK"的导出结果（docs/old_poses.json）
import os
if os.path.exists("docs/old_poses.json"):
    old = np.array(json.load(open("docs/old_poses.json"))["poses"])
else:
    q_start = env.q
    old_nodes = [q_start]
    for t in np.linspace(0, 1, 9)[1:]:
        p = (1-t)*p0 + t*p1
        q, err = S2.ik_solve(p, q_start)
        old_nodes.append(q)
    from chain_path import sample_traj as _st
    old = _st(np.array(old_nodes), n_per_seg=16)
# 新：链式
nodes = chain_nodes(p0, p1, env.q, K=9)
new = sample_traj(nodes, n_per_seg=16)
# 重采样到同长度便于对比
def resample(qs, n=160):
    idx = np.linspace(0, len(qs)-1, n).astype(int)
    return qs[idx]
old_r, new_r = resample(old, 160), resample(new, 160)

fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.9), dpi=130, sharey=True)
for ax, Q, title in ((axes[0], old_r, "修改前：独立解 IK（关节突变）"),
                     (axes[1], new_r, "修改后：链式解 IK（连续）")):
    for j in range(6):
        ax.plot(Q[:, j], lw=1.4, label=f"关节 {j+1}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("帧"); ax.grid(alpha=.25)
    ax.spines[["top","right"]].set_visible(False)
    mj = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).max())
    ax.text(0.02, 0.95, f"最大关节增量 {mj:.2f} rad", transform=ax.transAxes,
            fontsize=9, va="top", color="#b91c1c" if mj > 0.5 else "#15803d")
axes[0].set_ylabel("关节角 (rad)")
axes[0].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig("chain_compare.png")
print("已保存 chain_compare.png")
print("旧最大增量 %.3f rad，新 %.4f rad" % (max_jump(old_r), max_jump(new_r)))
