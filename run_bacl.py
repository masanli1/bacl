"""在被挡场景上跑基础动作组合学习"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from arm_env import ArmAvoidEnv, fk
from bacl import learn, ACTIONS

font_manager.fontManager.addfont("/System/Library/Fonts/Hiragino Sans GB.ttc")
plt.rcParams["font.family"] = ["Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

env = ArmAvoidEnv(seed=8); env.reset()   # 直线被挡、但绕行可行
Q, log, straight, final = learn(env, episodes=40)
nodes, L, col, jmp = final
print("直线 %.3f m | 学到 %.3f m（%.2f 倍）| 碰撞=%s | 最大关节增量 %.3f"
      % (straight, L, L/straight, col, jmp))
print("\n学到的策略：")
for s in sorted(Q.keys()):
    a = int(np.argmax(Q[s]))
    if abs(Q[s][a]) < 1e-6: continue
    print("  节点%d 长度档%d 碰撞%d 突变%d → %-8s Q=%.2f"
          % (s[0], s[1], s[2], s[3], ACTIONS[a], Q[s][a]))

# 碰撞解决率曲线
resolved = [0.0 if t[0][1] else 1.0 for t in log]
first = [t[-1][1] for t in log]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8), dpi=130)
ax = axes[0]
k = 5
ax.plot([t[-1][0] for t in log], alpha=.35, color="#3b82f6", label="每轮最终长度")
sm = np.convolve([t[-1][0] for t in log], np.ones(k)/k, mode="valid")
ax.plot(range(k-1, len(log)), sm, color="#1d4ed8", lw=2, label=f"{k} 轮平均")
ax.axhline(straight, color="#ef4444", ls="--", lw=1, label="直线（被挡，不可达）")
ax.set_xlabel("学习轮次"); ax.set_ylabel("路径长度 (m)")
ax.set_title("被挡场景：学会绕行后长度收敛", fontsize=10)
ax.grid(alpha=.25); ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)

ax2 = axes[1]
ok_flag = [0 if t[-1][1] else 1 for t in log]
sm2 = np.convolve(ok_flag, np.ones(k)/k, mode="valid")
ax2.plot(sm2, color="#16a34a", lw=2)
ax2.set_ylim(-0.05, 1.05); ax2.set_xlabel("学习轮次"); ax2.set_ylabel("找到无碰撞解的比例")
ax2.set_title("学会补全动作的比例上升", fontsize=10)
ax2.grid(alpha=.25); ax2.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("bacl_learning.png")
json.dump({"actions": ACTIONS, "straight": straight, "final_length": L,
           "qtable": {str(k_): v.tolist() for k_, v in Q.items()}},
          open("bacl_qtable.json", "w"), ensure_ascii=False, indent=1)
print("\n已保存 bacl_learning.png / bacl_qtable.json")
