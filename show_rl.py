"""展示：Q-learning 学"怎么改自己的解"——学习曲线 + 策略表"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from arm_env import ArmAvoidEnv
from rl_loop import train, ACTIONS

font_manager.fontManager.addfont("/System/Library/Fonts/Hiragino Sans GB.ttc")
plt.rcParams["font.family"] = ["Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

env = ArmAvoidEnv(seed=5); env.reset()
Q, curves, straight, final = train(env, episodes=30)
nodes, L, col, jmp = final

final_len = [c[0][-1] for c in curves]
first_len = [c[0][0] for c in curves]
steps = [c[1] for c in curves]
reached = [c[2] for c in curves]
k = 5
smooth = np.convolve(final_len, np.ones(k)/k, mode="valid")

fig, axes = plt.subplots(1, 3, figsize=(15.2, 3.9), dpi=130,
                         gridspec_kw={"width_ratios": [1.1, 1, 1]})
ax = axes[0]
ax.plot(first_len, alpha=.3, color="#f59e0b", label="每轮起始长度")
ax.plot(final_len, alpha=.35, color="#3b82f6", label="每轮最终长度")
ax.plot(range(k-1, len(final_len)), smooth, color="#1d4ed8", lw=2, label=f"{k} 轮滑动平均")
ax.axhline(straight, color="#ef4444", ls="--", lw=1, label=f"直线 {straight:.3f} m")
ax.set_xlabel("学习轮次（episode）"); ax.set_ylabel("路径长度 (m)")
ax.set_title("学习曲线：每一轮把解改得更短", fontsize=10)
ax.grid(alpha=.25); ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)

ax1b = axes[1]
ax1b.plot(steps, "o-", color="#16a34a", lw=1.6, ms=3)
ax1b.set_xlabel("学习轮次"); ax1b.set_ylabel("达到最优所需动作数")
ax1b.set_title("越学越快：收敛所需步数下降", fontsize=10)
ax1b.grid(alpha=.25); ax1b.spines[["top","right"]].set_visible(False)
print("前期平均步数 %.1f → 后期 %.1f" % (np.mean(steps[:10]), np.mean(steps[-10:])))

ax2 = axes[2]
ax2.axis("off")
rows = []
for s in sorted(Q.keys()):
    a = int(np.argmax(Q[s][:-1]))
    if abs(Q[s][a]) < 1e-6: continue
    rows.append((f"节点{s[0]} · 长度档{s[1]} · 碰撞{s[2]} · 突变{s[3]}",
                 ACTIONS[a], Q[s][a]))
rows.sort(key=lambda r: -r[2])
txt = "学到的策略（状态 → 动作）\n\n"
for st, ac, q in rows[:8]:
    txt += f"{st}\n   → {ac}   Q={q:.2f}\n\n"
ax2.text(0.02, 0.98, txt, va="top", fontsize=8.5)
fig.tight_layout(); fig.savefig("rl_learning.png")
print("已保存 rl_learning.png")
print("起始长度 %.3f → 最终 %.3f m（直线 %.3f m）" % (np.mean(first_len), L, straight))
json.dump({"qtable": {str(k_): v.tolist() for k_, v in Q.items()},
           "actions": ACTIONS, "final_length": L, "straight": straight},
          open("qtable.json", "w"), ensure_ascii=False, indent=1)
print("Q 表已保存 qtable.json")
