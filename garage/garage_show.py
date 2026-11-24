"""可视化：学习后的倒车入库轨迹 + 成功率-精度曲线"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Polygon
from garage_env import (GarageEnv, GOAL, TX0, TX1, SLOT_X0, SLOT_W, SLOT_DIV, N_SLOT,
                        AISLE_Y, SLOT_BACK, LEN, WID, REAR_OH, FRONT_OH, wrap_pi)
from garage_bacl import train, s_key

font_manager.fontManager.addfont("/System/Library/Fonts/Hiragino Sans GB.ttc")
plt.rcParams["font.family"] = ["Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False


def car_poly(S):
    x, y, th = S[0], S[1], S[2]
    fx, fy = np.cos(th), np.sin(th)
    lx, ly = -fy, fx
    pts = []
    for dx, dy in ((FRONT_OH, WID/2), (FRONT_OH, -WID/2), (-REAR_OH, -WID/2), (-REAR_OH, WID/2)):
        pts.append((x + dx*fx + dy*lx, y + dx*fy + dy*ly))
    return np.array(pts)


def draw_garage(ax):
    for i in range(N_SLOT):
        x0 = SLOT_X0 + i*(SLOT_W + SLOT_DIV)
        if i + 1 == 4:
            ax.add_patch(Rectangle((x0, AISLE_Y), SLOT_W, SLOT_BACK - AISLE_Y,
                                   facecolor="#dcfce7", edgecolor="#16a34a", lw=1.6, zorder=1))
        else:
            ax.add_patch(Rectangle((x0, AISLE_Y), SLOT_W, SLOT_BACK - AISLE_Y,
                                   facecolor="#e2e8f0", edgecolor="#94a3b8", lw=0.8, zorder=1))
            cy = AISLE_Y + 0.15 + 4.40/2
            ax.add_patch(Rectangle((x0 + 0.5, cy - 4.40/2), 1.80, 4.40,
                                   facecolor="#cbd5e1", edgecolor="#64748b", lw=0.8, zorder=2))
    ax.plot([0, 30], [AISLE_Y, AISLE_Y], color="#475569", lw=1.2, zorder=3)
    ax.set_xlim(0, 30); ax.set_ylim(0, 18)
    ax.set_aspect("equal"); ax.grid(alpha=.15)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


def run_policy(Q, start, arc=1.0, dt=0.10, max_steps=26):
    env = GarageEnv(arc_len=arc, max_steps=max_steps, dt=dt, seed=0)
    env.reset(start=np.array(start, float))
    poses, names = [env.S.copy()], []
    for _ in range(max_steps):
        a = int(np.argmax(Q.get(s_key(env.S), np.zeros(6))))
        _, r, done, info = env.step(a)
        poses.append(env.S.copy())
        names.append(GarageEnv.MANEUVERS[a][0])
        if done:
            return poses, names, info
    return poses, names, info


def main():
    print("训练 Q 表…")
    Q, hist = train(episodes=12000, start=None)

    # 找一个成功的学习轨迹
    env = GarageEnv(seed=7)
    found = None
    for k in range(60):
        env.reset()
        start = env.S.copy()
        poses, names, info = run_policy(Q, start)
        if info.get("success"):
            found = (start, poses, names, info)
            break
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=130,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    draw_garage(ax)
    if found:
        start, poses, names, info = found
        for i, S in enumerate(poses):
            P = car_poly(S)
            alpha = 0.15 + 0.5 * i / max(len(poses)-1, 1)
            ax.add_patch(Polygon(P, closed=True, facecolor="#3b82f6", alpha=alpha,
                                 edgecolor="#1e3a8a", lw=0.6, zorder=5))
        ax.plot([p[0] for p in poses], [p[1] for p in poses], color="#ef4444", lw=1.4, zorder=6)
        ax.scatter([start[0]], [start[1]], color="#f59e0b", s=60, zorder=7, label="起点")
        ax.scatter([GOAL[0]], [GOAL[1]], color="#16a34a", marker="*", s=160, zorder=7, label="目标")
        ax.set_title("学习后的倒车入库（%d 个动作）\n%s" % (len(names), " → ".join(names[:6]) + (" …" if len(names) > 6 else "")),
                     fontsize=9.5)
        ax.legend(fontsize=8, loc="lower left")
    else:
        ax.set_title("未找到成功轨迹（60 次尝试）")

    # 成功率 - 精度曲线
    ax2 = axes[1]
    env = GarageEnv(seed=11)
    cases = 120
    data = []
    for _ in range(cases):
        env.reset()
        poses, names, info = run_policy(Q, env.S.copy())
        data.append((np.linalg.norm(poses[-1][:2] - GOAL[:2]),
                     np.rad2deg(abs(wrap_pi(poses[-1][2] - GOAL[2])))))
    data = np.array(data)
    for tol_d, tol_a, color, label in ((0.35, 12, "#16a34a", "严格 0.35m/12°"),
                                       (0.60, 20, "#3b82f6", "较宽 0.60m/20°"),
                                       (1.00, 30, "#f59e0b", "宽松 1.00m/30°")):
        ok = ((data[:, 0] < tol_d) & (data[:, 1] < tol_a)).mean()
        print("%s：成功率 %.0f%%" % (label, ok * 100))
        ax2.bar(len(ax2.patches), ok * 100, color=color, label=f"{label}: {ok*100:.0f}%")
    ax2.set_xticks([]); ax2.set_ylabel("成功率 (%)"); ax2.set_ylim(0, 100)
    ax2.set_title("学习后在不同精度下的入库成功率（120 例随机起点）", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=.25)
    ax2.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("garage_result.png")
    print("已保存 garage_result.png")
    if found:
        print("示例轨迹动作:", " → ".join(found[2]))
        print("终点误差 %.2f m / %.0f°" % (np.linalg.norm(found[1][-1][:2]-GOAL[:2]),
              np.rad2deg(abs(wrap_pi(found[1][-1][2]-GOAL[2])))))


if __name__ == "__main__":
    main()
