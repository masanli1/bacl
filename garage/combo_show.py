"""画出组合动作拼出的入库路径"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Polygon
from garage_combo import ComboEnv
from garage_env import (GOAL, SLOT_X0, SLOT_W, SLOT_DIV, N_SLOT, AISLE_Y, SLOT_BACK,
                        WID, REAR_OH, FRONT_OH, TX0, TX1, wrap_pi)
from combo_search import beam

font_manager.fontManager.addfont("/System/Library/Fonts/Hiragino Sans GB.ttc")
plt.rcParams["font.family"] = ["Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False


def car_poly(S):
    x, y, th = S[0], S[1], S[2]
    fx, fy, lx, ly = np.cos(th), np.sin(th), -np.sin(th), np.cos(th)
    return np.array([(x + dx*fx + dy*lx, y + dx*fy + dy*ly)
                     for dx, dy in ((FRONT_OH, WID/2), (FRONT_OH, -WID/2),
                                    (-REAR_OH, -WID/2), (-REAR_OH, WID/2))])


def main():
    starts = [[6.0, 7.5, 0.0, 0.0], [9.0, 8.0, 0.0, 0.0], [7.676, 6.181, -0.18, 0.0]]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=125)
    for ax, st in zip(axes, starts):
        seq, S, ok = beam(st)
        # 重放并记录姿态
        env = ComboEnv(arc_len=1.0, max_steps=len(seq)+1, dt=0.05, seed=0)
        env.reset(start=np.array(st, float)); env.margin = 0.12
        poses = []
        for a in seq:
            for _ in range(6):                     # 段内插值，画得平滑
                env.step(a); poses.append(env.S.copy())
        env.reset(start=np.array(st, float)); poses = [env.S.copy()]
        for a in seq:
            env.step(a); poses.append(env.S.copy())

        for i in range(N_SLOT):
            x0 = SLOT_X0 + i*(SLOT_W + SLOT_DIV)
            tc = "#dcfce7" if i + 1 == 4 else "#e2e8f0"
            ax.add_patch(Rectangle((x0, AISLE_Y), SLOT_W, SLOT_BACK - AISLE_Y,
                                   facecolor=tc, edgecolor="#94a3b8", lw=0.7, zorder=1))
            if i + 1 != 4:
                cy = AISLE_Y + 0.15 + 2.2
                ax.add_patch(Rectangle((x0 + 0.5, cy - 2.2), 1.8, 4.4,
                                       facecolor="#cbd5e1", edgecolor="#64748b", lw=0.6, zorder=2))
        ax.plot([0, 30], [AISLE_Y, AISLE_Y], color="#475569", lw=1.0, zorder=3)
        for i, S in enumerate(poses):
            alpha = 0.12 + 0.55 * i / max(len(poses)-1, 1)
            ax.add_patch(Polygon(car_poly(S), closed=True, facecolor="#3b82f6",
                                 alpha=alpha, edgecolor="#1e3a8a", lw=0.5, zorder=5))
        ax.plot([p[0] for p in poses], [p[1] for p in poses], color="#ef4444", lw=1.3, zorder=6)
        ax.scatter([st[0]], [st[1]], color="#f59e0b", s=55, zorder=7)
        ax.scatter([GOAL[0]], [GOAL[1]], color="#16a34a", marker="*", s=150, zorder=7)
        err_d = np.linalg.norm(poses[-1][:2] - GOAL[:2])
        err_a = np.rad2deg(abs(wrap_pi(poses[-1][2] - GOAL[2])))
        ax.set_title("起点 (%.1f, %.1f) · %d 个动作\n终点误差 %.2f m / %.1f°"
                     % (st[0], st[1], len(seq), err_d, err_a), fontsize=9)
        ax.set_xlim(0, 30); ax.set_ylim(0, 17); ax.set_aspect("equal")
        ax.grid(alpha=.12); ax.tick_params(labelsize=7)
    fig.suptitle("动作组合拼出入库：每个动作 = 方向 × 转向 × 弧长（够到目标位）", fontsize=11)
    fig.tight_layout(); fig.savefig("combo_result.png")
    print("已保存 combo_result.png")


if __name__ == "__main__":
    main()
