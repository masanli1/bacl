"""动画：截肢学习过程（每轮切一段 → 重接 → 变短）"""
from __future__ import annotations
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image as PILImage

from arm_env import fk
from shorten import run, dense_len

font_manager.fontManager.addfont("/System/Library/Fonts/Hiragino Sans GB.ttc")
plt.rcParams["font.family"] = ["Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False


def path_pts(segs, n=26):
    P = []
    for s in segs:
        for t in np.linspace(0, 1, n, endpoint=False):
            P.append(fk(s.q(t))[1])
    P.append(fk(segs[-1].q(1.0))[1])
    return np.array(P)


def frame(env, nodes, segs, title, trail=None, cut=None, color="#22c55e", tick=0):
    fig = plt.figure(figsize=(5.4, 4.4), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    for ob in env.obstacles:
        c, r = ob[:3], ob[3]
        u, v = np.mgrid[0:2*np.pi:22j, 0:np.pi:14j]
        ax.plot_surface(c[0]+r*np.cos(u)*np.sin(v), c[1]+r*np.sin(u)*np.sin(v),
                        c[2]+r*np.cos(v), color="#64748b", alpha=.30, linewidth=0)
    P = path_pts(segs)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color=color, lw=2.3, label="轨迹")
    if cut is not None:
        C = path_pts([cut], 40)
        ax.plot(C[:, 0], C[:, 1], C[:, 2], color="#ef4444", lw=3.4, ls="--", label="要切掉的段")
    for i, q in enumerate(nodes):
        ax.scatter(*fk(q)[1], color="#3b82f6", s=26, depthshade=False)
    ax.scatter(*fk(nodes[0])[1], color="#f59e0b", s=55, depthshade=False, label="起点")
    ax.scatter(*fk(nodes[-1])[1], color="#ef4444", s=95, marker="*", depthshade=False, label="终点")
    L = sum(dense_len(s) for s in segs)
    ax.set_title(f"{title}\n节点 {len(nodes)} 个 · 长度 {L:.3f} m   [帧 {tick}]", fontsize=9)
    ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7); ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6); ax.view_init(elev=20, azim=-58)
    ax.legend(fontsize=6.5, loc="upper left")
    allp = np.vstack([P, np.array([ob[:3] for ob in env.obstacles])])
    c0 = allp.mean(0); rad = max(1.3, float(np.abs(allp-c0).max())+0.25)
    ax.set_xlim(c0[0]-rad, c0[0]+rad); ax.set_ylim(c0[1]-rad, c0[1]+rad); ax.set_zlim(0, c0[2]+rad)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).convert("P", palette=PILImage.ADAPTIVE, colors=128)


def main():
    env, nodes0, segs0, nodes_f, segs_f, lens_h, acts_h, history = run(
        seed=5, verbose=False, return_history=True)
    frames = []
    # 初始
    tick = 0
    for _ in range(6):
        tick += 1; frames.append(frame(env, nodes0, segs0, "初始解（三段式）",
                                       color="#ef4444", tick=tick))
    # 每轮：先高亮要切的段，再展示切完的结果
    for k in range(1, len(history)):
        prev_nodes, prev_segs, prev_len = history[k-1]
        cur_nodes, cur_segs, cur_len = history[k]
        # 找出被删掉的节点（prev 有而 cur 没有）
        cut = None
        for i in range(1, len(prev_nodes)-1):
            if not any(np.allclose(prev_nodes[i], q, atol=1e-9) for q in cur_nodes):
                cut = prev_segs[i-1] if i-1 < len(prev_segs) else None
                break
        for _ in range(8):
            tick += 1; frames.append(frame(env, prev_nodes, prev_segs,
                                           f"第 {k} 轮：找到可切的段", cut=cut,
                                           color="#3b82f6", tick=tick))
        for _ in range(8):
            tick += 1; frames.append(frame(env, cur_nodes, cur_segs,
                                           f"第 {k} 轮：切完重接 → 更短",
                                           color="#22c55e", tick=tick))
    for _ in range(10):
        tick += 1; frames.append(frame(env, nodes_f, segs_f, "收敛：不再有可切的段",
                                       color="#22c55e", tick=tick))

    durations = [110] * len(frames)
    frames[0].save("shorten_anim.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False, disposal=2)
    print("已保存 shorten_anim.gif，%d 帧" % len(frames))
    print("长度序列:", [round(x, 3) for x in lens_h])


if __name__ == "__main__":
    main()
