"""效果展示：3 个场景 × （加动作前 / 加动作后）"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from arm_env import ArmAvoidEnv, fk
import structured_search3 as S3
import structured_search2 as S2

font_manager.fontManager.addfont("/System/Library/Fonts/Hiragino Sans GB.ttc")
plt.rcParams["font.family"] = ["Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False


def dp_nodes(env, K=S3.K_LAYERS, seed=12345):
    q_start = env.q.copy()
    q_goal, ik = S2.ik_solve(env.target, q_start)
    cand = S2.free_candidates(env, seed=seed)
    layers = [np.array([q_start])] + [cand] * (K - 1) + [np.array([q_goal])]
    cost = np.zeros(1); parents = []
    for li in range(1, len(layers)):
        prev, cur = layers[li-1], layers[li]
        cc = np.full(len(cur), np.inf); par = np.full(len(cur), -1)
        ee_p = np.array([fk(q)[1] for q in prev]); ee_c = np.array([fk(q)[1] for q in cur])
        for j in range(len(cur)):
            for i in range(len(prev)):
                if not np.isfinite(cost[i]): continue
                c = cost[i] + float(np.linalg.norm(ee_c[j] - ee_p[i]))
                if c < cc[j]: cc[j] = c; par[j] = i
        if li < len(layers) - 1:
            keep = min(len(cur), max(24, len(cur)*3//4))
            order = np.argsort(cc)[:keep]
            layers[li] = cur[order]; cc = cc[order]; par = par[order]
        parents.append(par); cost = cc
    idx = int(np.argmin(cost)); path=[layers[-1][idx]]
    for li in range(len(layers)-1, 0, -1):
        idx = parents[li-1][idx]
        if idx < 0: break
        path.append(layers[li-1][idx])
    return np.array(path[::-1])


def draw(ax, env, nodes, segs, title, color, show_arm=True):
    for ob in env.obstacles:
        c, r = ob[:3], ob[3]
        u, v = np.mgrid[0:2*np.pi:26j, 0:np.pi:16j]
        x = c[0] + r*np.cos(u)*np.sin(v); y = c[1] + r*np.sin(u)*np.sin(v); z = c[2] + r*np.cos(v)
        ax.plot_surface(x, y, z, color="#64748b", alpha=.30, linewidth=0, shade=True)
    pts0, ee0 = fk(nodes[0]); pts1, ee1 = fk(nodes[-1])
    if show_arm:
        for i in range(len(pts0)-1):
            ax.plot(*zip(pts0[i], pts0[i+1]), color="#f59e0b", lw=3.2, alpha=.6)
    ax.scatter(*ee0, color="#f59e0b", s=55, label="起点", depthshade=False)
    ax.scatter(*ee1, color="#ef4444", s=90, marker="*", label="终点", depthshade=False)
    for q in nodes[1:-1]:
        ax.scatter(*fk(q)[1], color="#3b82f6", s=20, depthshade=False)
    for k, seg in enumerate(segs):
        ts = np.linspace(0, 1, 50)
        P = np.array([fk(seg.q(s))[1] for s in ts])
        ax.plot(P[:,0], P[:,1], P[:,2], color=color, lw=2.2, label="轨迹" if k == 0 else None)
        # 撞点（穿透）用红色 ✕ 标出
        hit = [fk(seg.q(s))[1] for s in ts if S3.penetration(seg.q(s), env.obstacles) > 0]
        if hit:
            H = np.array(hit)
            ax.scatter(H[:,0], H[:,1], H[:,2], marker="x", color="#b91c1c", s=55,
                       depthshade=False, label="碰撞点" if k == 0 else None)
        # 动作位置用绿色空心圈标出
        if seg.acts:
            A = np.array([fk(seg.q(t))[1] for (t, w, d) in seg.acts])
            ax.scatter(A[:,0], A[:,1], A[:,2], facecolors="none", edgecolors="#15803d",
                       s=90, linewidths=1.6, depthshade=False,
                       label="新增动作" if k == 0 else None)
    allp = np.concatenate([np.array([fk(q)[1] for q in nodes]),
                           np.array([ob[:3] for ob in env.obstacles])])
    c = allp.mean(0); rad = max(1.6, float(np.abs(allp - c).max()) + 0.35)
    ax.set_xlim(c[0]-rad, c[0]+rad); ax.set_ylim(c[1]-rad, c[1]+rad); ax.set_zlim(max(0, c[2]-rad), c[2]+rad)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7); ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=20, azim=-60)
    ax.legend(fontsize=6.5, loc="upper left")


def main():
    cases = []
    seed = 5
    tries = 0
    while len(cases) < 3 and tries < 40:
        tries += 1
        env = ArmAvoidEnv(seed=seed + tries)
        env.reset()
        nodes = dp_nodes(env)
        rng = np.random.default_rng(100 + tries)
        before, after, added = [], [], 0
        for i in range(len(nodes)-1):
            before.append(S3.ActionSegment(nodes[i], nodes[i+1]))
            sg = S3.ActionSegment(nodes[i], nodes[i+1])
            ok, n = S3.repair_with_actions(sg, env.obstacles, rng)
            added += n; after.append(sg)
        if added >= 1:                      # 只保留"确实需要加动作"的场景
            cases.append((env.obstacles.copy(), nodes, before, after, added))

    fig = plt.figure(figsize=(13.2, 8.6), dpi=125)
    for k, (obs, nodes, before, after, added) in enumerate(cases):
        e = ArmAvoidEnv(seed=5); e.obstacles = obs
        ax1 = fig.add_subplot(3, 2, 2*k+1, projection="3d")
        ax2 = fig.add_subplot(3, 2, 2*k+2, projection="3d")
        e.q = nodes[0]; e.target = fk(nodes[-1])[1]
        draw(ax1, e, nodes, before, f"场景 {k+1} · 加动作前（红=直线穿障碍）", "#ef4444")
        draw(ax2, e, nodes, after, f"场景 {k+1} · 加动作后（绿=绕开，节点仍 {len(nodes)} 个，追加 {added} 个动作）",
             "#22c55e", show_arm=False)
    fig.suptitle("三段式求解效果：固定位置（起点/终点）+ 固定数量（7 个节点）+ 不固定（每段动作）", fontsize=12)
    fig.tight_layout()
    fig.savefig("solve_demo.png")
    print("已保存 solve_demo.png")
    for k, (_, nodes, _, _, added) in enumerate(cases):
        print("场景 %d：节点 %d 个，追加动作 %d 个" % (k+1, len(nodes), added))


if __name__ == "__main__":
    main()
