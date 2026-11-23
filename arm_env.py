"""六自由度机械臂避障环境（numpy 版）

修正了原 MATLAB 版本的两个问题：
1. 原版 `CurrentQ = TargetQ` 直接跳到目标，**动作不影响状态**（MDP 退化，学不了）；
   这里动作是关节速度，状态逐步演化。
2. 原版 PPO 更新是空壳（返回原参数）；本包 ppo.py 是真实现。
"""
from __future__ import annotations

import numpy as np

# 连杆长度（与原版一致）
L = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.4])
BASE = np.array([0.0, 0.0, 0.0])
Q_HOME = np.array([0.0, -np.pi / 3, np.pi / 3, 0.0, np.pi / 2, 0.0])


def _trotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)


def _troty(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], float)


def _trans(x, y, z):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def fk(q: np.ndarray):
    """正运动学：返回 6 个关节点位置 + 末端位置。"""
    T = np.eye(4)
    T[:3, 3] = BASE
    pts = [T[:3, 3].copy()]
    T = T @ _trotz(q[0]) @ _trans(0, 0, L[0])
    pts.append(T[:3, 3].copy())
    T = T @ _troty(q[1]) @ _trans(L[1], 0, 0)
    pts.append(T[:3, 3].copy())
    T = T @ _troty(q[2]) @ _trans(L[2], 0, 0)
    pts.append(T[:3, 3].copy())
    T = T @ _trotz(q[3]) @ _trans(0, 0, L[3])
    pts.append(T[:3, 3].copy())
    T = T @ _troty(q[4]) @ _trans(L[4], 0, 0)
    pts.append(T[:3, 3].copy())
    T = T @ _trotz(q[5]) @ _trans(L[5], 0, 0)
    ee = T[:3, 3].copy()
    pts.append(ee)
    return np.array(pts), ee


def _seg_point_dist(a, b, p):
    ab = b - a
    denom = float(ab @ ab) + 1e-12
    t = np.clip(float((p - a) @ ab) / denom, 0.0, 1.0)
    return float(np.linalg.norm(a + t * ab - p))


class ArmAvoidEnv:
    """观测（10 维）：q(6) + 目标末端(3) + 最近障碍物距离(1)。
    动作（6 维）：关节增量，范围 [-1,1]，乘 0.15 rad。
    奖励：向目标靠近给正奖励，碰撞 -2 并终止，到达 +5 并终止，每步 -0.01 时间成本。
    """

    def __init__(self, obstacles=None, max_steps: int = 60, success_tol: float = 0.25, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        # 障碍物：球 (x,y,z,r)
        self.obstacles = np.array(obstacles, float) if obstacles is not None else None
        self.max_steps = max_steps
        self.success_tol = success_tol
        self.q = Q_HOME.copy()
        self.target = None
        self.steps = 0

    # ---------------------------------------------------------------- reset
    def reset(self):
        self.steps = 0
        self.q = np.clip(Q_HOME + self.rng.normal(0, 0.15, 6), -np.pi, np.pi)
        # 目标从"可达末端位置"里取：随机一组关节角，用它的正运动学结果当目标，
        # 这样目标一定可达（原版随机取空间点，常常超出可达范围）
        for _ in range(20):
            qs = np.clip(Q_HOME + self.rng.normal(0, 0.9, 6), -np.pi, np.pi)
            _, ee = fk(qs)
            if 0.8 < np.linalg.norm(ee - BASE) < 4.5:
                self.target = ee
                break
        else:
            self.target = fk(np.clip(Q_HOME + self.rng.normal(0, 0.6, 6), -np.pi, np.pi))[1]
        # 障碍物：① 不能压在起始姿态上 ② 不能把目标封死（可行性检查）
        for _ in range(60):
            cand = self._random_obstacles()
            self.obstacles = cand
            if self._min_dist() <= 0.25:
                continue                      # 压在机械臂身上 → 重采样
            if self.check_feasible():
                break
        return self._obs()

    def check_feasible(self, trials=3, iters=400):
        """粗判：起点到终点之间是否存在无碰撞通路（排除"障碍封死目标"的布局）。"""
        from structured_search2 import rrt_connect, ik_solve
        qg, ik = ik_solve(self.target, self.q)
        if ik >= 0.05:
            return False
        for sd in range(trials):
            if rrt_connect(self.q, qg, self.obstacles, max_iter=iters, step=0.3, seed=sd * 31 + 5) is not None:
                return True
        return False

    def _random_obstacles(self):
        """在起点末端与目标之间放 1–2 个球（挡在路径上，但不压在起始姿态上）。"""
        _, ee0 = fk(self.q)
        n = int(self.rng.integers(1, 3))
        out = []
        for _ in range(n):
            t = self.rng.uniform(0.35, 0.75)
            c = ee0 * (1 - t) + self.target * t
            c = c + self.rng.normal(0, 0.18, 3)
            out.append([*c, self.rng.uniform(0.15, 0.25)])
        return np.array(out, float)

    # ------------------------------------------------------------- helpers
    def _min_dist(self):
        pts, _ = fk(self.q)
        best = np.inf
        for ob in self.obstacles:
            c, r = ob[:3], ob[3]
            for i in range(len(pts) - 1):
                d = _seg_point_dist(pts[i], pts[i + 1], c) - r
                best = min(best, d)
        return float(best)

    def _obs(self):
        _, ee = fk(self.q)
        return np.concatenate([self.q, self.target, [self._min_dist()]]).astype(np.float32)

    # ---------------------------------------------------------------- step
    def step(self, action):
        a = np.clip(np.asarray(action, float).reshape(6), -1.0, 1.0)
        _, ee0 = fk(self.q)
        d0 = float(np.linalg.norm(ee0 - self.target))

        self.q = np.clip(self.q + 0.15 * a, -np.pi, np.pi)     # 状态真的变了
        self.steps += 1

        _, ee1 = fk(self.q)
        d1 = float(np.linalg.norm(ee1 - self.target))
        md = self._min_dist()

        reward = 2.0 * (d0 - d1) - 0.01                        # 靠近给奖励
        done = False
        info = {"dist": d1, "min_dist": md}
        if md < 0.0:
            reward -= 2.0
            done = True
            info["collision"] = True
        elif d1 < self.success_tol:
            reward += 5.0
            done = True
            info["success"] = True
        if self.steps >= self.max_steps:
            done = True
        return self._obs(), float(reward), bool(done), info
