"""倒车入库环境（移植自 MATLAB AutoParking 的模型）

后轴自行车模型：状态 [x, y, theta, delta]，dt=0.02
车位：8 个垂直车位，开口 y=11.0，进深 5.2，宽 2.8
目标：停进 4 号车位，车头朝向 +y，车尾距后墙 0.30 m
"""
from __future__ import annotations
import numpy as np

# ---- 车辆 ----
LEN, WID, WB = 4.50, 1.85, 2.70
REAR_OH, FRONT_OH = 0.90, 0.90
MAX_STEER = np.deg2rad(35)
MAX_SPEED, MAX_REV = 1.00, 0.40

# ---- 车库 ----
AISLE_Y = 11.0
SLOT_DEPTH, SLOT_W, SLOT_DIV = 5.2, 2.8, 0.10
SLOT_X0, N_SLOT = 4.0, 8
SLOT_BACK = AISLE_Y + SLOT_DEPTH

TARGET = 4
TX0 = SLOT_X0 + (TARGET - 1) * (SLOT_W + SLOT_DIV)
TX1 = TX0 + SLOT_W
GOAL = np.array([(TX0 + TX1) / 2, SLOT_BACK - 0.30 - REAR_OH, np.pi / 2])


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GarageEnv:
    """动作 = 一个基本动作（方向 × 转向），执行固定弧长。"""

    MANEUVERS = [
        ("前进左打", +1, +1.0), ("前进直行", +1, 0.0), ("前进右打", +1, -1.0),
        ("后退左打", -1, +1.0), ("后退直行", -1, 0.0), ("后退右打", -1, -1.0),
    ]

    def __init__(self, arc_len=1.6, max_steps=24, dt=0.02, seed=0, fine_dist=0.0, fine_arc=0.4):
        self.arc_len, self.max_steps, self.dt = arc_len, max_steps, dt
        self.fine_dist, self.fine_arc = fine_dist, fine_arc
        self.rng = np.random.default_rng(seed)
        self.S = None
        self.steps = 0

    # ---------------- 几何 ----------------
    def corners(self, S=None):
        S = self.S if S is None else S
        x, y, th = S[0], S[1], S[2]
        fx, fy = np.cos(th), np.sin(th)
        lx, ly = -fy, fx                      # 左向量
        pts = []
        for dx, dy in ((FRONT_OH, WID/2), (FRONT_OH, -WID/2),
                       (-REAR_OH, -WID/2), (-REAR_OH, WID/2)):
            px = x + dx * fx + dy * lx
            py = y + dx * fy + dy * ly
            pts.append((px, py))
        return np.array(pts)

    def in_free(self, S=None, margin=0.06):
        """车是否完全在可行驶区域（通道 + 目标车位）。"""
        pts = self.corners(S)
        for px, py in pts:
            in_aisle = (0.3 + margin <= px <= 29.7 - margin and 0.3 + margin <= py <= AISLE_Y + 0.05)
            in_slot = (TX0 + margin <= px <= TX1 - margin and AISLE_Y - 0.05 <= py <= SLOT_BACK - margin)
            if not (in_aisle or in_slot):
                return False
        return True

    def dist_to_goal(self, S=None):
        S = self.S if S is None else S
        return float(np.linalg.norm(S[:2] - GOAL[:2]))

    def at_goal(self, S=None):
        S = self.S if S is None else S
        return (np.linalg.norm(S[:2] - GOAL[:2]) < 0.35 and
                abs(wrap_pi(S[2] - GOAL[2])) < np.deg2rad(12))

    # ---------------- 环境 ----------------
    def reset(self, start=None):
        if start is None:
            # 通道里随机起步（在车位前方一带）
            x = self.rng.uniform(3.0, 12.0)
            y = self.rng.uniform(6.0, 9.5)
            th = self.rng.uniform(-0.3, 0.3)
            start = np.array([x, y, th, 0.0])
        self.S = np.array(start, float)
        self.steps = 0
        return self.obs()

    def obs(self):
        S = self.S
        return np.array([S[0], S[1], np.sin(S[2]), np.cos(S[2]),
                         GOAL[0] - S[0], GOAL[1] - S[1]], float)

    def step(self, a_idx):
        """执行第 a 个基本动作，返回 (obs, reward, done, info)"""
        name, direction, steer_frac = self.MANEUVERS[a_idx]
        steer = steer_frac * MAX_STEER
        v = (MAX_SPEED if direction > 0 else -MAX_REV)
        # 动态弧长：离目标近时用短弧，保证入库精度
        arc = self.fine_arc if self.dist_to_goal() < self.fine_dist else self.arc_len
        S = self.S.copy()
        travelled = 0.0
        d0 = self.dist_to_goal(S)
        a0 = abs(wrap_pi(S[2] - GOAL[2]))
        while travelled < arc:
            dmax = MAX_STEER * self.dt * 40.0 / 40.0
            S[3] = steer
            S[2] = wrap_pi(S[2] + (v / WB) * np.tan(S[3]) * self.dt)
            S[0] += v * np.cos(S[2]) * self.dt
            S[1] += v * np.sin(S[2]) * self.dt
            travelled += abs(v) * self.dt
            if not self.in_free(S):
                self.steps += 1
                info = {"collision": True, "dist": self.dist_to_goal(S), "name": name}
                self.S = S
                return self.obs(), -20.0, True, info
        self.S = S
        self.steps += 1
        d1 = self.dist_to_goal(S)
        a1 = abs(wrap_pi(S[2] - GOAL[2]))
        reward = (d0 - d1) * 8.0 + (a0 - a1) * 3.0 - 0.05
        done = False
        info = {"dist": d1, "name": name}
        if self.at_goal(S):
            reward += 30.0
            done = True
            info["success"] = True
            info["dist"] = d1
            return self.obs(), reward, True, info
        if self.steps >= self.max_steps:
            done = True
        return self.obs(), reward, done, info
