"""动作组合拼出入库（够到目标位，不用连续控制）

每个动作 = (方向, 转向, 弧长)：
  方向 ∈ {前进, 后退}；转向 ∈ {左打, 直行, 右打}；弧长 ∈ {0.3, 0.6, 1.0, 1.5, 2.0}
  → 2 × 3 × 5 = 30 个基本动作

求解：从目标位反向也可以，但这里用正向组合搜索——
      每一步在 30 个动作里选一个，看能不能把车"拼"到目标位（误差 ≤0.10 m / 3°）。
这仍然是"动作组合"，不是连续控制。
"""
from __future__ import annotations
import numpy as np
from garage_env import GarageEnv, GOAL, wrap_pi, MAX_STEER, MAX_SPEED, MAX_REV, WB


class ComboEnv(GarageEnv):
    """在基础环境上加"动作 = 方向 × 转向 × 弧长"的组合动作集"""

    STEERS = [-1.0, 0.0, 1.0]
    LENGTHS = [0.15, 0.3, 0.6, 1.0, 1.5, 2.0]
    DIRS = [+1, -1]

    N_ACT = 2 * 3 * len(LENGTHS)

    @staticmethod
    def decode(a):
        li = a % len(ComboEnv.LENGTHS)
        si = (a // len(ComboEnv.LENGTHS)) % len(ComboEnv.STEERS)
        di = a // (len(ComboEnv.LENGTHS) * len(ComboEnv.STEERS))
        return (ComboEnv.DIRS[di], ComboEnv.STEERS[si], ComboEnv.LENGTHS[li])

    @staticmethod
    def name(a):
        d, s, L = ComboEnv.decode(a)
        dn = "前进" if d > 0 else "后退"
        sn = {1.0: "左打", 0.0: "直行", -1.0: "右打"}[s]
        return f"{dn}{sn}{L:g}m"

    def step(self, a):
        d, s_frac, arc = self.decode(a)
        if not hasattr(self, 'margin'):
            self.margin = 0.06
        steer = s_frac * MAX_STEER
        v = MAX_SPEED if d > 0 else -MAX_REV
        S = self.S.copy()
        travelled = 0.0
        d0 = self.dist_to_goal(S)
        a0 = abs(wrap_pi(S[2] - GOAL[2]))
        while travelled < arc:
            S[3] = steer
            S[2] = wrap_pi(S[2] + (v / WB) * np.tan(S[3]) * self.dt)
            S[0] += v * np.cos(S[2]) * self.dt
            S[1] += v * np.sin(S[2]) * self.dt
            travelled += abs(v) * self.dt
            if not self.in_free(S):
                self.steps += 1
                self.S = S
                return self.obs(), -20.0, True, {"collision": True, "dist": self.dist_to_goal(S)}
        self.S = S
        self.steps += 1
        d1 = self.dist_to_goal(S)
        a1 = abs(wrap_pi(S[2] - GOAL[2]))
        reward = (d0 - d1) * 8.0 + (a0 - a1) * 3.0 - 0.02
        if self.at_goal(S):
            return self.obs(), reward + 30.0, True, {"success": True, "dist": d1}
        if self.steps >= self.max_steps:
            return self.obs(), reward, True, {"dist": d1}
        return self.obs(), reward, False, {"dist": d1}
