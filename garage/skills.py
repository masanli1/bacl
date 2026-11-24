"""技能库：学习 = 获得新技能，之后把技能当原材料复用

一条技能 = 一段动作序列 + 它的"适用姿态" + "效果"（姿态变化）
  适用姿态：这段技能在什么状态下好用（均值 + 允许偏差）
  效果：执行完姿态变了多少

学习来源（都算学）：
  · 从模仿中学：看一次成功的示范，把示范切成几段存成技能
  · 从组合中学：把已有技能首尾相接，得到新的长技能
  · 从传统中学：一开始给两条手工技能（前进绕行、倒车对齐）
""" 
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from garage_combo import ComboEnv
from garage_env import GOAL, wrap_pi


def pose_dist(a, b, w_ang=1.2):
    return float(np.linalg.norm(a[:2] - b[:2]) + w_ang * abs(wrap_pi(a[2] - b[2])))


@dataclass
class Skill:
    actions: tuple                 # 基本动作序列
    pre: np.ndarray                # 适用姿态 [x, y, theta]
    effect: np.ndarray             # 姿态变化 [dx, dy, dtheta]
    radius: float = 1.5            # 适用范围
    tries: int = 0
    wins: int = 0
    source: str = "imitate"        # imitate / compose / hand

    @property
    def value(self):
        return self.wins / max(self.tries, 1)

    def applies(self, S):
        return pose_dist(S, self.pre) < self.radius

    def predicted(self, S):
        """用技能效果预测执行后的姿态"""
        out = np.array(S, float).copy()
        out[0] += self.effect[0]; out[1] += self.effect[1]
        out[2] = wrap_pi(out[2] + self.effect[2])
        return out


class SkillLibrary:
    def __init__(self):
        self.skills: list[Skill] = []

    # ---------------- 学习 ----------------
    def add(self, actions, pre, effect, radius=1.5, source="imitate"):
        self.skills.append(Skill(tuple(actions), np.asarray(pre, float),
                                 np.asarray(effect, float), radius, source=source))

    def learn_from_demo(self, poses, actions, chunk=2):
        """从模仿中学：把示范切成 chunk 段，每段存一条技能"""
        i = 0
        while i < len(actions):
            j = min(i + chunk, len(actions))
            seg = actions[i:j]
            pre = poses[i][:3]
            endp = poses[j][:3]
            eff = np.array([endp[0]-pre[0], endp[1]-pre[1],
                            wrap_pi(endp[2]-pre[2])])
            self.add(seg, pre, eff, radius=2.2, source="imitate")
            i = j

    def compose(self, A: Skill, B: Skill):
        """从组合中学：两个技能接起来变成一条长技能"""
        pre = A.pre.copy()
        effA = A.effect
        preB_shifted = np.array([pre[0]+effA[0], pre[1]+effA[1],
                                 wrap_pi(pre[2]+effA[2])])
        gap = pose_dist(preB_shifted, B.pre, w_ang=0.8)
        if gap > 1.8:
            return None
        eff = np.array([effA[0] + B.effect[0], effA[1] + B.effect[1],
                        wrap_pi(effA[2] + B.effect[2])])
        s = Skill(A.actions + B.actions, pre, eff,
                  radius=max(A.radius, B.radius), source="compose")
        self.skills.append(s)
        return s

    # ---------------- 使用 ----------------
    def pick(self, S, goal=GOAL):
        """在库里挑一条对当前状态最有用的技能（不求解，只用学到的）"""
        d0 = np.linalg.norm(S[:2] - goal[:2])
        best, best_gain = None, 0.0
        for s in self.skills:
            if not s.applies(S):
                continue
            pred = s.predicted(S)
            d1 = np.linalg.norm(pred[:2] - goal[:2])
            gain = d0 - d1 + 0.3 * (abs(wrap_pi(S[2]-goal[2])) -
                                    abs(wrap_pi(pred[2]-goal[2])))
            gain *= (0.5 + s.value)          # 用得越成功越优先
            if gain > best_gain:
                best, best_gain = s, gain
        return best
