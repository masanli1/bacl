"""在技能库上学策略：技能是原材料，学的是"什么状态用哪条技能"

状态：姿态离散化 (x, y, theta)
动作：技能库里的每一条技能（不是基本动作）
奖励：靠近目标 + 入库 + 碰撞罚
学习：Q-learning
"""
from __future__ import annotations
import json
import numpy as np
import garage_env as GE
import garage_combo as GC

DT, ARC, MAX_STEPS = 0.05, 1.0, 24
GE.GOAL = np.array([14.1, 15.0, np.pi / 2])
GC.GOAL = GE.GOAL
from garage_combo import ComboEnv
from combo_search import beam
from skills import SkillLibrary

BX, BY, BTH = 0.8, 0.8, np.deg2rad(15)


def key(S):
    return (int(np.round(S[0]/BX)), int(np.round(S[1]/BY)),
            int(np.round(GE.wrap_pi(S[2])/BTH)))


def build_library(n_demos=8, seed=7):
    lib = SkillLibrary()
    rng = np.random.default_rng(seed)
    starts = [[6.0, 7.5, 0.0, 0.0], [9.0, 8.0, 0.0, 0.0], [7.6, 6.2, -0.18, 0.0]]
    for _ in range(n_demos - 3):
        starts.append([float(rng.uniform(4, 12)), float(rng.uniform(6, 9.5)),
                       float(rng.uniform(-0.2, 0.2)), 0.0])
    n_ok = 0
    for st in starts:
        seq, S, ok = beam(st, depth=14, width=40, max_steps=20)
        if not ok:
            continue
        env = ComboEnv(arc_len=ARC, max_steps=len(seq)+1, dt=DT, seed=0)
        env.reset(start=np.array(st, float)); env.margin = 0.12
        poses = [env.S.copy()]
        for a in seq:
            env.step(a); poses.append(env.S.copy())
        lib.learn_from_demo(poses, seq, chunk=2)
        n_ok += 1
    return lib, n_ok


def train(lib, episodes=4000, seed=0, alpha=0.3, gamma=0.97, eps0=0.5):
    rng = np.random.default_rng(seed)
    env = ComboEnv(arc_len=ARC, max_steps=MAX_STEPS, dt=DT, seed=0)
    skills = lib.skills
    Q = {}
    hist = []
    for ep in range(episodes):
        eps = max(0.05, eps0 * (1 - ep/episodes))
        env.reset()
        ok = False
        for _ in range(MAX_STEPS):
            k = key(env.S)
            qa = Q.setdefault(k, np.zeros(len(skills)))
            # 只考虑"适用"的技能
            valid = [i for i, s in enumerate(skills) if s.applies(env.S)]
            if not valid:
                break
            if rng.random() < eps:
                a = int(valid[rng.integers(0, len(valid))])
            else:
                a = max(valid, key=lambda i: qa[i])
            S0 = env.S.copy()
            d0 = np.linalg.norm(S0[:2]-GE.GOAL[:2])
            done = False; coll = False
            for act in skills[a].actions:
                _, r, done, info = env.step(act)
                if info.get("collision"):
                    coll = True; break
                if done: break
            d1 = np.linalg.norm(env.S[:2]-GE.GOAL[:2])
            rew = (d0-d1)*8.0 - 0.05 - (10.0 if coll else 0.0)
            if info.get("success"): rew += 25.0; ok = True
            k2 = key(env.S)
            q2 = Q.setdefault(k2, np.zeros(len(skills)))
            tgt = rew if (done or coll) else rew + gamma*float(np.max(q2))
            qa[a] += alpha*(tgt - qa[a])
            if coll or done: break
        hist.append(1 if ok else 0)
    return Q, hist


def eval_policy(lib, Q, cases=80, seed=11):
    rng = np.random.default_rng(seed)
    env = ComboEnv(arc_len=ARC, max_steps=MAX_STEPS, dt=DT, seed=0)
    skills = lib.skills
    ok = 0
    for _ in range(cases):
        env.reset()
        for _ in range(MAX_STEPS):
            valid = [i for i, s in enumerate(skills) if s.applies(env.S)]
            if not valid: break
            qa = Q.get(key(env.S), np.zeros(len(skills)))
            a = max(valid, key=lambda i: qa[i])
            done = False; coll = False
            for act in skills[a].actions:
                _, r, done, info = env.step(act)
                if info.get("collision"): coll = True; break
                if done: break
            if coll or done: break
        if (np.linalg.norm(env.S[:2]-GE.GOAL[:2]) < 0.35 and
                abs(GE.wrap_pi(env.S[2]-GE.GOAL[2])) < np.deg2rad(12)):
            ok += 1
    return ok/cases


if __name__ == "__main__":
    print("① 模仿：建技能库 …")
    lib, n_ok = build_library()
    print("   %d 条示范 → 技能库 %d 条技能" % (n_ok, len(lib.skills)))
    print("② 在技能库上学选择策略 …")
    Q, hist = train(lib, episodes=4000)
    print("   训练末 500 轮成功率 %.1f%%" % (100*np.mean(hist[-500:])))
    print("③ 评估（80 例随机起点）…")
    acc = eval_policy(lib, Q)
    print("   成功率 %.0f%%（只用学到的技能，无任何几何求解）" % (acc*100))
    json.dump({"skills": len(lib.skills), "acc": acc,
               "train_recent": float(np.mean(hist[-500:]))},
              open("skill_policy_stats.json","w"), ensure_ascii=False, indent=1)
