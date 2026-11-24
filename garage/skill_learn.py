"""技能库学习：模仿 → 存技能 → 组合 → 复用（越用越强）

阶段一（模仿）：老师示范几条成功轨迹（这里用搜索当老师）
阶段二（自学）：新起点上只用库里的技能；成功就把新轨迹切段入库，
              再把库里两条技能组合成更长的技能
阶段三（评估）：在没见过的起点上测成功率
"""
from __future__ import annotations
import json
import numpy as np
import garage_env as GE
import garage_combo as GC
from skills import SkillLibrary, pose_dist

DT, ARC, MAX_STEPS = 0.05, 1.0, 24

# 本演示用"车头朝里"的目标（搜索在这个目标下能解出示范）
GE.GOAL = np.array([14.1, 15.0, np.pi / 2])
GC.GOAL = GE.GOAL

from garage_combo import ComboEnv
from combo_search import beam


def run_episode(lib, start, rng, explore=0.2):
    env = ComboEnv(arc_len=ARC, max_steps=MAX_STEPS, dt=DT, seed=0)
    env.reset(start=np.array(start, float)); env.margin = 0.12
    poses, acts = [env.S.copy()], []
    for _ in range(MAX_STEPS):
        sk = None if rng.random() < explore else lib.pick(env.S, goal=GE.GOAL)
        seq = (int(rng.integers(0, ComboEnv.N_ACT)),) if sk is None else sk.actions
        for a in seq:
            _, r, done, info = env.step(a)
            if info.get("collision"):
                if sk: sk.tries += 1
                return poses, acts, False, True
            poses.append(env.S.copy()); acts.append(a)
            if done:
                break
        if sk: sk.tries += 1
        if info.get("success"):
            if sk: sk.wins += 1
            return poses, acts, True, False
        if done:
            break
    return poses, acts, False, False


def eval_policy(lib, starts):
    ok = 0
    for st in starts:
        env = ComboEnv(arc_len=ARC, max_steps=MAX_STEPS, dt=DT, seed=0)
        env.reset(start=np.array(st, float)); env.margin = 0.06
        done = False
        for _ in range(MAX_STEPS):
            sk = lib.pick(env.S, goal=GE.GOAL)
            if sk is None:
                break
            for a in sk.actions:
                _, r, done, info = env.step(a)
                if info.get("collision") or done:
                    break
            if info.get("collision") or done:
                break
        if (np.linalg.norm(env.S[:2]-GE.GOAL[:2]) < 0.35 and
                abs(GE.wrap_pi(env.S[2]-GE.GOAL[2])) < np.deg2rad(12)):
            ok += 1
    return ok / len(starts)


def main(rounds=300, seed=0):
    rng = np.random.default_rng(seed)
    lib = SkillLibrary()

    # ---------- 阶段一：模仿 ----------
    print("① 模仿阶段：老师示范 …")
    rng0 = np.random.default_rng(7)
    teachers = [[6.0, 7.5, 0.0, 0.0], [9.0, 8.0, 0.0, 0.0], [7.6, 6.2, -0.18, 0.0]]
    for _ in range(5):
        teachers.append([float(rng0.uniform(4, 12)), float(rng0.uniform(6, 9.5)),
                         float(rng0.uniform(-0.2, 0.2)), 0.0])
    n_demo = 0
    for st in teachers:
        seq, S, ok = beam(st, depth=14, width=40, max_steps=20)
        if not ok:
            continue
        env = ComboEnv(arc_len=ARC, max_steps=len(seq)+1, dt=DT, seed=0)
        env.reset(start=np.array(st, float)); env.margin = 0.12
        poses = [env.S.copy()]
        for a in seq:
            env.step(a); poses.append(env.S.copy())
        lib.learn_from_demo(poses, seq, chunk=2)
        n_demo += 1
    print("   学到了 %d 条示范 → 技能库 %d 条" % (n_demo, len(lib.skills)))

    # ---------- 阶段二：自学 ----------
    starts = [[float(rng.uniform(3.5, 13)), float(rng.uniform(6, 9.5)),
               float(rng.uniform(-0.2, 0.2)), 0.0] for _ in range(rounds + 60)]
    train, holdout = starts[:rounds], starts[rounds:]
    curve = []
    for it in range(rounds):
        poses, acts, ok, coll = run_episode(lib, train[it], rng)
        if ok:
            lib.learn_from_demo(poses, acts, chunk=2)      # 从新成功中再学
            for _ in range(3):
                if len(lib.skills) > 6:
                    a = lib.skills[int(rng.integers(0, len(lib.skills) // 2))]
                    b = lib.skills[int(rng.integers(0, len(lib.skills) // 2))]
                    lib.compose(a, b)                      # 从组合中学
        if (it + 1) % 30 == 0:
            acc = eval_policy(lib, holdout)
            curve.append(acc)
            print("   第 %3d 轮：技能库 %3d 条，留出集成功率 %2.0f%%"
                  % (it + 1, len(lib.skills), acc * 100), flush=True)
    src = {s: sum(1 for k in lib.skills if k.source == s) for s in ("hand", "imitate", "compose")}
    print("\n技能来源:", src)
    json.dump({"skills": len(lib.skills), "curve": [round(float(c), 3) for c in curve],
               "sources": src}, open("skill_stats.json", "w"), ensure_ascii=False, indent=1)
    return lib, curve


if __name__ == "__main__":
    main()
