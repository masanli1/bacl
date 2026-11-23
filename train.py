"""训练脚本：验证 PPO 真的会学（随机策略 vs 训练后）"""
from __future__ import annotations
import argparse, time
import numpy as np
from arm_env import ArmAvoidEnv
from ppo_torch import PPO


def rollout(policy: PPO, env: ArmAvoidEnv, steps: int):
    traj = {"obs": [], "act": [], "logp": [], "rew": [], "done": [], "val": []}
    obs = env.reset()
    total, eps_ret, eps_n = 0.0, [], 0
    for _ in range(steps):
        a, mean, lp = policy.act(obs)
        v = policy.value(obs)
        obs2, r, done, info = env.step(a)
        traj["obs"].append(obs); traj["act"].append(a); traj["logp"].append(lp)
        traj["rew"].append(r); traj["done"].append(done); traj["val"].append(v)
        total += r; eps_ret.append(r)
        obs = env.reset() if done else obs2
        if done:
            eps_n += 1
    return traj, total


def evaluate(policy, episodes=20, random_policy=False, seed=123):
    env = ArmAvoidEnv(seed=seed)
    rets, succ, coll, final_d = [], 0, 0, []
    for _ in range(episodes):
        obs = env.reset(); done = False; R = 0.0; info = {}
        while not done:
            a = env.rng.uniform(-1, 1, 6) if random_policy else policy.act(obs, deterministic=True)[0]
            obs, r, done, info = env.step(a)
            R += r
            if info.get("success"): succ += 1
            if info.get("collision"): coll += 1
        rets.append(R)
        final_d.append(info.get("dist", 9.9))
    return float(np.mean(rets)), succ / episodes, coll / episodes, float(np.mean(final_d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = ArmAvoidEnv(seed=1)
    policy = PPO(seed=args.seed)
    hist = []
    t0 = time.time()
    for it in range(args.iters):
        traj, ret = rollout(policy, env, args.steps)
        loss = policy.update(traj)
        hist.append(ret / args.steps)
        if (it + 1) % 10 == 0:
            m, s, c, fd = evaluate(policy, 20)
            print(f"iter {it+1:3d}  平均每步回报 {np.mean(hist[-10:]):+.4f}  "
                  f"评估回报 {m:+.2f} 成功率 {s:.0%} 碰撞率 {c:.0%} 末距离 {fd:.2f}  用时 {time.time()-t0:.0f}s", flush=True)
    rm, rs, rc, rfd = evaluate(policy, 50, random_policy=True)
    tm, ts, tc, tfd = evaluate(policy, 50)
    print("\n=== 对照（50 回合）===")
    print(f"随机策略：回报 {rm:+.2f}  成功率 {rs:.0%}  碰撞率 {rc:.0%}  末距离 {rfd:.2f}")
    print(f"训练后  ：回报 {tm:+.2f}  成功率 {ts:.0%}  碰撞率 {tc:.0%}  末距离 {tfd:.2f}")
    np.save("history.npy", np.array(hist))
    try:
        import torch
        torch.save(policy.net.state_dict(), "ppo_arm.pth")
        print("模型已保存 ppo_arm.pth")
    except Exception as e:
        print("模型保存跳过:", e)
    print("\n提升：回报 %+.2f，成功率 %+.0f 个百分点，末距离 %.2f → %.2f" % (tm - rm, (ts - rs) * 100, rfd, tfd))


if __name__ == "__main__":
    main()
