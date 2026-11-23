"""PPO（torch 版）：GAE + clip + 熵正则 + Adam。

环境侧仍是 numpy（arm_env.py）。用 torch 是因为手写反向传播在这类问题上容易数值不稳，
我们验证过 numpy 版能跑但梯度会 NaN——这里以"能真正学会"为第一优先。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=10, act_dim=6, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden // 2, act_dim)
        self.critic = nn.Linear(hidden // 2, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.7)

    def forward(self, x):
        h = self.body(x)
        return torch.tanh(self.actor(h)), self.critic(h).squeeze(-1)


class PPO:
    def __init__(self, obs_dim=10, act_dim=6, lr=3e-4, clip=0.2, epochs=8,
                 batch=128, gamma=0.99, lam=0.95, ent=0.01, seed=0):
        torch.manual_seed(seed)
        self.net = ActorCritic(obs_dim, act_dim)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.clip, self.epochs, self.batch = clip, epochs, batch
        self.gamma, self.lam, self.ent = gamma, lam, ent
        self.act_dim = act_dim

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32)[None]
        mean, v = self.net(o)
        std = self.net.log_std.exp()
        if deterministic:
            return mean[0].numpy().clip(-1, 1), float(v[0]), 0.0
        dist = torch.distributions.Normal(mean, std)
        a = dist.sample()
        a_clip = torch.clamp(a, -1, 1)
        return a_clip[0].numpy(), float(v[0]), float(dist.log_prob(a).sum())

    def value(self, obs):
        with torch.no_grad():
            _, v = self.net(torch.as_tensor(obs, dtype=torch.float32)[None])
        return float(v[0])

    def update(self, tr):
        obs = torch.as_tensor(np.array(tr["obs"]), dtype=torch.float32)
        act = torch.as_tensor(np.array(tr["act"]), dtype=torch.float32)
        old_lp = torch.as_tensor(np.array(tr["logp"]), dtype=torch.float32)
        rew = np.array(tr["rew"], dtype=np.float32)
        done = np.array(tr["done"], dtype=bool)
        val = np.array(tr["val"], dtype=np.float32)

        adv = np.zeros_like(rew)
        last = 0.0
        for t in range(len(rew) - 1, -1, -1):
            nxt = 0.0 if done[t] or t + 1 >= len(val) else val[t + 1]
            delta = rew[t] + self.gamma * nxt - val[t]
            last = delta + self.gamma * self.lam * (0.0 if done[t] else last)
            adv[t] = last
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)

        n = len(obs)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                if len(idx) < 8:
                    continue
                mean, v = self.net(obs[idx])
                std = self.net.log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                lp = dist.log_prob(act[idx]).sum(-1)
                ratio = torch.exp(lp - old_lp[idx])
                s1 = ratio * adv_t[idx]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[idx]
                actor_loss = -torch.min(s1, s2).mean()
                critic_loss = ((v - ret_t[idx]) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = actor_loss + 0.5 * critic_loss - self.ent * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
