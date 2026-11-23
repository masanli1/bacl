"""numpy 版 PPO（真实现：GAE + clip 裁剪 + Adam 更新）

对齐原 MATLAB 版的设计（actor 输出 6 维动作均值、critic 输出价值、clip_epsilon、
多 epoch 多 batch 更新），但**梯度是真的**——手写反向传播，参数每步都变。
"""
from __future__ import annotations

import numpy as np


def orthogonal(shape, gain=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    a = rng.normal(0, 1, shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    return (u if u.shape == shape else v) * gain


class MLP:
    def __init__(self, sizes, out_act=None, rng=None, out_gain=0.01):
        rng = rng or np.random.default_rng(0)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            gain = np.sqrt(2.0) if i < len(sizes) - 2 else out_gain
            self.W.append(orthogonal((sizes[i], sizes[i + 1]), gain, rng))
            self.b.append(np.zeros(sizes[i + 1]))
        self.out_act = out_act

    def forward(self, x):
        self._cache = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(z)
            else:
                h = z if self.out_act is None else self.out_act(z)
            self._cache.append(h)
        return h

    def backward(self, dout):
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in range(len(self.W) - 1, -1, -1):
            h_in = self._cache[i]
            if i < len(self.W) - 1:
                d = d * (1 - self._cache[i + 1] ** 2)     # tanh 导数
            grads_W[i] = h_in.T @ d
            grads_b[i] = d.sum(0)
            d = d @ self.W[i].T
        return grads_W, grads_b

    def params(self):
        return self.W + self.b


class Adam:
    def __init__(self, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class PPO:
    def __init__(self, obs_dim=10, act_dim=6, hidden=(128, 64), lr=3e-4,
                 clip=0.2, epochs=6, batch=64, gamma=0.99, lam=0.95,
                 std=0.5, seed=0):
        rng = np.random.default_rng(seed)
        self.actor = MLP([obs_dim, *hidden, act_dim], out_act=np.tanh, rng=rng)
        self.critic = MLP([obs_dim, *hidden, 1], rng=rng)
        self.log_std = np.full(act_dim, np.log(std))
        self.opt_a, self.opt_c = Adam(lr), Adam(lr)
        self.clip, self.epochs, self.batch = clip, epochs, batch
        self.gamma, self.lam = gamma, lam

    # ------------------------------------------------------------------
    def act(self, obs, deterministic=False):
        mean = self.actor.forward(obs[None, :])[0]
        if deterministic:
            return np.clip(mean, -1, 1), mean, 0.0
        std = np.exp(self.log_std)
        a = mean + std * np.random.randn(*mean.shape)
        a = np.clip(a, -1, 1)
        logp = self._gauss_logp(a, mean, std)
        return a, mean, logp

    @staticmethod
    def _gauss_logp(a, mean, std):
        return float(np.sum(-0.5 * ((a - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)))

    # ------------------------------------------------------------------
    def update(self, traj):
        obs = np.array(traj["obs"], np.float32)
        act = np.array(traj["act"], np.float32)
        old_logp = np.array(traj["logp"], np.float32)
        rew = np.array(traj["rew"], np.float32)
        done = np.array(traj["done"], bool)
        val = np.array(traj["val"], np.float32)

        # GAE
        adv = np.zeros_like(rew)
        last = 0.0
        for t in range(len(rew) - 1, -1, -1):
            nxt = 0.0 if done[t] else (val[t + 1] if t + 1 < len(val) else 0.0)
            delta = rew[t] + self.gamma * nxt - val[t]
            last = delta + self.gamma * self.lam * (0.0 if done[t] else last)
            adv[t] = last
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        std = np.exp(self.log_std)
        losses = []
        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, self.batch):
                b = idx[s:s + self.batch]
                if len(b) < 4:
                    continue
                x, a, lp_old, ad, rt = obs[b], act[b], old_logp[b], adv[b], ret[b]

                mean = self.actor.forward(x)
                v = self.critic.forward(x)[:, 0]
                lp_new = self._gauss_logp_batch(a, mean, std)
                ratio = np.exp(lp_new - lp_old)

                # 裁剪目标
                surr1 = ratio * ad
                surr2 = np.clip(ratio, 1 - self.clip, 1 + self.clip) * ad
                use_min = surr1 <= surr2
                actor_loss = -np.mean(np.minimum(surr1, surr2))
                losses.append(float(actor_loss))

                # d(loss)/d(mean)
                coef = np.where(use_min, ratio * ad, 0.0)     # ∂surr/∂ratio · ∂ratio/∂lp
                d_lp = -coef / len(b)                          # ∂loss/∂lp
                # ∂lp/∂mean = (a-mean)/std²；输出层是 tanh，还要乘 (1-mean²)
                d_mean = ((a - mean) / (std ** 2) * d_lp[:, None]) * (1.0 - mean ** 2)
                gW, gb = self.actor.backward(d_mean)
                self.opt_a.step(self.actor.W + self.actor.b, gW + gb)

                # critic
                d_v = ((v - rt) * 2.0 / len(b))[:, None]
                gWc, gbc = self.critic.backward(d_v)
                self.opt_c.step(self.critic.W + self.critic.b, gWc + gbc)

        return float(np.mean(losses)) if losses else 0.0

    @staticmethod
    def _gauss_logp_batch(a, mean, std):
        return np.sum(-0.5 * ((a - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi), axis=1)

    def value(self, obs):
        return float(self.critic.forward(obs[None, :])[0, 0])
