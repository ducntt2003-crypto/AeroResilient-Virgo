#!/usr/bin/env python3
"""
Virgo_RL: PPO (Proximal Policy Optimization) Training for 12-DOF MorphQuad
Features:
- Multi-environment parallel rollout across CPU cores
- PyTorch GPU-accelerated PPO with Generalized Advantage Estimation (GAE)
- Decoupled 6-DOF tracking (arbitrary 3D orientation & position)
- Null-Space Gimbal Lock and Downwash Avoidance
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from virgo_env import VirgoEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Shared or decoupled feature representation
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh()
        )
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        hidden = self.actor_backbone(x)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            action = torch.clamp(action, -1.0, 1.0)

        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.critic(x)
        return action, log_prob, entropy, value


class VectorVirgoEnv:
    """
    Lightweight, synchronous vectorized environment runner for fast PyTorch training
    """
    def __init__(self, num_envs, action_mode='wrench', randomize_dynamics=True):
        self.num_envs = num_envs
        self.envs = [
            VirgoEnv(
                action_mode=action_mode,
                randomize_targets=True,
                randomize_dynamics=randomize_dynamics
            )
            for _ in range(num_envs)
        ]
        self.obs_dim = self.envs[0].obs_dim
        self.action_dim = self.envs[0].action_space.shape[0]

    def reset(self):
        obs_list = []
        for env in self.envs:
            o, _ = env.reset()
            obs_list.append(o)
        return np.array(obs_list, dtype=np.float32)

    def step(self, actions):
        obs_list, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            act = actions[i]
            obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)
        return (
            np.array(obs_list, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos
        )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"[Virgo_RL] Initializing training on: {device}")
    if torch.cuda.is_available():
        print(f"[Virgo_RL] GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize Vector Environment
    vec_env = VectorVirgoEnv(
        num_envs=args.num_envs,
        action_mode=args.action_mode,
        randomize_dynamics=True
    )

    obs_dim = vec_env.obs_dim
    action_dim = vec_env.action_dim
    print(f"[Virgo_RL] Action Mode: {args.action_mode.upper()} (Action Dim: {action_dim}, Obs Dim: {obs_dim})")
    print(f"[Virgo_RL] Parallel Envs: {args.num_envs}, Rollout Steps: {args.num_steps}")

    # Model & Optimizer
    agent = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage buffers
    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs, action_dim), dtype=torch.float32, device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    # Start environment
    obs = torch.tensor(vec_env.reset(), dtype=torch.float32, device=device)
    done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
    global_step = 0
    start_time = time.time()
    best_eval_reward = -float('inf')

    print(f"[Virgo_RL] Total updates to perform: {num_updates}")
    print("=" * 80)

    for update in range(1, num_updates + 1):
        # Annealing learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        # 1. Rollout Collection
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = obs
            dones_buf[step] = done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # Step physical environment
            next_obs, reward, next_done, infos = vec_env.step(action.cpu().numpy())
            rewards_buf[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            done = torch.tensor(next_done, dtype=torch.float32, device=device)

        # 2. Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # Flatten batch
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1, action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # 3. PPO Optimization Epochs
        b_inds = np.arange(args.num_steps * args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.num_steps * args.num_envs, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        if update % args.log_interval == 0 or update == 1:
            fps = int(global_step / (time.time() - start_time))
            mean_rew = rewards_buf.mean().item()
            mean_dist = np.mean([info['dist_to_target'] for info in infos])
            print(
                f"Update {update:4d}/{num_updates} | "
                f"Step: {global_step:7d} | "
                f"Mean Rew: {mean_rew:6.2f} | "
                f"Pos Err: {mean_dist:5.2f}m | "
                f"FPS: {fps:4d} | "
                f"Loss: {loss.item():6.3f}"
            )

        # Evaluation & Checkpointing
        if update % args.eval_interval == 0 or update == num_updates:
            eval_score = evaluate_agent(agent, args.action_mode, device)
            print(f"--> [EVAL Update {update}] Evaluation Reward: {eval_score:6.2f}")
            if eval_score > best_eval_reward:
                best_eval_reward = eval_score
                ckpt_path = os.path.join(args.save_dir, "virgo_ppo_best.pt")
                torch.save({
                    'model_state_dict': agent.state_dict(),
                    'obs_dim': obs_dim,
                    'action_dim': action_dim,
                    'action_mode': args.action_mode,
                    'eval_reward': eval_score,
                    'update': update
                }, ckpt_path)
                print(f"--> [CHECKPOINT] Saved new best model to: {ckpt_path}")

    # Final save
    final_ckpt = os.path.join(args.save_dir, "virgo_ppo_latest.pt")
    torch.save({
        'model_state_dict': agent.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'action_mode': args.action_mode,
        'update': num_updates
    }, final_ckpt)
    print(f"\n[Virgo_RL] Training Complete! Final model saved to: {final_ckpt}")


def evaluate_agent(agent, action_mode, device, num_eval_episodes=5):
    """
    Evaluates policy over fixed benchmark targets
    """
    env = VirgoEnv(action_mode=action_mode, randomize_targets=False, randomize_dynamics=False)
    total_rewards = []
    
    with torch.no_grad():
        for ep in range(num_eval_episodes):
            obs, _ = env.reset()
            ep_rew = 0.0
            for _ in range(300):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                hidden = agent.actor_backbone(obs_t)
                action = agent.actor_mean(hidden).cpu().numpy()[0]
                action = np.clip(action, -1.0, 1.0)
                obs, r, term, trunc, _ = env.step(action)
                ep_rew += r
                if term or trunc:
                    break
            total_rewards.append(ep_rew)
    return float(np.mean(total_rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virgo_RL PPO Training")
    parser.add_argument("--action-mode", type=str, default="wrench", choices=["wrench", "direct"])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default="/home/duc/Raptor_UAV/Virgo_RL/checkpoints")

    args = parser.parse_args()
    train(args)
