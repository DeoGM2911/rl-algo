import gymnasium as gym
import torch
from torch import nn, optim
from copy import deepcopy


# Reuse some classes for convience
from trpo import TRPO


class PPOLoss(nn.Module):
    def __init__(self, clip_param):
        super(PPOLoss, self).__init__()
        self.clip_param = clip_param

    def forward(self, action_logits_old, actions_logits, actions, advantages):
        ratio = torch.exp(
            torch.gather(action_logits_old, dim=1, index=actions.view(-1, 1)).view(-1) - \
            torch.gather(actions_logits, dim=1, index=actions.view(-1, 1)).view(-1))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        return -torch.mean(torch.min(surr1, surr2))


class PPO(TRPO):
    def __init__(self, env, policy, state_dim, episode_len, gamma=0.99, lambda_ = 0.95, k=None, lr=0.001, clip_param=0.2, num_batches=5):
        # Create old policy and baseline
        super(PPO, self).__init__(env, policy, state_dim, episode_len, gamma, lambda_, k, lr, None, None, None, num_batches)

        # Old policy
        self.policy_old = deepcopy(self.policy)

        # PPO loss
        self.policy_loss = PPOLoss(clip_param)

    def train(self, num_episodes, track_history):
        policy_loss_history = []
        for i in range(num_episodes):
            for _ in range(self.num_batches):
                # Collect roll-outs
                states, actions_logits, actions, returns, _ = self._play_episodes(i, 5)

                # Compute the baselines for the states - Also forward pass for baseline
                self.baseline_optimizer.zero_grad()
                baselines = self.baseline(states)

                # Compute the advantage for each transition
                # Detach to avoid policy network backprop through the baseline network
                advantages = (returns - baselines.view(-1)).detach()

                # Normalize the advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Refit the baseline
                baseline_loss = self.baseline_criterion(baselines.view(-1), returns)
                baseline_loss.backward()
                self.baseline_optimizer.step()

                # Refit the policy by sampling batches of data from the old policy
                self.optimizer.zero_grad()
                policy_loss = self.policy_loss(self.policy_old(states).detach(), 
                                                    actions_logits,
                                                    actions,
                                                    advantages)
                if track_history:
                    policy_loss_history.append(policy_loss.item())
                
                policy_loss.backward()
                self.optimizer.step()

            # Update the old policy
            self.policy_old.load_state_dict(self.policy.state_dict())

        return policy_loss_history

# Unit test
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    policy = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 128),
        nn.Tanh(),
        nn.Linear(128, env.action_space.n),
    )
    ppo = PPO(env, policy, env.observation_space.shape[0], env.action_space.n)
    ppo.train(10, True)