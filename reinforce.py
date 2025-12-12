import gymnasium as gym
import torch
from torch import nn, optim


class Policy(nn.Module):
    """
    Neural Network for Policy.
    """
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def feature_extractor(state):
    return torch.from_numpy(state).float()


class PolicyLoss(nn.Module):
    def __init__(self):
        super(PolicyLoss, self).__init__()

    def forward(self, actions_logits, advantages, lengths):
        log_probs = torch.log(actions_logits)

        # Gradient for each timestep in a trajectory
        grads = log_probs * advantages

        # Sum the advantages * log prob along a trajectory then take the mean
        eps_ids = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
        out = torch.zeros(len(lengths), dtype=grads.dtype)
        out = out.index_add(0, eps_ids, grads)

        return torch.mean(out)


class REINFORCE:
    """
    Vanilla Policy Gradient
    """
    def __init__(self, env, policy, state_dim, episode_len, gamma=0.99, lr=0.001):
        self.policy = policy
        self.gamma = gamma
        self.state_dim = state_dim

        # An NN baseline
        self.baseline = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Optimizer for policy and baseline
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=lr)

        # Criterions for policy and baseline
        self.policy_criterion = PolicyLoss()
        self.baseline_criterion = nn.MSELoss()

        self.env = env
        self.episode_len = episode_len

    def _play_episode(self, num_iter):
        state, _ = self.env.reset()
        states = []
        action_logits = []
        rewards = []
        steps = 0

        while steps < self.episode_len:
            state = feature_extractor(state)
            logits = self.policy(state)
            # Use stochastic action selection for exploration
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            states.append(state)
            # Store probability of the chosen action (for log prob calculation)
            action_logits.append(probs[action])
            rewards.append(reward)
            
            if terminated or truncated:
                break

            state = next_state
            steps += 1
        
        # Compute the return
        returns = [rewards[-1]]
        for r in reversed(rewards[:-1]):
            returns.append(r + self.gamma * returns[-1])
        
        # Reverse the returns array
        returns = returns[::-1]

        # Convert to torch tensor
        states = torch.stack(states)
        action_logits = torch.stack(action_logits)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        return states, action_logits, returns, states.size(0)
    
    def _play_episodes(self, num_iter, num_trajectories):
        """Run multiple episode"""
        states = []
        action_logits = []
        returns = []
        eps_lens = []
        for _ in range(num_trajectories):
            eps_states, eps_action_logits, eps_returns, eps_len = self._play_episode(num_iter)
            states.extend(eps_states)
            action_logits.extend(eps_action_logits)
            returns.extend(eps_returns)
            eps_lens.append(eps_len)

        # Stack the states, action logits and returns
        lengths = torch.tensor(eps_lens)
        states = torch.stack(states)
        action_logits = torch.stack(action_logits)
        returns = torch.tensor(returns, dtype=torch.float32)

        return states, action_logits, returns, lengths

    def train(self, num_episodes, track_history=True):
        # Training History
        policy_loss_history = []
        baselines_loss_history = []

        for i in range(num_episodes):
            # Collect roll-outs from current policy. Also forward pass for policy
            self.optimizer.zero_grad()
            states, action_logits, returns, lengths = self._play_episodes(i, 5)
            
            # Compute the baselines for the states - Also forward pass for baseline
            self.baseline_optimizer.zero_grad()
            baselines = self.baseline(states)

            # Compute the advantage for each transition
            # Detach to avoid policy network backprop through the baseline network
            advantages = (returns - baselines.view(-1)).detach()

            # Refit the baselines
            baselines_loss = self.baseline_criterion(baselines.view(-1), returns)
            baselines_loss.backward()
            self.baseline_optimizer.step()
            if track_history:
                baselines_loss_history.append(baselines_loss.item())

            # Refit the policy
            policy_loss = self.policy_criterion(action_logits, advantages, lengths)
            policy_loss.backward()
            self.optimizer.step()   
            if track_history:
                policy_loss_history.append(policy_loss.item())
            
        return policy_loss_history, baselines_loss_history


# Unit test
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    agent = REINFORCE(env, policy, env.observation_space.shape[0], episode_len=500)
    agent.train(100)
