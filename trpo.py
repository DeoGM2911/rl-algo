import gymnasium as gym
import torch
from torch import nn, optim
from torch.distributions import Categorical

# Reuse some code for convenience
from reinforce import REINFORCE, Policy, PolicyLoss, feature_extractor


class TRPOLoss(PolicyLoss):
    def __init__(self, beta):
        super(TRPOLoss, self).__init__()
        self.beta = beta
    
    def forward(self, action_logits_old, actions_logits, actions, advantages, lengths):
        grads = super().forward(actions, advantages, lengths)
        
        # Compute the KL penalty
        kl = torch.distributions.kl.kl_divergence(\
                        Categorical(logits=action_logits_old),\
                        Categorical(logits=actions_logits)
        )
        kl = torch.max(kl)  # Take the max since the original constraint is KL < delta
        return grads + self.beta * kl, kl  # + KL since we're minimizing the objective


class TRPO(REINFORCE):
    def __init__(self, env, policy, state_dim, episode_len, gamma=0.99, lr=0.001, beta=2, KL_clip_max=0.2, KL_clip_min=0.1, num_batches=5):
        super(TRPO, self).__init__(env, policy, state_dim, episode_len, gamma, lr)
        
        self.KL_clip_max = KL_clip_max
        self.KL_clip_min = KL_clip_min
        self.num_batches = num_batches
        self.policy_loss = TRPOLoss(beta)
        self.policy_old = Policy(state_dim, env.action_space.n)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _play_episode(self, num_iter):
        state, _ = self.env.reset()
        states = []
        action_logits = []
        actions = []
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
            action_logits.append(logits)
            actions.append(probs[action])
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
        
        return states, action_logits, actions, returns, len(states)
    
    def _play_episodes(self, num_iter, num_trajectories):
        """Run multiple episode"""
        states = []
        action_logits = []
        actions = []
        returns = []
        eps_lens = []
        for _ in range(num_trajectories):
            eps_states, eps_action_logits, eps_actions, eps_returns, eps_len = self._play_episode(num_iter)
            states.extend(eps_states)
            action_logits.extend(eps_action_logits)
            actions.extend(eps_actions)
            returns.extend(eps_returns)
            eps_lens.append(eps_len)

        # Stack the states, action logits and returns
        lengths = torch.tensor(eps_lens)
        states = torch.stack(states)
        action_logits = torch.stack(action_logits)
        actions = torch.stack(actions)
        returns = torch.tensor(returns, dtype=torch.float32)

        return states, action_logits, actions, returns, lengths 

    def train(self, num_episodes, track_history=True):
        policy_loss_history = []
        
        for i in range(num_episodes):
            # Collect roll-outs from current policy. Also forward pass for policy
            # Repeat for num_batches times
            for _ in range(self.num_batches):
                self.optimizer.zero_grad()
                states, action_logits, actions, returns, lengths = self._play_episodes(i, 5)
            
                # Compute the baselines for the states - Also forward pass for baseline
                self.baseline_optimizer.zero_grad()
                baselines = self.baseline(states)

                # Compute the advantage for each transition
                # Detach to avoid policy network backprop through the baseline network
                advantages = (returns - baselines.view(-1)).detach()

                # Refit the baseline
                baseline_loss = self.baseline_criterion(baselines.view(-1), returns)
                baseline_loss.backward()
                self.baseline_optimizer.step()

                # Refit the policy by sampling batches of data from the old policy
                self.optimizer.zero_grad()
                policy_loss, kl = self.policy_loss(self.policy_old(states).detach(), 
                                                    action_logits, 
                                                    actions, 
                                                    advantages, 
                                                    lengths)
                policy_loss.backward()
                self.optimizer.step()   
                
                if track_history:
                    policy_loss_history.append(policy_loss.item())
                
                # Update KL penalty if needed
                if kl > self.KL_clip_max:
                    self.policy_loss.beta *= 2
                elif kl < self.KL_clip_min:
                    self.policy_loss.beta *= 0.5

            # Update old policy
            self.policy_old.load_state_dict(self.policy.state_dict())
            
        return policy_loss_history


# Unit test
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    agent = TRPO(env, policy, env.observation_space.shape[0], episode_len=500)
    agent.train(10)
