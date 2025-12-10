import torch
from torch import nn, optim
import gymnasium as gym
import copy
from typing import Callable
import random
from dataclasses import dataclass


def feature(state):
    """
    Feature extractor. Return feature vector of the state. Here, it's the identity function.
    """
    return state


@dataclass
class Config:
    """
    Configuration class for DQN
    """
    # DQN hyperparameters
    feature: Callable = feature
    num_episodes: int = 1000
    replay_buffer_size: int = 100  # Positive Integers
    batch_size: int = 32
    gamma: float = 0.99
    epsilon_0: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.99
    prioritize: bool = False  # Prioritized experience replay
    update_target_freq: int = 10  # Positive Integers
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer
    optimizer: optim.Optimizer = optim.RMSprop
    optimizer_lr: float = 0.001
    optimizer_momentum: float = 0.95


class DQN:
    """
    DQN algorithm.

    @params:
    env: gym.Env
        Environment
    q_model: nn.Module
        Q model. Expect it takes in feature vector and outputs Q values for each action. That is,
        the output of the model should be a vector of size action space.
    optimizer: optim.Optimizer
        Optimizer
    config: Config
        Configuration. See Config class for details.
    """
    def __init__(
        self,
        env: gym.Env,
        q_model: nn.Module,
        config: Config
    ):
        self.env = env
        self.q_model = q_model
        self.optimizer = config.optimizer(q_model.parameters(), 
                                            lr=config.optimizer_lr,
                                            momentum=config.optimizer_momentum)
        self.config = config
        self.device = config.device
        self._step = 0

        # Initially target model is the same as Q model
        self.target_q_model = copy.deepcopy(q_model)
        
        # Replay buffer
        self.replay_buffer = []

        assert self.config.update_target_freq > 0, "update_target_freq must be positive"
        assert self.config.replay_buffer_size > 0, "replay_buffer_size must be positive"
        
    def _epsilon(self, episode: int) -> float:
        """
        Epsilon decay
        """
        return max(self.config.epsilon_min, self.config.epsilon_0 * (self.config.epsilon_decay ** episode))
    
    def _update_target_model(self):
        self.target_q_model.load_state_dict(self.q_model.state_dict())
        self._step = 0
    
    def _sample_batch(self):
        """
        Sample data points from the replay buffer for training.
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        if self.config.prioritize:  # Prioritized experience replay
            # Compute Bellman error for each transition in the replay buffer
            errs = []
            for transition in self.replay_buffer:
                f_s, a, r, f_s_next, done = transition
                errs.append(torch.abs(r + self.config.gamma * torch.max(self.target_q_model(f_s_next)) - self.q_model(f_s)[a]).item())

            # Sample batch with probability proportional to Bellman error
            batch = random.choices(self.replay_buffer, weights=errs, k=self.config.batch_size)
        else:
            batch = random.sample(self.replay_buffer, self.config.batch_size)

        f_states = torch.stack([transition[0] for transition in batch]).to(self.device)
        actions = torch.tensor([transition[1] for transition in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([transition[2] for transition in batch], dtype=torch.float32).to(self.device)
        f_next_states = torch.stack([transition[3] for transition in batch]).to(self.device)
        dones = torch.tensor([transition[4] for transition in batch], dtype=torch.float32).to(self.device)
        
        return f_states, actions, rewards, f_next_states, dones
    
    def _train_step(self, f_states, actions, rewards, f_next_states, dones):
        """
        Train the DQN agent for one step.
        """
        # Compute the targets
        target_q_values = self.target_q_model(f_next_states)
        target_q_values, _ = torch.max(target_q_values, dim=1)
        target_q_values = rewards + self.config.gamma * target_q_values * (1 - dones)

        # Perform GD
        self.optimizer.zero_grad()
        q_values = self.q_model(f_states)
        q_values = q_values.gather(1, actions.view(-1, 1)).squeeze()
        loss = nn.HuberLoss()(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        """
        Train the DQN agent.
        """
        loss_history = []
        for episode in range(self.config.num_episodes):
            state, _ = self.env.reset()
            state = self.config.feature(state)
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            done = False
            episode_reward = 0

            while not done:
                # Sample action
                if random.random() < self._epsilon(episode):
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(self.q_model(state)).item()

                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.config.feature(next_state)
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward

                if self.config.replay_buffer_size > 0:
                    batch = self._sample_batch()
                    if batch is not None:
                        loss = self._train_step(*batch)
                        print(f"Episode {episode}: Reward {episode_reward}, Loss {loss}")
                        loss_history.append(loss)
                        break
                
                if self.config.update_target_freq > 0:
                    self._step += 1
                    if self._step % self.config.update_target_freq == 0:
                        self._update_target_model()
            print(f"Episode {episode}: Reward {episode_reward}")


# Unit test
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    q_model = nn.Sequential(
        nn.Linear(4, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )
    config = Config(
            feature=lambda x: x,
            num_episodes=10,
            replay_buffer_size=10,
            batch_size=4,
            gamma=0.99,
            epsilon_0=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.99,
            prioritize=False,
            update_target_freq=5,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            optimizer=optim.RMSprop,
            optimizer_lr=0.001,
            optimizer_momentum=0.95
    )
    agent = DQN(env, q_model, config)
    agent.train()