from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np


class QLearning:
    r"""
    Trainer for Q-learning algorithm.

    @params:
        env: Environment
        gamma: Discount factor
        max_iter: Maximum number of iterations
        horizon: Number of steps to sample per iteration
    
    THe Q-learning algorithm is an off-policy algorithm which samples data by interacting with the environment and uses
    that to update the Q-value function. The update rule is given by:
    \[
        Q(s, a) \leftarrow Q(s, a) + \alpha [R(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a)]
    \]
    where \(\alpha\) is the learning rate.

    In this implementation, we only sample one action per iteration. One can generalize to sample batches of data
    by playing self.horizon steps per iteration. Note that this is somewhat close to DQN, however, in this implementation,
    the number of iterations for Q learning is self.horizon * self.max_iter. The only difference is the change in the value 
    of \(\epsilon\) in epsilon-greedy policy.

    Note that for action sampling we use epsilon-greedy policy and for learning rate we use decay.
    """
    def __init__(
        self, 
        env: Env, 
        gamma: float,
        alpha: float,
        horizon: int,
        max_iter: int
    ):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter
        self.horizon = horizon
        self.epsilon_0 = 1  # Initially favor exploration
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

        assert isinstance(env.observation_space, spaces.Discrete), "Observation space must be discrete"
        assert isinstance(env.action_space, spaces.Discrete), "Action space must be discrete"
    
    def _alpha(self, num_iter):
        """
        Learning rate decay
        """
        return self.alpha / (num_iter + 1)

    def _epsilon_greedy(
        self, 
        s: int, 
        num_iter: int
    ):
        """
        Epsilon-greedy action selection.
        """
        if np.all(np.random.random() < (self.epsilon_0 / (num_iter + 1) ** 2)):
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[s])

    def _play_episode(self, s, horizon, current_iter):
        """
        Play an episode with length horizon
        """
        rewards, states, actions = [], [], []
        for _ in range(horizon):
            a = self._epsilon_greedy(s, current_iter)
            next_s, r, terminated, truncated, _ = self.env.step(a)
            rewards.append(r)
            states.append(s)
            actions.append(a)
            s = next_s
            if terminated or truncated:
                break
        states.append(next_s)
        return rewards, states, actions        

    def train(
        self
    ):
        """
        Train the Q-value function for one sample
        """
        # Initialize
        s, _ = self.env.reset()
        for i in range(self.max_iter):
            # Sample an action
            rewards, states, actions = self._play_episode(s, self.horizon, i)
            
            # Update Q-value
            for s, a, r, next_s in zip(states[:-1], actions, rewards, states[1:]):
                self.Q[s, a] += self._alpha(i) * (r + self.gamma * np.max(self.Q[next_s]) - self.Q[s, a])

            # Reset the env
            s, _ = self.env.reset()

        return self.Q


if __name__ == "__main__":
    env = gym.make('CliffWalking-v1', render_mode="human")
    agent = QLearning(env, gamma=0.99, max_iter=10000, alpha=0.1, horizon=100)
    agent.train()
    print(agent.Q)