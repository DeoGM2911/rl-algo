import numpy as np
from dataclasses import dataclass


class ValueIteration:
    r"""
    Value iteration algorithm.

    @params:
        states: set of states
        actions: set of actions
        transition: transition probability matrix. Shape: (len(states), len(actions), len(states))
        reward: reward function. Shape: (len(states), len(actions), len(states))
        gamma: discount factor
        horizon: maximum number of iterations
    
    The implementation of the value iteration algorithm. In short, given an MDP, we will iteratively
    recalculate the value function until it converges. At iteration \(\tau\), the value function is
    denoted by \(V(s, \tau)\). The value iteration algorithm is given by the following update rule:

    \[
        V_(s, \tau) = max_a \sum_{s'} P(s' | s, a) [R(s, a, s') + \gamma V(s', \tau - 1)]
    \]

    The optimal policy is given by
    \[
        \pi^*(s) = argmax_a \sum_{s'} P(s' | s, a) [R(s, a, s') + \gamma V(s', \tau - 1)]
    \]
    """
    def __init__(
        self, 
        states: set, 
        actions: set, 
        transition: np.ndarray, 
        reward: np.ndarray, 
        gamma: float, 
        horizon: int
    ):
        self.states = states
        self.actions = actions
        self.transition = transition
        self.reward = reward
        self.gamma = gamma  # Discount factor
        self.horizon = horizon

        # Some assertions for the input
        assert self.transition.shape == (len(self.states), len(self.actions), len(self.states)), \
            "Transition matrix shape must be (len(states), len(actions), len(states))"
        assert self.reward.shape == (len(self.states), len(self.actions), len(self.states)), \
            "Reward matrix shape must be (len(states), len(actions), len(states))"
        assert 0 <= self.gamma <= 1, "Discount factor must be between 0 and 1"
        assert self.horizon > 0, "Horizon must be greater than 0"
    
    def train(self):
        """
        Run the value iteration algorithm
        """
        # Initialize the table
        V = np.zeros((len(self.states)))
        pi = np.zeros((len(self.states)))

        for tau in range(self.horizon):
            # Sum across all possible next states, then take the max along the action dimension
            # Shape (1, 1, -1) for V since we want the V for the next state (dim=2)
            target = np.sum(
                (self.reward + self.gamma * V.reshape((1, 1, -1))) * self.transition,
                axis=2
            )
            V = np.max(target, axis=1)
            pi = np.argmax(target, axis=1)

        return pi, V


class PolicyIteration:
    r"""
    Policy iteration algorithm

    @params:
        states: set of states
        actions: set of actions
        transition: transition probability matrix. Shape: (len(states), len(actions), len(states))
        reward: reward function. Shape: (len(states), len(actions), len(states))
        gamma: discount factor
        horizon: maximum number of iterations

    The implementation of the policy iteration algorithm. Given an MDP and an initial policy, we will iteratively
    update the policy until it converges. At iteration \(\tau\), the policy is denoted by \(\pi(s, \tau)\). 
    The policy iteration algorithm is given by the following update rule:
    \[
        \pi(s, \tau + 1) = argmax_a \sum_{s'} P(s' | s, a) [R(s, a, s') + \gamma V^{\pi_\tau}(s', \tau)]
    \]
    where \(V^{\pi_\tau}(s, \tau) is the value function under policy \(\pi_\tau\)\).
    """
    def __init__(
        self, 
        states: set, 
        actions: set, 
        transition: np.ndarray, 
        reward: np.ndarray, 
        gamma: float,
        max_iter: int,
        horizon: int
    ):
        self.states = states
        self.actions = actions
        self.transition = transition
        self.reward = reward
        self.gamma = gamma  # Discount factor
        self.max_iter = max_iter
        self.horizon = horizon
        # Some assertions for the input
        assert self.transition.shape == (len(self.states), len(self.actions), len(self.states)), \
            "Transition matrix shape must be (len(states), len(actions), len(states))"
        assert self.reward.shape == (len(self.states), len(self.actions), len(self.states)), \
            "Reward matrix shape must be (len(states), len(actions), len(states))"
        assert 0 <= self.gamma <= 1, "Discount factor must be between 0 and 1"
    
    def _evaluate(self, pi):
        """
        Evaluate the given policy
        """
        V = np.zeros((len(self.states)))
        for _ in range(self.horizon):
            # Fix an action based on the policy, then calculate the value function
            # Select probability and reward for the chosen action for each state
            pi_expanded = pi[:, None, None]
            P_pi = np.take_along_axis(self.transition, pi_expanded, axis=1)
            R_pi = np.take_along_axis(self.reward, pi_expanded, axis=1)

            # Calculate expected value: sum_{s'} P(s'|s, pi(s)) * [ R(s, pi(s), s') + gamma * V(s') ]
            V = np.sum(P_pi * (R_pi + self.gamma * V.reshape((1, 1, -1))), axis=2).squeeze()

        return V

    def train(
        self,
        pi_0: np.ndarray=None
    ):
        """
        Run the policy iteration algorithm
        """
        if pi_0 is None:
            pi = np.array([np.random.choice(len(self.actions)) for _ in range(len(self.states))])
        else:
            pi = pi_0
        
        assert pi.shape == (len(self.states),), "Initial policy shape must be (len(states),)"

        # Evaluate initial policy
        V = self._evaluate(pi).reshape((1, 1, -1))
        converge = False
        patience = 0

        # Policy evaluation
        while not converge and patience < self.max_iter:
            next_pi = np.argmax(
                np.sum(
                    (self.reward + self.gamma * V) * self.transition, axis=2
                ), axis=1
            )
            # Check if converge
            if np.all(next_pi == pi):
                converge = True
            else:
                patience += 1
            pi = next_pi
            V = self._evaluate(pi).reshape((1, 1, -1))

        return pi, V.squeeze()


# Unit test
if __name__ == "__main__":
    states = set(range(5))
    actions = set(range(2))
    transition = np.array([
        [
            [0.7, 0.3, 0, 0, 0],
            [0.2, 0.8, 0, 0, 0]
        ],
        [
            [0.6, 0.4, 0, 0, 0],
            [0.1, 0.9, 0, 0, 0]
        ],
        [
            [0.5, 0.5, 0, 0, 0],
            [0.4, 0.6, 0, 0, 0]
        ],
        [
            [0.4, 0.6, 0, 0, 0],
            [0.3, 0.7, 0, 0, 0]
        ],
        [
            [0.3, 0.7, 0, 0, 0],
            [0.2, 0.8, 0, 0, 0]
        ]
    ])
    reward = np.array([
        [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ],
        [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ],
        [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ],
        [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ],
        [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ]
    ])
    gamma = 0.9
    max_iter = 1000
    horizon = 1000
    print("Testing value iteration...")
    value_iter = ValueIteration(states, actions, transition, reward, gamma, horizon)
    pi, V = value_iter.train()
    print(pi)
    print(V)

    print("Testing policy iteration...")
    policy_iter = PolicyIteration(states, actions, transition, reward, gamma, max_iter, horizon)
    pi, V = policy_iter.train()
    print(pi)
    print(V)

