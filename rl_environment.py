import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class DistributionPathEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nodes, transitions, node_to_idx, max_steps=20):
        super(DistributionPathEnv, self).__init__()
        self.nodes = nodes
        self.transitions = transitions
        self.node_to_idx = node_to_idx
        self.observation_space = spaces.Discrete(len(nodes))  # Current node index
        self.action_space = spaces.Discrete(len(nodes))       # Next node index

        self.max_steps = max_steps
        self.current_node = None
        self.goal_node = None
        self.steps_taken = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # For training (TODO: for now) sample random start-goal pairs
        self.current_node = random.choice(self.nodes)
        self.goal_node = random.choice(self.nodes)
        while self.goal_node == self.current_node:
            self.goal_node = random.choice(self.nodes)
        self.steps_taken = 0
        return self.node_to_idx[self.current_node], {}

    def step(self, action):
        # Action is the index of the next node. Convert back to DEA code:
        next_node = self.nodes[action]

        # Check if transition is valid
        if self.current_node not in self.transitions or \
           not any(t[0] == next_node for t in self.transitions[self.current_node]):
            # Invalid transition, large negative reward
            reward = -100.0
            done = True
            info = {'reason': 'invalid transition'}
        else:
            # Valid transition: find the distance
            dist = [t[1] for t in self.transitions[self.current_node] if t[0] == next_node][0]
            reward = -dist
            self.current_node = next_node
            self.steps_taken += 1
            done = False
            info = {}

            # Check if reached goal
            if self.current_node == self.goal_node:
                reward += 100.0  # Give a positive reward for reaching the goal
                done = True

        # Check for max steps
        if self.steps_taken >= self.max_steps:
            done = True

        return self.node_to_idx[self.current_node], reward, done, info

    def render(self, mode='human'):
        print(f"Current node: {self.current_node}, Goal: {self.goal_node}")

    def close(self):
        pass
