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

    def _get_reachable_nodes(self, start_node):
        visited = set()
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                if node in self.transitions:
                    neighbors = [t[0] for t in self.transitions[node]]
                    stack.extend(neighbors)
        return visited

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Choose a start node only from those that have outgoing transitions
        valid_start_nodes = list(self.transitions.keys())
        self.current_node = random.choice(valid_start_nodes)
        
        # Ensure the goal node is reachable
        reachable_nodes = self._get_reachable_nodes(self.current_node)
        attempts = 0
        # If we can't find a start node with a reachable set > 1, keep trying
        while len(reachable_nodes) <= 1 and attempts < 50:
            self.current_node = random.choice(valid_start_nodes)
            reachable_nodes = self._get_reachable_nodes(self.current_node)
            attempts += 1

        if len(reachable_nodes) <= 1:
            # If we still can't find a better scenario, fallback: goal = current
            # This means no movement, but it's better than guaranteed -100.
            self.goal_node = self.current_node
        else:
            # Pick a goal different from current, from reachable set
            if self.current_node in reachable_nodes:
                reachable_nodes.remove(self.current_node)
            self.goal_node = random.choice(list(reachable_nodes))

        self.steps_taken = 0
        return self.node_to_idx[self.current_node], {}

    def step(self, action):
        # Action is the index of the next node. Convert back to DEA code:
        next_node = self.nodes[action]

        # Check if transition is valid
        if (self.current_node not in self.transitions) or \
           (not any(t[0] == next_node for t in self.transitions[self.current_node])):
            # Invalid transition: now just give a smaller penalty and do NOT end the episode.
            reward = -10.0
            done = False
            info = {'reason': 'invalid transition'}
            # Do not change current_node or steps; agent tries again
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
                reward += 100.0  # Positive reward for reaching the goal
                done = True

        # Check for max steps
        if self.steps_taken >= self.max_steps:
            done = True

        return self.node_to_idx[self.current_node], reward, done, info

    def render(self):
        print(f"Current node: {self.current_node}, Goal: {self.goal_node}")

    def close(self):
        pass
