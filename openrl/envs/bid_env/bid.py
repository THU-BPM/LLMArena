# noqa: D212, D415
from __future__ import annotations

import os

import gymnasium
import numpy as np
import random
import math
from copy import deepcopy
from collections import Counter


from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.classic.tictactoe.board import Board
from pettingzoo.utils import agent_selector, wrappers

def find_random_max_index(arr):
    if not arr:
        return None 

    max_value = max(arr)  
    max_indices = [i for i, x in enumerate(arr) if x == max_value]  # 找出所有最大值的索引

    return random.choice(max_indices)  

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "bid_v1",
        "is_parallelizable": False,
        "render_fps": 1,
    }


    def __init__(
        self, render_mode: str | None = None, num_players: int = 2, max_value: float = 10.0
    ):
        super().__init__()
        EzPickle.__init__(self, render_mode)
        self.num_players = num_players
        self.max_value = max_value
        self.max_action = math.floor(max_value * 100)
        self.agents = [f"player_{i}" for i in range(0,num_players)]
        self.agent_values = [round(np.random.uniform(0,max_value) ,2) for i in range(0,num_players)]
        self.possible_agents = self.agents[:]
        self.action_spaces = {i: spaces.Discrete(self.max_action) for i in self.agents}
        self.round = 0
        self.agent1_plan = []
        self.agent2_plan = []
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(1,), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(self.max_action,), dtype=np.int8),
                }
            )
            for i in self.agents
        }
        self.bid_list = []
        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {"legal_moves": list(range(0, num_players + 1))} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode

    def observe(self, agent):
        action_mask = np.zeros(self.max_action, "int8")
        cur_player = self.possible_agents.index(agent)
        value = self.agent_values[cur_player]
        max_action_value = math.floor(value * 100)
        action_mask[:max_action_value + 1] = 1
        action_mask
        return {"observation": np.ones(1, "int8"), "action_mask": action_mask, "value":self.agent_values[cur_player]}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _legal_moves(self):
        return [i for i in range(len(self.board.squares)) if self.board.squares[i] == 0]
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        #if isinstance(action,str):
            #self.talk(action)
        target = self.agent_values[self.agents.index(self.agent_selection)] / 2
        reward = (action/100 - target) / target
        self.rewards[self.agent_selection] += reward
        next_agent = self.agents[(self.agents.index(self.agent_selection) + 1) % len(self.agents)]
        next_agent = self._agent_selector.next()
        self.bid_list.append(action/100)
        print(f"reward is {reward}")
        if len(self.bid_list) == self.num_players:
            max_index = find_random_max_index(self.bid_list)
            #self.rewards[self.agents[max_index]] += (self.agent_values[max_index] - max(self.bid_list))
            self.terminations = {i: True for i in self.agents}
            self._accumulate_rewards()
            print(f"\033[31m Player {max_index} wins. Bid { max(self.bid_list)} win:{self.agent_values[max_index] - max(self.bid_list)}\033[0m")
        self.agent_selection = next_agent
        if self.render_mode == "human":
            self.render()
    def reset(self, seed=None, options=None):
        # reset environment
        np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()
        self.agent_values = [round(np.random.uniform(0,self.max_value) ,2) for i in range(0,self.num_players)]

        self.bid_list = []
    def close(self):
        pass

    def render(self):
        return