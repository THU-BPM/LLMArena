# noqa: D212, D415
from __future__ import annotations

import os

import gymnasium
import numpy as np
import random
from copy import deepcopy
from collections import Counter


from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.classic.tictactoe.board import Board
from pettingzoo.utils import agent_selector, wrappers

def find_modes(values):

    value_counts = Counter(values)
    max_count = max(value_counts.values())
    modes = [value for value, count in value_counts.items() if count == max_count]
    return modes

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
        "name": "bargain_v1",
        "is_parallelizable": False,
        "render_fps": 1,
    }


    def __init__(
        self, render_mode: str | None = None, num_players: int = 2, num_items: int = 3
    ):
        super().__init__()
        EzPickle.__init__(self, render_mode)
        self.num_players = num_players
        self.num_items = num_items
        self.word_list = open("instances.txt","r").readlines()
        self.agents = [f"player_{i}" for i in range(0,num_players)]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(1001) for i in self.agents}
        self.round = 0
        self.opponent_bargain = ""
        self.agent1_plan = []
        self.agent2_plan = []
        self.set_instances()
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(1001,), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(1001,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.voted = {i: False for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {"legal_moves": list(range(0, num_players + 1))} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode
    def set_instances(self):
        line = np.random.choice(self.word_list)
        line = line.replace("\n","")
        infos = line.split(" ")
        self.item_nums = [int(i) for i in infos[0].split(",")]
        self.agent1_value = [int(i) for i in infos[1].split(",")]
        self.agent2_value = [int(i) for i in infos[2].split(",")]
    def observe(self, agent):
        if agent == self.agents[0]:
            value = self.agent1_value
            if len(self.agent2_plan) != 0:
                oppo_plan = self.agent2_plan[-1]
            else:
                oppo_plan = None
        else:
            value = self.agent2_value
            if len(self.agent1_plan) != 0:
                oppo_plan = self.agent1_plan[-1]
            else:
                oppo_plan = None
        action_mask = np.zeros(1001, "int8")
        action_mask[-1] = 1
        for i in range(self.item_nums[0] + 1):
            for j in range(self.item_nums[1] + 1):
                for k in range(self.item_nums[2] + 1):
                    action_mask[i * 100 + j * 10 + k] = 1
        return {"observation": np.ones(1001, "int8"), "action_mask": action_mask,"round":self.round,"item_nums":self.item_nums,"value":value, "opponent_plan":oppo_plan,"opponent_bargain":self.opponent_bargain}

    def talk(self,message):
        self.opponent_bargain = message

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _legal_moves(self):
        return [i for i in range(len(self.board.squares)) if self.board.squares[i] == 0]
    def step(self, action):
        print(action)
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        #if isinstance(action,str):
            #self.talk(action)
        next_agent = self.agents[(self.agents.index(self.agent_selection) + 1) % len(self.agents)]
        next_agent = self._agent_selector.next()
        player = 1
        if next_agent == self.agents[1]:
            player = 0
        else:
            self.round += 1
        if self.round > 10:
            self.rewards[self.agents[0]] = 0
            self.rewards[self.agents[1]] = 0
            self.terminations = {i: True for i in self.agents}
            self._accumulate_rewards()
        else:
            if action == 1000:
                if player == 0 and len(self.agent2_plan) == 0:
                    self.rewards[self.agents[0]] = -1
                    self.rewards[self.agents[1]] = 1
                    print(f"\033[31m illegal action when player_1 deal first.\033[0m")
                    self.terminations = {i: True for i in self.agents}
                    self._accumulate_rewards()
                else:
                    plan = self.agent1_plan[-1] if player == 1 else self.agent2_plan[-1]
                    oppo_plan = (self.item_nums[0] - plan[0] ,self.item_nums[1] - plan[1], self.item_nums[2] - plan[2])
                    if player == 0:
                        self.rewards[self.agents[0]] += sum(x * y for x, y in zip(self.agent1_value, oppo_plan))
                        self.rewards[self.agents[0]] -= sum(x * y for x, y in zip(self.agent2_value, plan))
                        self.rewards[self.agents[1]] += sum(x * y for x, y in zip(self.agent2_value, plan))
                        self.rewards[self.agents[1]] -= sum(x * y for x, y in zip(self.agent1_value, oppo_plan))
                    else:
                        self.rewards[self.agents[0]] += sum(x * y for x, y in zip(self.agent1_value, plan))
                        self.rewards[self.agents[0]] -= sum(x * y for x, y in zip(self.agent2_value, oppo_plan))
                        self.rewards[self.agents[1]] += sum(x * y for x, y in zip(self.agent2_value, oppo_plan))
                        self.rewards[self.agents[1]] -= sum(x * y for x, y in zip(self.agent1_value, plan))
                    print(f"\033[31m Deal. Player is {player}. Player_0 value is {self.agent1_value} Player_1 value is {self.agent2_value}\033[0m")
                    print(f"\033[31m Deal. Plan is player {player} {oppo_plan}, player {1-player} {plan}. Player_0 reward is {self.rewards[self.agents[0]]} Player_1 reward is {self.rewards[self.agents[1]]}\033[0m")

                    self.terminations = {i: True for i in self.agents}
                    self._accumulate_rewards()
            else:
                a, b, c = action//100 , (action % 100) // 10, action % 10
                if player == 0:
                    self.agent1_plan.append((a,b,c))
                    print(f"agent 0 get {sum(x * y for x, y in zip(self.agent1_value, [a, b, c]))} value, plan is {[a,b,c]}, round is {self.round}")
                else:
                    self.agent2_plan.append((a,b,c))
                    print(f"agent 1 get {sum(x * y for x, y in zip(self.agent2_value, [a, b, c]))} value, plan is {[a,b,c]}, round is {self.round}")

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
        self.round = 1
        self.set_instances()
        self.agent1_plan = []
        self.agent2_plan = []

    def close(self):
        pass

    def render(self):
        return