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
        "name": "undercover_v1",
        "is_parallelizable": False,
        "render_fps": 1,
    }


    def __init__(
        self, render_mode: str | None = None, num_players: int = 5, num_undercovers: int = 1
    ):
        super().__init__()
        EzPickle.__init__(self, render_mode)
        self.board = Board()
        self.num_players = num_players
        self.word_list = open("words.txt","r").readlines()

        self.num_undercovers = num_undercovers
        self.agents = [f"player_{i}" for i in range(0,num_players)]
        self.possible_agents = self.agents[:]
        #self.undercover_set = set(random.sample(list(range(num_players)),num_undercovers))
        self.undercover_set = set(list(range(num_players))[self.num_agents-num_undercovers:])
        self.non_undercover_set = set(list(range(num_players))) - self.undercover_set
        self.alive_undercover = deepcopy(self.undercover_set)
        self.alive_non_undercover = deepcopy(self.non_undercover_set)

        self.action_spaces = {i: spaces.Discrete(num_players + 1) for i in self.agents}
        self.round = 0
        self.set_words()
        self.message_list = [[] for _ in range((num_players))]
        self.votes = [[] for _ in range((num_players))]
        self.phrase = "talk"
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(num_players + 1,), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(num_players + 1,), dtype=np.int8),
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
    def set_words(self):
        line = np.random.choice(self.word_list)
        line = line.replace("\n","")
        self.undercover_word = line.split(" - ")[0]
        self.non_undercover_word = line.split(" - ")[-1]

    def observe(self, agent):
        messages = {
            agent:message for agent,message in self.message_list[self.round]
        }
        cur_player = self.possible_agents.index(agent)
        action_mask = np.ones(self.num_agents + 1, "int8")
        action_mask[cur_player] = 0
        word = self.getcode(agent)
        for i,agent in enumerate(self.agents):
            if self.voted[agent]:
                action_mask[i] = 0
        return {"observation": np.ones(self.num_agents + 1, "int8"), "action_mask": action_mask,"phrase":self.phrase,"word":word,"messages":messages}

    def getcode(self,agent):
        if self.possible_agents.index(agent) in self.undercover_set:
            return self.undercover_word
        return self.non_undercover_word

    def talk(self,agent, message):
        self.message_list[self.round].append((agent,message))

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
        if action == self.num_players:
            if len(self.message_list[self.round]) == self.num_players - self.round:
                self.phrase = 'vote'
            next_agent = self._agent_selector.next()
            while self.voted[next_agent]:
                next_agent = self._agent_selector.next()
            # Switch selection to next agents
            self._cumulative_rewards[self.agent_selection] = 0
            self.agent_selection = next_agent
            self._accumulate_rewards()
            return 
        self.votes[self.round].append({self.agent_selection:action})
        if len(self.votes[self.round]) == self.num_players - self.round:
            values = [list(player.values())[0] for player in self.votes[self.round]]
            voting_agent = find_modes(values)
            if len(voting_agent) == 1:
                voting_agent = voting_agent[0]
            else:
                voting_agent = random.choice(voting_agent)
            print(f"voting is {values}")
            print(f"===============votes {voting_agent}===============")
            if voting_agent in self.alive_undercover:
                self.alive_undercover.remove(voting_agent)
                print(f"==============={voting_agent} is undercover===============")
                print(f"===============there are {len(self.alive_undercover)} undercovers and {len(self.alive_non_undercover)} non-undercovers===============")

                if len(self.alive_undercover) == 0:
                    for i in self.non_undercover_set:
                        self.rewards[self.agents[i]] += 1
                    for i in self.undercover_set:
                        self.rewards[self.agents[i]] -= 1
                    self.terminations = {i: True for i in self.agents}
                    print("!!!!!!!!!!!!!non-undercover wins!!!!!!!!!!!!!")
                elif len(self.alive_undercover) + len(self.alive_non_undercover) <= 3:
                    for i in self.undercover_set:
                        self.rewards[self.agents[i]] += 1
                    for i in self.non_undercover_set:
                        self.rewards[self.agents[i]] -= 1
                    self.terminations = {i: True for i in self.agents}
                    print("!!!!!!!!!!!!!undercover wins!!!!!!!!!!!!!")
            else:
                self.alive_non_undercover.remove(voting_agent) 
                print(f"==============={voting_agent} is non-undercover===============")
                print(f"===============there are {len(self.alive_undercover)} undercovers and {len(self.alive_non_undercover)} non-undercovers===============")
                if len(self.alive_undercover) + len(self.alive_non_undercover) <= 3:
                    for i in self.undercover_set:
                        self.rewards[self.agents[i]] += 1
                    for i in self.non_undercover_set:
                        self.rewards[self.agents[i]] -= 1
                    self.terminations = {i: True for i in self.agents}
                    print("!!!!!!!!!!!!!undercover wins!!!!!!!!!!!!!")
            self.voted[f'player_{voting_agent}'] = True
            self.round += 1
            self.phrase = 'talk'
        # update infos
        # list of valid actions (indexes in board)
        # next_agent = self.agents[(self.agents.index(self.agent_selection) + 1) % len(self.agents)]
        next_agent = self._agent_selector.next()
        while self.voted[next_agent]:
            next_agent = self._agent_selector.next()
        # Switch selection to next agents
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_agent

        self._accumulate_rewards()
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
        self.voted = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self.undercover_set = set(list(range(self.num_players))[self.num_agents-self.num_undercovers:])
        self.non_undercover_set = set(list(range(self.num_players))) - self.undercover_set
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()
        self.round = 0
        self.set_words()
        self.message_list = [[] for _ in range((self.num_players))]
        self.votes = [[] for _ in range((self.num_players))]
        self.alive_undercover = deepcopy(self.undercover_set)
        self.alive_non_undercover = deepcopy(self.non_undercover_set)

    def close(self):
        pass

    def render(self):
        return