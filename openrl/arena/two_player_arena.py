#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""
from typing import Any, Callable, Dict, Optional
from copy import deepcopy
from openrl.arena.base_arena import BaseArena
from openrl.arena.games.two_player_game import TwoPlayerGame
from openrl.selfplay.selfplay_api.opponent_model import BattleResult


class TwoPlayerArena(BaseArena):
    def __init__(
        self,
        env_fn: Callable,
        dispatch_func: Optional[Callable] = None,
        use_tqdm: bool = True,
    ):
        super().__init__(env_fn, dispatch_func, use_tqdm=use_tqdm)
        self.game = TwoPlayerGame()

    def _deal_result(self, result: Any):
        if len(result["winners"]) == 2:
            # drawn
            for agent_name in result["winners"]:
                self.agents[agent_name].add_battle_result(BattleResult.DRAW)
                
            agent1_name = result["winners"][0]
            agent2_name = result["winners"][1]

            agent1 = self.agents[agent1_name]
            agent2 = self.agents[agent2_name]
            
            agent1_rating = deepcopy(agent1.rating) 
            agent2_rating = deepcopy(agent2.rating)

            agent1.update_trueskill(BattleResult.DRAW,agent2_rating)
            agent2.update_trueskill(BattleResult.DRAW,agent1_rating)

        else:
            for agent_name in result["winners"]:
                self.agents[agent_name].add_battle_result(BattleResult.WIN)
            for agent_name in result["losers"]:
                self.agents[agent_name].add_battle_result(BattleResult.LOSE)

            winner_name = result["winners"][0]
            loser_name = result["losers"][0]

            winner = self.agents[winner_name]
            loser = self.agents[loser_name]

            winner_rating = deepcopy(winner.rating) 
            loser_rating = deepcopy(loser.rating)

            winner.update_trueskill(BattleResult.WIN,loser_rating)
            loser.update_trueskill(BattleResult.LOSE,winner_rating)

        for agent_name,reward in result['total_reward'].items():
            self.agents[agent_name].add_reward(reward)
            if reward > 0:
                self.agents[agent_name].update_nonzero_game()

    def _get_final_result(self) -> Dict[str, Any]:
        result = {}
        for agent_name in self.agents:
            result[agent_name] = self.agents[agent_name].get_battle_info()
            result[agent_name]["rating"] = {
                "mu": self.agents[agent_name].rating.mu,
                "sigma": self.agents[agent_name].rating.sigma
            }
        return result
