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
from openrl.arena.games.multiplayer_game import MultiPlayerGame
from openrl.selfplay.selfplay_api.opponent_model import BattleResult


class MultiPlayerArena(BaseArena):
    def __init__(
        self,
        env_fn: Callable,
        dispatch_func: Optional[Callable] = None,
        use_tqdm: bool = True,
    ):
        super().__init__(env_fn, dispatch_func, use_tqdm=use_tqdm)
        self.game = MultiPlayerGame()

    def _deal_result(self, result: Any):
        winner_ratings = []
        loser_ratings = []
        for winner in result["winners"]:    
            winner_ratings.append(deepcopy(self.agents[winner].rating))
            self.agents[winner].add_battle_result(BattleResult.WIN)
        for loser in result["losers"]:    
            loser_ratings.append(deepcopy(self.agents[loser].rating))
            self.agents[loser].add_battle_result(BattleResult.LOSE)
        for i,winner in enumerate(result["winners"]):
            self.agents[winner].update_team_trueskill(BattleResult.WIN,winner_ratings,loser_ratings,i)
        for i,loser in enumerate(result["losers"]):
            self.agents[loser].update_team_trueskill(BattleResult.LOSE,winner_ratings,loser_ratings,i)
        for agent_name,reward in result['total_reward'].items():
            self.agents[agent_name].add_reward(reward)

    def _get_final_result(self) -> Dict[str, Any]:
        result = {}
        for agent_name in self.agents:
            result[agent_name] = self.agents[agent_name].get_battle_info()
            result[agent_name]["rating"] = {
                "mu": self.agents[agent_name].rating.mu,
                "sigma": self.agents[agent_name].rating.sigma
            }
        return result
