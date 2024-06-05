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

from abc import ABC, abstractmethod
from typing import Any, Dict

from openrl.selfplay.opponents.base_opponent import BaseOpponent
from openrl.selfplay.selfplay_api.opponent_model import BattleHistory, BattleResult
import trueskill
class BaseAgent(ABC):
    def __init__(self):
        self.batch_history = BattleHistory()
        self.rating = trueskill.Rating()
    def new_agent(self) -> BaseOpponent:
        agent = self._new_agent()
        return agent

    @abstractmethod
    def _new_agent(self) -> BaseOpponent:
        raise NotImplementedError
    
    def add_reward(self,reward: float):
        self.batch_history.update_reward(reward)

    def add_battle_result(self, result: BattleResult):
        self.batch_history.update(result)

    def get_battle_info(self) -> Dict[str, Any]:
        return self.batch_history.get_battle_info()
    
    def update_team_trueskill(self,result:BattleResult,winner_ratings,loser_ratings,rank):
        if result == BattleResult.WIN:
            winner, loser = trueskill.rate([winner_ratings,loser_ratings], ranks=[0, 1])
            self.rating = winner[rank]

        elif result == BattleResult.LOSE:
            winner, loser = trueskill.rate([winner_ratings, loser_ratings], ranks=[0, 1])
            self.rating = loser[rank]

    def update_nonzero_game(self):
        self.batch_history.update_nonzero()

    def update_trueskill(self,result: BattleResult,rating):
        my_rating = (self.rating,)
        opponent_rating = (rating,)
        if result == BattleResult.WIN:
            winner, loser = trueskill.rate([my_rating,opponent_rating], ranks=[0, 1])
            self.rating = winner[0]
            print(f"Wins.\nBefore:\nSelf Rating: {my_rating[0].mu} Opponent Rating:{rating.mu}\nAfter:\nSelf Rating: {winner[0].mu} Opponent Rating:{loser[0].mu}")
        elif result == BattleResult.LOSE:
            winner, loser = trueskill.rate([opponent_rating, my_rating], ranks=[0, 1])
            self.rating = loser[0]
            print(f"Loses.\n Before:\nSelf Rating: {my_rating[0].mu} Opponent Rating:{rating.mu}\nAfter:\nSelf Rating: {loser[0].mu} Opponent Rating:{winner[0].mu}")
        elif result == BattleResult.DRAW:
            player1, player2 = trueskill.rate([my_rating, opponent_rating], ranks=[0, 0])
            self.rating = player1[0]
            print(f"Draw.\n Before:\nSelf Rating: {my_rating[0].mu} Opponent Rating:{rating.mu}\nAfter:\nSelf Rating: {player1[0].mu} Opponent Rating:{player2[0].mu}")
        # print(f'self rating is {self.rating}')