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
from openrl.selfplay.opponents.base_llm_opponent import BaseLLMOpponent
from pathlib import Path
from typing import Dict, Optional, Union
import openai
import re
import os
import time
import random

class Opponent(BaseLLMOpponent):
    opponent_action = None
    round_retry = 0
    round_illegal = 0
    def __init__(self,opponent_id: Optional[str] = None,
        opponent_path: Optional[Union[str, Path]] = None,
        opponent_info: Optional[Dict[str, str]] = None):
        super().__init__(opponent_id=opponent_id,opponent_path=opponent_path,opponent_info=opponent_info)
        self.round = 0
        self.flag = False
        self.idx2card = {
            0:"A",
            1:"2",
            2:"3",
            3:"4",
            4:"5",
            5:"6",
            6:"7",
            7:"8",
            8:"9",
            9:"10",
            10:"J",
            11:"Q",
            12:"K",
        }
        self.act2pos = {
            0:"Fold",
            1:"Check and Call",
            2:"Raise Half Pot",
            3:"Raise Full Pot",
            4:"All in"  
        }
        self.act2des = {
            0:"Fold: Choosing 'Fold' means the player is out of the hand, forfeiting any potential claim to the pot and not committing any more chips to the pot.",
            1:"Check and Call: If no bet has been made, a player can choose to 'Check', which means they do not wish to make a bet, and play passes to the next player. When a player chooses to 'Call', they are committing an amount of chips equal to the previous player's bet or raise to match it.",
            2:"Raise Half Pot: The player raises an amount equal to half the size of the current pot.",
            3:"Raise Full Pot: The player raises an amount equal to the full size of the current pot.",
            4:"All in: This is a bet where the player wagers all of their remaining chips.",
            
        }
        self.pos2act = {value.replace(" ","").lower():key for (key,value) in self.act2pos.items()}
        self.private = set()
    def act(self, player_name, observation, reward, termination, truncation, info):
        action = self.sample_random_action(
            player_name, observation, reward, termination, truncation, info
        )
        return action

    def _extract_card(self,observation):
        cards = set()
        observation = observation[:52]
        for idx,card in enumerate(observation[:13]):
            if card == 1:
                cards.add("Spades " + self.idx2card[idx])
        for idx,card in enumerate(observation[13:26]):
            if card == 1:
                cards.add("Hearts " + self.idx2card[idx])
        for idx,card in enumerate(observation[26:39]):
            if card == 1:
                cards.add("Diamonds " + self.idx2card[idx])
        for idx,card in enumerate(observation[39:52]):
            if card == 1:
                cards.add("Clubs " + self.idx2card[idx])
        return cards
    def _get_board_status(
        self,observation
    ):
        if self.flag is False:
            self.flag = True
            self.private = self._extract_card(observation['observation'])
        self.public = self._extract_card(observation['observation']) - self.private
    def _get_chat_action(
            self,action_mask,observation,info=None
    ):
        max_retries = 3
        current_retry = 0
        illegal_count = 0
        while current_retry < max_retries:
            try:
                self._get_board_status(observation=observation)
                chips = observation['observation'][52]
                message = f"Your observation now is :\n"
                message += f"The cards in your hands is [{', '.join(list(self.private))}]\n"
                message += f"The community cards is [{', '.join(list(self.public))}]\n"
                message += f"You now have {100-chips} chips, the chips that you has put in until now is {chips}\n"
                # print(Opponent.opponent_action)
                # if Opponent.opponent_action is not None:
                #     message += f"Your opponent's action is {Opponent.opponent_action}\n"
                message += f"Now you can choose one of the following actions:\n"
                available = [self.act2des[idx] for idx, action in enumerate(action_mask) if action == 1]
                message +=  "\n".join(available) + "\n"
                message += f"You can choose one of the following actions: "
                available = [self.act2pos[idx] for idx, action in enumerate(action_mask) if action == 1]
                message +=  f"[ {', '.join(available)} ]\n"
                message += "You should think step by step, explain why you choose the action first and then output your action. For example: 'Action: Check and Call'\n"
                #print(message)
                answer = self._get_chat_answer(message=message)
                answer = answer.lower()
                pos = re.findall(r'action:\s*(fold|check and call|raise half pot|raise full pot|all in)',answer)[0].replace(" ","")
                pos = pos.replace('action:','')
                action = self.pos2act[pos]
                Opponent.opponent_action = pos
                break
            except Exception as e:
                print(e.__context__,e)
                current_retry += 1 
                time.sleep(1)
        else:
            indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]
            action = random.choice(indices_of_ones)
        if action  not in [i for i, value in enumerate(action_mask) if value == 1]:
            illegal_count += 1
        Opponent.round_retry += current_retry
        Opponent.round_illegal += illegal_count
        return action
    
  #get_retry()用于统计retry和illegal action的次数 
def get_retry(filename):
    if os.path.exists(filename):
        pass
    else: 
        with open(filename, 'w') as json_file:
            data = {"round_retry": 0, "round_illegal": 0, "total_retry": 0, "total_illegal": 0}
            json.dump(data, json_file)
            json_file.write('\n')
    with open(filename, 'r') as json_file:
        lines = json_file.readlines()
    if lines:
        last_line = lines[-1]
        try:
            last_data = json.loads(last_line)
        except json.JSONDecodeError:
            last_data = {}
    desired_key1 = "total_retry"
    desired_key2 = "total_illegal"
    if desired_key1 in last_data:
        desired_value1 = last_data[desired_key1]
        total_retry = desired_value1 + round_retry
    if desired_key2 in last_data:
        desired_value2 = last_data[desired_key2]
        total_illegal= desired_value2 + round_illegal
    new_data = {"round_retry": round_retry, "round_illegal": round_illegal, "total_retry": total_retry, "total_illegal": total_illegal}  
    with open(filename, 'a') as json_file:
        json.dump(new_data, json_file)
        json_file.write('\n')

if __name__ == "__main__":
    from pettingzoo.classic import texas_holdem_no_limit_v6
    import json
    result1 = open("result1.txt","w")
    result2 = open("result2.txt","w")
    info = json.load(open('info.json','r'))
    total_retry = 0
    total_illegal = 0
    opponent1 = Opponent(
        opponent_info=info
    )
    opponent2 = Opponent(        
        opponent_info=info
    )
    env = texas_holdem_no_limit_v6.env(render_mode=None)
    opponent1.reset(env, "player_0")
    opponent2.reset(env, "player_1")
    player2opponent = {"player_0": opponent1, "player_1": opponent2}

    env.reset()
    for player_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination:
            print(reward)
            result1.write(str(opponent1.messages))
            result2.write(str(opponent2.messages))
            break
        action = player2opponent[player_name].act(
            player_name, observation, reward, termination, truncation, info
        )
        round_retry = player2opponent[player_name].round_retry
        round_illegal = player2opponent[player_name].round_illegal
        print(player_name, action, type(action))
        env.step(action)
    get_retry("retry&illegal.json")
    # from pettingzoo.classic import texas_holdem_no_limit_v6

    # env = texas_holdem_no_limit_v6.env(num_players = 8,render_mode=None)
    # env.reset(seed=42)

    # for agent in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()

    #     if termination or truncation:
    #         action = None
    #     else:
    #         mask = observation["action_mask"]
    #         # this is where you would insert your policy
    #         action = env.action_space(agent).sample(mask)
    #         print(observation['observation'].shape,action)

    #     env.step(action)
    # env.close()