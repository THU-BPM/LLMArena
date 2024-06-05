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
import os
import sys
import transformers
from openrl.selfplay.opponents.base_llm_opponent import BaseLLMOpponent
from pathlib import Path
from typing import Dict, Optional, Union
import openai
import re
import math
import random
import time
class Opponent(BaseLLMOpponent):
    bargaining = None
    def __init__(self,opponent_id: Optional[str] = None,
        opponent_path: Optional[Union[str, Path]] = None,
        opponent_info: Optional[Dict[str, str]] = None):
        super().__init__(opponent_id=opponent_id,opponent_path=opponent_path,opponent_info=opponent_info)
    def _get_chat_action(
            self,action_mask, observation, info = None
    ):
        value = observation['value']
        try:
            message = f"You are {self.player_name}.\nYour valuation of the current auction item is ${value}, your bid must lower than your valuation.\n"
            message += f"Your rewards will be equal to $({value} - your bid), and you should aim to maximize your rewards.\n"
            message += f"You should think step by step and then output \"{self.player_name}\" and your bid:\n"
            message += f"For example: {self.player_name}: $1.08\n"
            answer = self._get_chat_answer(message=message)
            answer = answer.replace("Player","player").replace("\\","")
            match = re.findall(rf'(.*){self.player_name}:\s+\$(\d+\.\d+)',answer)
            if len(match) == 0:
                try:
                    bid = re.findall(rf'\$(\d+.\d+)',answer)[0]
                    action = math.floor( float(bid) * 100)
                except:
                    print("=============error============")
                    indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]
                    action = 0
            else:
                bid = match[0][1]
                action = math.floor( float(bid) * 100)
        except Exception as e:
            print(e.__context__,e)
            time.sleep(1)
            indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]
            action = 0
        return action #, current_retry, illegal_count
    
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
    from openrl.envs.bid_env import bid_v1
    import json
    result1 = open("result1.txt","w")
    result2 = open("result2.txt","w")
    info = json.load(open('./info.json','r'))
    round_retry = 0
    round_illegal = 0
    opponent1 = Opponent(
        opponent_info = info
    )
    opponent2 = Opponent(        
        opponent_info = info
    )


    env = bid_v1.env(render_mode=None)
    opponent1.reset(env, "player_0")
    opponent2.reset(env, "player_1")


    player2opponent = {"player_0": opponent1, "player_1": opponent2}

    env.reset()
    for player_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination:
            result1.write(str(opponent1.messages))
            result2.write(str(opponent2.messages))
            break
        info["env"] = env.unwrapped
        action = player2opponent[player_name].act(
            player_name, observation, reward, termination, truncation, info
        )
        print(player_name, action, type(action),reward)
        env.step(action)    
    get_retry("retry&illegal.json")
