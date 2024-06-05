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
import random
import time
class Opponent(BaseLLMOpponent):
    def __init__(self,opponent_id: Optional[str] = None,
        opponent_path: Optional[Union[str, Path]] = None,
        opponent_info: Optional[Dict[str, str]] = None):
        super().__init__(opponent_id=opponent_id,opponent_path=opponent_path,opponent_info=opponent_info)
    def _get_chat_action(
            self,action_mask, observation, info = None
    ):
        self.word = observation["word"]
        messages_dict = observation['messages']
        messages = ""
        for player,clue in messages_dict.items():
            messages += f"{player}: {clue}\n"
        phrase = observation["phrase"]
        if phrase == 'talk':
            try:
                message = f"Now is the giving clues stage:\nYou are {self.player_name}. Your secret word is \"{self.word}\".\n"
                message += f"This round the other players say:\n{messages}"
                message += f"Now you gives one clue (but never include the secret word). You cannot repeat what other players has said.\n"
                message += f"You should only output {self.player_name} and the clue, for example: \"{self.player_name}: It's a fruit.\".\n"
                answer = self._get_chat_answer(message=message)

                answer = answer.replace("Player","player").replace("\\","")
                clue_to_give = re.findall(rf'{self.player_name}:\s+(.*)',answer)[0]
                info["env"].talk(self.player_name,clue_to_give)
                action = 5
            except Exception as e:
                print(e.__context__,e)
                time.sleep(1)
                action = 5
                clue_to_give = " "
                info["env"].talk(self.player_name,clue_to_give) 
        else:    
            try:
                message = f"Now is the accusation stage:\nYou are {self.player_name}. Your secret word is \"{self.word}\".\n"
                message += f"This round the other players say:\n{messages}"
                message += f"Now guess what other players' secret words are and vote which of the other players (excluding yourself) is the undercover. \n"
                message += f"You should guess what other players' secret words are first and then output \"vote\" and the undercover name, for example: \"vote: player_1.\".\n"
                answer = self._get_chat_answer(message=message)
                answer = answer.replace("Vote","vote")
                action = re.findall(rf'vote:\s+(.*)',answer)[0]
                #action = clue_to_give
                action = re.findall(rf'\d',action)[0]
                action = int(action)
                if action_mask[action] != 1:
                    indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]
                    action = random.choice(indices_of_ones[:-1]) 
            except Exception as e:
                print(e.__context__,e)
                time.sleep(1)
                indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]
                action = random.choice(indices_of_ones[:-1]) 
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
    from openrl.envs.undercover_env import undercover_v1
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
    opponent3 = Opponent(        
        opponent_info = info
    )
    opponent4 = Opponent(        
        opponent_info = info
    )
    opponent5 = Opponent(        
        opponent_info = info
    )

    env = undercover_v1.env(render_mode=None)
    opponent1.reset(env, "player_0")
    opponent2.reset(env, "player_1")
    opponent3.reset(env, "player_2")
    opponent4.reset(env, "player_3")
    opponent5.reset(env, "player_4")

    player2opponent = {"player_0": opponent1, "player_1": opponent2,"player_2": opponent3, "player_3": opponent4, "player_4": opponent5}

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
