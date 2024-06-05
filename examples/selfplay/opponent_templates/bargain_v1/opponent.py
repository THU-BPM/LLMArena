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
    bargaining = ""
    def __init__(self,opponent_id: Optional[str] = None,
        opponent_path: Optional[Union[str, Path]] = None,
        opponent_info: Optional[Dict[str, str]] = None):
        super().__init__(opponent_id=opponent_id,opponent_path=opponent_path,opponent_info=opponent_info)
        self.init_flag = True
        self.str2number = {
            "zero":0,
            "one":1,
            "two":2,
            "three":3,
            "four":4,
            "five":5,
            "six":6,
            "seven":7,
            "eight":8,
            "nine":9
        }
    def _get_chat_action(
            self,action_mask, observation, info = None
    ):
        round = observation['round']
        item_nums = observation['item_nums']
        value = observation['value']
        oppo_plan = observation['opponent_plan']
        opponent_bargain = observation['opponent_bargain']
        try:
            if oppo_plan is None:
                message = f"The game begins.\nNow is round {round}:\nYou are {self.player_name}.\nYou start to give a plan."
            else:
                message = f"Now is round {round}:\nYou are {self.player_name}.\n"
            message += f"There are {item_nums[0]} hats, {item_nums[1]} balls and {item_nums[2]} apples.\n"
            message += f"The value for each hat is {value[0]}, for each ball is {value[1]} and for each apple is {value[2]}.\n"
            if oppo_plan is not None:
                if opponent_bargain == "":
                    message += f"Your opponent make a plan that:\n"
                else:
                    message += f"This round your opponent says:\n\n\"{opponent_bargain}\"\n\n"
                message += f"He will get {oppo_plan[0]} hats, {oppo_plan[1]} balls and {oppo_plan[2]} apples.\n"
                message += f"And you will get {item_nums[0] - oppo_plan[0]} hats, {item_nums[1] - oppo_plan[1]} balls and {item_nums[2] - oppo_plan[2]} apples, the total value is {(item_nums[0] - oppo_plan[0]) * value[0] + (item_nums[1] - oppo_plan[1]) * value[1] + (item_nums[2] - oppo_plan[2]) * value[2]}.\n"
                message += f"Do you agree with his plan? If you agree, You should only output {self.player_name} and \"Deal\", for example: \"{self.player_name}: Deal.\".\n\n"
                message += f"If you do not agree, you should say something to bargain with your opponent first and then output {self.player_name} and your plan as the following format:\n"
            else:
                message += f"You should say something to bargain with your opponent first and then output {self.player_name} and your plan as the following format:\n"

            message += f"\"{self.player_name}: x hats y balls z apples\". x, y, z are Arabic number.\n"
            message += f"For example:\"I'd like 1 hats, 2 balls and 0 apples. {self.player_name}: 1 hats 2 balls 0 apples\".\n"
            message += "Do not tell you opponent your value of each item, try to maximum the value you get."

            answer = self._get_chat_answer(message=message)
            answer = answer.replace("Player","player").replace("\\","")
            match = re.findall(rf'(.*){self.player_name}:\s+(.*)',answer)
            print(match)
            if len(match) == 0:
                try:
                    if "Deal" in answer or "deal" in answer:
                        action = 1000
                    else:
                        num_hats = re.findall("(\d)\s+hat",answer)[0]
                        num_balls = re.findall("(\d)\s+ball",answer)[0]
                        num_apples = re.findall("(\d)\s+apple",answer)[0]
                        action = 100 * int(num_hats) + 10 * int(num_balls) + int(num_apples)
                        info['env'].talk(answer)
                        print(f"opponent_bargain is {opponent_bargain}")
                except:
                    print("=============error============")
                    indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]
                    action = random.choice(indices_of_ones)
                    info['env'].talk("")
            else:
                bargaining = match[0][0]
                info['env'].talk(bargaining)


                planing = match[0][1].lower()
                if "deal" in planing:
                    action = 1000
                    return action
                for key,val in self.str2number.items():
                    planing = planing.replace(key,str(val))
                num_hats = re.findall("(\d)\s+hat",planing)[0]
                num_balls = re.findall("(\d)\s+ball",planing)[0]
                num_apples = re.findall("(\d)\s+apple",planing)[0]
                action = 100 * int(num_hats) + 10 * int(num_balls) + int(num_apples)
        except Exception as e:
            print(e.__context__,e)
            time.sleep(1)
            indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]
            action = random.choice(indices_of_ones)
            opponent_bargain = ""
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
    from openrl.envs.bargain_env import bargain_v1
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


    env = bargain_v1.env(render_mode=None)
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
