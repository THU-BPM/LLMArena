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

from openrl.selfplay.opponents.base_llm_opponent import BaseLLMOpponent
from pathlib import Path
from typing import Dict, Optional, Union
import openai
import re
import random
import time
class Opponent(BaseLLMOpponent):
    round_retry = 0
    round_illegal = 0
    def __init__(self,opponent_id: Optional[str] = None,
        opponent_path: Optional[Union[str, Path]] = None,
        opponent_info: Optional[Dict[str, str]] = None):
        super().__init__(opponent_id=opponent_id,opponent_path=opponent_path,opponent_info=opponent_info)
        self.act2pos = {
            0:"1",
            1:"2",
            2:"3",
            3:"4",
            4:"5",
            5:"6",
            6:"7",
        }
        self.pos2act = {value.replace(" ",""):key for (key,value) in self.act2pos.items()}
    def _get_board_status(
        self,observation
    ):
        
        opposite_mark = "X" if self.player_type=="O" else "O"
        marks = [["_","_","_","_","_","_","_"],["_","_","_","_","_","_","_"],["_","_","_","_","_","_","_"],["_","_","_","_","_","_","_"],["_","_","_","_","_","_","_"],["_","_","_","_","_","_","_"]]
        for i,rows in enumerate(observation['observation']):
            for j,cols in enumerate(rows):
                if cols[0] == 1:
                    marks[i][j] = self.player_type
                if cols[1] == 1:
                    marks[i][j] = opposite_mark
        board_status = "\n  1   2   3   4   5   6   7  \n"

        for mark in marks:
            for m in mark:
                board_status = board_status + f"| {m} "
            board_status = board_status + "|\n"
        print(board_status)
        return board_status
    def _get_available(self, action_mask):
        available = [self.act2pos[idx] for idx, action in enumerate(action_mask) if action == 1]
        return ', '.join(available)
    def _get_chat_action(
            self,action_mask,observation, info = None
    ):
        self.player_type = "X" if self.player_name == 'player_0' else "O"
        max_retries = 3
        current_retry = 0
        illegal_count = 0
        while current_retry < max_retries:
            try:
                board_status = self._get_board_status(observation=observation)
                available = self._get_available(action_mask=action_mask)
                message = f"You play {self.player_type}.\nThe column you put the mark on cannot be full.\n\n"
                message += f"Don't say anything besides mark position.\nYou should only output {self.player_type} and the column of the move you choose to put your mark, for example: \"{self.player_type}: 1\"\n\n"
                message += f"The board status is {board_status}.\n"
                answer = self._get_chat_answer(message=message)
                pos = re.findall(r'[XO]\s*:\s*[1-7]',answer)[0].replace(" ","")
                pos = pos[-1]
                action = self.pos2act[pos]
                break
            except Exception as e:
                current_retry += 1 
                print(f"no action Exception!{current_retry}")
                time.sleep(1)
        else:
            indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]
            action = random.choice(indices_of_ones)
            print("no action  random action:", action)
        if action  not in [i for i, value in enumerate(action_mask) if value == 1]:
            illegal_count += 1
            print(f"no action illegal action!{illegal_count}")
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
    from pettingzoo.classic import connect_four_v3
    import json
    result1 = open("result1.txt","w")
    result2 = open("result2.txt","w")
    info = json.load(open('./info.json','r'))
    total_retry = 0
    total_illegal = 0
    opponent1 = Opponent(
        opponent_info=info
    )
    opponent2 = Opponent(        
        opponent_info=info
    )
    env = connect_four_v3.env(render_mode=None)
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
        action = player2opponent[player_name].act(
            player_name, observation, reward, termination, truncation, info
        )
        round_retry = player2opponent[player_name].round_retry
        round_illegal = player2opponent[player_name].round_illegal
        print(player_name, action, type(action))
        env.step(action)
    get_retry("retry&illegal.json")
