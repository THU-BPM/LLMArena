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
from openrl.selfplay.opponents.base_opponent import BaseOpponent
from pathlib import Path
from typing import Dict, Optional, Union
import openai
import re
import os

class LLMOpponent(BaseOpponent):
    def __init__(self,opponent_id: Optional[str] = None,
        opponent_path: Optional[Union[str, Path]] = None,
        opponent_info: Optional[Dict[str, str]] = None):
        super().__init__(opponent_id=opponent_id,opponent_path=opponent_path,opponent_info=opponent_info)
        self.model_name = opponent_info.get('model_name',None)
        self.temperature = float(opponent_info.get('temperature',None))
        #self.messages = [{"role": "system", "content": "You are playing Connect Four. \nConnect Four is played on a six-by-seven grid by two players, who alternately drop their makers X and O into one of the seven columns. Each marker will fall to the lowest available space within the selected column. \nThe player who succeeds in placing four of their makers in a horizontal, vertical, or diagonal line is the winner.\nX plays first.\nHere are some examples:\n1. X wins, get a vertical line at column 4.\n  1   2   3   4   5   6   7  \n| _ | _ | _ | _ | _ | _ | _ |\n| _ | _ | _ | _ | _ | _ | _ |\n| _ | _ | _ | X | _ | _ | _ |\n| _ | _ | _ | X | _ | _ | _ |\n| _ | _ | _ | X | _ | _ | _ |\n| O | O | O | X | _ | _ | _ |\n\n3. X wins, get a diagonal line.\n  1   2   3   4   5   6   7  \n| _ | _ | _ | _ | _ | _ | _ |\n| _ | _ | _ | _ | _ | _ | _ |\n| _ | _ | _ | X | _ | _ | _ |\n| _ | _ | X | O | _ | _ | _ |\n| _ | X | O | O | _ | _ | _ |\n| X | O | O | X | X | _ | _ |\n\n\nPlayers interact with the game by specifying the column number where they want to drop their marker. If a column is already full, players cannot drop a marker into that column.\nThe columns are numbered from 1 to 7, from left to right. Players cannot place a token outside this range.'"}]
        self.messages = [{"role":"system","content":opponent_info.get('system_prompt',"")}]
        self.act2pos = opponent_info.get('act2pos',{})
        self.api_base = opponent_info.get('api_base',None)
        self.api_key = opponent_info.get('api_key',None)

        # self.act2pos = {
        #     0:"1",
        #     1:"2",
        #     2:"3",
        #     3:"4",
        #     4:"5",
        #     5:"6",
        #     6:"7",
        # }
        self.pos2act = {value.replace(" ",""):key for (key,value) in self.act2pos.items()}
    def act(self, player_name, observation, reward, termination, truncation, info):
        action = self.sample_random_action(
            player_name, observation, reward, termination, truncation, info
        )
        return action

    def sample_random_action(
        self, player_name, observation, reward, termination, truncation, info
    ):
        return self._sample_random_action(
            player_name, observation, reward, termination, truncation, info
        )
    def _get_chat_answer(
        self,message
    ):
        messages = self.messages + [{'role':'user','content':message}]

        answer = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature = self.temperature
        )
        print(answer)
        answer = answer['choices'][0]['message']['content']
        #self.messages.append({"role":"assistant","content":answer})
        return answer

    def _get_chat_action(
            self,action_mask,observation
    ):

        board_status = self._get_board_status(observation=observation)
        message = f"You play {self.player_type}.\nThe column you put the mark on cannot be full.\n\nDon't say anything besides mark position.\nYou should only output {self.player_type} and the column of the move you choose to put your mark, for example: \"{self.player_type}: 1\"\n\nThe board status is {board_status}.\n Column [ "
        flag = False
        for idx,action in enumerate(action_mask):
                if action == 0:
                    flag = True
                    message = message + " " + self.act2pos[idx] + " ,"
        if flag:
            message = message[:-1]
        message = message + " ] is already Full\nYou can only choose one of following columns: ["
        for idx,action in enumerate(action_mask):
            if action == 1:
                message = message + " " + self.act2pos[idx] + " ,"
        message = message[:-1] + f"]."
        answer = self._get_chat_answer(message=message)
        pos = re.findall(r'[XO]\s*:\s*[1-7]',answer)[0].replace(" ","")
        pos = pos[-1]
        action = self.pos2act[pos]

        return action
    def _sample_random_action(
        self, player_name, observation, reward, termination, truncation, info
    ):
        
        openai.api_base = self.api_base
        openai.api_key = self.api_key
        self.player_type = "X" if player_name == 'player_0' else "O"
        action_space = self.env.action_space(player_name)
        if isinstance(action_space, list):
            if not isinstance(observation, list):
                observation = [observation]

            action = []

            for obs, space in zip(observation, action_space):
                mask = obs.get("action_mask", None)
                try:
                    action.append(self._get_chat_action(mask,obs))
                except:
                    action.append(space.sample(mask))
        else:
            mask = observation.get("action_mask", None)
            try:
                action = self._get_chat_action(mask,observation)
            except Exception as e:
                print(e)
                action = action_space.sample(mask)
        return action

    def _load(self, opponent_path):
        pass

    def _set_env(self, env, opponent_player: str):
        pass
if __name__ == "__main__":
    from pettingzoo.classic import connect_four_v3
    result1 = open("result1.txt","w")
    result2 = open("result2.txt","w")

    opponent1 = BaseOpponent(
        opponent_info={
            "api_base":"http://localhost:8000/v1",
            "api_key":"",
            "player_type": "X",
            "model_name": "vicuna-13b-v1.5",
            "temperature":0.1
            }
    )
    opponent2 = BaseOpponent(        
        opponent_info={
            "player_type": "O",
            "model_name": "vicuna-13b-v1.5",
            "temperature":0.1
            }
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
        print(player_name, action, type(action))
        env.step(action)
