
from openrl.selfplay.opponents.base_opponent import BaseOpponent
from pathlib import Path
from typing import Dict, Optional, Union
import openai
import random
import traceback
class BaseLLMOpponent(BaseOpponent):
    opponent_action = None
    def __init__(self,opponent_id: Optional[str] = None,
        opponent_path: Optional[Union[str, Path]] = None,
        opponent_info: Optional[Dict[str, str]] = None):
        super().__init__(opponent_id=opponent_id,opponent_path=opponent_path,opponent_info=opponent_info)
        self.need_history = False
        self.api_base = None
        self.stop_token = None
        for key,value in opponent_info.items():
            try:
                value = float(value)
            except ValueError:
                pass
            setattr(self,key,value)
        self.messages = [{"role": "system", "content": self.prompt_rule}]
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
    def _reset_message(
        self
    ):
        self.messages =  [{"role": "system", "content": self.prompt_rule}]
    def _get_chat_answer(
        self,message
    ):
        self.messages.append({"role":"user","content":message})
        if self.stop_token is not None:

            answer = openai.ChatCompletion.create(
                model=self.model_name,
                messages=self.messages,
                temperature = self.temperature,
                #stop = [self.stop_token]
            )
        else:
            answer = openai.ChatCompletion.create(
                model=self.model_name,
                messages=self.messages,
                temperature = self.temperature,
            )
        answer = answer['choices'][0]['message']['content']
        #print(self.messages)
        print(f'==========={self.model_name}:{self.player_name}===========')
        print(message)
        print(f'-----------------------------------------')
        print(answer)
        print(f'=========================================')

        if self.need_history:
            self.messages.append({"role":"assistant","content":answer})
        else:
            self._reset_message()
        return answer
    def _get_chat_action(
            self,action_mask, observation, info = None
    ):
        indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]

        if not indices_of_ones:
            return None
        return random.choice(indices_of_ones)
    def _sample_random_action(
        self, player_name, observation, reward, termination, truncation, info
    ):
        
        self.player_name = player_name  
        if self.api_base is not None:
            openai.api_base = self.api_base
        openai.api_key = self.api_key

        action_space = self.env.action_space(player_name)
        if isinstance(action_space, list):
            if not isinstance(observation, list):
                observation = [observation]

            action = []

            for obs, space in zip(observation, action_space):
                mask = obs.get("action_mask", None)
                try:
                    action.append(self._get_chat_action(mask,obs,info))
                except:
                   action.append(space.sample(mask))
        else:
            mask = observation.get("action_mask", None)
            try:
                action = self._get_chat_action(mask,observation,info)
            except Exception as e:
               print(traceback.format_exc())
               action = action_space.sample(mask)
        return action
    def _load(self, opponent_path):
        pass

    def _set_env(self, env, opponent_player: str):
        pass