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
from openrl.arena import make_arena
from openrl.arena.agents.local_agent import LocalAgent
from openrl.envs.wrappers.pettingzoo_wrappers import RecordWinner
from openrl.selfplay.selfplay_api.opponent_model import BattleResult
from pettingzoo.classic import connect_four_v3
from pettingzoo.classic import chess_v6
from pettingzoo.classic import go_v5
from pettingzoo.classic import texas_holdem_no_limit_v6
from pettingzoo.classic import hanabi_v5
from openrl.envs.bargain_env import bargain_v1
from openrl.envs.bid_env import bid_v1
from openrl.envs.undercover_env import undercover_v1

from openrl.envs.PettingZoo.registration import register

import json
import random
import numpy as np
import os
import requests
import trueskill
def get_models():
    url = 'http://172.26.1.16:31251/get_model_list'  # 修改为您的服务器地址和端口
    response = requests.post(url)

    if response.status_code == 200:
        data = response.json()
        models = data.get('models', [])
        print('Models:', models)
        return models
    else:
        print('Error:', response.status_code)
        return []
    

def ConnectFourEnv(render_mode, **kwargs):
    return connect_four_v3.env(render_mode=None)
def ChessEnv(render_mode, **kwargs):
    return chess_v6.env(render_mode=None)
def GoEnv(render_mode, **kwargs):
    return go_v5.env(board_size = 5)
def TexasEnv(render_mode, **kwargs):
    return texas_holdem_no_limit_v6.env(render_mode=None)
def BargainEnv(render_mode, **kwargs):
    return bargain_v1.env(render_mode=None)
def BidEnv(render_mode, **kwargs):
    return bid_v1.env(render_mode=None)
def HanabiEnv(render_mode, **kwargs):
    return hanabi_v5.env(colors=2, ranks=5, players=2, hand_size=2, max_information_tokens=3, max_life_tokens=1, observation_type='card_knowledge')
def UndercoverEnv(render_mode, **kwargs):
    return undercover_v1.env(render_mode=None)
def register_new_envs():
    register("connect_four_v3", ConnectFourEnv)
    register("chess_v6", ChessEnv)
    register("go_v5", GoEnv)
    register("texas_no_limit_v6", TexasEnv)
    register("hanabi_v5", HanabiEnv)
    register("bargain_v1", BargainEnv)
    register("bid_v1", BidEnv)
    register("undercover_v1", UndercoverEnv)

    return ["connect_four_v3","chess_v6","go_v5","texas_no_limit_v6","hanabi_v5"]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def default_dispatch_func(
    np_random: np.random.Generator,
    players,
    agent_names
) :
    assert len(players) == 2, "The number of players must be equal to 2."
    np_random.shuffle(agent_names)
    return dict(zip(players, agent_names[:2]))

def default_dispatch_func_multi(
    np_random: np.random.Generator,
    players,
    agent_names,
    num_agents = 5
) :
    assert len(players) == num_agents, f"The number of players must be equal to {num_agents}."
    np_random.shuffle(agent_names)
    return dict(zip(players, agent_names[:num_agents]))

           

def run_arena(
    render: bool = False,
    parallel: bool = True,
    seed=0,
    total_games: int = 10,
    max_game_onetime: int = 50,
    use_tqdm: bool = True,
    game_envs: list = [],
):

    results = {}
    if os.path.exists("result.json"):
        results = json.load(open("result.json",'r'))
    idx = 0
    while True:
        idx += 1
        models = get_models()
        models = [] #Your models here
        llms = [model.split('/')[-1].replace(".","") for model in models]
        #llms = [model for model in llms if "gpt" not in model]
        env = random.choice(game_envs)
        env_wrappers = [RecordWinner]
        arena = make_arena(env, env_wrappers=env_wrappers, use_tqdm=use_tqdm, dispatch_func = default_dispatch_func,dispatch_func_multi = default_dispatch_func_multi)    
        agents = {llm:LocalAgent(f"../selfplay/opponent_templates/{env}/{llm}") for llm in llms}
        if env in results:
            for llm in agents:
                if llm in results[env]:
                    agents[llm].rating = trueskill.Rating(mu = results[env][llm]["rating"]["mu"], sigma=results[env][llm]["rating"]["sigma"])
                    #print(agents[llm].rating)
        arena.reset(
            agents=agents,
            total_games=10,
            max_game_onetime=max_game_onetime,
            seed=seed + 10 * idx,
        )
        result = arena.run(parallel=parallel)
        print(result)
        arena.close()
        if os.path.exists("result.json"):
            results = json.load(open("result.json",'r'))
        if env in results:
            for llm in result:
                if llm in results[env]:
                    before_games = results[env][llm]['total_games']
                    after_games = result[llm]['total_games']
                    total_games = max(before_games + after_games, 1)
                    results[env][llm]['win_rate'] = (results[env][llm]['win_rate'] * before_games + result[llm]['win_rate'] * after_games ) / total_games 
                    results[env][llm]['loss_rate'] = (results[env][llm]['loss_rate'] * before_games + result[llm]['loss_rate'] * after_games ) / total_games 
                    results[env][llm]['draw_rate'] = (results[env][llm]['draw_rate'] * before_games + result[llm]['draw_rate'] * after_games ) / total_games 
                    #results[env][llm]['non_zero_rate'] = (results[env][llm]['non_zero_rate'] * before_games + result[llm]['non_zero_rate'] * after_games ) / total_games 
                    results[env][llm]['total_reward'] += result[llm]['total_reward']
                    results[env][llm]['avg_reward'] = results[env][llm]['total_reward'] / total_games
                    results[env][llm]['total_games'] = before_games + after_games
                    results[env][llm]['rating'] = result[llm]['rating']
                else:
                    results[env][llm] = result[llm]
        else:
            results[env] = result
        print("="*50)
        print(result)   
        print("="*50)
        print(results)
        print("="*50)
        json.dump(results,open("result.json",'w'),indent=4)


if __name__ == "__main__":
    register_new_envs()
    game_envs = [] #envs here
    for game_env in game_envs:
        results = run_arena(render=False, parallel=True, seed=42, total_games=400, max_game_onetime=10, game_envs=game_envs)
        print(results)
    json.dump(results,open('result.json','w'),indent=4)
    