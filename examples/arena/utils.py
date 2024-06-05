from pettingzoo.classic import connect_four_v3
from pettingzoo.classic import chess_v6
from pettingzoo.classic import go_v5
from pettingzoo.classic import texas_holdem_no_limit_v6
from pettingzoo.classic import hanabi_v5
import random
import numpy as np

from openrl.envs.PettingZoo.registration import register


def ConnectFourEnv(render_mode, **kwargs):
    return connect_four_v3.env(render_mode=None)
def ChessEnv(render_mode, **kwargs):
    return chess_v6.env(render_mode=None)
def GoEnv(render_mode, **kwargs):
    return go_v5.env(render_mode=None)
def TexasEnv(render_mode, **kwargs):
    return texas_holdem_no_limit_v6.env(render_mode=None)
def HanabiEnv(render_mode, **kwargs):
    return hanabi_v5.env(render_mode=None)
def register_new_envs():
    register("connect_four_v3", ConnectFourEnv)
    register("chess_v6", ChessEnv)
    register("go_v5", GoEnv)
    register("texas_no_limit_v6", TexasEnv)
    register("hanabi_v5", HanabiEnv)

    return ["connect_four_v3","chess_v6","go_v5","texas_no_limit_v6","hanabi_v5"]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def select_pair(llms):
    llm1 = random.choice(list(llms.keys()))
    llm1_ratings = llms[llm1][0].mu
    diffs = []
    names = []
    for name , rating in llms.items():
        if name != llm1:
            diff = abs(llm1_ratings-rating[0].mu)
            diffs.append(diff)
            names.append(name)
    probs = softmax(np.array(diffs))
    llm2 = np.random.choice(names,p=probs)
    return llm1,llm2

