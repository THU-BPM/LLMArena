from openrl.selfplay.opponents.base_llm_opponent import BaseLLMOpponent
from pathlib import Path
from typing import Dict, Optional, Union
import openai
import re
import os
import numpy as np
import random
import time

class Opponent(BaseLLMOpponent):
    opponent_action = None
    round_retry = 0
    round_illegal = 0
    def __init__(self,opponent_id: Optional[str] = None,
        opponent_path: Optional[Union[str, Path]] = None,
        opponent_info: Optional[Dict[str, str]] = None):
        super().__init__(opponent_id=opponent_id,opponent_path=opponent_path,opponent_info=opponent_info)
        self.idx2color = {
            0: "Red",
            1: "Yellow",
        }
        self.act2des = {
            0: "Discard Card at position 0",
            1: "Discard Card at position 1",
            2: "Play Card at position 0",
            3: "Play Card at position 1",
            4: "Reveal Red Cards for another player",
            5: "Reveal Yellow Cards for another player",
            6: "Reveal Rank 1 Cards for another player",
            7: "Reveal Rank 2 Cards for another player",
            8: "Reveal Rank 3 Cards for another player",
            9: "Reveal Rank 4 Cards for another player",
            10:"Reveal Rank 5 Cards for another player",
        }
        self.riskrank = {
            0: 5,
            1: 5,
            2: 10,
            3: 10,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10:0,
        }
        self.pos2act = {value:key for (key,value) in self.act2des.items()}
        self.otherhands = set()
    
    def _decode_cards(self,position,observation):
        cards = set()
        observation = observation[:10]
        for idx,card in enumerate(observation[:5]):
            if card == 1:
                cards.add(position + " : Red " + str(idx+1))
        for idx,card in enumerate(observation[5:10]):
            if card == 1:
                cards.add(position + " : Yellow " + str(idx+1))
        return cards
    
    def _obs_cards(self,observation):
        zero_card = self._decode_cards("position 0" , observation[:10])
        first_card = self._decode_cards("position 1" , observation[10:20])
        otherhands = set.union(zero_card, first_card)
        return otherhands
        

    def _obs_fireworks(self,observation):
        fireworks = set()
        observation = observation[38:48]
        for idx,firework in enumerate(observation[:5]):
            if firework == 1:
                fireworks.add("the score of Red Firework is " + str(idx+1))
        if all(num ==0 for num in observation[:5]):
            fireworks.add("the score of Red Firework is 0")
        for idx,firework in enumerate(observation[5:10]):
            if firework == 1:
                fireworks.add("the score of Yellow Firework is " + str(idx+1))
        if all(num ==0 for num in observation[5:10]):
            fireworks.add("the score of Yellow Firework is 0")
        return fireworks
    
    def _obs_token(self,observation):
        token = set()
        observation = observation[48:52]
        infonum = str(observation[:3]).count("1")
        token.add("infotoken: " + str(infonum))
        if all(num ==0 for num in observation[:3]):
            token.add("infotoken: 0")
        if observation[3] == 1:
            token.add("lifetoken: 1")
        else:
            token.add("lifetoken: 0")
        return token
    
    def _extract_info(self,order,observation):
        revealedcards = set()
        observation = observation[:17] 
        if all(num == 1 for num in observation[:5]):
            revealedcards.add("the " + order + " card could be red")
        if all(num == 1 for num in observation[5:10]):
            revealedcards.add("the " + order + " card could be yellow")
        for idx,info in enumerate(observation[10:12]):
            if info == 1:
                revealedcards.add("the " + order + " card was revealed to be " + self.idx2color[idx])
        for idx,info in enumerate(observation[12:17]):
            if info == 1:
                revealedcards.add("the " + order + " card was revealed to be of rank " + str(idx+1)) 
        return revealedcards
        
    def _obs_info(self,observation):
        observation = observation[103:147]
        firstcard = self._extract_info("position 0", observation[:17])
        secondcard = self._extract_info("position 1", observation[17:34])
        return firstcard, secondcard
    
    def _obs_deck(self,observation):
        decksize = 0
        observation = observation[20:38]
        decksize = str(observation).count("1")
        if all(num == 0 for num in observation):
            decksize = 0
        return decksize
        
    def _obs_lastplay(self,observation):
        observation = observation[91:100]
        lastplayed = None
        if Opponent.opponent_action is not None:
            if int(self.pos2act[Opponent.opponent_action]) == 2 or int(self.pos2act[Opponent.opponent_action]) == 3:
                lastplayed = self._decode_cards("last played card", observation)
            if int(self.pos2act[Opponent.opponent_action]) == 0 or int(self.pos2act[Opponent.opponent_action]) == 1:
                lastplayed = self._decode_cards("last discarded card", observation)    
        return lastplayed

    def _get_board_status(self,observation):
        board_status = "\n"
        self.otherhands = self._obs_cards(observation['observation'])
        self.fireworks = self._obs_fireworks(observation['observation'])
        self.token = self._obs_token(observation['observation'])
        self.lastplayed = self._obs_lastplay(observation['observation'])
        self.firstcard, self.secondcard = self._obs_info(observation['observation'])
        self.decksize = self._obs_deck(observation['observation'])
        

    def _get_chat_action(
            self,action_mask,observation,info = None
    ):
        max_retries = 1
        current_retry = 0
        illegal_count = 0


        while current_retry < max_retries:
            try:
                self._get_board_status(observation=observation)
                message = f"Your observation now is :\n"
                message += f"The cards in another player's hands is :\n {self.otherhands}\n"
                message += f"The current fireworks is :\n {self.fireworks}\n"
                message += f"The current number of tokens is {self.token}\n"
                message += f"The information of your own cards that was revealed is:\n {self.firstcard, self.secondcard}\n"
                message += f"The current deck size is {self.decksize}\n"
                if Opponent.opponent_action is not None:
                    message += f"Your opponent's action is {Opponent.opponent_action}\n"
                message += f"The card that opponent last played or discarded is {self.lastplayed}\n"
                message += "You should think step by step and output your action. For example: 'Reveal Yellow Cards for another player'\n"
                message += f"Now you can choose one of the following actions:\n"
                for idx,action in enumerate(action_mask):
                    if action == 1:
                        message += f"{self.act2des[idx]} \n"
                message += f"You will respond with an action, formatted as:\n Action: <action>\n where you replace <action> with your actual action.\n"
                message += f"\nYou should explain why you choose the action\n"
                answer = self._get_chat_answer(message=message)
                action_str = "|".join(list(self.act2des.values()))
                pattern = rf"Action:\s*({action_str})"
                match = re.search(pattern, answer)
                if match:
                    pos = match.group(1)
                    pos = pos.rstrip('.')
                    print("提取的行动:", pos)
                    action = int(self.pos2act[pos])

                    Opponent.opponent_action = pos
                    break
                else:
                    print("未找到匹配。")
                    current_retry += 1 
            except Exception as e:
                print(e.__context__,e)
                current_retry += 1 
                time.sleep(1)
        else:
            indices_of_ones = [i for i, value in enumerate(action_mask) if value == 1]
            action = random.choice(indices_of_ones)
            print("random action:", action)
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

def get_risk(filename, action_risk, avg_action_risk):
    risk_count = action_risk
    print(risk_count)
    if os.path.exists(filename):
        pass
    else: 
        with open(filename, 'w') as json_file:
            data = {"rounds": 0, "total_risk": 0, "round_risk": 0, "avg_risk": 0, "avg_action_risk": 0, "action_risk": {}}
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
    desired_key1 = "total_risk"
    rounds = last_data["rounds"] + 1
    if desired_key1 in last_data:
        desired_value1 = last_data[desired_key1]
        total_risk = desired_value1 + risk_score
        avg_risk = total_risk / rounds
    desired_key2 = "action_risk"
    if desired_key2 in last_data:
        for k,v in last_data[desired_key2].items():
            str_key = int(k)
            risk_count[str_key] += v
    desired_key3 = "avg_action_risk"
    if desired_key3 in last_data:
        avg_action_risk = (last_data[desired_key3]*(rounds-1) + avg_action_risk) / rounds
    new_data = {"rounds": rounds ,"total_risk": total_risk, "round_risk": risk_score, "avg_risk": avg_risk, "avg_action_risk": avg_action_risk, "action_risk": risk_count}  
    with open(filename, 'a') as json_file:
        json.dump(new_data, json_file)
        json_file.write('\n')

if __name__ == "__main__":
    from pettingzoo.classic import hanabi_v5
    import json
    result1 = open("result1.txt","w")
    result2 = open("result2.txt","w")
    info = json.load(open("./info.json","r"))
    total_retry = 0
    total_illegal = 0
    risk_score = 0 #风险分数
    average_risk = 0 #平均风险分数
    action_risk = {0:0, 5:0, 10:0} #不同风险等级动作的次数
    opponent1 = Opponent(
        opponent_info = info
    )
    opponent2 = Opponent(        
        opponent_info = info
    )
    env = hanabi_v5.env(colors=2, ranks=5, players=2, hand_size=2, max_information_tokens=3, max_life_tokens=1, observation_type='card_knowledge')
    opponent1.reset(env, "player_0")
    opponent2.reset(env, "player_1")
    player2opponent = {"player_0": opponent1, "player_1": opponent2}
    
    env.reset()
    index = 0
    for player_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination:
            result1.write(str(opponent1.messages))
            result2.write(str(opponent2.messages))
            break
        action = player2opponent[player_name].act(
            player_name, observation, reward, termination, truncation, info
        )
        index += 1 
        risk = player2opponent[player_name].riskrank[action]
        if risk in action_risk:
            action_risk[risk] += 1
        else:
            action_risk[risk] = 1
        risk_score += risk
        avg_action_risk = risk_score/index
        round_retry = player2opponent[player_name].round_retry
        round_illegal = player2opponent[player_name].round_illegal
        print(player_name, action, type(action))
        print("reward: ", reward)
        env.step(action)
    get_retry("retry&illegal.json")
    get_risk("risk.json", action_risk, avg_action_risk)
    

