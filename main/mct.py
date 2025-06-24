import copy
import sys

import numpy as np
import threading

import pygame

from board import Board
import game_core as gc
from config import CONFIG


def soft_max(x):
    prob=np.exp(x-np.max(x))
    prob/=np.sum(prob)
    return prob

class TreeNode:
    def __init__(self,parent,priority_rate):
        self.parent=parent
        self.children={}
        self.visit=0
        self.Q=0
        self.U=0
        self.P=priority_rate
    def is_root(self):
        return self.parent is None
    def is_leaf(self):
        return self.children=={}
    def update_tree(self,child_value):
        self.visit+=1
        self.Q+=1.0*(child_value-self.Q)/self.visit
    def expand(self,action_prob):
        for action,prob in action_prob:
            if action not in self.children:
                self.children[action]=TreeNode(self,prob)
    def get_value(self,_c):
        self.U=_c*self.P*np.sqrt(self.parent.visit)/(1+self.visit)
        return self.Q+self.U
    def select(self,_c):
        return max(self.children.items(),key=lambda node:node[1].get_value(_c))
    def update_reverse(self,child_value):
        if self.parent:
            self.parent.update_reverse(-child_value)
        self.update_tree(child_value)

class MCT_Tree:
    def __init__(self,policy_value_function,_c=5,search_num=2000):
        self.root=TreeNode(None,1.0)
        self.policy_value_function=policy_value_function
        self._c=_c
        self.search_num=search_num
        self.board=Board(PlayerAI(None),PlayerAI(None))
    def search(self):
        tree_node=self.root
        while True:
            if tree_node.is_leaf():
                break
            self.board.judge_action_count()
            action_id,tree_node=tree_node.select(self._c)
            move=gc.id_to_action_mapping[action_id]
            player_id=self.board.get_current_player_id()
            new_state=gc.state_change_by_move(self.board.state_deque[-1],move)
            opponent_move_action=gc.get_all_move_action(new_state,-player_id)
            self.board.player_agent[-player_id].move_action = opponent_move_action
            self.board.judge_final_hit(move,player_id,opponent_move_action)
        current_player=self.board.get_current_player_id()
        state_array, move_id = self.board.player_agent[current_player].get_step_data(self.board.state_deque,
                                                                                     current_player,
                                                                                     self.board.player_agent[
                                                                                        -current_player].general_position)
        policy, value=self.policy_value_function(state_array,move_id)
        if self.board.play:
            tree_node.expand(policy)
        else:
            value=self.board.winner/current_player*1.0
        tree_node.update_reverse(-value)

    def get_action_prob(self,state_deque,player_id,self_general_position,opponent_general_position,temp=1e-3):
        for _ in range(self.search_num):
            self.board.init(player_id,copy.deepcopy(state_deque),self_general_position,opponent_general_position)
            self.search()

        action_visit=[(action,node.visit)for action,node in self.root.children.items()]
        action,visit=zip(*action_visit)
        action_prob=soft_max(1.0/temp*np.log(np.array(visit)+1e-10))
        return action,action_prob

    def update_root_by_move(self,last_move):
        if last_move in self.root.children:
            self.root=self.root.children[last_move]
            self.root.parent=None
        else:
            self.root=TreeNode(None,1.0)


class MyThread(threading.Thread):
    def run(self):
        if self._target is not None:
            self._return=self._target(*self._args,**self._kwargs)
    def join(self):
        super().join()
        return self._return
class PlayerAI:
    def __init__(self,policy_value_function,_c=5,search_num=2000,use_noise=False):
        self.general_position = "04"

        #得分模式
        self.score = 0

        self.move_action = []
        self.choose = None
        self.choose_move = []
        self.last_move = None
        self.warn = False
        self.thread=None
        if policy_value_function:
            self.mct_tree=MCT_Tree(policy_value_function,_c,search_num)
        self.use_noise=use_noise
    def reset_tree(self):
        self.mct_tree.update_root_by_move(-1)
    def update_tree_by_action(self,action_id):
        self.mct_tree.update_root_by_move(action_id)
    def get_step_data(self,state_deque,current_player,opponent_general_position):
        move_id=[]
        for item in self.move_action:
            if item[0:2] == self.general_position:
                self_general_position = item[2:4]
            else:
                self_general_position = self.general_position
            temp_state=gc.state_change_by_move(state_deque[-1],item)
            if gc.general_meet(temp_state,self_general_position,opponent_general_position):
                continue
            elif gc.general_warn(temp_state,-current_player,self_general_position):
                continue
            move_id.append(gc.action_to_id_mapping[item])
        params=np.empty([17,10,9])
        for i in range(16):
            params[i] = copy.deepcopy(state_deque[i - 16])
        params[16][:, :] = current_player
        return params,move_id
    def get_action_long(self,state_deque,current_player,opponent_general_position,mct_prob=False,opponent_move=None):
        state_deque_copy = copy.deepcopy(state_deque)
        action, action_prob = self.mct_tree.get_action_prob(state_deque_copy, current_player, self.general_position,
                                                            opponent_general_position)

        move_prob = np.zeros(2086)
        move_prob[list(action)] = action_prob


        if self.use_noise:
            move_id = np.random.choice(action, p=0.75 * action_prob + 0.25 * np.random.dirichlet(
                CONFIG["dirichlet"] * np.ones(len(action_prob))))
            # self.update_tree_by_action(move_id)
            self.reset_tree()
        else:
            move_id = np.random.choice(action, p=action_prob)
            self.reset_tree()
        move = gc.id_to_action_mapping[move_id]
        opponent_move_action = gc.get_all_move_action(gc.state_change_by_move(state_deque[-1], move), -current_player)
        if mct_prob:
            return move, opponent_move_action, move_prob
        else:
            return move, opponent_move_action
    def get_action(self,state_deque,current_player,opponent_general_position,opponent_move=None,mct_prob=False):
        if self.thread is None:
            self.thread=MyThread(target=self.get_action_long,
                             args=(state_deque, current_player, opponent_general_position, mct_prob))
            self.thread.start()
            return None,[]
        elif self.thread.is_alive():
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    break
                elif event.type == pygame.QUIT:
                    sys.exit()
            return None, []
        else:
            _return=self.thread.join()
            self.thread=None
            return _return