import numpy as np
import game_core as gc
import copy
import threading
import sys
import pygame
import os

from tao_zero.main.chinese_chess_main.chinese_chess import message

class MyThread(threading.Thread):
    def run(self):
        if self._target is not None:
            self._return=self._target(*self._args,**self._kwargs)
    def join(self):
        super().join()
        return self._return

class ChessAIPlayer:
    def __init__(self,direction=1):
        self.general_position = "04"  # 初始将帅位置
        self.move_action = []
        self.choose = None
        self.choose_move = []
        self.last_move = None
        self.warn = False
        self.thread = None
        self.direction = direction
        message('start')

    def reset_tree(self):
        # 重置搜索树（如果需要）
        pass

    def update_tree_by_action(self, action_id):
        # 更新搜索树（如果需要）
        pass

    def to_ai_move(self,move):
        if move[0] == '-':
            return "9999"
        elif self.direction == 1:
            from_file = str(9-int(move[0]))
            from_rank = str(8-int(move[1]))
            to_file = str(9-int(move[2]))
            to_rank = str(8-int(move[3]))
            return from_rank + from_file + to_rank + to_file
        else:
            from_file = move[0]
            from_rank = move[1]
            to_file = move[2]
            to_rank = move[3]
            return from_rank + from_file + to_rank + to_file

    def parse_ai_move(self,move):
        if self.direction == 1:
            from_file = str(8-int(move[0]))
            from_rank = str(9-int(move[1]))
            to_file = str(8-int(move[2]))
            to_rank = str(9-int(move[3]))
            return from_rank + from_file + to_rank + to_file
        else:
            from_file = move[0]
            from_rank = move[1]
            to_file = move[2]
            to_rank = move[3]
            return from_rank + from_file + to_rank + to_file

    def get_action_long(self, state_deque, current_player, opponent_general_position, mct_prob=False,opponent_move=None):

        print(self.direction)
        # 获取所有合法走法
        self.move_action = gc.get_all_move_action(state_deque[-1], current_player)
        
        # 使用中国象棋.py的AI进行决策
        go = message(self.to_ai_move(opponent_move))
        move = self.parse_ai_move(go)
        
        # 获取对手可能的走法
        opponent_move_action = gc.get_all_move_action(
            gc.state_change_by_move(state_deque[-1], move),
            -current_player
        )

        # 如果需要返回概率分布
        if mct_prob:
            move_prob = np.zeros(2086)  # 总走法数
            move_id = gc.action_to_id_mapping[move]
            move_prob[move_id] = 1.0
            return move, opponent_move_action, move_prob
        else:
            return move, opponent_move_action

    def get_action(self, state_deque, current_player, opponent_general_position,opponent_move):
        if self.thread is None:
            self.thread=MyThread(target=self.get_action_long,
                             args=(state_deque, current_player, opponent_general_position, False,opponent_move))
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

    # def get_action(self, state_deque, current_player, opponent_general_position, opponent_move):
    #     return self.get_action_long(state_deque, current_player, opponent_general_position,opponent_move=opponent_move)