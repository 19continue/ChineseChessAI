import copy
import math
import sys
import time
import numpy as np
from collections import deque
import game_core as gc
import pygame

animation={"choose":False,"move":False,"choose_time":0.,"move_time":0.,"choose_position":(-1,-1),"move_begin":(-1,-1),"move_to":(-1,-1)}
message={
    "show":False,
    "show_time":0.,
    "leave_time":0.,
    "content":"",
    "long":1.0
}

score_table={1:100,2:10,3:10,4:20,5:40,6:20,7:5}

win_score = 230

soldier_upgrade=False

class Renderer:
    def __init__(self,offensive):
        pygame.init()
        pygame.mixer.init()
        self.width = 720
        self.height = 800
        self.screen=pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("中国象棋-TaoZero")
        self.beam_surface =self.screen.convert_alpha()
        self.board_surface=self.screen.convert_alpha()
        self.main_surface=self.screen.convert_alpha()
        font = pygame.font.match_font('fangsong')
        self.fontObj = pygame.font.Font(font, 30)
        if offensive==1:
            self.num_to_word={ -1:"将",-2:"士",-3:"象",-4:"马",-5:"车",-6:"炮",-7:"卒",1:"帅",2:"仕",3:"相",4:"马",5:"车",6:"炮",7:"兵"}
            self.word_color={-1:(0,0,0),1:(250, 21, 24)}
        else:
            self.num_to_word = {1: "将", 2: "士", 3: "象", 4: "马", 5: "车", 6: "炮", 7: "卒", -1: "帅", -2: "仕", -3: "相",
                                -4: "马", -5: "车", -6: "炮", -7: "兵"}
            self.word_color = {1: (0, 0, 0), -1: (250, 21, 24)}
        self.move_snd = pygame.mixer.Sound("../sound/move.wav")
        self.warn_snd = pygame.mixer.Sound("../sound/warn.mp3")
        self.eat_snd = pygame.mixer.Sound("../sound/eat.mp3")
        self.final_kill_snd = pygame.mixer.Sound("../sound/final_kill.mp3")
    def draw_line(self, color, point1, point2, weight):
        pygame.draw.line(self.board_surface, color, point1, point2, weight)
    def draw_one_chess(self,x,y,r,color,word):
        for i in range(8):
            pygame.draw.circle(self.main_surface, (169 + i * 10, 120 + i * 10, 69 + i * 10), (x, y), r - i/30*r)
        pygame.draw.arc(self.main_surface, (189, 140, 89), (x - r, y - r, round(2*r), round(2*r)), math.radians(260),
                        math.radians(50), round(1/30*r))
        pygame.draw.arc(self.main_surface, (189, 140, 89), (
        round(x - r + 1 / 30 * r), round(y - r + 1 / 30 * r), round(2 * r - 1 / 15 * r), round(2 * r - 1 / 15 * r)),
                        math.radians(260),
                        math.radians(50), round(1 / 30 * r))
        pygame.draw.arc(self.main_surface, (189, 140, 89), (
        round(x - r + 1 / 15 * r), round(y - r + 1 / 15 * r), round(2 * r - 2 / 15 * r), round(2 * r - 2 / 15 * r)),
                        math.radians(260),
                        math.radians(50), round(1 / 30 * r))
        pygame.draw.circle(self.main_surface, (255, 203, 138), (x, round(y - 2/15*r)), round(r-2/15*r))
        pygame.draw.circle(self.main_surface, color, (x, round(y - 2/15*r)), round(r-4/15*r), round(1/30*r))
        for i in range(4):
            pygame.draw.arc(self.main_surface, (255, 222, 148), (
            round(x - r + 1 / 6 * r - i / 30 * r), round(y - r - 1 / 30 * r - 2 * i / 30 * r),
            round(2 * r - 7 / 30 * r + 2 * i / 30 * r), round(2 * r - 7 / 30 * r + 2 * i / 30 * r)),
                            math.radians(200 + 5 * i), math.radians(250 - 2 * i), round(1 / 30 * r))
        pygame.draw.arc(self.main_surface, (255, 222, 148),
                        (round(x - r+2/15*r), round(y - r-0.1*r), round(2*r-1/6*r), round(2*r-1/6*r)), math.radians(180),
                        math.radians(250), round(1/30*r))
        pygame.draw.arc(self.main_surface, (255, 222, 148),
                        (round(x - r+1/15*r), round(y - r-1/6*r), round(2*r-1/30*r), round(2*r-0.1*r)), math.radians(230),
                        math.radians(260), round(1/30*r))
        pygame.draw.arc(self.main_surface, (255, 222, 148),
                        (round(x - r+0.1*r), round(y -r-1/6*r), round(2*r-0.1*r), round(2*r-0.1*r)), math.radians(210),
                        math.radians(260), round(1/30*r))
        font = pygame.font.match_font('fangsong')
        font_object = pygame.font.Font(font, round(r))
        text = font_object.render(word, True, color, None)
        self.main_surface.blit(text, (round(x - 0.5*r), round(y - 2/3*r)))
    def show_message(self):
        if message["show"]:
            message["show_time"] = time.time() + message["long"]
            message["leave_time"] = message["show_time"] + 0.15
            message["show"]=False
        now_time = time.time()
        d_time = message["show_time"] - now_time
        l_time = message["leave_time"] - now_time
        if d_time>0 or message["long"]==-1:
            alpha=255
            width = len(message["content"]) * 20 + 40
            px=self.width/2-width/2
            py=self.height/2-25
            if d_time-0.8*message["long"]>0:
                alpha=round((1.0-((d_time-0.8*message["long"])/0.2*message["long"]))*255)
                py=py+((d_time-0.8*message["long"])/0.2*message["long"])*50
            pygame.draw.rect(self.main_surface, (208,220,230,alpha),(px,py,width,50),25,10)
            font = pygame.font.match_font('fangsong')
            font_object = pygame.font.Font(font, 20)
            text = font_object.render(message["content"], True, (43,43,43), None)
            self.main_surface.blit(text, (px + 20, py + 14))
        elif l_time>0:
            alpha = round((l_time/0.2)*255)
            width = len(message["content"]) * 20 + 20
            px = self.width / 2 - width / 2
            py = self.height / 2 - 25
            pygame.draw.rect(self.main_surface, (208, 220, 230, alpha), (px, py, width, 50), 25, 10)
        pass
    def chess_move(self,color,word):
        if animation["move"]:
            animation["move_time"] = time.time() + 0.2
            animation["move"] = False
        now_time = time.time()
        d_time = animation["move_time"] - now_time
        if d_time>0:
            x,y=animation["move_begin"]
            tox,toy=animation["move_to"]
            x,y,tox,toy=40+x*80,40+80*y,40+80*tox,40+80*toy
            pygame.draw.ellipse(self.main_surface, (
            round(181 + d_time / 0.2 * 10), round(131 + d_time / 0.2 * 10), round(84 + d_time / 0.2 * 10),
            round(160 - d_time / 0.2 * 80)), (
                                round(tox + d_time / 0.2 * (x - tox) - 10), round(toy + d_time / 0.2 * (y - toy) - 10),
                                round(75 - d_time / 0.2 * 10), (85 - d_time / 0.2 * 10)))
            self.draw_one_chess(tox + d_time / 0.2 * (x - tox), toy + d_time / 0.2 * (y - toy),
                                30 + (d_time / 0.2) * 50 - ((d_time / 0.2) ** 2) * 40, color, word)
        else:
            animation["move_begin"]=(-1,-1)
            animation["move_to"] = (-1, -1)
    def choose_chess(self,x,y,color,word):
        if animation["choose"]:
            animation["choose_position"] = (x, y)
            animation["choose_time"] =time.time() + 0.2
            animation["choose"] = False
        now_time=time.time()
        d_time=animation["choose_time"]-now_time
        if d_time>0:
            pygame.draw.ellipse(self.main_surface, (
            round(191 - d_time / 0.2 * 10), round(141 - d_time / 0.2 * 10), round(94 - d_time / 0.2 * 10),
            round(80 + d_time / 0.2 * 80)), (round(x - 10 - d_time / 0.2 * 10), round(y - 10 - d_time / 0.2 * 10),
                                             round(65 - d_time / 0.2 * 10), (75 - d_time / 0.2 * 10)))
            pygame.draw.ellipse(self.main_surface, (191, 141, 94, 110), (
            round(x - d_time / 0.2 * 20), round(y - d_time / 0.2 * 25), round(45 - d_time / 0.2 * 35),
            round(55 - d_time / 0.2 * 45)))
            self.draw_one_chess(x, y, 40 - d_time/0.2*10,color,word)
        else:
            pygame.draw.ellipse(self.main_surface, (191, 141, 94, 80), (x-10, y-10, 65, 75))
            pygame.draw.ellipse(self.main_surface, (191, 141, 94, 110), (x , y , 45, 55))
            self.draw_one_chess(x, y, 40, color, word)
    def draw_chess(self,current_state,choose):
        move=None
        for l in range(10):
            for n in range(9):
                chess=current_state[l][n]
                if chess!=0:
                    x, y = 40+80*n,40+80*l
                    if choose == str(l)+str(n):
                        self.choose_chess(x,y,self.word_color[chess/abs(chess)],self.num_to_word[chess])
                        continue
                    elif (n,l) == animation["move_to"]:
                        move=chess
                        continue
                    for i in range(9):
                        pygame.draw.circle(self.main_surface, (117, 93, 66, 160), (x + i, y + (i+1)*1), 31-i*0.1)
                    for i in range(10):
                        pygame.draw.circle(self.main_surface, (117-4*i, 93-4*i, 66-4*i,160), (x+10, y+12), 29-i)
                    for i in range(8):
                        pygame.draw.circle(self.main_surface, (169+i*10, 120+i*10, 69+i*10), (x, y), 30 - i)
                    pygame.draw.arc(self.main_surface, (189, 140, 89), (x - 30, y - 30, 60, 60), math.radians(260),
                                    math.radians(50), 1)
                    pygame.draw.arc(self.main_surface, (189, 140, 89), (x - 29, y - 29, 58, 58), math.radians(260),
                                    math.radians(50), 1)
                    pygame.draw.arc(self.main_surface, (189, 140, 89), (x - 28, y - 28, 56, 56), math.radians(260),
                                    math.radians(50), 1)
                    pygame.draw.circle(self.main_surface, (255, 203, 138), (x, y - 4), 26)
                    pygame.draw.circle(self.main_surface, self.word_color[chess/abs(chess)], (x, y - 4), 22,1)
                    for i in range(4):
                        pygame.draw.arc(self.main_surface, (255, 222, 148),
                                        (x - 25 - i, y - 31 - 2 * i, 53 + 2 * i, 53 + 2 * i), math.radians(200 + 5 * i),
                                        math.radians(250 - 2 * i), 1)
                    pygame.draw.arc(self.main_surface, (255, 222, 148),
                                    (x - 26, y - 33, 55, 55), math.radians(180),
                                    math.radians(250), 1)
                    pygame.draw.arc(self.main_surface, (255, 222, 148),
                                    (x - 28, y - 35, 59, 57), math.radians(230),
                                    math.radians(260), 1)
                    pygame.draw.arc(self.main_surface, (255, 222, 148),
                                    (x - 27, y - 35, 57, 57), math.radians(210),
                                    math.radians(260), 1)
                    text = self.fontObj.render(self.num_to_word[chess], True, self.word_color[chess/abs(chess)], None)
                    self.main_surface.blit(text, (x - 15, y - 20))
        if move is not None:
            self.chess_move(self.word_color[move / abs(move)], self.num_to_word[move])
        self.show_message()
    def draw(self,player,current_state):
        self.screen.blit(self.board_surface, (0, 0))
        self.screen.blit(self.beam_surface, (0, 0))
        self.screen.blit(self.main_surface,(0,0))
        self.board_surface.fill((218,167,115))
        self.beam_surface.fill((255, 255, 255, 0))
        self.main_surface.fill((255,255,255,0))
        for i in range(40):
            pygame.draw.circle(self.beam_surface, (241,216,177,3*i), (360, 340), 360-i*5)
        for i in range(0, 10, 1):
            self.draw_line((180,134,94), (40, 80 * i + 40), (680, 80 * i + 40), 1)
        for i in range(1, 8, 1):
            self.draw_line((180,134,94), (80 * i + 40, 40), (80 * i + 40, 360), 1)
        for i in range(1, 8, 1):
            self.draw_line((180,134,94), (80 * i + 40, 440), (80 * i + 40, 760), 1)
        self.draw_line((180,134,94), (40, 40), (40, 760), 1)
        self.draw_line((180,134,94), (680, 40), (680, 760), 1)
        self.draw_line((180,134,94), (280, 40), (440, 200), 1)
        self.draw_line((180,134,94), (440, 40), (280, 200), 1)
        self.draw_line((180,134,94), (280, 600), (440, 760), 1)
        self.draw_line((180,134,94), (440, 600), (280, 760), 1)
        self.draw_chess(current_state,player.choose)
        if player.choose:
            for point in player.choose_move:
                point_y, point_x = 40 + 80 * int(point[2]), 40 + 80 * int(point[3])
                pygame.draw.circle(self.main_surface, (45, 113, 184), (point_x, point_y), 5)
    def draw_last_move(self,player):
        if player.last_move:
            point_y1, point_x1 = 40 + 80 * int(player.last_move[0]), 40 + 80 * int(player.last_move[1])
            point_y2, point_x2 = 40 + 80 * int(player.last_move[2]), 40 + 80 * int(player.last_move[3])
            pygame.draw.circle(self.main_surface, (209, 221, 231), (point_x1, point_y1), 3)
            pygame.draw.circle(self.main_surface, (209, 221, 231), (point_x1, point_y1), 6, 1)
            pygame.draw.circle(self.main_surface, (209, 221, 231), (point_x2, point_y2), 34,1)
class Player:
    def __init__(self):
        self.general_position="04"
        self.move_action=[]
        self.choose = None
        self.choose_move=[]
        self.last_move=None
        self.warn=False
        #得分
        self.score=0

    def get_action(self,state_deque,current_player,opponent_general_position):
        action=None
        opponent_move_action=[]
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                y,x = event.pos
                i_y=y//80
                i_x=x//80
                if 0<=i_x<10 and 0<=i_y<9:
                    if 10+80*i_y<=y<=70+80*i_y and 10+80*i_x<=x<=70+80*i_x:
                        x,y=i_x,i_y
                    else:
                        return action,opponent_move_action
                else:
                    return action,opponent_move_action
                position=str(x)+str(y)
                if state_deque[-1][x][y] * current_player > 0:
                    self.choose = position
                    self.choose_move.clear()
                    for item in self.move_action:
                        if position == item[0:2]:
                            self.choose_move.append(item)
                    animation["choose"] = True
                else:
                    if self.choose is None:
                        pass
                    else:
                        for item in self.choose_move:
                            if self.choose + position == item:
                                action = item
                                break
                        self.choose = None
                        self.choose_move.clear()
                        if action is None:
                            message["content"]="该棋子无法移动至此"
                            message["show"]=True
                        else:
                            if action[0:2] == self.general_position:
                                self_general_position=action[2:4]
                            else:
                                self_general_position=self.general_position
                            next_state=gc.state_change_by_move(state_deque[-1],action)
                            if gc.general_meet(next_state,self_general_position,opponent_general_position):
                                action=None
                                message["content"] = "不能送将哦"
                                message["show"] = True
                                return action,opponent_move_action
                            opponent_move_action=gc.get_all_move_action(next_state,-current_player)
                            for item in opponent_move_action:
                                if self_general_position == item[2:4]:
                                    action=None
                                    if self.warn:
                                        message["content"] = "您正在被将军"
                                        message["show"] = True
                                    else:
                                        message["content"] = "不能送将哦"
                                        message["show"] = True
                                    break
            elif event.type==pygame.QUIT:
                sys.exit()
        return action,opponent_move_action


class Board:
    def __init__(self,player1,player2,offensive=1,renderer:Renderer=None):
        self.state_deque = deque(maxlen=16)
        self.action_count = 0
        self.not_kill_count = 0
        self.winner = 0
        self.play=True
        self.player_agent={ 1:player1,-1:player2 }
        self.player_agent[1].general_position="04"
        self.player_agent[-1].general_position="94"
        self.offensive=offensive
        self.renderer=renderer
        for i in range(16):
            self.state_deque.append(np.array(gc.init_state))
        self.player_agent[offensive].move_action=gc.get_all_move_action(self.state_deque[-1],offensive)
    def init(self,offensive=1,state_deque=None,self_general_position=None,opponent_general_position=None):
        self.play=True
        self.action_count = 0
        self.not_kill_count = 0
        self.winner = 0
        if state_deque:
            self.state_deque=state_deque
        else:
            for i in range(16):
                self.state_deque.append(np.array(gc.init_state))
        if self_general_position:
            self.player_agent[offensive].general_position = self_general_position
            self.player_agent[-offensive].general_position = opponent_general_position
        else:
            self.player_agent[1].general_position = "04"
            self.player_agent[-1].general_position = "94"
        self.offensive = offensive
        self.player_agent[offensive].move_action = gc.get_all_move_action(self.state_deque[-1], offensive)
    def no_event(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                pass
            elif event.type==pygame.QUIT:
                sys.exit()
    def get_state_array(self):
        params = np.empty([17, 10, 9])
        for i in range(16):
            params[i] = copy.deepcopy(self.state_deque[i - 16])
        params[16][:, :] = ((-1) ** self.action_count)*self.offensive
        return params
    def game_end(self,msg,winner):
        if self.renderer is None:
            self.winner=winner
            self.play=False
        else:
            self.winner=winner
            self.play=False
            message["content"] = msg
            message["show"] = True
            message["long"] = -1
            print(msg)
            print(self.state_deque[-1])
    def draw_board(self):
        player_id = ((-1) ** self.action_count)*self.offensive
        self.renderer.draw(self.player_agent[player_id],self.state_deque[-1])
        self.renderer.draw_last_move(self.player_agent[-player_id])
        pygame.display.flip()
    def judge_action_count(self):
        if self.action_count >= 399:
            self.game_end("平局",0)
        elif self.not_kill_count >= 119:
            self.game_end("平局",0)


    def judge_final_hit(self,move,player_id,opponent_move_action):
        new_state = gc.state_change_by_move(self.state_deque[-1], move)
        self.state_deque.append(new_state)
        if move[0:2] == self.player_agent[player_id].general_position:
            self.player_agent[player_id].general_position = move[2:4]
        self.action_count += 1
        eat=False
        if self.state_deque[-2][int(move[2])][int(move[3])] * player_id < 0:

            #吃子增加self.player_agent[player_id].score分数，并判断分值大于一定值时，该玩家获胜
            self.player_agent[player_id].score+=score_table[(-player_id * self.state_deque[-2][int(move[2])][int(move[3])])]

            if self.player_agent[player_id].score >= win_score:
                self.game_end(str(player_id)+"得分获胜!", player_id)

            eat=True
            self.not_kill_count = 0
        else:
            self.not_kill_count += 1
        self.player_agent[-player_id].warn = gc.general_warn(new_state, player_id,
                                                             self.player_agent[-player_id].general_position)
        final_hit = True
        for item in opponent_move_action:
            opponent_general_position = self.player_agent[-player_id].general_position
            if item[0:2] == opponent_general_position:
                opponent_general_position = item[2:4]
            temp_state = gc.state_change_by_move(new_state, item)
            if gc.general_meet(temp_state, opponent_general_position, self.player_agent[player_id].general_position):
                continue
            if not gc.general_warn(temp_state, player_id, opponent_general_position):
                final_hit = False
                break
        if final_hit:
            self.game_end("绝杀!", player_id)
        if self.renderer is not None:
            if final_hit:
                self.renderer.final_kill_snd.play()
            elif self.player_agent[-player_id].warn and self.renderer is not None:
                self.renderer.warn_snd.play()
                message["content"] = "将军!"
                message["show"] = True
            elif eat:
                self.renderer.eat_snd.play()
                print("吃!")
            else:
                self.renderer.move_snd.play()
    def get_current_player_id(self):
        return ((-1) ** self.action_count)*self.offensive
    def do_move(self):
        player_id = ((-1) ** self.action_count)*self.offensive
        move, opponent_move_action = self.player_agent[player_id].get_action(self.state_deque, player_id,
                                                                             self.player_agent[
                                                                                 -player_id].general_position)
        if move is None:
            return
        else:
            print(move)
        if not opponent_move_action:
            self.game_end("绝杀!", player_id)
            return
        self.player_agent[-player_id].move_action=opponent_move_action
        self.player_agent[player_id].last_move=move
        animation["move_to"]=(int(move[3]),int(move[2]))
        animation["move_begin"] = (int(move[1]), int(move[0]))
        animation["move"]=True
        self.judge_final_hit(move,player_id,opponent_move_action)


