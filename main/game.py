from board import Board, Renderer, Player
from config import CONFIG
from mct import PlayerAI
from chess_ai_player import ChessAIPlayer
import numpy as np
import requests
from net import Net
import game_core as gc
import time
from collections import deque

# 记录完整的局面历史（包括双方落子后的局面）
state_history = deque(maxlen=8)

pv_sequence = None  # 存储PV序列
last_opponent_move = None  # 存储对手上一步的移动


def _convert_move_to_web_format(move):

    file_map = {'0': 'a', '1': 'b', '2': 'c', '3': 'd', '4': 'e',
                '5': 'f', '6': 'g', '7': 'h', '8': 'i'}
    rank_map = {'0': '9', '1': '8', '2': '7', '3': '6', '4': '5',
                '5': '4', '6': '3', '7': '2', '8': '1', '9': '0'}

    from_rank = rank_map[move[0]]
    from_file = file_map[move[1]]
    to_rank = rank_map[move[2]]
    to_file = file_map[move[3]]

    return from_file + from_rank + to_file + to_rank


class WebAIPlayer(Player):
    def __init__(self, offensive=1):
        super().__init__()
        self.offensive = offensive
        self.last_request_time = 0
        self.request_interval = 5.0  # 减少请求间隔到5秒
        self.repeat = 0  # 已重试的次数
        self.repeat_max = 5  # 获取落子动作失败时,重试的最大次数
        self.base_retry_delay = 2.0  # 基础重试延迟时间

        # 设置请求头，模拟来自xqipu.com的请求
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.xqipu.com/',
            'Origin': 'https://www.xqipu.com',
            'Host': 'engine.xqipu.com',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive'
        }

    def get_web_action(self, url, state_deque):
        global pv_sequence, last_opponent_move
        try:
            response = requests.get(url, headers=self.headers, timeout=15,verify=False)  # 添加超时设置
            self.last_request_time = time.time()
            data = response.json()

            if "error" in data:
                print(f"Web API error: {data['msg']}")
                return None

            if not data.get("moves"):
                print("No moves returned from web API")
                return None

            current_state = state_deque[-1]
            state_count = 0
            for state in state_history:
                if np.array_equal(state, current_state):
                    state_count += 1

            web_move = None
            if state_count >= 2:
                print("检测到重复局面")
                if pv_sequence is None:
                    # 首次重复，尝试获取PV序列
                    if data["moves"][0].get("pv"):
                        pv_sequence = data["moves"][0]["pv"].split()
                        web_move = data["moves"][0]["move"]
                        print(f"保存PV序列: {pv_sequence}")
                    else:
                        # 如果没有PV序列，选择第一个移动
                        web_move = data["moves"][0]["move"]
                        print("没有PV序列，选择第一个移动")
                elif pv_sequence and last_opponent_move:
                    # 检查对手上一步是否匹配PV序列中的预期移动
                    if len(pv_sequence) >= 2 and last_opponent_move == pv_sequence[1]:
                        # 使用PV序列中的下一步
                        if len(pv_sequence) >= 3:
                            web_move = pv_sequence[2]
                            pv_sequence = pv_sequence[2:]
                            print(f"使用PV序列中的下一步: {web_move}")
                            print(f"剩余PV序列:{pv_sequence}")
                        else:
                            # PV序列不足，选择第一个移动
                            web_move = data["moves"][0]["move"]
                            print("PV序列不足，选择第一个移动")
                            # 清除PV序列
                            pv_sequence = None
                            last_opponent_move = None
                    else:
                        # 对手移动不匹配预期，选择第二个移动
                        if len(data["moves"]) > 1:
                            web_move = data["moves"][1]["move"]
                            print(f"对手移动不匹配预期，选择第二步: {web_move}")
                        else:
                            web_move = data["moves"][0]["move"]
                            print("只有一个移动可用，选择第一个移动")
                        # 清除PV序列
                        pv_sequence = None
                        last_opponent_move = None
                else:
                    # 没有PV序列或对手移动，选择第一个移动
                    web_move = data["moves"][0]["move"]
                    print("没有PV序列或对手移动，选择第一个移动")
            else:
                # 正常情况，选择第一个移动
                web_move = data["moves"][0]["move"]
                print(f"选择移动: {web_move}")

            return self._convert_web_move(web_move)

        except requests.exceptions.Timeout:
            print("请求超时")
            return None
        except requests.exceptions.RequestException as e:
            print(f"请求异常: {e}")
            return None
        except Exception as e:
            print(f"未知错误: {e}")
            return None

    def get_action_long(self, state_deque, current_player, opponent_general_position, mct_prob=False, opponent_move=None):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_interval:
            time.sleep(self.request_interval - time_since_last_request)

        fen = self._state_to_fen(state_deque[-1])
        url = f"https://engine.xqipu.com/api/engine/getMoves?fen={fen}"
        print(f"请求URL: {url}")

        try:
            # Convert web move format to our format
            move = self.get_web_action(url, state_deque)
            if move is None:
                self.repeat = 0
                url = url + "&skip=1"
                while move is None and self.repeat < self.repeat_max:
                    # 使用指数退避策略
                    retry_delay = self.base_retry_delay * (2 ** self.repeat)
                    time.sleep(retry_delay)
                    print(f"第{self.repeat + 1}次重试，等待{retry_delay}秒，请求地址：{url}")
                    move = self.get_web_action(url, state_deque)
                    self.repeat += 1

            if move is None:
                raise Exception("多次重试后仍未能获取落子动作")

            opponent_move_action = gc.get_all_move_action(
                gc.state_change_by_move(state_deque[-1], move),
                -current_player
            )

            if mct_prob:
                move_prob = np.zeros(2086)
                move_id = gc.action_to_id_mapping[move]
                move_prob[move_id] = 1.0
                return move, opponent_move_action, move_prob
            else:
                return move, opponent_move_action, None

        except Exception as e:
            print(f"获取落子动作失败: {e}")
            return None, [], None

    def _state_to_fen(self, state):
        """Convert game state to FEN format"""
        fen = []
        for row in state:
            empty = 0
            row_str = ""
            for cell in row:
                if cell == 0:
                    empty += 1
                else:
                    if empty > 0:
                        row_str += str(empty)
                        empty = 0
                    # 把棋子数字转换成 FEN 字母
                    piece_map = {
                        1: 'k', 2: 'a', 3: 'b', 4: 'n', 5: 'r', 6: 'c', 7: 'p',
                        -1: 'K', -2: 'A', -3: 'B', -4: 'N', -5: 'R', -6: 'C', -7: 'P'
                    }
                    row_str += piece_map[cell]
            if empty > 0:
                row_str += str(empty)
            fen.append(row_str)
        return '/'.join(fen) + ' b'  # 'b' indicates black's turn

    def _convert_web_move(self, web_move):
        """转换 web API 的 move 变成自己的格式"""
        # Web API 的格式: 'h9g7' -> 自己的格式: '89' + '67'
        file_map = {'a': '0', 'b': '1', 'c': '2', 'd': '3', 'e': '4',
                    'f': '5', 'g': '6', 'h': '7', 'i': '8'}
        rank_map = {'9': '0', '8': '1', '7': '2', '6': '3', '5': '4',
                    '4': '5', '3': '6', '2': '7', '1': '8', '0': '9'}

        from_file = file_map[web_move[0]]
        from_rank = rank_map[web_move[1]]
        to_file = file_map[web_move[2]]
        to_rank = rank_map[web_move[3]]

        return from_rank + from_file + to_rank + to_file


class Game:
    def __init__(self, board: Board = None):
        self.board = board

    def start_with_renderer(self, fist_play = -1):
        if self.board is None or self.board.renderer is None:
            chess_renderer = Renderer(fist_play)
            if self.board is not None:
                player1 = self.board.player_agent[1]
                player2 = self.board.player_agent[-1]
            else:
                _net = Net(CONFIG["model_path"])
                player1 = PlayerAI(_net.get_policy_value, search_num=10, use_noise=False)
                player2 = Player()
            self.board = Board(player1, player2, fist_play, chess_renderer)
            print("已采用默认设置!")
        self.board.play = True
        while True:
            self.board.draw_board()
            self.board.judge_action_count()
            if self.board.play:
                self.board.do_move()
            else:
                self.board.no_event()

    def start_with_local_ai(self,fist_play = 1):
        if self.board is None or self.board.renderer is None:
            chess_renderer = Renderer(fist_play)
            if self.board is not None:
                player1 = self.board.player_agent[1]
                player2 = self.board.player_agent[-1]
            else:
                _net = Net(CONFIG["model_path"])
                player1 = ChessAIPlayer(fist_play)
                player2 = Player()
            self.board = Board(player1, player2, fist_play, chess_renderer)
        self.board.play = True
        while True:
            self.board.draw_board()
            self.board.judge_action_count()
            if self.board.play:
                self.board.do_move()
            else:
                self.board.no_event()

    def start_people_with_people(self, offensive=-1):
        if self.board is None:
            chess_renderer = Renderer(offensive)
            player1 = Player()
            player2 = Player()
            self.board = Board(player1, player2, offensive, chess_renderer)
        self.board.play = True
        while self.board.play:
            self.board.draw_board()
            self.board.judge_action_count()
            self.board.do_move()

    def start_training_no_renderer(self, player1=None, player2=None):
        if player1 is None or player2 is None:
            player1 = PlayerAI(None)
            player2 = PlayerAI(None)
        self.board = Board(player1, player2)
        self.board.play = True
        state_data, mct_prob_data, player_id_data = [], [], []
        while self.board.play:
            self.board.judge_action_count()
            player_id = self.board.get_current_player_id()
            move, opponent_move_action, mct_prob = self.board.player_agent[player_id].get_action_long(
                self.board.state_deque,
                player_id,
                self.board.player_agent[
                    -player_id].general_position,
                True,
                self.board.player_agent[
                    -player_id].last_move)
            if move is None:
                print("AI未能获取落子动作!")
                return
            else:
                self.board.player_agent[player_id].last_move = move
                print("第" + str(self.board.action_count) + "回合: " + str(player_id) + " -- " + move)
            state_data.append(self.board.get_state_array())
            mct_prob_data.append(mct_prob)
            player_id_data.append(player_id)
            if not opponent_move_action:
                self.board.game_end("绝杀!", player_id)
                print("绝杀!")
            self.board.player_agent[-player_id].move_action = opponent_move_action
            self.board.judge_final_hit(move, player_id, opponent_move_action)
            # self.board.player_agent[-player_id].update_tree_by_action(action_to_id_mapping[move])
            print(self.board.state_deque[-1])
        winner_data = np.zeros(len(player_id_data), dtype=float)
        if self.board.winner != 0:
            winner_data[np.array(player_id_data) == self.board.winner] = 1.0
            winner_data[np.array(player_id_data) != self.board.winner] = -1.0
        self.board.player_agent[1].reset_tree()
        self.board.player_agent[-1].reset_tree()
        print("--结束--")
        print(self.board.winner)
        return self.board.winner, zip(state_data, mct_prob_data, winner_data)

    def start_training_with_web_ai(self, player1=None, offensive=1):
        global pv_sequence, last_opponent_move
        if player1 is None:
            player1 = PlayerAI(None)
        self.board = Board(WebAIPlayer(), player1, -offensive)
        self.board.play = True
        state_data, mct_prob_data, player_id_data = [], [], []

        while self.board.play:
            self.board.judge_action_count()
            player_id = self.board.get_current_player_id()

            move, opponent_move_action, mct_prob = self.board.player_agent[player_id].get_action_long(
                self.board.state_deque,
                player_id,
                self.board.player_agent[-player_id].general_position,
                True
            )

            if move is None:
                print("未能获取落子动作!")
                raise Exception("未能获取落子动作!")

            if player_id == 1:
                time.sleep(10)
            elif opponent_move_action:
                # 保存对手的移动
                last_opponent_move = _convert_move_to_web_format(move)

            self.board.player_agent[player_id].last_move = move
            print(f"第{self.board.action_count}回合: {player_id} -- {move}")
            state_data.append(self.board.get_state_array())
            mct_prob_data.append(mct_prob)
            player_id_data.append(player_id)

            if not opponent_move_action:
                self.board.game_end("绝杀!", player_id)
                print("绝杀!")

            self.board.player_agent[-player_id].move_action = opponent_move_action
            self.board.judge_final_hit(move, player_id, opponent_move_action)
            state_history.append(self.board.state_deque[-1])
            print(self.board.state_deque[-1])

        winner_data = np.zeros(len(player_id_data), dtype=float)
        if self.board.winner != 0:
            winner_data[np.array(player_id_data) == self.board.winner] = 1.0
            winner_data[np.array(player_id_data) != self.board.winner] = -1.0

        if hasattr(self.board.player_agent[1], 'reset_tree'):
            self.board.player_agent[1].reset_tree()

        print("--结束--")
        print(self.board.winner)
        return self.board.winner, zip(state_data, mct_prob_data, winner_data)

    def start_training_with_chess_ai(self, player1=None, offensive=1):
        """使用本地AI进行训练"""
        if player1 is None:
            player1 = PlayerAI(None)
        self.board = Board(ChessAIPlayer(offensive), player1, offensive)
        self.board.play = True
        state_data, mct_prob_data, player_id_data = [], [], []

        while self.board.play:
            self.board.judge_action_count()
            player_id = self.board.get_current_player_id()

            move, opponent_move_action, mct_prob = self.board.player_agent[player_id].get_action_long(
                self.board.state_deque,
                player_id,
                self.board.player_agent[-player_id].general_position,
                True,
                self.board.player_agent[-player_id].last_move
            )

            if move is None:
                print("未能获取落子动作!")
                raise Exception("未能获取落子动作!")

            self.board.player_agent[player_id].last_move = move
            print(f"第{self.board.action_count}回合: {player_id} -- {move}")
            state_data.append(self.board.get_state_array())
            mct_prob_data.append(mct_prob)
            player_id_data.append(player_id)

            if not opponent_move_action:
                self.board.game_end("绝杀!", player_id)
                print("绝杀!")

            self.board.player_agent[-player_id].move_action = opponent_move_action
            self.board.judge_final_hit(move, player_id, opponent_move_action)
            state_history.append(self.board.state_deque[-1])
            print(self.board.state_deque[-1])

        winner_data = np.zeros(len(player_id_data), dtype=float)
        if self.board.winner != 0:
            winner_data[np.array(player_id_data) == self.board.winner] = 1.0
            winner_data[np.array(player_id_data) != self.board.winner] = -1.0

        if hasattr(self.board.player_agent[1], 'reset_tree'):
            self.board.player_agent[1].reset_tree()

        print("--结束--")
        print(self.board.winner)
        return self.board.winner, zip(state_data, mct_prob_data, winner_data)


if __name__ == "__main__":
    # net = Net()
    # player_ai_1 = PlayerAI(net.get_policy_value,use_noise=True,search_num=500)
    # player_ai_2 = PlayerAI(net.get_policy_value,use_noise=True,search_num=500)
    # chess_game=Game()
    # chess_game.start_training_no_renderer(player_ai_1, player_ai_2)

    chess_game = Game()


    #  人 VS 人

    # chess_game.start_people_with_people()

    #  AI VS 人
    #
    chess_game.start_with_renderer()

    # 单机简单AI vs 人

    # chess_game.start_with_local_ai()

    # AI VS AI

    # ai_net_1 = Net(CONFIG["model_path"])
    # # ai_net_1 = Net("tao_zero_train_100.model")
    # ai_net_2 = Net("tao_zero_pro.model")
    # chess_ai_renderer = Renderer(1)
    # player_ai_1 = PlayerAI(ai_net_1.get_policy_value, search_num=10, use_noise=False)
    # player_ai_2 = PlayerAI(ai_net_2.get_policy_value, search_num=10, use_noise=False)
    # chess_game.board = Board(player_ai_2, player_ai_1, 1, chess_ai_renderer)
    # chess_game.start_with_renderer()
