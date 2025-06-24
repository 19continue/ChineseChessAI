from board import Board, Renderer, Player
from config import CONFIG
from mct import PlayerAI
import numpy as np
import requests
from net_pro import Net
import game_core as gc
import time
from collections import deque

# 记录完整的局面历史（包括双方落子后的局面）
state_history = deque(maxlen=8)

class WebAIPlayer(Player):
    def __init__(self, offensive=1):
        super().__init__()
        self.offensive = offensive
        self.last_request_time = 0
        self.request_interval = 20.0  # 每次请求间隔20秒
        
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
        
    def get_action_long(self, state_deque, current_player, opponent_general_position, mct_prob=False):
        # Add delay between requests
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_interval:
            time.sleep(self.request_interval - time_since_last_request)
        
        # Convert current state to FEN format
        fen = self._state_to_fen(state_deque[-1])
        url = f"https://engine.xqipu.com/api/engine/getMoves?fen={fen}"
        print(url)
        try:
            response = requests.get(url, headers=self.headers)
            self.last_request_time = time.time()  # Update last request time
            data = response.json()
            
            if "error" in data:
                print("Web API error:", data["msg"])
                return None, [], None
                
            if not data.get("moves"):
                print("No moves returned from web API")
                return None, [], None
            
            # Check for repeated positions in the complete game history
            current_state = state_deque[-1]
            state_count = 0
            for state in state_history:
                if np.array_equal(state, current_state):
                    state_count += 1
            
            # If position has appeared twice and there are alternative moves
            if state_count >= 2 and len(data["moves"]) > 1:
                # Skip the first move and choose the second one
                web_move = data["moves"][1]["move"]
                print("重复落子, 选择第二步:", web_move)
            else:
                # Get first move from response as usual
                web_move = data["moves"][0]["move"]
                print(web_move)
            
            # Convert web move format to our format
            move = self._convert_web_move(web_move)
            opponent_move_action = gc.get_all_move_action(
                gc.state_change_by_move(state_deque[-1], move),
                -current_player
            )
            
            # Create mct_prob array with probability 1 for the chosen move
            if mct_prob:
                move_prob = np.zeros(2086)  # Total number of possible moves
                move_id = gc.action_to_id_mapping[move]
                move_prob[move_id] = 1.0
                return move, opponent_move_action, move_prob
            else:
                return move, opponent_move_action, None
            
        except Exception as e:
            print(f"Error calling web API: {e}")
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
                    # Convert piece number to FEN character
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
        """Convert web API move format to our format"""
        # Web format: 'h9g7' -> Our format: '89' + '67'
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

    def get_state_array(self):
        """获取9个局面的状态数组"""
        params = np.empty([9, 10, 9])
        # 获取前8个历史局面
        for i in range(8):
            if i < len(self.board.state_deque):
                params[i] = copy.deepcopy(self.board.state_deque[-(i+1)])
            else:
                params[i] = np.zeros((10, 9))
        # 第9个局面为当前落子方
        params[8] = np.ones((10, 9)) * ((-1) ** self.board.action_count) * self.board.offensive
        return params

    def start_with_renderer(self):
        if self.board is None or self.board.renderer is None:
            chess_renderer = Renderer(1)
            if self.board is not None:
                player1 = self.board.player_agent[1]
                player2 = self.board.player_agent[-1]
            else:
                _net = Net(CONFIG["model_path"])
                player1 = Player()
                player2 = PlayerAI(_net.get_policy_value, search_num=10, use_noise=False)
            self.board = Board(player1, player2, 1, chess_renderer)
            print("已采用默认设置!")
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
                self.board.player_agent[-player_id].general_position,
                True
            )
            if move is None:
                print("AI未能获取落子动作!")
                return
            else:
                print("第" + str(self.board.action_count) + "回合: " + str(player_id) + " -- " + move)
            state_data.append(self.get_state_array())
            mct_prob_data.append(mct_prob)
            player_id_data.append(player_id)
            if not opponent_move_action:
                self.board.game_end("绝杀!", player_id)
                print("绝杀!")
            self.board.player_agent[-player_id].move_action = opponent_move_action
            self.board.judge_final_hit(move, player_id, opponent_move_action)
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
        if player1 is None:
            player1 = PlayerAI(None)
        self.board = Board(WebAIPlayer(), player1, -offensive)
        self.board.play = True
        state_data, mct_prob_data, player_id_data = [], [], []
        
        while self.board.play:
            self.board.judge_action_count()
            player_id = self.board.get_current_player_id()

            if player_id == 1:
                time.sleep(15)
            move, opponent_move_action, mct_prob = self.board.player_agent[player_id].get_action_long(
                self.board.state_deque,
                player_id,
                self.board.player_agent[-player_id].general_position,
                True
            )
            
            if move is None:
                print("未能获取落子动作!")
                return
                
            print(f"第{self.board.action_count}回合: {player_id} -- {move}")
            state_data.append(self.get_state_array())
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