# parallel_mct.py
import copy
import sys
import numpy as np
import threading
from concurrent.futures import ProcessPoolExecutor  # 改为进程池
import pygame

from board import Board
import game_core as gc
from config import CONFIG


def soft_max(x):
    prob = np.exp(x - np.max(x))
    prob /= np.sum(prob)
    return prob


class TreeNode:
    def __init__(self, parent, priority_rate):
        self.parent = parent
        self.children = {}
        self.visit = 0
        self.Q = 0
        self.U = 0
        self.P = priority_rate
        self.cached_value = None

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return not self.children

    def update_tree(self, child_value):
        self.visit += 1
        self.Q += (child_value - self.Q) / self.visit
        self.cached_value = None

    def expand(self, action_prob):
        for action, prob in action_prob:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def get_value(self, _c):
        if self.cached_value is None:
            if self.parent:
                self.U = _c * self.P * np.sqrt(self.parent.visit) / (1 + self.visit)
            else:
                self.U = 0
            self.cached_value = self.Q + self.U
        return self.cached_value

    def select(self, _c):
        return max(self.children.items(), key=lambda item: item[1].get_value(_c))

    def update_reverse(self, child_value):
        current = self
        while current.parent is not None:
            current.parent.update_tree(-child_value)
            current = current.parent
        current.update_tree(child_value)


class MCT_Tree:
    def __init__(self, policy_value_function, _c=5, search_num=2000):
        self.root = TreeNode(None, 1.0)
        self.policy_value_function = policy_value_function
        self._c = _c
        self.search_num = search_num
        self.board = Board(PlayerAI(None), PlayerAI(None))
        self.id_to_action = gc.id_to_action_mapping
        self.action_to_id = gc.action_to_id_mapping

    def copy(self):
        """完全独立的副本，避免共享任何状态"""
        new_tree = MCT_Tree(self.policy_value_function, self._c, self.search_num)
        new_tree.root = copy.deepcopy(self.root)
        new_tree.board = Board(PlayerAI(None), PlayerAI(None))  # 新建独立棋盘
        return new_tree

    def search(self):
        node = self.root
        while not node.is_leaf():
            action_id, node = node.select(self._c)
            move = self.id_to_action.get(action_id)
            if not move:
                continue

            # 独立棋盘操作
            try:
                player_id = self.board.get_current_player_id()
                new_state = gc.state_change_by_move(self.board.state_deque[-1], move)
                opponent_actions = gc.get_all_move_action(new_state, -player_id)
                self.board.player_agent[-player_id].move_action = opponent_actions
                self.board.judge_final_hit(move, player_id, opponent_actions)
            except Exception as e:
                print(f"Board operation error: {str(e)}")
                continue

        state_array, move_id = self._get_step_data()
        if not move_id:
            return

        try:
            policy, value = self.policy_value_function(state_array, move_id)
            if self.board.play:
                node.expand(policy)
            else:
                value = self.board.winner / self.board.get_current_player_id()
            node.update_reverse(-value)
        except Exception as e:
            print(f"Policy value error: {str(e)}")

    def _get_step_data(self):
        current_player = self.board.get_current_player_id()
        move_id = []

        # 增强动作验证
        for item in list(self.board.player_agent[current_player].move_action):  # 使用副本遍历
            if len(item) != 4 or not item.isdigit():
                continue
            if item not in self.action_to_id:
                continue

            # 坐标范围严格检查
            try:
                x, y = int(item[0]), int(item[1])
                tox, toy = int(item[2]), int(item[3])
                if not (0 <= x < 10 and 0 <= y < 9 and 0 <= tox < 10 and 0 <= toy < 9):
                    continue
            except:
                continue

            # 状态验证
            try:
                temp_state = gc.state_change_by_move(self.board.state_deque[-1], item)
                self_general = item[2:4] if item.startswith(
                    self.board.player_agent[current_player].general_position
                ) else self.board.player_agent[current_player].general_position

                if gc.general_meet(temp_state, self_general,
                                   self.board.player_agent[-current_player].general_position):
                    continue
                if gc.general_warn(temp_state, -current_player, self_general):
                    continue
            except:
                continue

            # 获取合法ID
            action_id = self.action_to_id.get(item, -1)
            if 0 <= action_id < 2086:
                move_id.append(action_id)

        # 安全填充历史状态
        params = np.zeros((17, 10, 9), dtype=np.float32)
        try:
            states = list(self.board.state_deque)
            valid_states = min(len(states), 16)
            if valid_states > 0:
                params[:valid_states] = np.array(states[-valid_states:], dtype=np.float32)
            params[16] = current_player
        except Exception as e:
            print(f"State array error: {str(e)}")

        return params, move_id


class ParallelMCTS:
    def __init__(self, main_tree, num_threads=4):
        self.main_tree = main_tree
        self.num_threads = num_threads

    def parallel_search(self, state_deque, player_id, self_general, opponent_general):
        # 改用进程池避免GIL限制
        with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for _ in range(self.num_threads):
                thread_tree = self.main_tree.copy()
                # 深度初始化棋盘状态
                try:
                    thread_tree.board.init(
                        player_id,
                        copy.deepcopy(state_deque),
                        copy.deepcopy(self_general),
                        copy.deepcopy(opponent_general)
                    )
                except Exception as e:
                    print(f"Board init error: {str(e)}")
                    continue
                futures.append(executor.submit(self._thread_worker, thread_tree))

            merged_visits = {}
            for future in futures:
                try:
                    root = future.result()
                    for action, child in root.children.items():
                        merged_visits[action] = merged_visits.get(action, 0) + child.visit
                except Exception as e:
                    print(f"Thread error: {str(e)}")

            # 原子化更新
            with threading.Lock():
                for action, visits in merged_visits.items():
                    if action not in self.main_tree.root.children:
                        self.main_tree.root.children[action] = TreeNode(None, 1.0)
                    self.main_tree.root.children[action].visit += visits

    def _thread_worker(self, tree):
        try:
            for _ in range(tree.search_num // self.num_threads):
                tree.search()
        except Exception as e:
            print(f"Search error: {str(e)}")
        return tree.root

class MyThread(threading.Thread):
    def run(self):
        if self._target is not None:
            self._return=self._target(*self._args,**self._kwargs)
    def join(self):
        super().join()
        return self._return

class PlayerAI:
    def __init__(self, policy_value_function, _c=5, search_num=2000, use_noise=False, num_threads=4):
        self.general_position = "04"
        self.move_action = []
        self.choose = None
        self.choose_move = []
        self.last_move = None
        self.warn = False
        self.thread = None
        self.use_noise = use_noise

        if policy_value_function:
            self.mct_tree = MCT_Tree(policy_value_function, _c, search_num)
            self.parallel_mcts = ParallelMCTS(self.mct_tree, num_threads)

    def get_action_prob(self, state_deque, current_player, self_general, opponent_general, temp=1e-3):
        try:
            self.parallel_mcts.parallel_search(state_deque, current_player, self_general, opponent_general)
        except Exception as e:
            print(f"Parallel search error: {str(e)}")
            return [], np.array([])

        with threading.Lock():
            visits = np.array([node.visit for node in self.mct_tree.root.children.values()], dtype=np.float32)

        if visits.size == 0:
            return [], np.array([])

        try:
            action_prob = soft_max(1.0 / temp * np.log(visits + 1e-10))
        except:
            return [], np.array([])

        return list(self.mct_tree.root.children.keys()), action_prob

    def get_action_long(self, state_deque, current_player, opponent_general, mct_prob=False):
        action, action_prob = self.get_action_prob(state_deque, current_player,
                                                   self.general_position, opponent_general)
        if not action:
            return (None, []) if not mct_prob else (None, [], np.zeros(2086))

        # 安全概率处理
        valid_actions = [a for a in action if 0 <= a < 2086]
        valid_probs = action_prob[:len(valid_actions)]
        valid_probs = np.clip(valid_probs, 1e-8, None)  # 防止零概率
        valid_probs /= valid_probs.sum()

        try:
            if self.use_noise:
                noise = np.random.dirichlet(CONFIG["dirichlet"] * np.ones_like(valid_probs))
                final_prob = 0.75 * valid_probs + 0.25 * noise
            else:
                final_prob = valid_probs
            final_prob /= final_prob.sum()

            move_id = np.random.choice(valid_actions, p=final_prob)
        except:
            return (None, []) if not mct_prob else (None, [], np.zeros(2086))

        move = gc.id_to_action_mapping.get(move_id)
        if not move:
            return (None, []) if not mct_prob else (None, [], np.zeros(2086))

        try:
            opponent_actions = gc.get_all_move_action(gc.state_change_by_move(state_deque[-1], move), -current_player)
        except:
            opponent_actions = []

        if mct_prob:
            move_prob = np.zeros(2086)
            move_prob[valid_actions] = valid_probs
            return move, opponent_actions, move_prob
        return move, opponent_actions

    def get_action(self, state_deque, current_player, opponent_general_position, mct_prob=False):
        if self.thread is None:
            self.thread = MyThread(target=self.get_action_long,
                                   args=(state_deque, current_player, opponent_general_position, mct_prob))
            self.thread.start()
            return None, []
        elif self.thread.is_alive():
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    break
                elif event.type == pygame.QUIT:
                    sys.exit()
            return None, []
        else:
            _return = self.thread.join()
            self.thread = None
            return _return