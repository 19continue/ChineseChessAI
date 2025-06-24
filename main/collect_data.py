from collections import deque
import copy
import os
import pickle
import time
from game import Game
import game_core as gc
from net import Net
from mct import PlayerAI
from config import CONFIG
from file_lock import FileLock


def get_symmetry_data(play_data):
    extend_data = []
    for state, mct_prob, winner in play_data:
        extend_data.append((state, mct_prob, winner))
        state_copy = copy.deepcopy(state).transpose([1, 2, 0])
        state_flip = copy.deepcopy(state).transpose([1, 2, 0])
        state_invert_copy = copy.deepcopy(state).transpose([2, 1, 0])
        state_invert = copy.deepcopy(state).transpose([2, 1, 0])
        for i in range(10):
            for j in range(9):
                state_flip[i][j] = state_copy[i][8 - j]
        state_flip = state_flip.transpose([2, 0, 1])
        state_flip_copy = copy.deepcopy(state_flip).transpose([2, 1, 0])
        state_flip_invert = copy.deepcopy(state_flip).transpose([2, 1, 0])
        for i in range(9):
            for j in range(10):
                state_invert[i][j] = state_invert_copy[i][9 - j]
                state_flip_invert[i][j] = state_flip_copy[i][9 - j]
                for k in range(17):
                    if state_invert[i][j][k] != 0:
                        state_invert[i][j][k] *= -1
                    if state_flip_invert[i][j][k] != 0:
                        state_flip_invert[i][j][k] *= -1
        state_invert = state_invert.transpose([2, 1, 0])
        state_flip_invert = state_flip_invert.transpose([2, 1, 0])
        mct_prob_flip = copy.deepcopy(mct_prob)
        mct_prob_invert = copy.deepcopy(mct_prob)
        for i in range(len(mct_prob_flip)):
            mct_prob_flip[i] = mct_prob[gc.action_to_id_mapping[flip_map(gc.id_to_action_mapping[i])]]
            mct_prob_invert[i] = mct_prob[gc.action_to_id_mapping[invert_map(gc.id_to_action_mapping[i])]]
        mct_prob_flip_invert = copy.deepcopy(mct_prob_flip)
        for i in range(len(mct_prob_flip_invert)):
            mct_prob_flip_invert[i] = mct_prob_flip[gc.action_to_id_mapping[invert_map(gc.id_to_action_mapping[i])]]
        winner_invert = 0.
        if winner != 0:
            winner_invert = winner * -1
        extend_data.append((state_flip, mct_prob_flip, winner))
        extend_data.append((state_invert, mct_prob_invert, winner_invert))
        extend_data.append((state_flip_invert, mct_prob_flip_invert, winner_invert))
    return extend_data

def flip_map(string):
    new_str = ''
    for index in range(4):
        if index == 0 or index == 2:
            new_str += (str(string[index]))
        else:
            new_str += (str(8 - int(string[index])))
    return new_str

def invert_map(string):
    new_str = ''
    for index in range(4):
        if index == 0 or index == 2:
            new_str += (str(9 - int(string[index])))
        else:
            new_str += (str(string[index]))
    return new_str

class CollectPipeline:

    def __init__(self, init_model=CONFIG["model_path"]):
        self.count = 0
        self.game = Game()
        self.temp = 1                               # 温度
        self.search_num = CONFIG["search_num"]      # 每次移动的模拟次数
        self._c = CONFIG["_c"]                      # u的权重
        self.buffer_size = CONFIG["buffer_size"]    # 经验池大小
        self.current_buffer_size=0
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iterator = 0
        self.episode_len=0
        self.init_model=init_model
        self.net = None
        self.player_ai_1 = None
        self.player_ai_2 = None
    # 从主体加载模型
    def load_model(self):
        try:
            self.net = Net(model_path=self.init_model)
            print('已加载最新模型')
        except:
            self.net = Net()
            print('已加载初始模型')
        self.player_ai_1 = PlayerAI(self.net.get_policy_value,
                                    _c=self._c,
                                    search_num=self.search_num,
                                    use_noise=True)
        self.player_ai_2 = PlayerAI(self.net.get_policy_value,
                                    _c=self._c,
                                    search_num=self.search_num,
                                    use_noise=True)

    def collect_data(self, n_games=1):
        for i in range(n_games):
            self.load_model()
            winner, play_data = self.game.start_training_with_web_ai(self.player_ai_1,-1**self.count)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = get_symmetry_data(play_data)
            
            # 使用文件锁保护文件访问
            with FileLock(CONFIG["train_data_path"]) as lock:
                if os.path.exists(CONFIG["train_data_path"]):
                    try:
                        with open(CONFIG["train_data_path"], "rb") as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = data_file["data_buffer"]
                            self.iterator = data_file["iterator"]
                            del data_file
                            self.iterator += 1
                            self.data_buffer.extend(play_data)
                        print('成功载入数据')
                    except Exception as e:
                        print(f'载入数据失败: {e}')
                        time.sleep(5)
                else:
                    self.data_buffer.extend(play_data)
                    self.iterator += 1
                
                self.current_buffer_size = len(self.data_buffer)
                data_dict = {"data_buffer": self.data_buffer, "iterator": self.iterator}
                with open(CONFIG["train_data_path"], "wb") as data_file:
                    pickle.dump(data_dict, data_file)
            
            self.count = self.count + 1
        return self.iterator

    def run(self):
        try:
            while self.current_buffer_size<=self.buffer_size:
                iterator = self.collect_data()
                print('batch i: {},buffer current/max: {}/{}, episode_len: {}'.format(
                    iterator,self.current_buffer_size,self.buffer_size ,self.episode_len))
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == "__main__":
    collecting_pipeline = CollectPipeline(init_model=CONFIG["model_path"])
    collecting_pipeline.run()