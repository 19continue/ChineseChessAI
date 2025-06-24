import numpy as np
import pickle
import os
import time
from game_pro import Game
from net_pro import Net
from mct import PlayerAI
from config_pro import CONFIG

class TrainPipeline:
    def __init__(self, init_model=None):
        # 训练参数
        self.learn_rate = 2e-3
        self.temp = 1.0
        self.n_playout = CONFIG["search_num"]
        self.c_puct = CONFIG["_c"]
        self.buffer_size = CONFIG["buffer_size"]
        self.batch_size = CONFIG["batch_size"]
        self.data_buffer = []
        self.play_batch_size = 1
        self.epochs = CONFIG["epochs"]
        self.kl_targ = CONFIG["kl"]
        self.check_freq = 50
        self.game_batch_num = CONFIG["game_batch_num"]
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000
        self.init_model = init_model
        self.policy_value_net = Net(model_path=init_model)
        self.mcts_player = PlayerAI(self.policy_value_net.get_policy_value,
                                   _c=self.c_puct,
                                   search_num=self.n_playout,
                                   use_noise=True)

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mct_prob, winner in play_data:
            extend_data.append((state, mct_prob, winner))
            # 水平翻转
            state_flip = np.flip(state, axis=2)
            mct_prob_flip = np.flip(mct_prob.reshape(10, 9), axis=1).flatten()
            extend_data.append((state_flip, mct_prob_flip, winner))
            # 垂直翻转
            state_flip = np.flip(state, axis=1)
            mct_prob_flip = np.flip(mct_prob.reshape(10, 9), axis=0).flatten()
            extend_data.append((state_flip, mct_prob_flip, winner))
            # 水平垂直翻转
            state_flip = np.flip(np.flip(state, axis=1), axis=2)
            mct_prob_flip = np.flip(np.flip(mct_prob.reshape(10, 9), axis=0), axis=1).flatten()
            extend_data.append((state_flip, mct_prob_flip, winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.mcts_player.start_self_play(self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        mini_batch = np.random.choice(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch])
        mcts_probs_batch = np.array([data[1] for data in mini_batch])
        winner_batch = np.array([data[2] for data in mini_batch])
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)
            ), axis=1)
            )
            if kl > self.kl_targ * 4:
                break
        if kl > self.kl_targ * 2 and self.learn_rate > 1e-5:
            self.learn_rate /= 1.5
        elif kl < self.kl_targ / 2 and self.learn_rate < 1e-2:
            self.learn_rate *= 1.5
        explained_var_old = (1 -
                            np.var(np.array(winner_batch) - old_v.flatten()) /
                            np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                            np.var(np.array(winner_batch) - new_v.flatten()) /
                            np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr:{:.7f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.learn_rate,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                    i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    self.policy_value_net.save_model(CONFIG["model_path"])
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    training_pipeline = TrainPipeline(init_model=CONFIG["model_path"])
    training_pipeline.run() 