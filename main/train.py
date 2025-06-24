import random
import numpy as np
import pickle
import time
from net import Net
from config import CONFIG
from file_lock import FileLock


# 训练
class TrainPipeline:

    def __init__(self, init_model=None):
        # 训练参数
        self.learn_rate = 1e-3
        self.lr_multiplier = 1                          # 基于KL自适应的调整学习率
        self.temp = 1.0
        self.batch_size = CONFIG['batch_size']          # 训练的batch大小
        self.epochs = CONFIG['epochs']                  # 每次更新的train_step数量
        self.kl = CONFIG['kl']                          # kl散度控制
        self.check_freq = 100                           # 保存模型的频率
        self.game_batch_num = CONFIG['game_batch_num']  # 训练更新的次数
        self.data_buffer=None
        self.iterator=0

        try:
            self.lr_multiplier = np.load(CONFIG['lr_multiple'])  # 基于KL自适应的调整学习率
        except:
            self.lr_multiplier = 1  # 基于KL自适应的调整学习率

        if init_model:
            try:
                self.net = Net(model_path=init_model)
                print('已加载上次最终模型')
            except:
                print('模型路径不存在，从零开始训练')
                self.net = Net()
        else:
            print('从零开始训练')
            self.net = Net()




    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)

        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype('float32')

        mct_prob_batch = [data[1] for data in mini_batch]
        mct_prob_batch = np.array(mct_prob_batch).astype('float32')

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype('float32')
        old_prob, old_v = self.net.batch_policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.net.train_step(
                state_batch,
                mct_prob_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            new_prob, new_v = self.net.batch_policy_value(state_batch)
            kl = np.mean(np.sum(old_prob * (
                np.log(old_prob + 1e-10) - np.log(new_prob + 1e-10)),
                                axis=1))
            if kl > self.kl * 4:  # 如果KL散度很差，则提前终止
                break
        # 自适应调整学习率
        if kl > self.kl * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def run(self):
        try:
            for i in range(self.game_batch_num):
                time.sleep(30)  # 每30秒更新一次模型

                # 使用文件锁保护文件访问
                with FileLock(CONFIG["train_data_path"]) as lock:
                    try:
                        with open(CONFIG["train_data_path"], "rb") as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = data_file["data_buffer"]
                            self.iterator = data_file["iterator"]
                            del data_file
                        print('已载入数据')
                    except Exception as e:
                        print(f'载入数据失败: {e}')
                        time.sleep(5)
                        continue

                print("step i {}: ".format(self.iterator))
                if len(self.data_buffer) > self.batch_size:
                    print("第-- "+str(i)+" --次 训练")
                    loss, entropy = self.policy_update()
                # 保存模型
                self.net.save_model(CONFIG["model_path"])
                np.save(CONFIG['lr_multiple'], self.lr_multiplier)
                if (i + 1) % self.check_freq == 0:
                    print("current self play batch: {}".format(i + 1))
                    self.net.save_model("models/tao_zero_train_{}.model".format(i + 1))

        except KeyboardInterrupt:
            print('\n\rquit')


training_pipeline = TrainPipeline(init_model=CONFIG["model_path"])
training_pipeline.run()