CONFIG = {
    'dirichlet': 0.15,                                      # 噪声,国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'search_num': 200,                                     # 每次移动的模拟次数
    '_c': 5,                                                # u的权重
    'buffer_size': 50000,                                  # 经验池大小
    'model_path': 'tao_zero_pro.model',                    # 模型路径
    'train_data_path': 'tao_zero_pro_train_data.pkl',      # 数据容器的路径
    'batch_size': 512,                                     # 每次更新的train_step数量
    'kl': 0.02,                                            # kl散度控制
    'epochs': 5,                                           # 每次更新的train_step数量
    'game_batch_num': 100,                                 # 需要训练的次数
    'lr_multiple': 'lr_multiple_pro.npy',                  # lr缩放因子路径
    'embedding_dim': 256,                                  # 嵌入维度
    'history_length': 8,                                   # 历史局面长度
} 