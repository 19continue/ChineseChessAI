import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

class ChessEmbedding(nn.Layer):
    def __init__(self, num_pieces=15, embedding_dim=256):
        super().__init__()
        # 15种棋子类型（包括空位）的嵌入
        self.embedding = nn.Embedding(num_pieces, embedding_dim)
        
    def forward(self, x):
        # x shape: [batch_size, 10, 9]
        # 将输入转换为长整型
        x = paddle.cast(x, dtype='int64')
        # 将棋子编号映射到0-14的范围
        x = x + 7  # 将-7到7的范围映射到0-14
        # 获取嵌入
        return self.embedding(x)  # shape: [batch_size, 10, 9, embedding_dim]

class ResBlock(nn.Layer):
    def __init__(self, channels_num=256):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels=channels_num, out_channels=channels_num, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2D(num_features=channels_num)
        self.conv1_relu = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=channels_num, out_channels=channels_num, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2D(num_features=channels_num)
        self.conv2_relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_relu(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = paddle.add(x, y)
        return self.conv2_relu(y)

class PolicyValueNet(nn.Layer):
    def __init__(self, embedding_dim=256, channels_num=256, res_block_num=19):
        super().__init__()
        self.chess_embedding = ChessEmbedding(num_pieces=15, embedding_dim=embedding_dim)
        
        # 处理嵌入后的棋盘状态
        self.initial_conv = nn.Conv2D(in_channels=embedding_dim, out_channels=channels_num, 
                                     kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm2D(num_features=channels_num)
        self.initial_relu = nn.ReLU()
        
        # 残差块
        self.res_blocks = nn.LayerList([ResBlock(channels_num=channels_num) for _ in range(res_block_num)])
        
        # 策略头
        self.policy_conv = nn.Conv2D(in_channels=channels_num, out_channels=16, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2D(num_features=16)
        self.policy_relu = nn.ReLU()
        self.policy_linear = nn.Linear(in_features=16*10*9, out_features=2086)
        
        # 价值头
        self.value_conv = nn.Conv2D(in_channels=channels_num, out_channels=8, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2D(num_features=8)
        self.value_relu1 = nn.ReLU()
        self.value_linear1 = nn.Linear(in_features=8*10*9, out_features=256)
        self.value_relu2 = nn.ReLU()
        self.value_linear2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        # x shape: [batch_size, 9, 10, 9]
        batch_size = x.shape[0]
        
        # 处理前8个历史局面
        history_states = x[:, :8]  # [batch_size, 8, 10, 9]
        history_states = paddle.reshape(history_states, [-1, 10, 9])  # [batch_size*8, 10, 9]
        history_embeddings = self.chess_embedding(history_states)  # [batch_size*8, 10, 9, embedding_dim]
        history_embeddings = paddle.reshape(history_embeddings, [batch_size, 8, 10, 9, -1])
        history_embeddings = paddle.mean(history_embeddings, axis=1)  # [batch_size, 10, 9, embedding_dim]
        
        # 处理当前局面
        current_state = x[:, 8]  # [batch_size, 10, 9]
        current_embedding = self.chess_embedding(current_state)  # [batch_size, 10, 9, embedding_dim]
        
        # 合并历史信息和当前局面
        x = paddle.add(history_embeddings, current_embedding)  # [batch_size, 10, 9, embedding_dim]
        x = paddle.transpose(x, [0, 3, 1, 2])  # [batch_size, embedding_dim, 10, 9]
        
        # 通过卷积网络
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
            
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_relu(policy)
        policy = paddle.reshape(policy, [-1, 16*10*9])
        policy = self.policy_linear(policy)
        policy = F.log_softmax(policy)
        
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_relu1(value)
        value = paddle.reshape(value, [-1, 8*10*9])
        value = self.value_linear1(value)
        value = self.value_relu2(value)
        value = self.value_linear2(value)
        value = F.tanh(value)
        
        return policy, value

class Net:
    def __init__(self, model_path=None, use_gpu=True):
        self.weight_decay = 2e-3
        self.use_gpu = use_gpu
        self.policy_value_net = PolicyValueNet()
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=0.001,
            parameters=self.policy_value_net.parameters(),
            weight_decay=self.weight_decay
        )
        if model_path:
            parameters = paddle.load(model_path)
            self.policy_value_net.set_dict(parameters)

    def batch_policy_value(self, state_array):
        self.policy_value_net.eval()
        state = paddle.to_tensor(state_array)
        policy, value = self.policy_value_net(state)
        probabilities = np.exp(policy.numpy())
        return probabilities, value.numpy()

    def get_policy_value(self, state_array, move_id):
        self.policy_value_net.eval()
        state = np.ascontiguousarray(state_array.reshape(-1, 9, 10, 9)).astype("float32")
        state = paddle.to_tensor(state)
        policy, value = self.policy_value_net(state)
        probabilities = np.exp(policy.numpy().flatten())
        probabilities = zip(move_id, probabilities[move_id])
        return probabilities, value.numpy()

    def train_step(self, state_array, mct_prob, winner_array, learning_rate=0.001):
        self.policy_value_net.train()
        state_array = paddle.to_tensor(state_array)
        mct_prob = paddle.to_tensor(mct_prob)
        winner_array = paddle.to_tensor(winner_array)
        self.optimizer.clear_gradients()
        self.optimizer.set_lr(learning_rate)
        policy, value = self.policy_value_net(state_array)
        value = paddle.reshape(value, [-1])
        policy_loss = -paddle.mean(paddle.sum(mct_prob * policy, axis=1))
        value_loss = F.mse_loss(input=value, label=winner_array)
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.minimize(loss)
        entropy = -paddle.mean(paddle.sum(paddle.exp(policy) * policy, axis=1))
        return loss.numpy(), float(entropy)

    def save_model(self, model_path):
        parameters = self.policy_value_net.state_dict()
        paddle.save(parameters, model_path) 