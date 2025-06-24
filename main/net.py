import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

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
        y = paddle.add(x,y)
        return self.conv2_relu(y)

class PolicyValueNet(nn.Layer):
    def __init__(self,in_channels=17,channels_num=256,res_block_num=19):
        super().__init__()
        self.initial_conv=nn.Conv2D(in_channels=in_channels,out_channels=channels_num,kernel_size=3,stride=1,padding=1)
        self.initial_bn=nn.BatchNorm2D(num_features=256)
        self.initial_relu=nn.ReLU()
        self.res_blocks=nn.LayerList([ResBlock(channels_num=channels_num) for _ in range(res_block_num)])
        self.policy_conv=nn.Conv2D(in_channels=channels_num,out_channels=16,kernel_size=1,stride=1)
        self.policy_bn=nn.BatchNorm2D(num_features=16)
        self.policy_relu=nn.ReLU()
        self.policy_linear=nn.Linear(in_features=16*10*9,out_features=2086)
        self.value_conv=nn.Conv2D(in_channels=channels_num,out_channels=8,kernel_size=1,stride=1)
        self.value_bn=nn.BatchNorm2D(num_features=8)
        self.value_relu1=nn.ReLU()
        self.value_linear1=nn.Linear(in_features=8*10*9,out_features=256)
        self.value_relu2=nn.ReLU()
        self.value_linear2=nn.Linear(in_features=256,out_features=1)

    def forward(self, x):
        x=self.initial_conv(x)
        x=self.initial_bn(x)
        x=self.initial_relu(x)
        for res_block in self.res_blocks:
            x=res_block(x)
        policy=self.policy_conv(x)
        policy=self.policy_bn(policy)
        policy=self.policy_relu(policy)
        policy=paddle.reshape(policy,[-1,16*10*9])
        policy=self.policy_linear(policy)
        policy=F.log_softmax(policy)
        value=self.value_conv(x)
        value=self.value_bn(value)
        value=self.value_relu1(value)
        value=paddle.reshape(value,[-1,8*10*9])
        value=self.value_linear1(value)
        value=self.value_relu2(value)
        value=self.value_linear2(value)
        value=F.tanh(value)

        return policy,value



class Net:
    def __init__(self,model_path=None,use_gpu=True):
        self.weight_decay=2e-3
        self.use_gpu=use_gpu
        self.policy_value_net=PolicyValueNet()
        self.optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=self.policy_value_net.parameters(),
                                         weight_decay=self.weight_decay)
        if model_path:
            parameters=paddle.load(model_path)
            self.policy_value_net.set_dict(parameters)

    def batch_policy_value(self,state_array):
        self.policy_value_net.eval()
        state=paddle.to_tensor(state_array)
        policy,value=self.policy_value_net(state)
        probabilities=np.exp(policy.numpy())
        return probabilities,value.numpy()

    def get_policy_value(self,state_array,move_id):
        self.policy_value_net.eval()
        state=np.ascontiguousarray(state_array.reshape(-1,17,10,9)).astype("float32")
        state=paddle.to_tensor(state)
        policy,value=self.policy_value_net(state)
        probabilities = np.exp(policy.numpy().flatten())
        probabilities = zip(move_id,probabilities[move_id])
        return probabilities,value.numpy()

    def train_step(self,state_array,mct_prob,winner_array,learning_rate=0.001):
        self.policy_value_net.train()
        state_array=paddle.to_tensor(state_array)
        mct_prob=paddle.to_tensor(mct_prob)
        winner_array=paddle.to_tensor(winner_array)
        self.optimizer.clear_gradients()
        self.optimizer.set_lr(learning_rate)
        policy,value=self.policy_value_net(state_array)
        value=paddle.reshape(value,[-1])
        policy_loss = -paddle.mean(paddle.sum(mct_prob * policy, axis=1))
        value_loss=F.mse_loss(input=value,label=winner_array)
        loss=value_loss+policy_loss
        loss.backward()
        self.optimizer.minimize(loss)
        entropy=-paddle.mean(paddle.sum(paddle.exp(policy)*policy,axis=1))
        return loss.numpy(),float(entropy)

    def save_model(self,model_path):
        parameters=self.policy_value_net.state_dict()
        paddle.save(parameters,model_path)



if __name__ == "__main__":
    policy_value_net=PolicyValueNet()
    test_data=paddle.ones([8,17,10,9])
    po, va=policy_value_net(test_data)
    print(po.shape)
    print(va.shape)