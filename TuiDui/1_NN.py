import torch
from torch import nn

#神经网络取名为Test1，继承nn.Module
class Test1(nn.Module):
    #初始化函数
    def __init__(self):
        super(Test1, self).__init__()
        #自己定义的神经网络结构
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    #前向传播函数
    def forward(self, x):
        #自己设置的前向传播过程
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
#实例化神经网络
model = Test1()
print(model)

#打印神经网络的参数
for name, param in model.named_parameters():
    print(name, param.size())

#打印神经网络的结构
print(model.fc1)
print(model.fc2)

# #打印神经网络的前向传播过程
print(model.forward)

# 创建一个随机输入张量，假设批量大小为1，输入大小为28x28（即784）
input_tensor = torch.randn(1, 28, 28)

# 将输入张量传递给模型，获取输出
output = model(input_tensor)

# 打印输出结果
print(output)