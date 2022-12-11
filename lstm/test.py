import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os

# 设置超参数，并下载手写数据集
epoch = 2
lr = 0.01
batch_size=50
train_data = torchvision.datasets.MNIST(
    root='./datas/',   #下载到该目录下
    train=True,                                     #为训练数据
    transform=torchvision.transforms.ToTensor(),    #将其装换为tensor的形式
    download=False, #第一次设置为true表示下载，下载完成后，将其置成false
)

test_data = torchvision.datasets.MNIST(
    root='./datas/',
    train=False, # this is testing data
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

# 载入数据
train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_x = test_data.data.type(torch.FloatTensor)[:2000]/255
test_y = test_data.targets[:2000]

class LSTMnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMnet, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)

    def forward(self, x):                  # x‘s shape (batch_size, 序列长度, 序列中每个数据的长度)
        out, _ = self.lstm(x)              # out‘s shape (batch_size, 序列长度, hidden_dim)
        out = out[:, -1, :]                # 中间的序列长度取-1，表示取序列中的最后一个数据，这个数据长度为hidden_dim，
                                            # 得到的out的shape为(batch_size, hidden_dim)
        out = self.linear(out)             # 经过线性层后，out的shape为(batch_size, n_class)
        return out

model = LSTMnet(28, 56, 2, 10)             # 图片大小28*28，lstm的每个隐藏层56（自己设定数量大小）个节点，2层隐藏层
if torch.cuda.is_available():
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# training and testing
for epoch in range(2):
    for iteration, (train_x, train_y) in enumerate(train_loader):    # train_x‘s shape (BATCH_SIZE,1,28,28)
        train_x = train_x.squeeze()# after squeeze, train_x‘s shape (BATCH_SIZE,28,28),
        #print(train_x.size())  # 第一个28是序列长度(看做句子的长度)，第二个28是序列中每个数据的长度(词纬度)。
        # train_x = train_x.cuda()
        # print(train_x[0])
        # train_y = train_y.cuda()
        # test_x = test_x.cuda()
        #print(test_x[0])
        # test_y = test_y.cuda()
        output = model(train_x)
        loss = criterion(output, train_y)  # cross entropy loss
        optimizer.zero_grad()              # clear gradients for this training step
        loss.backward()                    # backpropagation, compute gradients
        optimizer.step()                   # apply gradients
 
        if iteration % 100 == 0:
            test_output = model(test_x)
            predict_y = torch.max(test_output, 1)[1].cpu().numpy()
            accuracy = float((predict_y == test_y.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
            print('epoch:{:<2d} | iteration:{:<4d} | loss:{:<6.4f} | accuracy:{:<4.2f}'.format(epoch, iteration, loss, accuracy))