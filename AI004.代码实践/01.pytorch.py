# 0.定义训练数据
xs = [[0,0,0,0,0,0,0,0],
      [1,0,0,0,0,0,0,0],
      [0,1,0,0,0,0,0,0],
      [0,0,1,0,0,0,0,0],
      [1,1,0,0,0,0,0,0],
      [1,0,1,0,0,0,0,0],
      [0,1,1,0,0,0,0,0],
      [1,1,1,0,0,0,0,0],
      [0,0,0,1,0,0,0,0],
      [1,0,0,1,0,1,0,0],
      [0,0,0,0,1,0,0,0],
      [0,0,1,0,1,0,0,1],
      [0,0,0,0,0,1,0,0],
      [0,0,0,0,0,0,1,0],
      [0,0,0,0,0,0,0,1],
      [0,0,0,0,0,1,1,0],
      [0,0,0,0,0,1,0,1],
      [0,0,0,0,0,1,1,1],
      [1,1,1,1,1,1,1,1]]

ys = [[1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [1,0,0,0],
      [0,1,0,0],
      [0,1,0,0],
      [0,0,0,1],
      [0,0,0,1],
      [0,0,1,0],
      [0,0,1,0],
      [0,0,1,0],
      [0,0,1,0],
      [0,0,1,0],
      [0,0,1,0],
      [0,0,1,0]]

import torch
import torch.nn as nn
import torch.optim as optim

# 1.定义神经网络结构
model = nn.Sequential(
    nn.Linear(8, 5),
    nn.Sigmoid(),
    nn.Linear(5, 4),
    nn.Sigmoid()
)

# 2.训练神经网络

# 将数据转换为 tensor
inputs = torch.tensor(xs, dtype=torch.float32)
labels = torch.tensor(ys, dtype=torch.float32)

# 损失与优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练神经网络
for epoch in range(1000):
    # 1.前向计算
    logits = model(inputs)
    # 2.计算均方误差损失  
    loss = criterion(logits, labels)
    # 3.初始化梯度
    optimizer.zero_grad()
    # 4.反向传播
    loss.backward()
    # 5.调整参数
    optimizer.step()
    print(f"Epoch {epoch} - loss: {loss.item():.6f}")

# 输出训练后对输入数据的最终输出（概率和预测）
with torch.no_grad():
    logits = model(inputs)

print("Final predictions:")
print(torch.round(logits * 1000) / 1000)