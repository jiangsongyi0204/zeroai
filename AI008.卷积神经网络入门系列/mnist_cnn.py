import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 下载并加载训练数据
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)

# 2. 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层：输入通道1（灰度图），输出通道4，卷积核3x3
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # 第二个卷积层：输入通道4，输出通道8，卷积核3x3
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 32)  # 经过两次池化，28x28 -> 14x14 -> 7x7
        self.fc2 = nn.Linear(32, 10)  # 10个数字类别
    
    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 展平
        x = x.view(-1, 16 * 7 * 7)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# 3. 创建模型实例
model = SimpleCNN().to(device)
print("模型结构:")
print(model)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    accuracy = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    return avg_loss, accuracy

# 6. 测试函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\n测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return test_loss, accuracy


# 8. 训练模型
num_epochs = 5
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

print("开始训练...")

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test()
    
    # 记录训练过程
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

# 9. 可视化训练过程
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(range(1, num_epochs + 1), train_losses, 'b-', label='训练损失')
ax1.plot(range(1, num_epochs + 1), test_losses, 'r-', label='测试损失')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('训练和测试损失')
ax1.legend()
ax1.grid(True)

# 准确率曲线
ax2.plot(range(1, num_epochs + 1), train_accuracies, 'b-', label='训练准确率')
ax2.plot(range(1, num_epochs + 1), test_accuracies, 'r-', label='测试准确率')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('训练和测试准确率')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 11. 保存模型
torch.save(model.state_dict(), './model/mnist_cnn.pth')
print("模型已保存为 './model/mnist_cnn.pth'")