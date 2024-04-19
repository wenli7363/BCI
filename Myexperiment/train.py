import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from preprocessing import import_data
from model import Feature, Predictor1, Predictor2
from dataloader import dataloader

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 导入数据
file_path = 'path/to/your/h5py/file'
X, y = import_data(file_path)

# 2. 创建数据加载器
batch_size = 128
train_loader, test_loader = dataloader(X, y, batch_size=batch_size, shuffle=True)

# 3. 实例化模型
num_classes = 4  # 假设你有4个类别
feature_extractor = Feature(num_classes).to(device)
classifier1 = Predictor1(num_classes).to(device)
classifier2 = Predictor2(num_classes).to(device)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer_feature = optim.Adam(feature_extractor.parameters(), lr=0.001)
optimizer_classifier1 = optim.Adam(classifier1.parameters(), lr=0.001)
optimizer_classifier2 = optim.Adam(classifier2.parameters(), lr=0.001)

# 5. 训练模型
num_epochs = 100
train_losses = []
test_accuracies = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 前向传播
        features = feature_extractor(inputs)
        output1 = classifier1(features)
        output2 = classifier2(features)
        
        # 计算损失
        loss1 = criterion(output1, labels)
        loss2 = criterion(output2, labels)
        loss = loss1 + loss2
        
        # 反向传播和优化
        optimizer_feature.zero_grad()
        optimizer_classifier1.zero_grad()
        optimizer_classifier2.zero_grad()
        loss.backward()
        optimizer_feature.step()
        optimizer_classifier1.step()
        optimizer_classifier2.step()
        
        running_loss += loss.item()
    
    train_losses.append(running_loss / len(train_loader))
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}', end=' ')

    # 测试模型
    feature_extractor.eval()
    classifier1.eval()
    classifier2.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            features = feature_extractor(inputs)
            output1 = classifier1(features)
            output2 = classifier2(features)
            output = output1 + output2
            _, predicted = torch.max(output.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    correct = sum(np.array(all_predictions) == np.array(all_labels))
    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

# 7. 可视化
# 绘制损失曲线
plt.figure(figsize=(8, 6))
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# 绘制测试集准确率曲线
plt.figure(figsize=(8, 6))
plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Curve')
plt.legend()
plt.show()

# 绘制最后一个epoch的混淆矩阵
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()