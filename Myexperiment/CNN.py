import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataloader import dataloader_train_val_test, dataloader_test
from matplotlib import pyplot as plt
from preprocessing import import_data, append_data

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256000, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(batch_size, -1)  # 将特征图拉平
        # print(x.shape[1])
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, reg_lambda=0.001):
    model.train()
    train_loss_history = []
    val_loss_history = []
    test_acc_history = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        running_val_loss = 0.0
        
        # 训练阶段
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 计算L2正则项
            reg_loss = 0
            for param in model.parameters():
                reg_loss += torch.sum(param ** 2)
            reg_loss *= reg_lambda
            loss += reg_loss
            
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)
        train_accuracy = train_correct / train_total

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(epoch_val_loss)

        # 测试阶段,计算准确率
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_test_acc = correct / total
        test_acc_history.append(epoch_test_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}, Test Acc: {epoch_test_acc}')

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state_dict = model.state_dict()

    # 加载最佳模型
    model.load_state_dict(best_model_state_dict)
    
    # 测试阶段
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_accuracy = test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # 绘制损失值趋势图
    plt.figure()
    plt.plot(range(num_epochs), train_loss_history, label='Train Loss')
    plt.plot(range(num_epochs), val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Trend')
    plt.legend()
    plt.show()

    return model

if __name__ == '__main__':

    # 使用示例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    file_path = 'D:\\Desktop\\2\\WH1.h5'
    X, y = import_data(file_path)
    X, y = append_data("D:/Desktop/2/WH2.h5", X, y)
    X, y = append_data("D:/Desktop/2/WH3.h5", X, y)
    # X, y = append_data("D:/Desktop/2/WH4.h5", X, y)
    # X, y = append_data("D:/Desktop/2/WH5.h5", X, y)
    # X, y = append_data("D:/Desktop/2/WH6.h5", X, y)
    # X,y = append_data("D:/Desktop/2/WH7.h5",X,y)

    train_loader, val_loader, test_loader = dataloader_train_val_test(X, y, val_ratio=0.2)
    # test_loader = dataloader_test(X, y, batch_size=200)

    # for batch in train_loader:
    #     inputs, labels = batch
    #     print(f"Inputs shape: {inputs.shape}")
    #     print(f"Labels shape: {labels.shape}")
    #     break
    # for batch in val_loader:
    #     inputs, labels = batch
    #     print(f"Inputs shape: {inputs.shape}")
    #     print(f"Labels shape: {labels.shape}")
    #     break
    # for batch in test_loader:
    #     inputs, labels = batch
    #     print(f"Inputs shape: {inputs.shape}")
    #     print(f"Labels shape: {labels.shape}")
    #     # print(inputs)
    #     # print(labels)
    #     break

    num_epochs = 100

    model = train(model, train_loader, val_loader,test_loader, criterion, optimizer, device, num_epochs)