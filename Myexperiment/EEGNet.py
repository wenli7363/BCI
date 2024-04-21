import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import Feature, Predictor1
from dataloader4EGGNet import dataloader
from matplotlib import pyplot as plt
from preprocessing import import_data

class EEGNet(nn.Module):
    def __init__(self, num_classes):
        super(EEGNet, self).__init__()
        self.feature_extractor = Feature(num_classes)
        self.predictor1 = Predictor1(num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.predictor1(x)
        return x

def train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs):
    model.train()
    train_loss_history = []
    val_loss_history = []
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
        val_accuracy = val_correct / val_total
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

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
    model = EEGNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    file_path = '../dataset/200.h5'    
    X,y = import_data(file_path)
    train_loader, val_loader, test_loader = dataloader(X, y, val_ratio=0.2)

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
    #     break


    num_epochs = 100

    model = train(model, train_loader, val_loader,test_loader, criterion, optimizer, device, num_epochs)