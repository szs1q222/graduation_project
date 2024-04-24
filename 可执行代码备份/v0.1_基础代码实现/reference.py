# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets, transforms
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
#
# # 假设我们有一个简单的神经网络模型
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         # ... 定义网络层 ...
#
#     def forward(self, x):
#         # ... 定义前向传播 ...
#         return x
#
#     # 初始化模型、损失函数和优化器
#
#
# model = SimpleNet()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 假设我们有一些训练数据和测试数据
# # 这里仅为示例，实际使用时您需要从数据集中加载数据
# train_data = TensorDataset(torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,)))
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# test_data = TensorDataset(torch.randn(20, 3, 32, 32), torch.randint(0, 10, (20,)))
# test_loader = DataLoader(test_data, batch_size=20, shuffle=False)
#
# # 初始化日志文件
# log_file = open('training_log.txt', 'w')
#
# # 训练循环
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#         # 计算训练集上的准确率、精确率、召回率和F1值（这里仅作示例，通常不会这么做）
#     # 因为没有标签，这里假设我们已经有了预测结果和真实标签
#     # 实际使用时，您需要在验证集或测试集上评估模型
#     # preds = ... (从模型获得预测结果)
#     # true_labels = ... (真实标签)
#     # accuracy = accuracy_score(true_labels, preds.argmax(dim=1))
#     # precision = precision_score(true_labels, preds.argmax(dim=1), average='macro')
#     # recall = recall_score(true_labels, preds.argmax(dim=1), average='macro')
#     # f1 = f1_score(true_labels, preds.argmax(dim=1), average='macro')
#
#     # 在测试集上评估模型
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         all_preds = []
#         all_labels = []
#         for data in test_loader:
#             images, labels = data
#             outputs = model(images)
#             _, predicted = torch.max(outputs.dataload, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             all_preds.extend(predicted.tolist())
#             all_labels.extend(labels.tolist())
#
#         # 计算测试集上的评估指标
#         accuracy = 100 * correct / total
#         precision = precision_score(all_labels, all_preds, average='macro')
#         recall = recall_score(all_labels, all_preds, average='macro')
#         f1 = f1_score(all_labels, all_preds, average='macro')
#
#     # 将结果打印到控制台并写入日志文件
#     log_file.write(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, '
#                    f'Test Accuracy: {accuracy:.2f}%, '
#                    f'Test Precision: {precision:.4f}, '
#                    f'Test Recall: {recall:.4f}, '
#                    f'Test F1: {f1:.4f}\n')
#     log_file.flush()  # 确保内容被写入文件
#     print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, '
#           f'Test Accuracy: {accuracy:.2f}%, '
#           f'Test Precision: {precision:.4f}, '
#           f'Test Recall: {recall:.4f}, '
#           f'Test F1: {f1:.4f}')
#
# # 关闭日志文件
# log_file.close()
#
# # 假设我们有一些函数来加载数据和模型，这里只是一个示例结构
# # train_model(model, train_loader, criterion, optimizer, num_epochs)
# # test_model(model, test_loader)
#
# # 在实际应用中，您可能希望将模型和数据加载、训练、测试过程封装成函数
# # 然后调用这些函数来执行训练和评估
# # train_model(model, train_loader, criterion, optimizer, num_epochs)
# # test_results = test_model(model, test_loader)
# # 然后使用test_results中的预测和真实标签来计算准确率、精确率、召回率和F1值
#
# # 注意：上面的代码示例仅用于说明目的，并没有实现完整的训练和测试流程。
# # 在实际应用中，您需要根据您的数据集和模型架构来定制这些流程。
import os

import torch
from torch import nn

# T1 = torch.tensor([1, 2, 5, 4])
# T2 = torch.tensor([1, 2, 3, 4])
# T3 = torch.tensor([2, 2, 3, 4])
# a, b = [], []
# a.append(T1.unsqueeze_(0))
# a.append(T2.unsqueeze_(0))
# a.append(T3.unsqueeze_(0))
# b.append(T1)
# b.append(T2)
# b.append(T3)
# print(a)
# print(b)
#
# c = torch.cat(a, dim=0)
# print(c)
# d = torch.cat(b, dim=0)
# print(d)

# targets = torch.tensor([[1, ], [0, ], [2, ]])
# print(targets)
# targets = targets.flatten()
# print(targets)

# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# print(input)
# print(input.shape)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(target)
# print(target.shape)
# output = loss(input, target)
# print(output)
# print(output.shape)

# y_axit = {"total_train_losses": 1,
#           "total_loss_losses": 2,
#           "accuracies": 3,
#           "precisions": 4,
#           "recalls": 5,
#           "f1_scores": 6}
# for i, j in enumerate(y_axit):
#     print(i)

# folder_path = os.path.join(os.getcwd(), )
# print(folder_path)
# print(os.path.exists("./visualization"))
# # os.makedirs("./visualization")

