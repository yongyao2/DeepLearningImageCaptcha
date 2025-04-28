# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import dataset
from model import CNN
from evaluate import main as evaluate

num_epochs = 30
batch_size = 100
learning_rate = 0.001


def main():
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 实例化 CNN 类，创建一个卷积神经网络模型
    cnn = CNN()

    # 如果有多个GPU，使用DataParallel包装模型
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        cnn = nn.DataParallel(cnn)
    # 将模型移动到GPU上
    cnn = CNN().to(device)
    # 将模型设置为训练模式
    cnn.train()

    # 定义损失函数为多标签软边距损失
    criterion = nn.MultiLabelSoftMarginLoss()
    # 定义优化器为 Adam
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # 初始化最大评估准确率为-1
    max_eval_acc = -1

    # 调用 dataset 模块中的 get_train_data_loader 函数，获取训练数据的 DataLoader
    train_dataloader = dataset.get_train_data_loader()

    # 开始训练循环，遍历每个epoch
    for epoch in range(num_epochs):
        # 遍历每个批次的数据
        for i, (images, labels) in enumerate(train_dataloader):
            # 将图像和标签数据移动到GPU上
            images = images.to(device)
            labels = labels.to(device)

            # 将图像数据传入模型 cnn，得到预测的标签
            predict_labels = cnn(images)

            # 计算预测标签与真实标签之间的损失值
            loss = criterion(predict_labels, labels)

            # 清除优化器中所有参数的梯度，防止梯度累积
            optimizer.zero_grad()

            # 进行反向传播，计算所有参数的梯度
            loss.backward()

            # 使用优化器更新模型参数
            optimizer.step()

            if (i + 1) % 10 == 0:  # 每处理10个批次，打印当前的epoch、step和损失值
                print("epoch:", epoch, "step:", i, "loss:", loss.item())  # 输出训练信息

            if (i + 1) % 100 == 0:  # 每处理100个批次，保存当前模型的参数到 model.pkl 文件中
                torch.save(cnn.state_dict(), "./model.pkl")  # 保存模型的状态字典（即模型参数）
                print("save model")

        print("epoch:", epoch, "step:", i, "loss:", loss.item())

        # 调用 evaluate 函数，评估当前模型在验证集上的准确率
        eval_acc = evaluate()

        # 如果当前评估准确率高于之前的最高记录，则更新最高准确率并保存当前模型为 best_model.pkl
        if eval_acc > max_eval_acc:
            torch.save(cnn.state_dict(), "./best_model.pkl")  # 保存最佳模型的参数
            print("save best model")

    # 训练结束后，保存最终的模型参数到 model.pkl 文件中
    torch.save(cnn.state_dict(), "./model.pkl")
    print("save last model")


if __name__ == '__main__':
    main()