# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import setting
import dataset
from model import CNN
import encoding


def main():
    cnn = CNN()
    cnn.eval()#将模型设置为评估模式。这会影响某些层的行为，如 Dropout 和 Batch Normalization，确保在评估时不进行随机操作。
    cnn.load_state_dict(torch.load('model.pkl'))#加载之前训练好的模型参数。torch.load('model.pkl') 读取保存的模型状态字典，cnn.load_state_dict(...) 将这些参数加载到模型中。
    # 打印一条消息，表示模型已成功加载。
    print("load cnn net.")
    # 调用 dataset 模块中的 get_eval_data_loader 函数，获取评估数据的 DataLoader。DataLoader 负责批量加载数据，并支持数据打乱和并行加载（尽管在评估时通常不打乱）。
    eval_dataloader = dataset.get_eval_data_loader()
    # 初始化正确预测的数量为0，用于统计模型在评估集上的正确预测数。
    correct = 0
    # 初始化总样本数为0，用于统计评估集上的总样本数。
    total = 0
    # 遍历评估数据加载器中的每个批次。i 是批次的索引，images 是当前批次的图像数据，labels 是对应的标签。
    for i, (images, labels) in enumerate(eval_dataloader):
        # 将当前批次的图像数据赋值给变量 image（这一步实际上没有必要，可以直接使用 images）。
        image = images
        # 将图像数据包装成 Variable（在较新版本的 PyTorch 中，可以直接使用 Tensor，无需显式地包装成 Variable）。
        vimage = Variable(image)
        # 将图像数据传入模型 cnn，得到预测的标签。predict_label 是一个张量，包含每个样本的预测结果。
        predict_label = cnn(vimage)
        # c0~c3 这四行代码分别提取模型预测的四个字符位置的概率分布，并找到概率最大的字符作为预测结果。
        # **predict_label[0, 0:setting.ALL_CHAR_SET_LEN]**​：提取第一个样本的第一个字符位置的预测概率分布
        # **np.argmax(...)**：找到概率最大的索引，即预测的字符在 ALL_CHAR_SET 中的位置。
        # **setting.ALL_CHAR_SET[...]**：根据索引获取对应的字符。
        # **c0, c1, c2, c3** 分别对应验证码中的四个字符位置
        c0 = setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 0:setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, setting.ALL_CHAR_SET_LEN:2 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * setting.ALL_CHAR_SET_LEN:3 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * setting.ALL_CHAR_SET_LEN:4 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        # 将四个预测的字符拼接成一个字符串，表示整个验证码的预测结果。
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        # labels.numpy()[0]将标签张量转换为 NumPy 数组，并提取第一个样本的真实标签（假设 labels 的形状为 [batch_size, ...]）。
        # 调用 encoding 模块中的 decode 函数，将编码后的真实标签解码为实际的字符串
        true_label = encoding.decode(labels.numpy()[0])
        # 将当前批次的样本数量加到总样本数中。labels.size(0) 返回当前批次的样本数。
        total += labels.size(0)
        # 如果预测的验证码字符串与真实的验证码字符串相同，则增加正确预测的数量。
        if (predict_label == true_label):
            correct += 1
        # 每处理200个样本，打印一次当前的评估准确率。
        if (total % 200 == 0):
            # 输出当前已评估的样本总数和准确率（百分比形式）
            print('Test Accuracy of the model on the %d eval images: %f %%' %
                  (total, 100 * correct / total))
    # 在所有批次评估完成后，打印最终的评估准确率
    print('Test Accuracy of the model on the %d eval images: %f %%' %
          (total, 100 * correct / total))
    # **return correct / total**：返回最终的评估准确率（正确预测数除以总样本数）。这在某些情况下可能用于进一步的分析或记录。
    return correct / total


if __name__ == '__main__':
    main()
