# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
# from visdom import Visdom # pip install Visdom
import setting
import dataset
from model import CNN


def main():
    # 创建 CNN 类的一个实例，即加载你之前训练好的模型结构。
    cnn = CNN()
    # 将模型设置为评估模式。这会影响某些层的行为，如 Dropout 和 Batch Normalization，使其在评估时表现稳定。
    cnn.eval()
    # 从文件 best_model.pkl 中加载模型的参数（即权重）。这假设你已经通过训练过程保存了最佳模型的参数。
    cnn.load_state_dict(torch.load('best_model.pkl'))
    # 输出提示信息，表示模型已成功加载。
    print("load cnn net.")

    # 调用 dataset 模块中的 get_predict_data_loader 函数，获取用于预测的数据加载器。这个 DataLoader 会批量加载待预测的图像数据。
    predict_dataloader = dataset.get_predict_data_loader()
    
    # vis = Visdom()
    # 开始遍历预测数据的 DataLoader。images 是当前批次的图像数据，labels 是对应的标签（虽然在预测时可能不需要标签，但 DataLoader 可能仍然返回它）。
    for i, (images, labels) in enumerate(predict_dataloader):
        # 将当前批次的图像数据赋值给变量 image。注意，这里假设批次大小为1，因为后续代码处理单个样本。
        image = images
        # 或者，如果需要计算梯度（虽然预测时通常不需要），可以保留，但一般预测时不需要 Variable。
        vimage = Variable(image)

        # 将图像数据传入模型 cnn，得到预测的标签。predict_label 应该是一个形状为 [batch_size, ALL_CHAR_SET_LEN * MAX_CAPTCHA] 的张量。
        predict_label = cnn(vimage)

        # 从 predict_label 中提取第一个样本的第一个字符的预测概率分布。
        # 使用 np.argmax 找到概率最大的索引，然后通过 setting.ALL_CHAR_SET 将该索引转换为对应的字符。
        # c0 表示第一个字符的预测结果。
        c0 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:setting.ALL_CHAR_SET_LEN].data.numpy())]

        # 提取第一个样本的第二个字符的预测概率分布，并转换为对应的字符。
        # c1 表示第二个字符的预测结果。
        c1 = setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, setting.ALL_CHAR_SET_LEN:2 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        # 提取第一个样本的第三个字符的预测概率分布，并转换为对应的字符。
        # c2 表示第三个字符的预测结果。
        c2 = setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * setting.ALL_CHAR_SET_LEN:3 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        # 提取第一个样本的第四个字符的预测概率分布，并转换为对应的字符。
        # c3 表示第四个字符的预测结果。
        c3 = setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * setting.ALL_CHAR_SET_LEN:4 * setting.ALL_CHAR_SET_LEN].data.numpy())]

        # 将四个字符拼接成一个完整的验证码字符串。
        c = '%s%s%s%s' % (c0, c1, c2, c3)

        # 输出预测的验证码字符串。
        print(c)
        # vis.images(image, opts=dict(caption=c))


if __name__ == '__main__':
    main()
