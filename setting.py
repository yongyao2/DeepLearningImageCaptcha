# -*- coding: UTF-8 -*-
import os

# 验证码中的字符
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
UPPER_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
LOWER_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

ALL_CHAR_SET = NUMBER + UPPER_ALPHABET + LOWER_ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

TRAIN_DATASET_PATH = 'dataset' + os.path.sep + 'train'
EVAL_DATASET_PATH = 'dataset' + os.path.sep + 'eval'
PREDICT_DATASET_PATH = 'dataset' + os.path.sep + 'predict'