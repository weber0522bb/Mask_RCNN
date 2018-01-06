# -*- coding: utf-8 -*-
import pandas as pd
from PIL import Image
import numpy as np

TRAIN_FILE = 'kaggle/training.csv'
TEST_FILE = 'kaggle/test.csv'
SAVE_PATH = 'kaggle/model'


VALIDATION_SIZE = 100    #验证集大小
EPOCHS = 100             #迭代次数
BATCH_SIZE = 64          #每个batch大小，稍微大一点的batch会更稳定
EARLY_STOP_PATIENCE = 10 #控制early stopping的参数


def input_data(test=False):
    file_name = TEST_FILE if test else TRAIN_FILE
    df = pd.read_csv(file_name)
    cols = df.columns[:-1]

    df = df.dropna()
    df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' ') / 255.0)

    X = np.vstack(df['Image'])
    X = X.reshape((-1, 96, 96, 1))

    if test:
        y = None
    else:
        y = df[cols].values / 96.0
    return X, y


#将文件中的数值转成图片并保存
def data2img(test=False):
    file_name = TEST_FILE if test else TRAIN_FILE
    df = pd.read_csv(file_name)
    df = df.dropna()
    df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' '))

    X = np.vstack(df['Image'])
    for i in range(0, len(X)):
        data = np.reshape(X[i], (96, 96))
        new_im = Image.fromarray(data)
        #new_im.show()
        new_im = new_im.convert('RGB')
        new_im.save("kaggle/face/face" + str(i) + ".png", "png")


#最后生成提交结果的时候要用到
keypoint_index = {
        'left_eye_center_x': 0,
        'left_eye_center_y': 1,
        'right_eye_center_x': 2,
        'right_eye_center_y': 3,
        'left_eye_inner_corner_x': 4,
        'left_eye_inner_corner_y': 5,
        'left_eye_outer_corner_x': 6,
        'left_eye_outer_corner_y': 7,
        'right_eye_inner_corner_x': 8,
        'right_eye_inner_corner_y': 9,
        'right_eye_outer_corner_x': 10,
        'right_eye_outer_corner_y': 11,
        'left_eyebrow_inner_end_x': 12,
        'left_eyebrow_inner_end_y': 13,
        'left_eyebrow_outer_end_x': 14,
        'left_eyebrow_outer_end_y': 15,
        'right_eyebrow_inner_end_x': 16,
        'right_eyebrow_inner_end_y': 17,
        'right_eyebrow_outer_end_x': 18,
        'right_eyebrow_outer_end_y': 19,
        'nose_tip_x': 20,
        'nose_tip_y': 21,
        'mouth_left_corner_x': 22,
        'mouth_left_corner_y': 23,
        'mouth_right_corner_x': 24,
        'mouth_right_corner_y': 25,
        'mouth_center_top_lip_x': 26,
        'mouth_center_top_lip_y': 27,
        'mouth_center_bottom_lip_x': 28,
        'mouth_center_bottom_lip_y': 29
}

if __name__ == '__main__':
    data2img(test=True)
