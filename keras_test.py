#coding=utf-8
'''
测试车牌识别程序。

'''


from tensorflow.python import keras
from tensorflow.python.keras.models import load_model, model_from_json
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import win_unicode_console
import numpy as np
import sys
win_unicode_console.enable()


image_height = 36
image_width = 136

# 车牌共有如下字符类型
chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ]
n_class=len(chars)


def decode(y_):
    '''
    将one-hot码列表转为字符串。
    '''
    #y = np.argmax(np.array(y_), axis=2)[:,0]
    y = np.argmax(np.array(y_), axis=2)[0]
    print(y.shape)
    print(y)
    return ''.join([chars[x] for x in y])


#model_filename = './ckpt/cifar10_resnet_ckpt.h5'
model_filename = './ckpt/plate_resnet-40e-val_acc_0.88.h5'
json_filename = './resnetv1.json'

img = load_img(sys.argv[1], target_size=(image_height, image_width))
img = img_to_array(img)/255.   # 转到0-1
img = np.expand_dims(img, 0)


model = load_model(model_filename)
#model.load_weights(model_filename)
output = model.predict(img)
print(output.shape)
plate_str = decode(output)

print(plate_str)                       

