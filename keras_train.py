#coding=utf-8
'''
根据生成的车牌样本，将数据载入为dataframe格式，然后训练车牌识别算法。 
用到了keras的自定义样本输入迭代器，不用已有的迭代器；
用到了keras的自定义loss函数，定制字符串的loss函数，将每个字符的loss求平均作为总loss

'''
from tensorflow.python import keras
from tensorflow.python.keras import models, layers, optimizers
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os          
#from tensorflow.python.keras import
import pandas as pd 
import numpy as np
import random
import math

train_image_data_path = './plate_images/train'
val_image_data_path = './plate_images/val'
test_image_data_path = './plate_images/test'
image_height = 36
image_width = 136
ckpt_path = './ckpt'
log_dir = './log'
plate_str_length = 7    # 车牌字符个数


class DataGenerator(keras.utils.Sequence):
    '''
    Keras的数据样本batch生成迭代器，可作为model.fit_generator函数的参数输入。
    必须继承自Sequence

    '''
    def __init__(self, df_datas, batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.datas = df_datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle 
    
    def __len__(self):
        # 计算每个epoch的迭代次数
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，根据自己对数据的读取方式自由发挥
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas.iloc[k] for k in batch_indexs]

        # 生成数据
        x,y = self.data_generation(batch_datas)

        return x,y
    
    def on_epoch_end(self):
        # 在每次epoch结束时进行依次随机排序，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def data_generation(self, batch_datas):
        images = []
        labels = []

        for i,data in enumerate(batch_datas):
            # 从dataframe中提取出文件名和标签信息
            image_filename = data['filename']
            str_labels = data['target']
            # x_train数据
            img = load_img(image_filename, target_size=(image_height, image_width))
            img = img_to_array(img)/255.   # 转到0-1
            images.append(img)

            # y_train数据
            labels.append(str_labels)

        return np.array(images), np.array(labels)






# 车牌共有如下字符类型
chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ]
n_class=len(chars)

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64}


def decode(y_):
    '''
    将one-hot码列表转为字符串。
    '''
    y = np.argmax(np.array(y_), axis=2)[:,0]
    return ''.join([chars[x] for x in y])

def encode(y_):
    '''
    将字符串转化为one-hot码
    '''
    y = np.zeros((plate_str_length, n_class), dtype=np.float32)
    for i in range(plate_str_length):
        y[i, index[y_[i]]] = 1
    
    return y
    

def load_images_to_pd(plate_image_path):
    '''
    载入车牌图片，图片名字即为车牌号，将数据转为pd格式，以待输入。
    需要输入one-hot码？

    '''
    plate_images = os.listdir(plate_image_path)
    print("file number: {}".format(len(plate_images)))
    plate_images = [xx for xx in plate_images if os.path.splitext(xx)[-1] in ['.jpg']]
    print('find {} plates'.format(len(plate_images)))
    #plate_dict = {}
    filename_list = []
    plate_str_list = []
    for plate_filename in plate_images:
        # 取出中间的车牌号
        _,plate_str,_ = plate_filename.split('.')
        #plate_dict[os.path.join(plate_image_path, plate_filename)] = plate_str
        filename_list.append(os.path.join(plate_image_path, plate_filename))
        plate_str_list.append(encode(plate_str))   # 先转为onehot码
    #onehot_plate_str = encode()
    print("target shape: ", plate_str_list[0].shape)
    
    df = pd.DataFrame({"filename":filename_list, "target":plate_str_list})

    return df


def my_str_cross_entropy_loss(target, y_pred):
    '''
    字符比较，每个字符用crossentropy loss，所有字符求平均
    '''
    str_loss_list = []
    for i in range(plate_str_length):
        # 计算出每个字符的loss
        str_loss = K.categorical_crossentropy(target, y_pred, from_logits=False)
        str_loss_list.append(str_loss)
    # 计算loss平均值
    mean_loss = K.mean(str_loss_list)
    return mean_loss



# 自定义loss，使用ctc loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_simple_cnn_model():
    '''
    创建cnn分类模型，输入为包含7个车牌字符的整张图像，每个字符的输出为分类one-hot编码。
    '''
    input_tensor = models.Input((image_height, image_width, 3))
    x = input_tensor

    for i in range(3):
        x = layers.Conv2D(32*2**i, (3,3), activation='relu')(x)
        x = layers.Conv2D(32*2**i, (3,3), activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.25)(x)

    x_list = []
    # 经过多个全连接层，收集所有输出结果，保存为list，7个元素表示7个字符的预测结果
    #xx = [layers.Reshape((-1,65))(layers.Dense(n_class ,activation='softmax', name='c%d'%(i+1))(x)) for i in range(plate_str_length)]
    #print("#### ", type(xx))
    for i in range(plate_str_length):
        xi = layers.Dense(n_class, activation='softmax', name='c%d'%(i+1))(x)
        xi = layers.Reshape((1,65))(xi)   # 维度 (1, 65)，reshape函数不考虑batch维

        #print('### xi shape: ', xi.output_shape)
        x_list.append(xi)


    # 对x进行concat，在第2维上，第一维是batch_size
    concated = layers.concatenate(x_list, axis=1)
    #print("#### output shape: ", K.shape(concated))
    
    model = models.Model(inputs=input_tensor, outputs=concated)
    print("#### model output shape: ", model.output_shape)

    # 绘制模型结构图
    print("### plot model")
    plot_model(model, to_file='simple_cnn.png', show_shapes=True)
    return model




def gen_image_batch_from_path_list(filename_list):
    '''
    根据文件名列表，打开所有图片文件，生成image_batch
    '''

plate_count = 0
# 创建数据样本迭代器
def plate_generator(df, batch_size, batch_number_per_epoch):
    '''
    样本生成迭代器，df是dataframe格式的数据，batch_size是生成的batch_size

    '''
    df = df.copy()
    global plate_count # 声明count是全局变量  
    sample_number = len(df)  # 样本个数
    old_index = list(range(sample_number))
    while True:
        # 随机生成随机开始index
        index_start = np.random.randint(sample_number-batch_size)
        index_end = index_start + batch_size
        index_list = list(range(index_start, index_end))
        plate_count += 1

        filename_list = []
        image_data_list = []
        target_list = []
        # 获取filename_list和target_list
        for ii in index_list:
            filename_list.append(df['filename'][ii])
            target_list.append(df['target'][ii])
            img = load_img(df['filename'][ii], target_size=(image_height, image_width))
            img = img_to_array(img)
            img = np.expand_dims(img, 0)  # 扩充一维
            image_data_list.append(img)
        # 将图片list合并为一个BHWC的四维数组
        image_batch = np.concatenate(image_data_list, axis=0)

        yield image_batch, target_list

        # 如果执行完一个epoch，重新排列
        if plate_count % batch_number_per_epoch == 0:
            # 用random.sample函数生成新的index，注意不是np.random
            new_index = random.sample(old_index, sample_number)
            # 重新排序
            df = df.reindex(new_index)



    

def train_plate():
    '''
    训练车牌识别算法
    '''
    adam = optimizers.Adam(lr=0.001)
    model = get_simple_cnn_model()

    # 
    model.compile(loss='categorical_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])

    ### train data from dataframe
    batch_size = 96
    steps_per_epoch = 10000 // batch_size

    train_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                shear_range=0.2,
                zoom_range = 0.2,
                validation_split=0.1,
                horizontal_flip=True)

    test_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_image_df = load_images_to_pd(train_image_data_path)
    val_image_df = load_images_to_pd(val_image_data_path)
    test_image_df = load_images_to_pd(test_image_data_path)

    train_dataset = DataGenerator(train_image_df, batch_size=batch_size)
    val_dataset = DataGenerator(val_image_df, batch_size=batch_size)

    print(train_image_df.head())


    ckpt_name = 'plate_resnet-{epoch:02d}e-val_acc_{val_acc:.2f}.h5'
    #ckpt_name = 'plate_resnet-{epoch:02d}e.h5'
    #ckpt_name = 'plate_best.h5'
    checkpoint = ModelCheckpoint(filepath=os.path.join(ckpt_path, ckpt_name), 
                                    monitor='val_acc',
                                    verbose=1, save_best_only=False, save_weights_only=False, period=2)


    # 动态调整学习率
    total_epochs = 40
    def lr_sch(epoch):
        if epoch < total_epochs//4:
            return 1e-3
        elif total_epochs//4 <= epoch < total_epochs//2:
            return 1e-4
        else:
            return 1e-5

    lr_scheduler = LearningRateScheduler(lr_sch)
    #lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
    #                                patience=5, mode='max', min_lr=1e-3)
    train_log = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #callbacks = [checkpoint, lr_scheduler, lr_reducer]
    
    callbacks = [checkpoint, lr_scheduler]

    history = model.fit_generator(train_dataset,
                    steps_per_epoch=steps_per_epoch,
                    epochs=total_epochs,
                    verbose=1,
                    validation_data=val_dataset,
                    callbacks=callbacks)

    history_dict = history.history
    df = pd.DataFrame.from_dict(history_dict)
    df.to_csv(os.path.join(ckpt_path,'scene_resnet_dict.csv'))


if '__main__' == __name__:
    train_plate()
