from collections import defaultdict
from glob import glob
from random import choice, sample
from myUtils import gen, gen_over_sampling, gen_completely_separated, read_img
import time
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras import backend as K
import math
import random
import threading
import matplotlib.pyplot as pt
from tqdm import tqdm
import seaborn as sns
from sklearn import linear_model
import copy
import tensorflow as tf
from keras.callbacks import TensorBoard


def nextTime(rateParameter):
    return -math.log(1.0 - random.random()) / rateParameter


def myLoss(margin):
    def Loss(y_true, y_pred):
        return (1-y_true)*0.5*(y_pred)^2 + y_true*0.5*K.max(0, margin-y_pred)^2
    return Loss


def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def baseline_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model1 = VGGFace(model='resnet50', include_top=False, name="vggface_resnet50_leg1")
    base_model2 = VGGFace(model='resnet50', include_top=False, name="vggface_resnet50_leg2")

    for x in base_model1.layers[:-3]:
        x.trainable = True

    for x in base_model2.layers[:-3]:
        x.trainable = True

    x1 = base_model1(input_1)
    x2 = base_model2(input_2)

    # x1_ = Reshape(target_shape=(7*7, 2048))(x1)
    # x2_ = Reshape(target_shape=(7*7, 2048))(x2)
    #
    # x_dot = Dot(axes=[2, 2], normalize=True)([x1_, x2_])
    # x_dot = Flatten()(x_dot)

    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x = Multiply()([x1, x2])

    x = Concatenate(axis=-1)([x, x3])

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    # loss = myLoss(0.5)

    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=Adam(1e-5))  # default 1e-5

    # model.summary()

    return model


def detect_outliers2(df):
    outlier_indices = []

    # 1st quartile (25%)
    Q1 = np.percentile(df, 25)
    # 3rd quartile (75%)
    Q3 = np.percentile(df, 75)
    # Interquartile range (IQR)
    IQR = Q3 - Q1

    # outlier step
    outlier_step = 1.5 * IQR
    for nu in df:
        if (nu < Q1 - outlier_step) | (nu > Q3 + outlier_step):
            df.remove(nu)
    return df


def add_task():
    global task_num, request_end_flag# , workload_time, workload_num, lock
    for wt in arriving_proccess:
        lock.acquire()
        task_queue.append(choice(picture_files))
        task_num += 1
        cur_time = time.time()
        try:
            workload_num.append(workload_num[-1])
            workload_time.append(cur_time)
        except:
            pass
        workload_time.append(cur_time)
        workload_num.append(task_num)
        lock.release()
        time.sleep(wt)  # wait until next arrival
        print("plus:", task_num)
    request_end_flag = True



def do_task():
    global task_num, request_end_flag  # , workload_time, workload_num, lock
    while task_queue or not request_end_flag:
        if not task_queue:
            continue
        try:
            lock.acquire()
            pictures_tmp = copy.copy(task_queue)
            task_num -= len(task_queue)
            task_queue[:] = []
            cur_time = time.time()
            try:
                workload_num.append(workload_num[-1])
                workload_time.append(cur_time)
            except:
                pass
            workload_time.append(cur_time)
            workload_num.append(task_num)
        except:
            pass
        finally:
            lock.release()
        ### do tasks with schedule
        start_time = time.time()
        model = baseline_model()
        model.load_weights(file_path)
        end_time = time.time()
        Bjob_times.append(end_time - start_time)

        picture1 = [read_img(x[0]) for x in pictures_tmp]
        picture2 = [read_img(x[1]) for x in pictures_tmp]
        model.predict([picture1, picture2])

        # print("do task", tmp)
        # print("minus:", task_num)


if __name__ == '__main__':
    ########### prepare
    request_end_flag = False
    Bjob_times = []
    basestr = 'splitmodel'
    train_file_path = "./data/train_relationships.csv"
    train_folders_path = "./data/train/"
    val_famillies = "F09"
    all_images = glob(train_folders_path + "*/*/*.jpg")
    train_images = [x for x in all_images if val_famillies not in x]
    val_images = [x for x in all_images if val_famillies in x]
    train_person_to_images_map = defaultdict(list)
    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]
    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)
    val_person_to_images_map = defaultdict(list)
    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)
    relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]
    train = [x for x in relationships if val_famillies not in x[0]]
    val = [x for x in relationships if val_famillies in x[0]]
    file_path = './data' + "/vgg_face_" + basestr + ".h5"
    test_path = "./data/test/"
    submission = pd.read_csv('./data/sample_submission.csv')
    picture_files_tmp = submission.img_pair.values
    X1 = [test_path + x.split("-")[0] for x in picture_files_tmp]
    X2 = [test_path + x.split("-")[1] for x in picture_files_tmp]
    picture_files = list(zip(X1, X2))

    ########### simulation
    area_list = []
    for _ in tqdm(range(1)):
        arriving_proccess = []
        total_arriving_time = 0
        while total_arriving_time < 3600*0 + 60:
            next_time = nextTime(83.333)  # nextTime(lambda)
            arriving_proccess.append(next_time)
            total_arriving_time += next_time
        task_queue = []
        workload_time = []
        workload_num = []
        task_num = 0
        lock = threading.Lock()
        add_task_t = threading.Thread(target=add_task)
        do_task_t = threading.Thread(target=do_task)
        add_task_t.start()
        do_task_t.start()
        add_task_t.join()
        do_task_t.join()
        workload_data = np.array([workload_time, workload_num])
        workload_data.T[np.lexsort(workload_data[::-1, :])].T
        area = 0
        for i in range(len(workload_time)):
            if i == len(workload_time)-1:
                break
            area += workload_data[1, i]*(workload_data[0, i+1] - workload_data[0, i])
        area_list.append(area)
        pt.plot(workload_data[0, :], workload_data[1, :])
    pt.figure()
    sns.distplot(area_list)
    print("finish")

    # ########### plot A(n)
    # model = baseline_model()
    # model.load_weights(file_path)
    # computing_time_tmp = []
    # for batchsize in tqdm(list(range(1, 500, 50))):
    #     predictions = []
    #     time_per_batch = []
    #     for batch in tqdm(chunker(submission.img_pair.values, batchsize)):
    #         time_start = time.time()
    #
    #         X1 = [x.split("-")[0] for x in batch]
    #         X1 = [read_img(test_path + x) for x in X1]
    #         X2 = [x.split("-")[1] for x in batch]
    #         X2 = [read_img(test_path + x) for x in X2]
    #
    #         model.predict([X1, X2])
    #
    #         time_end = time.time()
    #
    #         time_per_batch.append(time_end - time_start)
    #     computing_time_tmp.append(detect_outliers2(time_per_batch))
    #     batch_size_tmp = list(range(1, 500, 50))
    #     computing_time = []
    #     batch_size = []
    #     for id, el in enumerate(computing_time_tmp):
    #         for elel in el:
    #             computing_time.append([elel])
    #             batch_size.append([batch_size_tmp[id]])
    #     pt.plot(batch_size, computing_time, 'r*')
    #     regression_model = linear_model.LinearRegression()
    #     regression_model.fit(batch_size, computing_time)
    #     predictions = regression_model.predict([[x] for x in batch_size_tmp])
    #     pt.plot(batch_size_tmp, predictions.ravel())
    #     print("K: ", regression_model.coef_.ravel()[0])
        # sns.violinplot(data=pd.DataFrame(full_data_predict).T)

