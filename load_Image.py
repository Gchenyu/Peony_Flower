# -*- coding:utf-8 -*-

# @Author : GCY

# @Date  :   2018/10/27 15:09

# @Flie  :  load_Image.py

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 去除系统返回的gpu信息


# 获取图片路径及其标签
def get_all_image_label(file_path, is_random=True):
    image_list = []
    label_list = []
    feature_list = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ]

    dhh_count, dj_count, jph_count, lhhc_count, lhzl_count, \
    mrjl_count, rfr_count, wlps_count, xy_count, xyth_count, \
    yh_count = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for item in os.listdir(file_path):
        item_path = file_path + '\\' + item
        item_label = item.split('_')[0]  # 文件名形如  dhh_0.jpg,只需要取第一个

        if os.path.isfile(item_path):
            image_list.append(item_path)
        else:
            raise ValueError('路径下有不可读取项.')
        # 标记
        if item_label == 'dhh':
            label_list.append(feature_list[0])
            dhh_count += 1
        elif item_label == 'dj':
            label_list.append(feature_list[1])
            dj_count += 1
        elif item_label == 'jph':
            label_list.append(feature_list[2])
            jph_count += 1
        elif item_label == 'lhhc':
            label_list.append(feature_list[3])
            lhhc_count += 1
        elif item_label == 'lhzl':
            label_list.append(feature_list[4])
            lhzl_count += 1
        elif item_label == 'mrjl':
            label_list.append(feature_list[5])
            mrjl_count += 1
        elif item_label == 'rfr':
            label_list.append(feature_list[6])
            rfr_count += 1
        elif item_label == 'wlps':
            label_list.append(feature_list[7])
            wlps_count += 1
        elif item_label == 'xy':
            label_list.append(feature_list[8])
            xy_count += 1
        elif item_label == 'xyth':
            label_list.append(feature_list[9])
            xyth_count += 1
        else:
            label_list.append(feature_list[10])
            yh_count += 1

    print('数据集中有{}株dhh,{}株dj,{}株jgh,{}株lhhc,{}株lhzl,{}株mrjl,{}株rfr,{}株wlps,{}株xy,{}株xyth,{}株yh.'
          .format(dhh_count, dj_count, jph_count, lhhc_count, lhzl_count, mrjl_count, rfr_count, wlps_count, xy_count,
                  xyth_count, yh_count))

    label_list = np.asarray(label_list)
    image_list = np.asarray(image_list)  # 将列表转换为数组

    # 图像乱序
    if is_random:
        rnd_index = np.arange(len(image_list))  # 为图像排号
        np.random.shuffle(rnd_index)
        image_list = image_list[rnd_index]
        label_list = label_list[rnd_index]

    return image_list, label_list


# 获取训练批次
def get_batch(train_list, image_size, batch_size, capacity, is_random=True):
    # slice_input_producer函数用来每次产生一个切片。shuffle = False则不打乱（在get_all_files中已经打乱）
    # 每次从一个tensor列表(train_list)中按顺序或者随机抽取出一个tensor放入文件名队列，即抽一组[image_list, label_list]。
    # 表中tensor的第一维度的值必须相等，即个数必须相等，有多少个图像，就应该有多少个对应的标签。
    intput_queue = tf.train.slice_input_producer(train_list, shuffle=False)

    # 从路径中读取图片
    image_train = tf.read_file(intput_queue[0])
    image_train = tf.image.decode_jpeg(image_train, channels=3)  # 这里是jpg格式，3通道
    image_train = tf.image.resize_images(image_train, [image_size, image_size])  # 统一图片尺寸
    image_train = tf.cast(image_train, tf.float32) / 255.  # 转换数据类型并归一化

    # 图片标签
    label_train = intput_queue[1]

    # tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)按顺序组合成一个batch
    # [example, label]表示样本和样本标签，这个可以是一个样本和一个样本标签
    # batch_size是返回的一个batch样本集的样本个数
    # capacity是队列中的容量

    # tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity, min_after_dequeue)
    # 是一个乱序的样本排列的batch
    # 一定要保证min_after_dequeue小于capacity参数的值，否则会出错
    if is_random:
        image_train_batch, label_train_batch = tf.train.shuffle_batch([image_train, label_train], batch_size=batch_size,
                                                                      capacity=capacity, min_after_dequeue=100,
                                                                      num_threads=2)
    else:
        image_train_batch, label_train_batch = tf.train.batch([image_train, label_train],
                                                              batch_size=11,
                                                              capacity=capacity,
                                                              num_threads=1)
    return image_train_batch, label_train_batch


'''
tf中的数据读取机制:
1.调用 tf.train.slice_input_producer，从 本地文件里抽取tensor，准备放入Filename Queue（文件名队列）中;
2.调用 tf.train.batch，从文件名队列中提取tensor，使用单个或多个线程，准备放入文件队列;
3.调用 tf.train.Coordinator() 来创建一个线程协调器，用来管理之后在Session中启动的所有线程;
4.调用tf.train.start_queue_runners, 启动入队线程，由多个或单个线程，按照设定规则，把文件读入Filename Queue中。
  函数返回线程ID的列表，一般情况下，系统有多少个核，就会启动多少个入队线程（入队具体使用多少个线程在tf.train.batch中定义）;
5.文件从 Filename Queue中读入内存队列的操作不用手动执行，由tf自动完成;
6.调用sess.run 来启动数据出列和执行计算;
7.使用 coord.should_stop()来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，
  会抛出一个 OutofRangeError 的异常，这时候就应该停止Sesson中的所有线程了;
8.使用coord.request_stop()来发出终止所有线程的命令，使用coord.join(threads)把线程加入主线程，等待threads结束
'''
# 测试图片读取
if __name__ == '__main__':

    image_dir = 'data\\test'
    train_list = get_all_image_label(image_dir, True)
    image_train_batch, label_train_batch = get_batch(train_list, 200, 11, 200, False)
    # print('-------------------------------------------------------')
    # print(image_train_batch)
    # print('---------------------------------------')
    # print(label_train_batch)
    # print('-----------------------------------------------------')
    sess = tf.Session()
    # Coordinator类用来管理在Session中的多个线程，可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常，
    # 该线程捕获到这个异常之后就会终止所有线程
    coord = tf.train.Coordinator()  # 协调器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 入队线程启动器

    try:
        for step in range(11):
            if coord.should_stop():
                break

            image_batch, label_batch = sess.run([image_train_batch, label_train_batch])
            print("批次中图像:{}  标签：{}".format(image_batch, label_batch))
            print("------------------------------------------")
            for i in range(11):
                # (numpy-array).any()数组进行比较时，True表示不同，False表示相同
                if not ((label_batch[i] - [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).any()):
                    label = 'dhh'
                elif not ((label_batch[i] - [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).any()):
                    label = 'dj'
                elif not ((label_batch[i] - [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).any()):
                    label = 'jph'
                elif not ((label_batch[i] - [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).any()):
                    label = 'lhhc'
                elif not ((label_batch[i] - [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).any()):
                    label = 'lhzl'
                elif not ((label_batch[i] - [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).any()):
                    label = 'mrjl'
                elif not ((label_batch[i] - [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).any()):
                    label = 'rfr'
                elif not ((label_batch[i] - [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).any()):
                    label = 'wlps'
                elif not ((label_batch[i] - [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).any()):
                    label = 'xy'
                elif not ((label_batch[i] - [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).any()):
                    label = 'xyth'
                elif not ((label_batch[i] - [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).any()):
                    label = 'yh'
                else:
                    print("无效标签")
                plt.imshow(image_batch[i]), plt.title(label)
                plt.show()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()
