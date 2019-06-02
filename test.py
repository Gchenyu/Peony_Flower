# -*- coding:utf-8 -*-

# @Author : GCY

# @Date  :   2018/10/29 11:14

# @Flie  :  test.py
import tensorflow as tf
import load_Image as lg
import model as m
import numpy as np
import matplotlib.pyplot as plt


def test():
    N_CLASSES = 11
    IMG_SIZE = 200
    BATCH_SIZE = 1
    CAPACITY = 200
    # MAX_STEP = 10
    test_dir = './data/test'
    logs_dir = './log_1'  # 检查点目录

    sess = tf.Session()

    test_list = lg.get_all_image_label(test_dir, is_random=True)
    image_test_batch, label_train_batch = lg.get_batch(test_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)
    test_logits = m.inference(image_test_batch, N_CLASSES)
    print('------------------1---------------------')
    print(test_logits)
    print('-------------------1----------------------------------')
    test_logits = tf.nn.softmax(test_logits)  # 用softmax转化为百分比数值
    print('-----------------2----------------------')
    print(test_logits)
    print('------------------2-----------------------------------')
    # 载入检查点
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(11):
            if coord.should_stop():
                break
            image, prediction = sess.run([image_test_batch, test_logits])
            # 取出a中元素最大值所对应的索引，（索引值默认从0开始）
            max_index = np.argmax(prediction)
            if max_index == 0:
                label = '%.2f%% is a dhh.' % (prediction[0][0] * 100)
            elif max_index == 1:
                label = '%.2f%% is a dj.' % (prediction[0][1] * 100)
            elif max_index == 2:
                label = '%.2f%% is a jph.' % (prediction[0][2] * 100)
            elif max_index == 3:
                label = '%.2f%% is a lhhc.' % (prediction[0][3] * 100)
            elif max_index == 4:
                label = '%.2f%% is a lhzl.' % (prediction[0][4] * 100)
            elif max_index == 5:
                label = '%.2f%% is a mrjl.' % (prediction[0][5] * 100)
            elif max_index == 6:
                label = '%.2f%% is a rfr.' % (prediction[0][6] * 100)
            elif max_index == 7:
                label = '%.2f%% is a wlps.' % (prediction[0][7] * 100)
            elif max_index == 8:
                label = '%.2f%% is a xy.' % (prediction[0][8] * 100)
            elif max_index == 9:
                label = '%.2f%% is a xyth.' % (prediction[0][9] * 100)
            elif max_index == 10:
                label = '%.2f%% is a yh.' % (prediction[0][10] * 100)
            plt.imshow(image[0])
            plt.title(label)
            plt.show()
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()
