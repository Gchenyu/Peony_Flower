# -*- coding:utf-8 -*-

# @Author : GCY

# @Date  :   2018/10/29 11:44

# @Flie  :  train.py
import time
import load_Image as lg
import model as m
import tensorflow as tf
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SUMMARY_DIR = "./log_1/summary"


# 训练模型
def training():
    N_CLASSES = 11
    IMG_SIZE = 200
    BATCH_SIZE = 10
    CAPACITY = 200
    MAX_STEP = 15000
    LEARNING_RATE = 1e-4

    # 测试图片读取
    image_dir = './data/train'
    logs_dir = './log_1'  # 检查点保存路径

    sess = tf.Session()

    train_list = lg.get_all_image_label(image_dir, True)
    image_train_batch, label_train_batch = lg.get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)

    # CNN
    train_logits = m.inference(image_train_batch, N_CLASSES)
    train_loss = m.losses(train_logits, label_train_batch)
    tf.summary.scalar('classify/loss', train_loss)

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(train_loss)

    var_list = tf.trainable_variables()
    paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_list])
    print('参数数目:%d' % sess.run(paras_count), end='\n\n')

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    s_t = time.time()
    try:

        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            summary_op = tf.summary.merge_all()
            summary, _, loss = sess.run([summary_op, train_op, train_loss])
            if (step + 1) % 100 == 0:  # 实时记录训练过程并显示
                runtime = time.time() - s_t
                print('Step: %6d, loss: %.8f, time:%.2fs, time left: %.2fhours'
                      % (step, loss, runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()
            if step % 1000 == 0 or step == MAX_STEP - 1:  # 保存检查点
                checkpoint_path = os.path.join(logs_dir, 'calssify_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
            summary_writer.add_summary(summary, step)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()
