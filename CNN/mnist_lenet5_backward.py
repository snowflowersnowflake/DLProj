
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import os
import numpy as np
# 定义训练过程中的超参数
BATCH_SIZE = 100 # 一个 batch 的数量
LEARNING_RATE_BASE = 0.005 # 初始学习率
LEARNING_RATE_DECAY = 0.99 # 学习率的衰减率
REGULARIZER = 0.0001 # 正则化项的权重
STEPS = 50000 # 最大迭代次数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均的衰减率
MODEL_SAVE_PATH="./model/" # 保存模型的路径
MODEL_NAME="mnist_model" # 模型命名
# 训练过程·
def backward(mnist):
    # x, y_是定义的占位符，需要指定参数的类型，维度（要和网络的输入与输出维度一致），类似于函数的形参，运行时必须传入值
    x = tf.placeholder(tf.float32,[BATCH_SIZE,
                                   mnist_lenet5_forward.IMAGE_SIZE,
                                   mnist_lenet5_forward.IMAGE_SIZE,
                                   mnist_lenet5_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
    y = mnist_lenet5_forward.forward(x,True, REGULARIZER) # 调用前向传播网络得到维度为10 的 tensor
    global_step = tf.Variable(0, trainable=False) # 声明一个全局计数器，并输出化为 0
    # 先是对网络最后一层的输出 y 做 softmax，通常是求取输出属于某一类的概率，其实就是一个num_classes 大小的向量，再将此向量和实际标签值做交叉熵，需要说明的是该函数返回的是一个向量
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce) # 再对得到的向量求均值就得到 loss
    loss = cem + tf.add_n(tf.get_collection('losses')) # 添加正则化中的 losses
    # 实现指数级的减小学习率，可以让模型在训练的前期快速接近较优解，又可以保证模型在训练后期不会有太大波动
    # 计算公式：decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY,
                                               staircase=True) # 当 staircase=True 时，（global_step/decay_steps）则被转化为整数，以此来选择不同的衰减方式

    # 传入学习率，构造一个实现梯度下降算法的优化器，再通过使用 minimize 更新存储要训练的变量的列表来减小 loss
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # 实现滑动平均模型，参数 MOVING_AVERAGE_DECAY 用于控制模型更新的速度。训练过程中会对每一个变量维护一个影子变量，这个影子变量的初始值
    # 就是相应变量的初始值，每次变量更新时，影子变量就会随之更新
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]): # 将 train_step 和 ema_op 两个训练操作绑定到 train_op 上
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver() # 实例化一个保存和恢复变量的 saver
    with tf.Session() as sess: # 创建一个会话，并通过 python 中的上下文管理器来管理这个会话
        init_op = tf.global_variables_initializer() # 初始化计算图中的变量
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) # 通过 checkpoint 文件定位到最新保存的模型
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) # 加载最新的模型

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE) # 读取一个 batch 的数据
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_lenet5_forward.IMAGE_SIZE,
                                          mnist_lenet5_forward.IMAGE_SIZE,
                                          mnist_lenet5_forward.NUM_CHANNELS))  # 将输入数据 xs 转换成与网络输入相同形状的矩阵
            # 喂入训练图像和标签，开始训练
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs,y_: ys})
            if i % 100 == 0: # 每迭代 100 次打印 loss 信息，并保存最新的模型
                print("After %d training step(s), loss on training batch is %g." % (step,loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True) # 读入 mnist 数据
    backward(mnist)

if __name__ == '__main__':
    main()


