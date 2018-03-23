import os
import pickle
from datetime import datetime
from random import shuffle

import numpy as np
import tensorflow as tf

from data_utils import get_points_and_class


class Model:
    def __init__(self, args):
        self.args = args
        self.data = pickle.load(open(os.path.join(args.data_dir, 'data.pickle'), "rb" ))
        self.train_list = self.data['train_list']
        self.eval_list = self.data['eval_list']
        self.test_list = self.data['test_list']
        self.class_dict = self.data['class_dict']
        self.build_point_net()

    def build_point_net(self):
        n_dims = 3
        xavier_init = tf.contrib.layers.xavier_initializer_conv2d()
        self.X = tf.placeholder(tf.float32, shape=(None, 1024, n_dims, 1), name='X')
        self.y = tf.placeholder(tf.int32, shape=(None))

        with tf.name_scope('point_net'):
            # Implement T-net here
            self.net = tf.layers.conv2d(inputs=self.X, filters=64, kernel_size=(1,3), padding='valid',
                                        activation=tf.nn.relu, kernel_initializer=xavier_init)
            self.net = tf.layers.conv2d(inputs=self.net, filters=64, kernel_size=(1,1), padding='valid',
                                        activation=tf.nn.relu, kernel_initializer=xavier_init)
            # Implement second T-net here
            self.net = tf.layers.conv2d(inputs=self.net, filters=64, kernel_size=(1,1), padding='valid',
                                        activation=tf.nn.relu, kernel_initializer=xavier_init)
            self.net = tf.layers.conv2d(inputs=self.net, filters=128, kernel_size=(1, 1), padding='valid',
                                        activation=tf.nn.relu, kernel_initializer=xavier_init)
            self.net = tf.layers.conv2d(inputs=self.net, filters=1024, kernel_size=(1, 1), padding='valid',
                                        activation=tf.nn.relu, kernel_initializer=xavier_init)
            self.net = tf.layers.max_pooling2d(self.net, pool_size=[self.args.n_points, 1],
                                               strides=(2,2), padding='valid')
            self.net = tf.layers.dense(self.net, 512, activation=tf.nn.relu,
                                       kernel_initializer=xavier_init)
            self.net = tf.layers.dense(self.net, 256, activation=tf.nn.relu,
                                       kernel_initializer=xavier_init)
            self.net = tf.nn.dropout(self.net, keep_prob=self.args.keep_prob)
            self.logits = tf.layers.dense(self.net, 10, activation=None,
                                       kernel_initializer=xavier_init)

        with tf.name_scope('loss'):
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            self.loss = tf.reduce_mean(self.xentropy)

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
            self.training_op = self.optimizer.minimize(self.loss)

        with tf.name_scope("eval"):
            self.correct = tf.nn.in_top_k(tf.squeeze(self.logits), self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

    def train(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.args.load_checkpoint:
            self.load()

        print('[*] Initializing training.')

        n_epochs = self.args.n_epochs
        batch_size = self.args.batch_size
        best_loss = np.infty
        max_checks_without_progress = self.args.early_stopping_max_checks
        checks_without_progress = 0

        for epoch in range(n_epochs):
            shuffle(self.train_list)
            for iteration in range(len(self.train_list) // batch_size):
                average_loss = list()
                iter_indices_begin = iteration * batch_size
                iter_indices_end = (iteration + 1) * batch_size
                X_batch, y_batch = get_points_and_class(self.train_list[iter_indices_begin:iter_indices_end],
                                                        self.class_dict, self.args.n_points,
                                                        rotate=self.args.augment_training)
                self.sess.run(self.training_op, feed_dict={self.X: X_batch[:,:,:,np.newaxis],
                                                      self.y: y_batch})
                iter_loss = self.sess.run(self.loss, feed_dict={self.X: X_batch[:,:,:,np.newaxis],
                                                  self.y: y_batch})
                average_loss.append(iter_loss)
            average_loss = sum(average_loss) / len(average_loss)
            if average_loss < best_loss:
                best_loss = average_loss
                checks_without_progress = 0
            else:
                checks_without_progress += 1
                if checks_without_progress > max_checks_without_progress:
                    print("Early stopping!")
                    batch_accuracy = self.sess.run(self.accuracy, feed_dict={self.X: X_batch[:, :, :, np.newaxis],
                                                                             self.y: y_batch})
                    print('Epoch: %d\tAverage Loss: %.3f\tBatch accuracy: %.3f' % (epoch, average_loss, batch_accuracy))
                    self.save(epoch)
                    break
            if epoch % 50 == 0:
                average_accuracy = list()
                for iteration in range(len(self.eval_list) // batch_size):

                    iter_indices_begin = iteration * batch_size
                    iter_indices_end = (iteration + 1) * batch_size
                    X_batch, y_batch = get_points_and_class(self.eval_list[iter_indices_begin:iter_indices_end],
                                                            self.class_dict, self.args.n_points)
                    eval_accuracy = self.sess.run(self.accuracy, feed_dict={self.X: X_batch[:,:,:,np.newaxis],
                                                                            self.y: y_batch})
                    average_accuracy.append(eval_accuracy)
                average_accuracy = sum(average_accuracy) / len(average_accuracy)
                print('Epoch: %d\tAverage Loss: %.3f\tBatch accuracy: %.3f' % (epoch, average_loss, average_accuracy))
                self.save(epoch)

    def test(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        batch_size = 20
        average_acc = list()
        if self.args.model_name is None:
            print('Saved model needs to be loaded for testing.')
        else:
            self.load()
            for iteration in range(len(self.test_list) // batch_size):
                iter_indices_begin = iteration * batch_size
                iter_indices_end = (iteration + 1) * batch_size
                X_batch, y_batch = get_points_and_class(self.test_list[iter_indices_begin:iter_indices_end],
                                                        self.class_dict, self.args.n_points)
                iter_accuracy = self.sess.run(self.accuracy, feed_dict={self.X: X_batch[:, :, :, np.newaxis],
                                                                        self.y: y_batch})
                average_acc.append(iter_accuracy)
            average_acc = sum(average_acc) / len(average_acc)
            print('Test accuracy: %.3f' % average_acc)

    def save(self, epoch):
        print('[*] Saving checkpoint ....')
        model_name = 'model_{}_epoch_{}.ckpt'.format(datetime.now().strftime("%d:%H:%M:%S"), epoch)
        self.saver = tf.train.Saver()
        save_path = self.saver.save(self.sess, os.path.join(self.args.saved_model_directory, model_name))
        print('[*] Checkpoint saved in file {}'.format(save_path))

    def load(self):
        print("[*] Loading checkpoint...")
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.args.saved_model_directory, self.args.model_name))