import numpy as np
import tensorflow as tf
import math


class Network:
    def __init__(
            self,
            name, model, inf, optimizer, data_provider,
            feed_dict_train, feed_dict_test, feed_dict_inf,
            saver=None, inference=False
    ):
        self.network_name = name  # Network name
        self.model = model  # Main model of network
        self.inf = inf  # Desirous output of network
        self.data_provider = data_provider  # Data provider of network
        self.log_dir = './log_seg/logs_' + self.network_name
        self.weight_dir = './log_seg/weight_' + self.network_name

        # Tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Input, parameters of network
        self.feed_dict_train = feed_dict_train
        self.feed_dict_test = feed_dict_test
        self.feed_dict_inf = feed_dict_inf
        self.optimizer = optimizer

        if not inference:
            self.summary_merged = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if saver is None:
            self.saver = tf.train.Saver()
        else:
            self.saver = saver

        self.ckpt = tf.train.get_checkpoint_state(self.weight_dir)
        if self.ckpt and tf.train.checkpoint_exists(self.ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
            print('[%s weights restored]' % self.network_name)
        else:
            print('Initialize variables')
            self.sess.run(tf.global_variables_initializer())

    def learning(self, epoch, batch_size=None):
        if self.data_provider is not None:
            self.data_provider.load_data()
            self.data_provider.set_batch_size(batch_size)
        else:
            print('Need to provide data provider!')
            exit(1)

        print('[%s] Start learning' % self.network_name)

        for e in range(epoch):
            for batch in range(self.data_provider.n_batch):
                print('Epoch:%d, Batch:%d' % (e + 1, batch + 1))
                data, label = self.data_provider.next_batch(batch_size)
                self.optimizing(data, label, e * self.data_provider.n_batch + batch, epoch * self.data_provider.n_batch)
                self.write_summary(data, label)
                if (batch + 1) % 100 == 0:
                    self.save_network()

    def inference(self, data):
        inf_keys = list(self.feed_dict_inf.keys())
        self.feed_dict_inf[inf_keys[0]] = data
        prediction = self.sess.run(
            self.inf,
            feed_dict=self.feed_dict_inf
        )
        prediction = np.reshape(
            a=np.array(prediction),
            newshape=[-1, 96, 624, 1]
        )
        return prediction

    def optimizing(self, data, label, iter, whole_iter):
        train_keys = list(self.feed_dict_train.keys())
        self.feed_dict_train[train_keys[0]] = data
        self.feed_dict_train[train_keys[1]] = label
        self.sess.run(
            self.optimizer,
            feed_dict=self.feed_dict_train
        )

    def write_summary(self, data, label):
        test_keys = list(self.feed_dict_test.keys())
        self.feed_dict_test[test_keys[0]] = data
        self.feed_dict_test[test_keys[1]] = label

        summary = self.sess.run(
            self.summary_merged,
            feed_dict=self.feed_dict_test
        )
        self.summary_writer.add_summary(
            summary,
            global_step=self.sess.run(self.global_step)
        )

    def save_network(self):
        self.saver.save(
            self.sess,
            self.weight_dir + '/dnn.ckpt',
            global_step=self.sess.run(self.global_step)
        )
        print('[%s saved]' % self.network_name)
