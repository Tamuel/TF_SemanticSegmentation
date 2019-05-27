from network.network import Network
from utils.tf_util import *
from utils.base_util import *
from network.resnet_v2 import *

debug = False
base_architecture = 'resnet_v2_50'


def resnet_network(input, is_training, output_stride=None):
    if base_architecture == 'resnet_v2_101':
        logits, end_points = resnet_v2_101(
            inputs=input,
            num_classes=None,
            is_training=is_training,
            global_pool=False,
            output_stride=output_stride
        )
    else:
        logits, end_points = resnet_v2_50(
            inputs=input,
            num_classes=None,
            is_training=is_training,
            global_pool=False,
            output_stride=output_stride
        )
    # if is_training:
    #     print('[Load resnet pretrained parameters]')
    #     exclude = [
    #         base_architecture + '/logits', 'global_step'
    #     ]
    #     variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
    #     assign_map = {
    #         v.name.split(':')[0]: v for v in variables_to_restore
    #         if 'encoder_weights' not in v.name.split(':')[0] and
    #            'decoder_weights' not in v.name.split(':')[0]
    #     }
    #     tf.train.init_from_checkpoint(
    #         ckpt_dir_or_file=base_architecture + '_model/' + base_architecture + '.ckpt',
    #         assignment_map=assign_map
    #     )
    #     print('[Finished]')

    net = end_points

    return net


class Segmentator(Network):
    def __init__(self, num_classes, input_size, data_provider=None, inference=False):
        """
        Segmentator for life boat segmentation
        :param input_size: Input image size
        :param data_provider: Data provider for training. It can be None in inference stage.
        :param inference: Indicate inference or training
        """
        weight_decay = 5e-4
        learning_rate = 0.000001
        stddev = 0.002
        self.network_name = 'Segmentator'
        print('[Load %s]' % self.network_name)

        self.input_size = input_size

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.data = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.input_size[0], self.input_size[1], 3],
            name='Data'
        )
        self.label = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.input_size[0], self.input_size[1], num_classes],
            name='Label'
        )
        self.is_training = tf.placeholder(
            dtype=tf.bool,
            name='Is_Train'
        )
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=self.global_step,
            decay_steps=50000,
            decay_rate=0.96
        )
        self.drop_out_keep_prob = tf.placeholder(
            dtype=tf.float32,
            name='Keep_Prob'
        )

        with tf.name_scope('encoder_weights'):
            conv6_filter = kernels(shape=[3, 3, 2048, num_classes], stddev=stddev, name='conv6_weights')

        with tf.name_scope('network'):
            with tf.name_scope('encoder'):
                resnet = resnet_network(
                    input=self.data,
                    is_training=not inference
                )
                conv6 = scale_conv2d(
                    resnet[base_architecture + '/block4'], conv6_filter,
                    initial_step=3, number_of_step=5
                )
                conv6 = batch_norm(conv6, self.is_training)
                conv6 = tf.nn.relu(conv6)
                output_size = tf.shape(self.data)
                resize_tensor = tf.image.resize_bilinear(conv6, size=output_size[1:3])

            with tf.name_scope('Model'):
                self.model = resize_tensor

        train_var_list = [v for v in tf.trainable_variables()]

        with tf.name_scope('Loss'):
            resize_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=resize_tensor,
                    labels=tf.argmax(self.label, axis=3),
                    name="entropy"
                )
            )
            fine_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.model,
                    labels=tf.argmax(self.label, axis=3),
                    name="entropy"
                )
            )
            alpha = 0.1  # alpha * fine_loss +
            self.loss = (1 - alpha) * resize_loss + alpha * fine_loss
            # weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])

        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            ).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope('Summary'):
            # For voc 2012 color map
            cmap = list(voc_color_map())
            color_plane = list()
            for i in range(21):
                color_plane.append(
                    tf.tile(
                        tf.constant(
                            value=cmap[i],
                            shape=[1, 1, 3],
                            dtype=tf.float32
                        ),
                        [self.input_size[0], self.input_size[1], 1]
                    )
                )
            mask = list()
            for i in range(21):
                mask.append(tf.constant(i, dtype=tf.float32, shape=[self.input_size[0], self.input_size[1], 3]))

            def tf_voc_label_to_color(label):
                image = tf.cast(tf.tile(label, [1, 1, 3]), tf.float32)
                result = tf.zeros(shape=[self.input_size[0], self.input_size[1], 3])
                for c in range(21):
                    m = tf.cast(tf.equal(mask[c], image), tf.float32)
                    result += m * color_plane[c]

                result = tf.reshape(result, [1, self.input_size[0], self.input_size[1], 3])
                result = tf.cast(result, tf.uint8)
                return result

            self.prediction = tf.argmax(resize_tensor, axis=3)
            self.prediction = tf.reshape(self.prediction, [-1, input_size[0], input_size[1], 1])
            self.prediction2 = tf.argmax(self.model, axis=3)
            self.prediction2 = tf.reshape(self.prediction2, [-1, input_size[0], input_size[1], 1])
            tf.summary.image(
                'Prediction',
                tf_voc_label_to_color(self.prediction[0])
            )
            tf.summary.image(
                'Prediction2',
                tf_voc_label_to_color(self.prediction2[0])
            )
            tf.summary.scalar('Loss', resize_loss + fine_loss)
            tf.summary.scalar('Resize Loss', resize_loss)
            tf.summary.scalar('Fine Loss', fine_loss)
            tf.summary.image(
                'Label',
                tf_voc_label_to_color(tf.reshape(tf.argmax(self.label[0], axis=2), [input_size[0], input_size[1], 1]))
            )
            tf.summary.image('Input', self.data, max_outputs=1)

        self.feed_dict_train = {
            self.data: None,
            self.label: None,
            self.is_training: True,
            self.drop_out_keep_prob: 1.0
        }
        self.feed_dict_test = {
            self.data: None,
            self.label: None,
            self.is_training: False,
            self.drop_out_keep_prob: 1.0
        }
        self.feed_dict_inf = {
            self.data: None,
            self.is_training: False,
            self.drop_out_keep_prob: 1.0
        }
        self.feed_dict_list = {
            'data': self.data,
            'label': self.label,
            'learning_rate': self.learning_rate,
            'is_training': self.is_training,
            'keep_prob': self.drop_out_keep_prob
        }
        super().__init__(
            name=self.network_name,
            model=[self.model],
            inf=[self.prediction],
            optimizer=self.optimizer,
            data_provider=data_provider,
            feed_dict_train=self.feed_dict_train,
            feed_dict_test=self.feed_dict_test,
            feed_dict_inf=self.feed_dict_inf,
            inference=inference
        )
