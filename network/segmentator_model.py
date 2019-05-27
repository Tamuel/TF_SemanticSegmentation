from utils.tf_util import *
from utils.preprocessing import *
from network.resnet_v2 import resnet_v2_50
from network.resnet_v2 import resnet_v2_101
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib import layers as layers_lib


FLAGS = tf.app.flags.FLAGS


class Segmentator:
    def __init__(self, params: dict):
        self.network_name = 'Segmentator_res'
        print('[Load %s]' % self.network_name)

        self.params = params

        self.model = self.network

    def network(self, inputs, is_training):
        params = self.params
        batch_norm_decay = params['batch_norm_decay']

        if params['base_architecture'] == 'resnet_v2_50':
            base_model = resnet_v2_50
        elif params['base_architecture'] == 'resnet_v2_101':
            base_model = resnet_v2_101
        else:
            raise ValueError('Base architecture must be resnet_v2_50 or resnet_v2_101')

        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            logits, end_points = base_model(
                inputs=inputs,
                num_classes=None,
                is_training=is_training,
                global_pool=False,
                output_stride=params['output_stride'],
                multi_grid=(1, 2, 4)
            )

        resnet = end_points

        # Load pretrained variables
        if is_training:
            if params['pre_trained_model'] is not None:
                exclude = [
                    params['base_architecture'] + '/logits',
                    'global_step'
                ]
                variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
                variables_to_restore = [
                    v for v in variables_to_restore if params['base_architecture'] in
                       v.name.split(':')[0] and 'conv2/bias' not in v.name.split(':')[0]
                ]

                tf.train.init_from_checkpoint(
                    ckpt_dir_or_file=params['pre_trained_model'],
                    assignment_map={v.name.split(':')[0]: v for v in variables_to_restore}
                )

        with tf.variable_scope('encoder_weights', reuse=tf.AUTO_REUSE):
            conv6_filter = kernels(shape=[1, 1, 256, params['num_classes']], name='conv6_weights')

        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                sspp = self.atrous_spatial_pyramid_pooling(
                    inputs=resnet[params['base_architecture'] + '/block4'],
                    output_stride=params['output_stride'],
                    batch_norm_decay=params['batch_norm_decay'],
                    is_training=is_training
                )
                conv6 = conv2d(  # Performance improved by remove batch normalization and activation
                    sspp, conv6_filter
                )
                input_size = tf.shape(inputs)
                logits = tf.image.resize_bilinear(conv6, size=input_size[1:3], align_corners=True)

        return logits

    def atrous_spatial_pyramid_pooling(self, inputs, output_stride, batch_norm_decay, is_training, depth=256):
        """Atrous Spatial Pyramid Pooling.

        Args:
          inputs: A tensor of size [batch, height, width, channels].
          output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
            the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
          batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
          is_training: A boolean denoting whether the input is for training.
          depth: The depth of the ResNet unit output.

        Returns:
          The atrous spatial pyramid pooling output.
        """
        with tf.variable_scope("aspp"):
            if output_stride not in [8, 16]:
                raise ValueError('output_stride must be either 8 or 16.')

            atrous_rates = [6, 12, 18]
            if output_stride == 8:
                atrous_rates = [2 * rate for rate in atrous_rates]

            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                with arg_scope([layers.batch_norm], is_training=is_training):
                    inputs_size = tf.shape(inputs)[1:3]
                    # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18)
                    # when output stride = 16.
                    # the rates are doubled when output stride = 8.
                    conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
                    conv_3x3_1 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0],
                                                          scope='conv_3x3_1')
                    conv_3x3_2 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1],
                                                          scope='conv_3x3_2')
                    conv_3x3_3 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2],
                                                          scope='conv_3x3_3')

                    # (b) the image-level features
                    with tf.variable_scope("image_level_features"):
                        # global average pooling
                        image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling',
                                                              keepdims=True)
                        # 1x1 convolution with 256 filters( and batch normalization)
                        image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1,
                                                                 scope='conv_1x1')
                        # bilinearly upsample features
                        image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size,
                                                                        name='upsample')

                    net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3,
                                    name='concat')
                    net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

                    return net

    def model_fn(self, features, labels, mode):
        params = self.params

        if isinstance(features, dict):
            features = features['feature']

        images = tf.cast(
            tf.map_fn(mean_image_addition, features),
            tf.uint8
        )

        logits = self.model(
            inputs=features,
            is_training=(mode == tf.estimator.ModeKeys.TRAIN)
        )

        predict_classes = tf.expand_dims(
            tf.argmax(logits, axis=3, output_type=tf.int32),
            axis=3
        )

        predict_decoded_labels = tf.py_func(
            decode_labels,
            [predict_classes, params['batch_size'], params['num_classes']],
            tf.uint8
        )

        predictions = {
            'classes': predict_classes,
            'probabilities': tf.nn.softmax(logits, name='probabilities'),
            'decoded_labels': predict_decoded_labels
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
            predictions_without_decoded_labels = predictions.copy()
            del predictions_without_decoded_labels['decoded_labels']

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'preds': tf.estimator.export.PredictOutput(
                        predictions_without_decoded_labels)
                })

        gt_decoded_labels = tf.py_func(
            decode_labels,
            [labels, params['batch_size'], params['num_classes']],
            tf.uint8
        )

        labels_flat = tf.reshape(labels, [-1])
        preds_flat = tf.reshape(predict_classes, [-1])

        not_ignore_mask = tf.to_float(tf.not_equal(labels_flat, params['ignore_label']))

        labels_flat = labels_flat * tf.to_int32(not_ignore_mask)

        confusion_matrix = tf.confusion_matrix(
            labels_flat, preds_flat,
            num_classes=params['num_classes'],
            weights=not_ignore_mask
        )

        predictions['valid_preds'] = preds_flat
        predictions['valid_labels'] = labels_flat
        predictions['confusion_matrix'] = confusion_matrix

        if not params['fine_tune_batch_norm']:
            train_var_list = [v for v in tf.trainable_variables()]
        else:
            train_var_list = [v for v in tf.trainable_variables()
                              if 'beta' not in v.name and 'gamma' not in v.name]

        # Loss
        one_hot_labels = slim.one_hot_encoding(labels_flat, params['num_classes'], 1.0, 0.0)
        cross_entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=one_hot_labels,
            logits=tf.reshape(logits, [-1, params['num_classes']]),
            weights=not_ignore_mask
        )

        l2_losses = params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])

        loss = cross_entropy + l2_losses

        accuracy = tf.metrics.accuracy(
            labels_flat, preds_flat,
            weights=not_ignore_mask
        )
        mean_iou = tf.metrics.mean_iou(
            labels_flat, preds_flat, params['num_classes'],
            weights=not_ignore_mask
        )
        metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou}

        if mode == tf.estimator.ModeKeys.TRAIN:

            global_step = tf.train.get_or_create_global_step()

            learning_rate = tf.train.polynomial_decay(
                params['initial_learning_rate'],
                tf.cast(global_step, tf.int32) - params['initial_global_step'],
                params['max_iter'], params['end_learning_rate'],
                power=params['power']
            )

            def compute_mean_iou_per_classes(total_cm, name='mean_iou'):
                """Compute the mean intersection-over-union via the confusion matrix."""
                sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
                sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
                cm_diag = tf.to_float(tf.diag_part(total_cm))
                denominator = sum_over_row + sum_over_col - cm_diag

                # The mean is only computed over classes that appear in the
                # label or prediction tensor. If the denominator is 0, we need to
                # ignore the class.
                num_valid_entries = tf.reduce_sum(tf.cast(
                    tf.not_equal(denominator, 0), dtype=tf.float32))

                # If the value of the denominator is 0, set it to 1 to avoid
                # zero division.
                denominator = tf.where(
                    tf.greater(denominator, 0),
                    denominator,
                    tf.ones_like(denominator))
                iou = tf.div(cm_diag, denominator)

                for i in range(params['num_classes']):
                    tf.identity(iou[i], name='train_iou_class{}'.format(i))
                    tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

                # If the number of valid entries is 0 (no classes) we return 0.
                result = tf.where(
                    tf.greater(num_valid_entries, 0),
                    tf.reduce_sum(iou, name=name) / num_valid_entries,
                    0)
                return result

            train_mean_iou = compute_mean_iou_per_classes(mean_iou[1])

            with tf.name_scope('Summary'):
                tf.summary.image(
                    'Result',
                    tf.concat([images, gt_decoded_labels, predict_decoded_labels], axis=2),
                    max_outputs=1
                )
                # Create a tensor for logging purposes.
                tf.identity(cross_entropy, name='cross_entropy')
                tf.summary.scalar('cross_entropy', cross_entropy)
                tf.identity(learning_rate, name='learning_rate')
                tf.summary.scalar('learning_rate', learning_rate)
                tf.identity(accuracy[1], name='train_px_accuracy')
                tf.summary.scalar('train_px_accuracy', accuracy[1])
                tf.identity(train_mean_iou, name='train_mean_iou')
                tf.summary.scalar('train_mean_iou', train_mean_iou)

            with tf.name_scope('Optimizer'):
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate,
                    momentum=params['momentum']
                )
                optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

                # Batch norm requires update ops to be added as a dependency to the train_op
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(
                        loss=loss,
                        global_step=global_step,
                        var_list=train_var_list
                    )
        else:
            train_op = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics
        )
