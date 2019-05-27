from utils.tf_util import *
from network.resnet_v2 import bottleneck
from tensorflow.contrib.framework.python.ops import arg_scope
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops

FLAGS = tf.app.flags.FLAGS


# def oracle_module(inputs, labels):


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    if weights is not None:
        per_entry_cross_ent *= tf.tile(tf.expand_dims(weights, axis=1), [1, 21])
    return tf.reduce_sum(per_entry_cross_ent)


def simple_decoder(features,
                   decoded_height,
                   decoded_width,
                   number_of_classes,
                   weight_decay=0.0001):
    decode = conv2d_layer(features, number_of_classes, 1, to_batch_norm=False, activation_fn=None,
                          weights_regularizer=slim.l2_regularizer(weight_decay))
    decode = tf.image.resize_bilinear(decode, size=[decoded_height, decoded_width], align_corners=True)

    return decode


def refine_by_decoder(features,
                      raw_features,
                      decoder_height,
                      decoder_width,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False,
                      decoder_use_separable_conv=False):
    """Adds the decoder to obtain sharper segmentation results.

    Args:
      features: A tensor of size [batch, features_height, features_width,
        features_channels].
      raw_features: A tensor of early layer
      decoder_height: The height of decoder feature maps.
      decoder_width: The width of decoder feature maps.
      weight_decay: The weight decay for model variables.
      reuse: Reuse the model variables or not.
      is_training: Is training or not.
      fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
      decoder_use_separable_conv: Use separable convolution for decode.

    Returns:
      Decoder output with size [batch, decoder_height, decoder_width,
        decoder_channels].
    """
    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
    }

    with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding='SAME',
            stride=1,
            reuse=reuse):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            resized_features = tf.image.resize_bilinear(
                    features, [decoder_height, decoder_width], align_corners=True)
            raw_features = slim.conv2d(
                    raw_features,
                    48,
                    1,
                    scope='feature_projection')
            decoder_depth = 256
            if decoder_use_separable_conv:
                decoder_features = split_separable_conv2d(
                    tf.concat([resized_features, raw_features], 3),
                    filters=decoder_depth,
                    rate=1,
                    weight_decay=weight_decay,
                    scope='decoder_conv0')
                decoder_features = split_separable_conv2d(
                    decoder_features,
                    filters=decoder_depth,
                    rate=1,
                    weight_decay=weight_decay,
                    scope='decoder_conv1')
            else:
                decoder_features = slim.conv2d(
                    tf.concat([resized_features, raw_features], 3),
                    decoder_depth,
                    3,
                    scope='decoder_conv_1'
                )
                decoder_features = slim.conv2d(
                    decoder_features,
                    decoder_depth,
                    3,
                    scope='decoder_conv_2'
                )

            return decoder_features


def pyramid_atrous_convolution(inputs, batch_norm_decay, weight_decay, is_training, feature_depth=256,
                               output_depth=256):
    with tf.variable_scope("feature_extraction"):
        with arg_scope([conv2d_layer, multi_conv2d_layer, global_avg_pooling_layer, depthwise_conv2d_layer],
                       to_batch_norm=True, batch_norm_decay=batch_norm_decay, is_training=is_training,
                       activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(weight_decay)):
            conv1x1 = conv2d_layer(inputs, feature_depth, 1)
            mul_conv2d = multi_conv2d_layer(inputs, feature_depth, 3, basis_rate=[1, 3, 5, 7, 9])
            global_avg_pooling = global_avg_pooling_layer(inputs, feature_depth, upsample=True)

            concat = tf.concat([conv1x1, mul_conv2d, global_avg_pooling], axis=3)
            output = conv2d_layer(concat, output_depth, 1)

            return output


def scale_invariant_feature_extraction2(inputs, batch_norm_decay, weight_decay, is_training, feature_depth=256,
                                        output_depth=256):
    with tf.variable_scope("feature_extraction"):
        with arg_scope([conv2d_layer, multi_conv2d_layer, global_avg_pooling_layer, depthwise_conv2d_layer],
                       to_batch_norm=True, batch_norm_decay=batch_norm_decay, is_training=is_training,
                       activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(weight_decay)):
            conv1x1 = conv2d_layer(inputs, feature_depth, 1)
            mul_conv2d = separable_multi_conv2d(inputs, feature_depth, basis_rate=[1, 3, 5, 7, 9])
            global_avg_pooling = global_avg_pooling_layer(inputs, feature_depth, upsample=True)

            concat = tf.concat([conv1x1, mul_conv2d, global_avg_pooling], axis=3)
            output = conv2d_layer(concat, output_depth, 1)

            return output


def slim_decoder(inputs, batch_norm_decay, weight_decay, is_training, feature_depth=256, output_depth=256):
    with tf.variable_scope('slim_decoder'):
        with arg_scope([depthwise_conv2d_layer, conv2d_layer], to_batch_norm=True,
                       batch_norm_decay=batch_norm_decay, is_training=is_training, activation_fn=tf.nn.relu,
                       weights_regularizer=slim.l2_regularizer(weight_decay)):

            net = depthwise_conv2d_layer(inputs, 3)
            net = conv2d_layer(net, feature_depth, 1)
            net = depthwise_conv2d_layer(net, 3)
            net = conv2d_layer(net, output_depth, 1)

            return net


def segmentation_discriminator(inputs, batch_norm_decay, weight_decay, is_training):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        with arg_scope([fc_layer, conv2d_layer, global_avg_pooling_layer], to_batch_norm=True,
                       batch_norm_decay=batch_norm_decay, is_training=is_training, activation_fn=tf.nn.leaky_relu,
                       weights_regularizer=slim.l2_regularizer(weight_decay)):

            net = conv2d_layer(inputs, 32, 5, strides=[1, 2, 2, 1])  # 256 x 256
            net = conv2d_layer(net, 64, 5, strides=[1, 2, 2, 1])  # 128 x 128
            net = conv2d_layer(net, 128, 3, strides=[1, 2, 2, 1])  # 64 x 64
            net = conv2d_layer(net, 256, 3, strides=[1, 2, 2, 1])  # 32 x 32
            net = conv2d_layer(net, 512, 3, strides=[1, 2, 2, 1])  # 16 x 16
            net = global_avg_pooling_layer(net, upsample=False, keepdims=False)
            net = fc_layer(net, 2, to_batch_norm=False, activation_fn=tf.nn.sigmoid)

        return net


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
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
    if output_stride not in [8, 16]:
        raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
        atrous_rates = [2 * rate for rate in atrous_rates]

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': 1e-5,
        'scale': True
    }

    with tf.variable_scope("aspp"):
        with arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                features = list()

                conv_1x1 = slim.conv2d(inputs, depth, 1, scope="conv_1x1")
                features.append(conv_1x1)

                # Atrous convolutions
                for a in atrous_rates:
                    conv_3x3 = slim.conv2d(inputs, depth, 3, rate=a, scope='conv_3x3_' + str(a))
                    features.append(conv_3x3)

                with tf.variable_scope("image_level_features"):
                    image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling',
                                                          keepdims=True)
                    image_level_features = slim.conv2d(image_level_features, depth, 1, stride=1,
                                                       scope='conv_1x1')
                    image_level_features = tf.image.resize_bilinear(image_level_features, tf.shape(inputs)[1:3],
                                                                    name='upsample', align_corners=True)
                    features.append(image_level_features)

                net = tf.concat(features, axis=3, name='concat')
                net = slim.conv2d(net, depth, 1, scope='conv_1x1_concat')

                return net


def gaussian_kernel(size, size_y=None, sigma=1.0):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
    g = np.exp(-(x ** 2 / (float(size) * sigma) + y ** 2 / (float(size_y) * sigma)))
    return g / g.sum()


def gaussianize_image(inputs, filter_size, sigma=3.0):
    gaussian_filter_value = gaussian_kernel(int(filter_size[0]/2), int(filter_size[1]/2), sigma=sigma)
    gaussian_filter_value = np.reshape(gaussian_filter_value, [filter_size[0], filter_size[1], 1, 1])
    gaussian_filter = tf.constant(
        value=gaussian_filter_value,
        dtype=tf.float32,
        name='gaussian_filter'
    )
    gaussian_filter = tf.tile(gaussian_filter, [1, 1, inputs.get_shape().as_list()[3], 1])
    gaussian_image = tf.nn.depthwise_conv2d(inputs, gaussian_filter, strides=[1, 1, 1, 1],
                                            padding='SAME')

    return gaussian_image


def seg_modify_gradient_weight(labels, remain_rate=0.5, edge_multiplier=1.5, image_summary=False):
    labels_remove_bak = labels * tf.to_int32(tf.not_equal(labels, 255))
    labels_flat = tf.reshape(labels_remove_bak, [-1])
    one_hot_labels = slim.one_hot_encoding(labels_flat, FLAGS.num_classes, 1.0, 0.0)
    label_size = tf.shape(labels)
    one_hot_label_images = tf.reshape(one_hot_labels, [-1, label_size[1], label_size[2], FLAGS.num_classes])

    filter_size = np.array([35, 35])  # Must be odd

    edge_check_filter_value = np.array(
        [
            [1, -0.5],
            [-0.5, 0]
        ]
    )
    edge_check_filter = tf.constant(
        value=edge_check_filter_value,
        dtype=tf.float32,
        name='edge_filter'
    )
    edge_check_filter = tf.reshape(edge_check_filter, [2, 2, 1, 1])
    edge_check_filter = tf.tile(edge_check_filter, [1, 1, FLAGS.num_classes, 1])
    padded_label = tf.pad(one_hot_label_images, [[0, 0], [0, 1], [0, 1], [0, 0]], mode='SYMMETRIC')
    padded_label = tf.cast(padded_label, tf.float32)

    edge_image = tf.nn.depthwise_conv2d(padded_label, edge_check_filter, strides=[1, 1, 1, 1], padding='VALID')
    edge_image = tf.cast(tf.not_equal(edge_image, 0), tf.float32)

    compress_filter_value = np.array(
        [
            [1, 1],
            [1, 1]
        ]
    )
    compress_filter = tf.constant(
        value=compress_filter_value,
        dtype=tf.float32,
        name='compress_filter'
    )
    compress_filter = tf.reshape(compress_filter, [2, 2, 1, 1])
    edge_image_not_sum = edge_image
    edge_image = tf.reduce_sum(edge_image, axis=3, keepdims=True)
    compress_image = conv2d(edge_image, compress_filter, strides=[1, 2, 2, 1])
    compress_image = conv2d(compress_image, compress_filter, strides=[1, 2, 2, 1])

    gaussian_edge = gaussianize_image(compress_image, filter_size)
    gaussian_edge = tf.image.resize_bilinear(gaussian_edge, size=label_size[1:3], align_corners=True)
    label_weights = tf.clip_by_value(gaussian_edge * edge_multiplier + remain_rate,
                                     clip_value_min=0.0, clip_value_max=4.0)

    if image_summary:
        tf.summary.image(
            'edge_image',
            tf.reduce_sum(edge_image_not_sum, axis=3, keepdims=True),
            max_outputs=1
        )
        tf.summary.image(
            'label_weights',
            label_weights,
            max_outputs=1
        )

    return label_weights, edge_image_not_sum



