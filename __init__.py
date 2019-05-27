from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from network.seg_resnetV1_pac_bal_simpleDecoder import Segmentator
from utils import preprocessing

import shutil


flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_gpu', 1,
                     'Number of GPUs to use.')

flags.DEFINE_string('base_architecture', 'resnet_v1_101',
                    'The architecture of base Resnet building block.')

flags.DEFINE_string('pre_trained_model',
                    './init_checkpoints/' + FLAGS.base_architecture + '/model.ckpt',
                    'The architecture of base Resnet building block.')

flags.DEFINE_string('model_dir', './model',
                    'Base directory for the model')

flags.DEFINE_string('data_dir', './dataset/',
                    'Path to the directory containing the PASCAL VOC data tf record.')

flags.DEFINE_boolean('clean_model_dir', False,
                     'Whether to clean up the model directory if present.')

flags.DEFINE_integer('train_epochs', 46,
                     'Number of training epochs.')

flags.DEFINE_integer('epochs_per_eval', 1,
                     'The number of training epochs to run between evaluations.')

flags.DEFINE_integer('batch_size', 16,
                     'Size of batch.')

flags.DEFINE_integer('max_iter', 30000,
                     'Number of maximum iteration used for "poly" learning rate policy.')

flags.DEFINE_integer('initial_global_step', 0,
                     'Initial global step for controlling learning rate when fine-tuning model.')

flags.DEFINE_integer('output_stride', 16,
                     'Output stride for DeepLab v3. Currently 8 or 16 is supported.')

flags.DEFINE_float('initial_learning_rate', 0.007,
                   'Initial learning rate for the optimizer.')

flags.DEFINE_float('end_learning_rate', 0,
                   'End learning rate for the optimizer.')

flags.DEFINE_float('power', 0.9,
                   'Parameter for polynomial learning rate policy.')

flags.DEFINE_float('momentum', 0.9,
                   'Parameter for momentum optimizer.')

# 0.00004 for Xception or mobileNetV2, 0.0001 for ResNet
flags.DEFINE_float('weight_decay', 0.0001,
                   'The weight decay to use for regularizing the model.')

flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Whether fine tune parameters of batch normalization.')

flags.DEFINE_float('batch_norm_decay', 0.9997,
                   'Batch normalization decay rate.')

flags.DEFINE_boolean('debug', False,
                     'Whether to use debugger to track down bad values during training.')

flags.DEFINE_integer('num_classes', 21,
                     'Number of classes to predict.')

flags.DEFINE_integer('input_height', 513,
                     'Input images height.')

flags.DEFINE_integer('input_width', 513,
                     'Input images width.')

flags.DEFINE_integer('input_depth', 3,
                     'Input images depth.')

flags.DEFINE_float('min_scale', 0.5,
                   'Minimum scale for multi scale input.')

flags.DEFINE_float('max_scale', 2.0,
                   'Maximum scale for multi scale input.')

flags.DEFINE_integer('ignore_label', 255,
                     'Maximum scale for multi scale input.')

_NUM_IMAGES = {
    'train': 10582,
    'validation': 1449,
}
_PROB_OF_FLIP = 0.5
_MEAN_RGB = [123.15, 115.90, 103.06]


def get_filenames(is_training, data_dir):
    """Return a list of filenames.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: path to the the directory containing the input data.

    Returns:
      A list of file names.
    """
    if is_training:
        return [os.path.join(data_dir, 'voc_train.record')]
    else:
        return [os.path.join(data_dir, 'voc_val.record')]


def parse_record(raw_record):
    """Parse PASCAL image and label from a tf record."""
    keys_to_features = {
        'image/height':
            tf.FixedLenFeature((), tf.int64),
        'image/width':
            tf.FixedLenFeature((), tf.int64),
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'label/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'label/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    # height = tf.cast(parsed['image/height'], tf.int32)
    # width = tf.cast(parsed['image/width'], tf.int32)

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]), FLAGS.input_depth)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    label = tf.image.decode_image(
        tf.reshape(parsed['label/encoded'], shape=[]), 1)
    label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
    label.set_shape([None, None, 1])

    return image, label


def preprocess_image_and_label(image,
                               label,
                               crop_height,
                               crop_width,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0.,
                               ignore_label=255,
                               is_training=True,
                               model_variant=None):
    """Preprocesses the image and label.

    Args:
      image: Input image.
      label: Ground truth annotation label.
      crop_height: The height value used to crop the image and label.
      crop_width: The width value used to crop the image and label.
      min_resize_value: Desired size of the smaller image side.
      max_resize_value: Maximum allowed size of the larger image side.
      resize_factor: Resized dimensions are multiple of factor plus one.
      min_scale_factor: Minimum scale factor value.
      max_scale_factor: Maximum scale factor value.
      scale_factor_step_size: The step size from min scale factor to max scale
        factor. The input is randomly scaled based on the value of
        (min_scale_factor, max_scale_factor, scale_factor_step_size).
      ignore_label: The label value which will be ignored for training and
        evaluation.
      is_training: If the preprocessing is used for training or not.
      model_variant: Model variant (string) for choosing how to mean-subtract the
        images. See feature_extractor.network_map for supported model variants.

    Returns:
      original_image: Original image (could be resized).
      processed_image: Preprocessed image.
      label: Preprocessed ground truth segmentation label.

    Raises:
      ValueError: Ground truth label not provided during training.
    """
    if is_training and label is None:
        raise ValueError('During training, label must be provided.')
    if model_variant is None:
        tf.logging.warning('Default mean-subtraction is performed. Please specify '
                           'a model_variant. See feature_extractor.network_map for '
                           'supported model variants.')

    # Keep reference to original image.
    original_image = image

    processed_image = tf.cast(image, tf.float32)

    if label is not None:
        label = tf.cast(label, tf.int32)

    # Resize image and label to the desired range.
    if min_resize_value is not None or max_resize_value is not None:
        [processed_image, label] = (
            preprocessing.resize_to_range(
                image=processed_image,
                label=label,
                min_size=min_resize_value,
                max_size=max_resize_value,
                factor=resize_factor,
                align_corners=True))
        # The `original_image` becomes the resized image.
        original_image = tf.identity(processed_image)

    # Data augmentation by randomly scaling the inputs.
    if is_training:
        scale = preprocessing.get_random_scale(
            min_scale_factor, max_scale_factor, scale_factor_step_size)
        processed_image, label = preprocessing.randomly_scale_image_and_label(
            processed_image, label, scale)
        processed_image.set_shape([None, None, 3])

    # Pad image and label to have dimensions >= [crop_height, crop_width]
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

    # Pad image with mean pixel value.
    mean_pixel = tf.reshape(_MEAN_RGB, [1, 1, 3])
    processed_image = preprocessing.pad_to_bounding_box(
        processed_image, 0, 0, target_height, target_width, mean_pixel)

    if label is not None:
        label = preprocessing.pad_to_bounding_box(
            label, 0, 0, target_height, target_width, ignore_label)

    # Randomly crop the image and label.
    if is_training and label is not None:
        processed_image, label = preprocessing.random_crop(
            [processed_image, label], crop_height, crop_width)

    processed_image.set_shape([crop_height, crop_width, 3])

    if label is not None:
        label.set_shape([crop_height, crop_width, 1])

    if is_training:
        # Randomly left-right flip the image and label.
        processed_image, label, _ = preprocessing.flip_dim(
            [processed_image, label], _PROB_OF_FLIP, dim=1)

    return processed_image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.

    Returns:
      A tuple of images and labels.
    """
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        # is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

    dataset = dataset.map(parse_record)
    dataset = dataset.map(
        lambda image, label: preprocess_image_and_label(
            image, label, FLAGS.input_height, FLAGS.input_width,
            min_scale_factor=FLAGS.min_scale,
            max_scale_factor=FLAGS.max_scale,
            scale_factor_step_size=0.25,
            is_training=is_training),
        num_parallel_calls=FLAGS.num_gpu
    )

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels


def input_fn_for_eval(data_dir):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
      data_dir: The directory containing the input data.
      scale: Rescale image and label by factor.
      flip: Whether flip image and label or not.

    Returns:
      A tuple of images and labels.
    """
    is_training = False
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    dataset = dataset.map(parse_record)
    dataset = dataset.map(
        lambda image, label: preprocess_image_and_label(
            image, label, FLAGS.input_height, FLAGS.input_width,
            is_training=is_training),
        num_parallel_calls=FLAGS.num_gpu
    )

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.batch(FLAGS.batch_size * 2)
    dataset = dataset.prefetch(FLAGS.batch_size * 2)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels


def main(argv):
    # Set GPU to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Remove tensorflow basic logs 3: remove all
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    if FLAGS.clean_model_dir:
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    print(FLAGS)
    # Set up network
    segmentator = Segmentator(
        params={
            'batch_norm_decay': FLAGS.batch_norm_decay,
            'base_architecture': FLAGS.base_architecture,
            'output_stride': FLAGS.output_stride,
            'pre_trained_model': FLAGS.pre_trained_model,
            'num_classes': FLAGS.num_classes,
            'batch_size': int(FLAGS.batch_size / FLAGS.num_gpu),
            'weight_decay': FLAGS.weight_decay,
            'initial_learning_rate': FLAGS.initial_learning_rate,
            'initial_global_step': FLAGS.initial_global_step,
            'max_iter': FLAGS.max_iter,
            'end_learning_rate': FLAGS.end_learning_rate,
            'power': FLAGS.power,
            'momentum': FLAGS.momentum,
            'fine_tune_batch_norm': FLAGS.fine_tune_batch_norm,
            'ignore_label': FLAGS.ignore_label
        }
    )
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_summary_steps=10,
        log_step_count_steps=10,
        save_checkpoints_secs=10000
    )

    estimator = tf.estimator.Estimator(
        model_fn=segmentator.model_fn if FLAGS.num_gpu == 1 else
        tf.contrib.estimator.replicate_model_fn(segmentator.model_fn),
        config=run_config
    )

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'Summary/learning_rate',
            'cross_entropy': 'Summary/loss',
            'train_px_accuracy': 'Summary/train_px_accuracy',
            'train_mean_iou': 'Summary/train_mean_iou',
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1)
        train_hooks = [logging_hook]
        eval_hooks = None

        tf.logging.info("Start training.")
        estimator.train(
            input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
            hooks=train_hooks,
            # steps=1  # For debug
        )

        tf.logging.info("Start evaluation.")
        eval_results = estimator.evaluate(
            input_fn=lambda: input_fn_for_eval(FLAGS.data_dir),
            hooks=eval_hooks,
            # steps=1  # For debug
        )
        print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
