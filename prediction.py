import os
from utils.preprocessing import *
from network.seg_resnetV1_pac_bal_simpleDecoder import Segmentator
import numpy as np
from PIL import Image
from utils.base_util import Timer

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_gpu', 1,
                     'Number of GPUs to use.')

model_dir = './test_model'
input_dir = './test_input'
output_dir = './test_output'

sess = tf.Session()
input_image = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])

segmentation = Segmentator(
    params={
        'base_architecture': 'resnet_v2_101',
        'batch_size': 1,
        'fine_tune_batch_norm': False,
        'num_classes': 21,
        'weight_decay': 0.0001,
        'output_stride': 16,
        'batch_norm_decay': 0.9997
    }
)

logits = segmentation.network(inputs=input_image, is_training=False)

predict_classes = tf.expand_dims(
    tf.argmax(logits, axis=3, output_type=tf.int32),
    axis=3
)

variables_to_restore = tf.contrib.slim.get_variables_to_restore()
get_ckpt = tf.train.init_from_checkpoint(
    ckpt_dir_or_file=model_dir,
    assignment_map={v.name.split(':')[0]: v for v in variables_to_restore}
)

sess.run(tf.global_variables_initializer())

print('[Read images]')
file_list = os.listdir(input_dir)
images = list()
for i, f in enumerate(file_list):
    if i == 30:
        break
    f = os.path.join(input_dir, f)
    img = np.array(Image.open(f)).astype(np.float32)
    images.append(img)

num_images = len(images)
print('[Done]')

print('[Do segment]')
timer = Timer(as_progress_notifier=False)


def print_fn():
    print('Elapsed: %f' % timer.elapsed_time)


timer.print_fn = print_fn

for idx, i in enumerate(images):
    image_data = i
    timer.start()
    predictions = sess.run(
        predict_classes,
        feed_dict={
            input_image: np.expand_dims(image_data, 0)
        }
    )
    timer.check()
    decoded = decode_labels(np.array(predictions).astype(np.uint8))
    timer.check()
    segment_image = decoded[0]
    Image.fromarray(segment_image.astype(np.uint8)).save(os.path.join(output_dir, file_list[idx]))

