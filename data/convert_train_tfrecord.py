from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import glob
import random
import tensorflow as tf


flags.DEFINE_string('hr_dataset_path', './data/DIV2K/DIV2K800_sub',
                    'path to high resolution dataset')
flags.DEFINE_string('lr_dataset_path', './data/DIV2K/DIV2K800_sub_bicLRx4',
                    'path to low resolution dataset')
flags.DEFINE_string('output_path', './data/DIV2K800_sub_bin.tfrecord',
                    'path to ouput tfrecord')
flags.DEFINE_boolean('is_binary', True, 'whether save images as binary files'
                     ' or load them on the fly.')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example_bin(img_name, hr_img_str, lr_img_str):
    # Create a dictionary with features that may be relevant (binary).
    feature = {'image/img_name': _bytes_feature(img_name),
               'image/hr_encoded': _bytes_feature(hr_img_str),
               'image/lr_encoded': _bytes_feature(lr_img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def make_example(img_name, hr_img_path, lr_img_path):
    # Create a dictionary with features that may be relevant.
    feature = {'image/img_name': _bytes_feature(img_name),
               'image/hr_img_path': _bytes_feature(hr_img_path),
               'image/lr_img_path': _bytes_feature(lr_img_path)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(_):
    hr_dataset_path = FLAGS.hr_dataset_path
    lr_dataset_path = FLAGS.lr_dataset_path

    if not os.path.isdir(hr_dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(hr_dataset_path))

    samples = []
    logging.info('Reading data list...')
    for hr_img_path in glob.glob(os.path.join(hr_dataset_path, '*.png')):
        img_name = os.path.basename(hr_img_path).replace('.png', '')
        lr_img_path = os.path.join(lr_dataset_path, img_name + '.png')
        samples.append((img_name, hr_img_path, lr_img_path))
    random.shuffle(samples)

    if os.path.exists(FLAGS.output_path):
        logging.info('{:s} already exists. Exit...'.format(
            FLAGS.output_path))
        exit(1)

    logging.info('Writing {} sample to tfrecord file...'.format(len(samples)))
    with tf.io.TFRecordWriter(FLAGS.output_path) as writer:
        for img_name, hr_img_path, lr_img_path in tqdm.tqdm(samples):
            if FLAGS.is_binary:
                hr_img_str = open(hr_img_path, 'rb').read()
                lr_img_str = open(lr_img_path, 'rb').read()
                tf_example = make_example_bin(img_name=str.encode(img_name),
                                              hr_img_str=hr_img_str,
                                              lr_img_str=lr_img_str)
            else:
                tf_example = make_example(img_name=str.encode(img_name),
                                          hr_img_path=str.encode(hr_img_path),
                                          lr_img_path=str.encode(lr_img_path))
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
