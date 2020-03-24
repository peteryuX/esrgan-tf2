import time
import cv2
from absl import app, flags
from absl.flags import FLAGS

from modules.dataset import load_tfrecord_dataset


flags.DEFINE_boolean('using_bin', True, 'whether use binary file or not')
flags.DEFINE_boolean('visualization', True, 'whether visualize dataset or not')


def main(_):
    if FLAGS.using_bin:
        train_dataset = load_tfrecord_dataset(
            './data/DIV2K800_sub_bin.tfrecord', 16, 128, 4,
            using_bin=True, using_flip=True, using_rot=False, buffer_size=10)
    else:
        train_dataset = load_tfrecord_dataset(
            './data/DIV2K800_sub.tfrecord', 16, 128, 4,
            using_bin=False, using_flip=True, using_rot=False, buffer_size=10)

    num_samples = 100
    start_time = time.time()
    for idx, (inputs, labels) in enumerate(train_dataset.take(num_samples)):
        print("{} inputs:".format(idx), inputs.shape, "outputs:", labels.shape)

        if FLAGS.visualization:
            cv2.imshow('inputs', cv2.cvtColor(inputs[0].numpy(),
                       cv2.COLOR_RGB2BGR))
            cv2.imshow('labels', cv2.cvtColor(labels[0].numpy(),
                       cv2.COLOR_RGB2BGR))

            if cv2.waitKey(0) == ord('q'):
                exit()

    print("data fps: {:.2f}".format(num_samples / (time.time() - start_time)))


if __name__ == '__main__':
    app.run(main)
