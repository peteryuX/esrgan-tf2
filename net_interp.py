"""Implementation network interpolation and image interpolation proposed
   from ESRGAN paper."""
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import pathlib
import numpy as np
import tensorflow as tf

from modules.models import RRDB_Model
from modules.utils import (load_yaml, tensor2img, create_lr_hr_pair,
                           change_weight)


flags.DEFINE_string('cfg_path1', './configs/psnr.yaml', 'config file path 1')
flags.DEFINE_string('cfg_path2', './configs/esrgan.yaml', 'config file path 2')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', './data/PIPRM_3_crop.png',
                    'path to input image')
flags.DEFINE_boolean('save_image', True, 'save the result images.')
flags.DEFINE_boolean('save_ckpt', False, 'save all alpha ckpt.')


def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    cfg1 = load_yaml(FLAGS.cfg_path1)
    cfg2 = load_yaml(FLAGS.cfg_path2)

    # define network
    model = RRDB_Model(None, cfg1['ch_size'], cfg1['network_G'])

    # load checkpoint
    checkpoint_dir1 = './checkpoints/' + cfg1['sub_name']
    checkpoint1 = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir1):
        checkpoint1.restore(tf.train.latest_checkpoint(checkpoint_dir1))
        print("[*] load ckpt 1 from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir1)))
    else:
        print("[*] Cannot find ckpt 1 from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir1)))
    vars1 = [v.numpy() for v in checkpoint1.model.trainable_variables]

    checkpoint_dir2 = './checkpoints/' + cfg2['sub_name']
    checkpoint2 = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir2):
        checkpoint2.restore(tf.train.latest_checkpoint(checkpoint_dir2))
        print("[*] load ckpt 2 from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir2)))
    else:
        print("[*] Cannot find ckpt 2 from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir2)))
    vars2 = [v.numpy() for v in checkpoint2.model.trainable_variables]

    # evaluation
    print("[*] Processing on single image {}".format(FLAGS.img_path))
    if not os.path.exists(FLAGS.img_path):
        raise ValueError('Can not find image from {}.'.format(FLAGS.img_path))
    raw_img = cv2.imread(FLAGS.img_path)
    lr_img, hr_img = create_lr_hr_pair(raw_img, cfg1['scale'])

    results_path = './results_interp/{}_{}/'.format(
        cfg1['sub_name'], cfg2['sub_name'])
    pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

    interp_w = [np.zeros([hr_img.shape[0], 5, 3], np.uint8)]
    interp_i = [np.zeros([hr_img.shape[0], 5, 3], np.uint8)]
    for alpha in [1., 0.8, 0.6, 0.4, 0.2, 0.]:
        print("[*] Process alpha = {:.1f}".format(alpha))
        # interpolation weights
        change_weight(model, vars1, vars2, alpha)
        interp_w.append(tensor2img(model(lr_img[np.newaxis, :] / 255)))
        interp_w.append(np.zeros([hr_img.shape[0], 5, 3], np.uint8))
        if FLAGS.save_ckpt:
            checkpoint2.save(results_path + 'alpha_{}'.format(alpha))

        # interpolation image
        change_weight(model, vars1, vars2, 0.0)
        sr_img1 = tensor2img(model(lr_img[np.newaxis, :] / 255))
        change_weight(model, vars1, vars2, 1.0)
        sr_img2 = tensor2img(model(lr_img[np.newaxis, :] / 255))
        interp_i.append((sr_img1.astype(np.float32) * (1 - alpha) \
            + sr_img2.astype(np.float32) * alpha).astype(np.uint8))
        interp_i.append(np.zeros([hr_img.shape[0], 5, 3], np.uint8))


    if FLAGS.save_image:
        base_name = os.path.basename(FLAGS.img_path)
        result_interp_w_path = results_path + 'weight_interp_' + base_name
        result_interp_i_path = results_path + 'image_interp_' + base_name
        print("[*] write the weight interp {}".format(result_interp_w_path))
        cv2.imwrite(result_interp_w_path, np.concatenate(interp_w, 1))
        print("[*] write the image interp {}".format(result_interp_i_path))
        cv2.imwrite(result_interp_i_path, np.concatenate(interp_i, 1))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
