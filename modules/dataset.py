import tensorflow as tf


def _parse_tfrecord(gt_size, scale, using_bin, using_flip, using_rot):
    def parse_tfrecord(tfrecord):
        if using_bin:
            features = {
                'image/img_name': tf.io.FixedLenFeature([], tf.string),
                'image/hr_encoded': tf.io.FixedLenFeature([], tf.string),
                'image/lr_encoded': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            lr_img = tf.image.decode_jpeg(x['image/lr_encoded'], channels=3)
            hr_img = tf.image.decode_jpeg(x['image/hr_encoded'], channels=3)
        else:
            features = {
                'image/img_name': tf.io.FixedLenFeature([], tf.string),
                'image/hr_img_path': tf.io.FixedLenFeature([], tf.string),
                'image/lr_img_path': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            hr_image_encoded = tf.io.read_file(x['image/hr_img_path'])
            lr_image_encoded = tf.io.read_file(x['image/lr_img_path'])
            lr_img = tf.image.decode_png(lr_image_encoded, channels=3)
            hr_img = tf.image.decode_png(hr_image_encoded, channels=3)

        lr_img, hr_img = _transform_images(
            gt_size, scale, using_flip, using_rot)(lr_img, hr_img)

        return lr_img, hr_img
    return parse_tfrecord


def _transform_images(gt_size, scale, using_flip, using_rot):
    def transform_images(lr_img, hr_img):
        lr_img_shape = tf.shape(lr_img)
        hr_img_shape = tf.shape(hr_img)
        gt_shape = (gt_size, gt_size, tf.shape(hr_img)[-1])
        lr_size = int(gt_size / scale)
        lr_shape = (lr_size, lr_size, tf.shape(lr_img)[-1])

        tf.Assert(
            tf.reduce_all(hr_img_shape >= gt_shape),
            ["Need hr_image.shape >= gt_size, got ", hr_img_shape, gt_shape])
        tf.Assert(
            tf.reduce_all(hr_img_shape[:-1] == lr_img_shape[:-1] * scale),
            ["Need hr_image.shape == lr_image.shape * scale, got ",
             hr_img_shape[:-1], lr_img_shape[:-1] * scale])
        tf.Assert(
            tf.reduce_all(hr_img_shape[-1] == lr_img_shape[-1]),
            ["Need hr_image.shape[-1] == lr_image.shape[-1]], got ",
             hr_img_shape[-1], lr_img_shape[-1]])

        # randomly crop
        limit = lr_img_shape - lr_shape + 1
        offset = tf.random.uniform(tf.shape(lr_img_shape), dtype=tf.int32,
                                   maxval=tf.int32.max) % limit
        lr_img = tf.slice(lr_img, offset, lr_shape)
        hr_img = tf.slice(hr_img, offset * scale, gt_shape)

        # randomly left-right flip
        if using_flip:
            flip_case = tf.random.uniform([1], 0, 2, dtype=tf.int32)
            def flip_func(): return (tf.image.flip_left_right(lr_img),
                                     tf.image.flip_left_right(hr_img))
            lr_img, hr_img = tf.case(
                [(tf.equal(flip_case, 0), flip_func)],
                default=lambda: (lr_img, hr_img))

        # randomly rotation
        if using_rot:
            rot_case = tf.random.uniform([1], 0, 4, dtype=tf.int32)
            def rot90_func(): return (tf.image.rot90(lr_img, k=1),
                                      tf.image.rot90(hr_img, k=1))
            def rot180_func(): return (tf.image.rot90(lr_img, k=2),
                                       tf.image.rot90(hr_img, k=2))
            def rot270_func(): return (tf.image.rot90(lr_img, k=3),
                                       tf.image.rot90(hr_img, k=3))
            lr_img, hr_img = tf.case(
                [(tf.equal(flip_case, 0), rot90_func),
                 (tf.equal(flip_case, 1), rot180_func),
                 (tf.equal(flip_case, 2), rot270_func)],
                default=lambda: (lr_img, hr_img))

        # scale to [0, 1]
        lr_img = lr_img / 255
        hr_img = hr_img / 255

        return lr_img, hr_img
    return transform_images


def load_tfrecord_dataset(tfrecord_name, batch_size, gt_size,
                          scale, using_bin=False, using_flip=False,
                          using_rot=False, shuffle=True, buffer_size=10240):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(gt_size, scale, using_bin, using_flip, using_rot),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    import time

    using_bin = True
    if using_bin:
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
    print("data fps: {:.2f}".format(num_samples / (time.time() - start_time)))
