import functools
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, LeakyReLU


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer with scale."""
    scale = 2. * scale
    return tf.keras.initializers.VarianceScaling(
        scale=scale, mode='fan_in', distribution="truncated_normal", seed=seed)


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True,
                 scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs)

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ResDenseBlock_5C(tf.keras.layers.Layer):
    """Residual Dense Block"""
    def __init__(self, nf=64, gc=32, res_beta=0.2, wd=0., name='RDB5C',
                 **kwargs):
        super(ResDenseBlock_5C, self).__init__(name=name, **kwargs)
        # gc: growth channel, i.e. intermediate channels
        self.res_beta = res_beta
        lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
        _Conv2DLayer = functools.partial(
            Conv2D, kernel_size=3, padding='same',
            kernel_initializer=_kernel_init(0.1), bias_initializer='zeros',
            kernel_regularizer=_regularizer(wd))
        self.conv1 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv2 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv3 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv4 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv5 = _Conv2DLayer(filters=nf, activation=lrelu_f())

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(tf.concat([x, x1], 3))
        x3 = self.conv3(tf.concat([x, x1, x2], 3))
        x4 = self.conv4(tf.concat([x, x1, x2, x3], 3))
        x5 = self.conv5(tf.concat([x, x1, x2, x3, x4], 3))
        return x5 * self.res_beta + x


class ResInResDenseBlock(tf.keras.layers.Layer):
    """Residual in Residual Dense Block"""
    def __init__(self, nf=64, gc=32, res_beta=0.2, wd=0., name='RRDB',
                 **kwargs):
        super(ResInResDenseBlock, self).__init__(name=name, **kwargs)
        self.res_beta = res_beta
        self.rdb_1 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.rdb_2 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.rdb_3 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)

    def call(self, x):
        out = self.rdb_1(x)
        out = self.rdb_2(out)
        out = self.rdb_3(out)
        return out * self.res_beta + x


def RRDB_Model(size, channels, cfg_net, gc=32, wd=0., name='RRDB_model'):
    """Residual-in-Residual Dense Block based Model """
    nf, nb = cfg_net['nf'], cfg_net['nb']
    lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
    rrdb_f = functools.partial(ResInResDenseBlock, nf=nf, gc=gc, wd=wd)
    conv_f = functools.partial(Conv2D, kernel_size=3, padding='same',
                               bias_initializer='zeros',
                               kernel_initializer=_kernel_init(),
                               kernel_regularizer=_regularizer(wd))
    rrdb_truck_f = tf.keras.Sequential(
        [rrdb_f(name="RRDB_{}".format(i)) for i in range(nb)],
        name='RRDB_trunk')

    # extraction
    x = inputs = Input([size, size, channels], name='input_image')
    fea = conv_f(filters=nf, name='conv_first')(x)
    fea_rrdb = rrdb_truck_f(fea)
    trunck = conv_f(filters=nf, name='conv_trunk')(fea_rrdb)
    fea = fea + trunck

    # upsampling
    size_fea_h = tf.shape(fea)[1] if size is None else size
    size_fea_w = tf.shape(fea)[2] if size is None else size
    fea_resize = tf.image.resize(fea, [size_fea_h * 2, size_fea_w * 2],
                                 method='nearest', name='upsample_nn_1')
    fea = conv_f(filters=nf, activation=lrelu_f(), name='upconv_1')(fea_resize)
    fea_resize = tf.image.resize(fea, [size_fea_h * 4, size_fea_w * 4],
                                 method='nearest', name='upsample_nn_2')
    fea = conv_f(filters=nf, activation=lrelu_f(), name='upconv_2')(fea_resize)
    fea = conv_f(filters=nf, activation=lrelu_f(), name='conv_hr')(fea)
    out = conv_f(filters=channels, name='conv_last')(fea)

    return Model(inputs, out, name=name)


def DiscriminatorVGG128(size, channels, nf=64, wd=0.,
                        name='Discriminator_VGG_128'):
    """Discriminator VGG 128"""
    lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
    conv_k3s1_f = functools.partial(Conv2D,
                                    kernel_size=3, strides=1, padding='same',
                                    kernel_initializer=_kernel_init(),
                                    kernel_regularizer=_regularizer(wd))
    conv_k4s2_f = functools.partial(Conv2D,
                                    kernel_size=4, strides=2, padding='same',
                                    kernel_initializer=_kernel_init(),
                                    kernel_regularizer=_regularizer(wd))
    dese_f = functools.partial(Dense, kernel_regularizer=_regularizer(wd))

    x = inputs = Input(shape=(size, size, channels))

    x = conv_k3s1_f(filters=nf, name='conv0_0')(x)
    x = conv_k4s2_f(filters=nf, use_bias=False, name='conv0_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn0_1')(x))

    x = conv_k3s1_f(filters=nf * 2, use_bias=False, name='conv1_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn1_0')(x))
    x = conv_k4s2_f(filters=nf * 2, use_bias=False, name='conv1_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn1_1')(x))

    x = conv_k3s1_f(filters=nf * 4, use_bias=False, name='conv2_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn2_0')(x))
    x = conv_k4s2_f(filters=nf * 4, use_bias=False, name='conv2_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn2_1')(x))

    x = conv_k3s1_f(filters=nf * 8, use_bias=False, name='conv3_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn3_0')(x))
    x = conv_k4s2_f(filters=nf * 8, use_bias=False, name='conv3_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn3_1')(x))

    x = conv_k3s1_f(filters=nf * 8, use_bias=False, name='conv4_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn4_0')(x))
    x = conv_k4s2_f(filters=nf * 8, use_bias=False, name='conv4_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn4_1')(x))

    x = Flatten()(x)
    x = dese_f(units=100, activation=lrelu_f(), name='linear1')(x)
    out = dese_f(units=1, name='linear2')(x)

    return Model(inputs, out, name=name)
