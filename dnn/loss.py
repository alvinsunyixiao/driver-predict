import tensorflow as tf
import tensorflow.keras as K
import tensorflow_probability as tfp

from dnn.model import DPNet

class LossManager:
    def __init__(self, net: DPNet):
        self.net = net

    def image_recon_loss(self, img_bhw3, img_recon_bhw3):
        img_loss_bhw = tf.reduce_sum(tf.square(img_bhw3 - img_recon_bhw3), axis=-1)
        return tf.reduce_mean(img_loss_bhw)

    def state_recon_loss(self, state_b3, state_recon_b3):
        # position
        pos_recon_b2 = state_recon_b3[:, :2]
        pos_b2 = state_b3[:, :2]
        pos_loss_b = tf.reduce_sum(tf.square(pos_recon_b2 - pos_b2), axis=-1)

        # rotation
        rot_recon_b = state_recon_b3[:, 2]
        rot_b = state_b3[:, 2]
        rot_diff_b = rot_recon_b - rot_b
        rot_diff_b = tf.atan2(tf.sin(rot_diff_b), tf.cos(rot_diff_b))
        rot_loss_b = tf.square(rot_diff_b)

        return tf.reduce_mean(pos_loss_b + rot_loss_b)

    def dynamics_loss(self, z, y):
        z = tf.clip_by_value(z, -1.0 + 1e-4, 1.0 - 1e-4)
        y = tf.clip_by_value(y, -1.0 + 1e-4, 1.0 - 1e-4)
        loss_b = tf.reduce_sum(tf.square(tf.atanh(z) - tf.atanh(y)), axis=-1)
        return tf.reduce_mean(loss_b)

