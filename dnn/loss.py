import tensorflow as tf
import tensorflow_probability as tfp

from dnn.model import DPNet

class LossManager:
    def __init__(self, net: DPNet):
        self.net = net

    def compute(self, y_proc, Py_proc, y, img_bhw3, state_b3):
        y_proc_dist = tfp.distributions.MultivariateNormalTriL(
            loc=y_proc, scale_tril=tf.linalg.cholesky(Py_proc))

        tf.print(y_proc_dist.stddev())

        y_loss = self.latent_meas_loss(y, y_proc_dist)
        img_loss, state_loss = self.decoded_loss(img_bhw3, state_b3, y, y_proc_dist)

        return y_loss, img_loss, state_loss

    def latent_meas_loss(self, y, y_proc_dist):
        return -tf.reduce_mean(y_proc_dist.log_prob(y))

    def decoded_loss(self, img_bhw3, state_b3, y, y_proc_dist):
        y_proc_sample = y

        # decoded image loss
        img_code = y_proc_sample[:, :self.net.p.img_enc_units]
        img_recon = self.net.img_decoder(img_code)
        img_loss = tf.reduce_sum(tf.square(img_recon - img_bhw3), axis=-1)
        img_loss = tf.reduce_mean(img_loss)

        # decoded state loss
        state_code = y_proc_sample[:, self.net.p.img_enc_units:]
        state_recon = self.net.state_decoder(state_code)

        # position
        pos_recon = state_recon[:, :2]
        pos = state_b3[:, :2]
        pos_loss = tf.reduce_sum(tf.square(pos_recon - pos), axis=-1)

        # rotation
        rot_recon = state_recon[:, 2]
        rot = state_b3[:, 2]
        rot_diff = rot_recon - rot
        rot_diff = tf.atan2(tf.sin(rot_diff), tf.cos(rot_diff))
        rot_loss = tf.square(rot_diff)

        state_loss = tf.reduce_mean((pos_loss + rot_loss))

        return img_loss, state_loss

