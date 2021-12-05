import argparse
import math
import os
import time

import tensorflow as tf
import tensorflow.keras as K

from dnn.data import NuScenesDataset
from dnn.loss import LossManager
from dnn.model import DPNet
from utils.params import ParamDict

class Trainer:

    DEFAULT_PARAMS = ParamDict(
        num_epochs = 3000,
        num_steps_per_epoch = 25,
        num_eval_per_epoch = 2,
        save_freq = 10,
    )

    def __init__(self):
        self.args = self._parse_args()
        self.p = ParamDict.from_file(self.args.params)

        self.network = DPNet(self.p.model, self.p.data)
        if self.args.checkpoint:
            self.network.model.load_weights(self.args.checkpoint)

        self.loss = LossManager(self.network)
        self.data = NuScenesDataset(self.p.data)
        lr_schedule = K.optimizers.schedules.PiecewiseConstantDecay([30000, 50000], [1e-3, 1e-4, 1e-5])
        self.optimizer = K.optimizers.Adam(lr_schedule, epsilon=1e-5)
        self.train_dataiter = iter(self.data.train_dataset)
        self.val_dataiter = iter(self.data.val_dataset)
        self.sess_dir = os.path.join(self.args.output, time.strftime("sess_%y-%m-%d_%H-%M-%S"))
        self.ckpt_dir = os.path.join(self.sess_dir, "checkpoints")
        self.log_dir = os.path.join(self.sess_dir, "logs")
        self.train_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "train"))
        self.val_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "val"))
        self.model_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "model"))

        self._save_model_graph()

    def _save_model_graph(self):
        data_dict = self.data.train_dataset.take(1).get_single_element()
        data_dict = {key: tf.zeros_like(data_dict[key]) for key in data_dict}

        @tf.function
        def init_as_graph(inputs):
            z, _, img_code, _ = self.network.encode(inputs["image_bhw3"], inputs["state_b3"], training=True)
            return z, img_code

        @tf.function
        def model_as_graph(inputs):
            return self.network.model(inputs, training=True)

        tf.summary.trace_on()

        init_state, img_code = init_as_graph(data_dict)
        data_dict["init_state"] = init_state

        output_dict = model_as_graph(data_dict)

        with self.model_writer.as_default():
            tf.summary.trace_export("model_trace", step=0)
        tf.summary.trace_off()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--params", type=str, required=True,
                            help="path to the parameter file")
        parser.add_argument("-o", "--output", type=str, required=True,
                            help="path to store weights")
        parser.add_argument("--checkpoint", type=str,
                            help="optionally load previous checkpoint")

        return parser.parse_args()

    def _preprocess_offset(self, data, offset):
        ret = data.copy()
        state_b3 = data["state_b3"]
        pos = state_b3[:, :2] - offset["pos"]
        rot = state_b3[:, 2]
        rot = tf.atan2(tf.sin(rot), tf.cos(rot))
        ret["state_b3"] = tf.concat([pos, rot[:, None]], axis=-1)

        return ret

    def _generate_state_offset(self, data):
        pos = data["state_b3"][:, :2]
        rot = data["state_b3"][:, 2]

        return {
            "pos": pos + tf.random.normal(tf.shape(pos), stddev=5.0),
            "rot": tf.random.uniform(tf.shape(rot), maxval=2*math.pi),
        }

    def _single_time_loss(self, step, seq_id, data, offset, init_state, init_img_feats):
        # TODO(alvin): adding training=False
        data = self._preprocess_offset(data, offset)

        # combine with states
        data["init_state"] = init_state

        # compute outputs and loss
        outputs = self.network.model(data, training=True)
        img_recon_loss = self.loss.image_recon_loss(data["image_bhw3"], outputs["img_recon_bhw3"])
        state_recon_loss = self.loss.state_recon_loss(data["state_b3"], outputs["state_recon_b3"])
        dynamic_loss = self.loss.dynamics_loss(outputs["z"], outputs["y"])

        if seq_id == 0:
            tf.summary.scalar("img_loss_0", img_recon_loss, step=step)
            tf.summary.scalar("state_loss_0", state_recon_loss, step=step)
            tf.summary.scalar("dynamic_loss_0", dynamic_loss, step=step)
        elif seq_id == self.data.p.seq_len - 2:
            tf.summary.scalar("img_loss_1", img_recon_loss, step=step)
            tf.summary.scalar("state_loss_1", state_recon_loss, step=step)
            tf.summary.scalar("dynamic_loss_1", dynamic_loss, step=step)

        return img_recon_loss + state_recon_loss + 1e-3*dynamic_loss, outputs["z"]

    @tf.function
    def _train_step(self,
                    step,
                    dataiter: tf.data.Iterator,
                    optimizer: K.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            # generate offset based on first measurement
            data = dataiter.get_next()
            offset = self._generate_state_offset(data)
            data = self._preprocess_offset(data, offset)
            init_state, _, img_feats, _ = self.network.encode(
                data["image_bhw3"], data["state_b3"], training=True)
            loss = 0.0

            # loop through subsequent timesteps
            for i in tf.range(self.data.p.seq_len - 1):
                data = dataiter.get_next()
                tmp_loss, init_state = self._single_time_loss(step, i, data, offset, init_state, img_feats)
                loss += tmp_loss

            # add regularization loss
            reg_loss = sum(self.network.model.losses)
            tf.summary.scalar("Regularization Loss", reg_loss, step=step)

            loss += reg_loss

        tf.print("Loss:", loss)
        tf.summary.scalar("Total Loss", loss, step=step)

        grads = tape.gradient(loss, self.network.model.trainable_variables)
        #for grad, var in zip(grads, self.network.model.trainable_variables):
        #    tf.print(var.name, tf.norm(grad))
        optimizer.apply_gradients(zip(grads, self.network.model.trainable_variables))

    @tf.function
    def _eval_step(self, step, dataiter: tf.data.Iterator):
        # generate offset based on first measurement
        data = dataiter.get_next()
        offset = self._generate_state_offset(data)
        data = self._preprocess_offset(data, offset)
        init_state, img_code, img_feats, _ = self.network.encode(
            data["image_bhw3"], data["state_b3"], training=False)
        loss = 0.0

        # plot image reconstruction
        image_recon_bhw3 = self.network.img_decoder((img_code, img_feats))
        images_concat = tf.concat([data["image_bhw3"], image_recon_bhw3], axis=2)
        tf.summary.image("image / reconstruction", images_concat / 2.0 + 0.5, step=step)

        # loop through subsequent timesteps
        for i in tf.range(self.data.p.seq_len - 1):
            data = dataiter.get_next()
            tmp_loss, init_state = self._single_time_loss(step, i, data, offset, init_state, img_feats)
            loss += tmp_loss

        # plot reconstruction of the dynamical inference
        img_recon_dym_bhw3, _ = self.network.decode(init_state, img_feats, training=False)
        img_recon_concat = tf.concat([data["image_bhw3"], img_recon_dym_bhw3], axis=2)
        tf.summary.image("image / dynamic reconstruction", img_recon_concat / 2.0 + 0.5, step=step)

        # add regularization loss
        reg_loss = sum(self.network.model.losses)
        tf.summary.scalar("Regularization Loss", reg_loss, step=step)

        loss += reg_loss

        tf.print("Validation Loss:", loss)
        tf.summary.scalar("Total Loss", loss, step=step)

    def train(self):
        for i in tf.range(self.p.trainer.num_epochs, dtype=tf.int64):
            # training
            with self.train_writer.as_default():
                for j in tf.range(self.p.trainer.num_steps_per_epoch, dtype=tf.int64):
                    step = i * self.p.trainer.num_steps_per_epoch + j
                    self._train_step(step, self.train_dataiter, self.optimizer)

                for var in self.network.model.trainable_variables:
                    tf.summary.histogram(var.name, var, step=step)

            # validation
            with self.val_writer.as_default():
                for j in tf.range(self.p.trainer.num_eval_per_epoch, dtype=tf.int64):
                    self._eval_step(step + j, self.val_dataiter)

            # save model
            if i % self.p.trainer.save_freq == 0:
                model_path = os.path.join(self.ckpt_dir, f"epoch-{i}")
                self.network.model.save_weights(model_path)

if __name__ == "__main__":
    Trainer().train()
