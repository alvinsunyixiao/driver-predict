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
        num_epochs = 1000,
        num_steps_per_epoch = 50,
        save_freq = 10,
    )

    def __init__(self):
        self.args = self._parse_args()
        self.p = ParamDict.from_file(self.args.params)
        self.network = DPNet(self.p.model, self.p.data)
        self.loss = LossManager(self.network)
        self.data = NuScenesDataset(self.p.data)
        self.optimizer = K.optimizers.Adam(1e-3, epsilon=1e-5, amsgrad=True)
        self.dataiter = iter(self.data.dataset)
        self.sess_dir = os.path.join(self.args.output, time.strftime("sess_%y-%m-%d_%H-%M-%S"))
        self.ckpt_dir = os.path.join(self.sess_dir, "checkpoints")
        self.log_dir = os.path.join(self.sess_dir, "logs")
        self.train_writer = tf.summary.create_file_writer(self.log_dir, name="train")
        self.val_writer = tf.summary.create_file_writer(self.log_dir, name="train")
        self.model_writer = tf.summary.create_file_writer(self.log_dir, name="model")

        self._save_model_graph()

    def _save_model_graph(self):
        data_dict = self.data.dataset.take(1).get_single_element()
        data_dict = {key: tf.zeros_like(data_dict[key]) for key in data_dict}

        @tf.function
        def init_as_graph(inputs):
            return self.network.get_init_state(inputs["image_bhw3"], inputs["state_b3"], training=True)

        @tf.function
        def model_as_graph(inputs):
            return self.network.model(inputs, training=True)

        tf.summary.trace_on()

        init_states = init_as_graph(data_dict)
        for key in init_states:
            data_dict[key] = tf.zeros_like(init_states[key])

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

        return parser.parse_args()

    def _preprocess_offset(self, data, offset):
        ret = data.copy()
        state_b3 = data["state_b3"]
        pos = state_b3[:, :2] - offset["pos"]
        rot = state_b3[:, 2] #- offset["rot"]
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

    def _single_time_loss(self, data, offset, init_states):
        data = self._preprocess_offset(data, offset)

        # combine with states
        data.update(init_states)

        # compute outputs and loss
        outputs = self.network.model(data, training=True)
        y_loss, img_loss, state_loss = self.loss.compute(
            y_proc = outputs["y_proc"],
            Py_proc = outputs["Py_proc"],
            y = outputs["y"],
            img_bhw3 = data["image_bhw3"],
            state_b3 = data["state_b3"],
        )

        tf.print("y_loss:", y_loss,
                 "img_loss:", img_loss,
                 "state_loss:", state_loss)

        return 1e-2*y_loss + img_loss + state_loss, {"init_z": outputs["z_proc"],
                                                     "init_Pz": outputs["Pz_proc"],
                                                     "init_z_hist": outputs["z_hist"]}

    @tf.function
    def _train_step(self,
                    dataiter: tf.data.Iterator,
                    optimizer: K.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            # generate offset based on first measurement
            data = dataiter.get_next()
            offset = self._generate_state_offset(data)
            data = self._preprocess_offset(data, offset)
            init_states = self.network.get_init_state(
                data["image_bhw3"], data["state_b3"], training=True)
            loss = 0.0

            # loop through subsequent timesteps
            for i in tf.range(self.data.p.seq_len - 1):
                data = dataiter.get_next()
                tmp_loss, init_states = self._single_time_loss(data, offset, init_states)
                loss += tmp_loss

            # add regularization loss
            loss += sum(self.network.model.losses)

        tf.print("Loss:", loss)

        grads = tape.gradient(loss, self.network.model.trainable_variables)
        #for grad, var in zip(grads, self.network.model.trainable_variables):
        #    tf.print(var.name, tf.norm(grad), tf.reduce_min(var), tf.reduce_max(var))
        optimizer.apply_gradients(zip(grads, self.network.model.trainable_variables))

    def train(self):
        for i in range(self.p.trainer.num_epochs):
            for j in range(self.p.trainer.num_steps_per_epoch):
                self._train_step(self.dataiter, self.optimizer)

            # save model
            if i % self.p.trainer.save_freq == 0:
                model_path = os.path.join(self.ckpt_dir, f"epoch-{i}")
                self.network.model.save_weights(model_path)

if __name__ == "__main__":
    Trainer().train()
