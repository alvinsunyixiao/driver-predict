import argparse
import math
import os
import tensorflow as tf
import tensorflow.keras as K

from dnn.data import NuScenesDataset
from dnn.loss import LossManager
from dnn.model import DPNet
from utils.params import ParamDict

class Trainer:

    DEFAULT_PARAMS = ParamDict(
        num_epochs = 100,
        save_freq = 10,
    )

    def __init__(self):
        self.args = self._parse_args()
        self.p = ParamDict.from_file(self.args.params)
        self.network = DPNet(self.p.model, self.p.data)
        self.loss = LossManager(self.network)
        self.data = NuScenesDataset(self.p.data)
        self.optimizer = K.optimizers.Adam(1e-3)

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--params", type=str, required=True,
                            help="path to the parameter file")
        parser.add_argument("-o", "--output", type=str, required=True,
                            help="path to store weights")

        return parser.parse_args()

    def _generate_state_offset(self, data):
        pos = data["state_b3"][:, :2]
        rot = data["state_b3"][:, 2]

        return {
            "pos": pos + tf.random.normal(tf.shape(pos), stddev=5.0),
            "rot": tf.random.uniform(tf.shape(rot), maxval=2*math.pi),
        }

    def _single_time_loss(self, data, offset, init_states):
        # preprocess state offset
        state_b3 = data["state_b3"]
        pos = state_b3[:, :2] - offset["pos"]
        rot = state_b3[:, 2] - offset["rot"]
        rot = tf.atan2(tf.sin(rot), tf.cos(rot))
        data["state_b3"] = tf.concat([pos, rot[:, None]], axis=-1)

        # combine with states
        data.update(init_states)

        # compute outputs and loss
        outputs = self.network.model(data)
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

        return y_loss + img_loss + state_loss, \
               {"init_state": tf.concat([outputs["z_meas"], outputs["z_log_var_meas"]], axis=-1)}

    @tf.function
    def _train_step(self,
                    dataiter: tf.data.Iterator,
                    optimizer: K.optimizers.Optimizer):

        optional = dataiter.get_next_as_optional()
        if not optional.has_value():
            return False

        with tf.GradientTape() as tape:
            # generate offset based on first measurement
            data = optional.get_value()
            offset = self._generate_state_offset(data)
            init_states = self.network.get_init_state(data["image_bhw3"], data["state_b3"])
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

        return True

    def train(self):
        for i in range(self.p.trainer.num_epochs):
            dataiter = iter(self.data.dataset)
            while True:
                if not self._train_step(dataiter, self.optimizer):
                    break

            # save model
            if i % self.p.trainer.save_freq == 0:
                model_path = os.path.join(self.args.output, f"epoch-{i}")
                self.network.model.save_weights(model_path)

if __name__ == "__main__":
    Trainer().train()
