import argparse
import math
import tensorflow as tf
import tensorflow.keras as K

from dnn.data import NuScenesDataset
from dnn.loss import LossManager
from dnn.model import DPNet
from utils.params import ParamDict
from utils.tf_utils import set_tf_memory_growth

class Trainer:

    DEFAULT_PARAMS = ParamDict(

    )

    def __init__(self):
        self.args = self._parse_args()
        self.p = ParamDict.from_file(self.args.params)
        self.network = DPNet(self.p.model, self.p.data)
        self.loss = LossManager(self.network)
        self.data = NuScenesDataset(self.p.data)
        self.dataiter = iter(self.data.dataset)
        self.optimizer = K.optimizers.Adam(1e-3)

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--params", type=str, required=True,
                            help="path to the parameter file")

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

        return y_loss + img_loss + state_loss, {"init_z": outputs["z_meas"],
                                                "init_Pz": outputs["Pz_meas"],
                                                "init_z_hist": outputs["z_hist"]}

    @tf.function
    def _train_step(self, dataiter: tf.data.Iterator):
        optional = dataiter.get_next_as_optional()
        if not optional.has_value():
            return False

        with tf.GradientTape() as tape:
            # generate offset based on first measurement
            data = optional.get_value()
            offset = self._generate_state_offset(data)
            init_states = self.network.get_init_state()
            loss, init_states = self._single_time_loss(data, offset, init_states)

            # loop through subsequent timesteps
            for i in tf.range(self.data.p.seq_len - 1):
                data = dataiter.get_next()
                if i < 10:
                    tmp_loss, init_states = self._single_time_loss(data, offset, init_states)
                    loss += tmp_loss

            # add regularization loss
            loss += sum(self.network.model.losses)

        grads = tape.gradient(loss, self.network.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.network.model.trainable_weights))

        return True

    def train(self):
        while True:
            if not self._train_step(self.dataiter):
                break
        self.dataiter = iter(self.data.dataset)

if __name__ == "__main__":
    set_tf_memory_growth()
    Trainer().train()
