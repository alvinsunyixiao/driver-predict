import typing as T
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL

from utils.params import ParamDict

class ImageEncoder(KL.Layer):
    def __init__(self, units: int, **kwargs):
        super(ImageEncoder, self).__init__(**kwargs)
        self.units = units
        self.mbnet = K.applications.MobileNetV3Small(
            minimalistic=True,
            include_top=False,
            pooling="avg",
        )
        self.mbnet.trainable = self.trainable
        self.fc = KL.Dense(units, trainable=self.trainable, name="fc")

    def call(self, image):
        x = self.mbnet(image)
        return self.fc(x)

class CorrectNet(KL.Layer):
    def __init__(self,
        z_units: int,
        num_hidden: int,
        num_layers: int,
        **kwargs
    ):
        super(CorrectNet, self).__init__(**kwargs)
        self.z_units = z_units
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.mlp = self.build_mlp()

    def call(self, inputs):
        a, a_hat = inputs

        # image residual
        r_img = a["image"] - a_hat["image"]

        # translation residual
        r_trans = a["state"][..., :2] - a_hat["state"][..., :2]

        # rotation residual
        r_theta = a["state"][..., 2, None] - a_hat["state"][..., 2, None]
        r_rot = tf.atan2(tf.sin(r_theta), tf.cos(r_theta))

        r_all = tf.concat([r_img, r_trans, r_rot], axis=-1)

        return self.mlp(r_all)

    def build_mlp(self):
        fcs = [KL.Dense(
            self.num_hidden,
            activation="relu",
            kernel_regularizer=K.regularizers.l2(1e-4),
            bias_regularizer=K.regularizers.l2(1e-4),
        ) for i in range(self.num_layers)]

        fcs.append(KL.Dense(self.z_units))

        return K.Sequential(fcs)

class DPNet:

    DEFAULT_PARAMS = ParamDict(
        batch_size = 16,
        img_enc_units = 128,
        img_units = 128,
        state_units = 3,
        z_units = 256,
        mlp_units = 256,
        mlp_layers = 4,
        img_size = (225, 400),
    )

    def __init__(self, params: ParamDict = DEFAULT_PARAMS):
        self.p = params
        self.img_encoder = ImageEncoder(self.p.img_enc_units)
        self.C_img = KL.Dense(self.p.img_units, name="C_img")
        self.C_state = KL.Dense(self.p.state_units, name="C_state")
        self.correct_net = CorrectNet(
            z_units=self.p.z_units,
            num_hidden=self.p.mlp_units,
            num_layers=self.p.mlp_layers,
        )
        self.gru = KL.GRU(self.p.z_units,
            kernel_regularizer=K.regularizers.l2(1e-4),
            bias_regularizer=K.regularizers.l2(1e-4),
            recurrent_regularizer=K.regularizers.l2(1e-4),
            stateful=True,
            return_state=True,
        )
        self.model = self.build_model()

    def build_model(self) -> K.Model:
        img_bhw3 = KL.Input(self.p.img_size + (3,), batch_size=self.p.batch_size)
        state_b3 = KL.Input((self.p.state_units,), batch_size=self.p.batch_size)
        ctrl_mask_bt = KL.Input((None,), batch_size=self.p.batch_size, dtype=tf.bool)
        ctrl_inputs_bt3 = KL.Input((None, 3), batch_size=self.p.batch_size)

        # process model
        _, latent_proc_bz = self.gru(ctrl_inputs_bt3, mask=ctrl_mask_bt)
        a_hat = {"image": self.C_img(latent_proc_bz), "state": self.C_state(latent_proc_bz)}

        # image
        img_code_by = self.img_encoder(img_bhw3)
        a = {"image": img_code_by, "state": state_b3}
        latent_delta_bz = self.correct_net((a, a_hat))
        latent_meas_bz = KL.Add()([latent_proc_bz, latent_delta_bz])

        return K.Model(
                inputs={
                    "img_bhw3": img_bhw3,
                    "state_b3": state_b3,
                    "ctrl_mask_bt": ctrl_mask_bt,
                    "ctrl_inputs_bt3": ctrl_inputs_bt3,
                },
                outputs={
                    "latent_proc_bz": latent_proc_bz,
                    "latent_meas_bz": latent_meas_bz,
                })

if __name__ == "__main__":
    dpnet = DPNet()
    dpnet.model.summary()
