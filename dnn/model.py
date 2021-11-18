import math
import typing as T
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL
import tensorflow_probability as tfp

from utils.params import ParamDict
from dnn.data import NuScenesDataset

class MLP(KL.Layer):
    def __init__(self,
                 output_units: int,
                 hidden_units: int,
                 num_layers: int,
                 dropout: T.Optional[float] = None,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout

        if dropout is not None:
            assert dropout >= 0 and dropout <= 1, r"dropout = {dropout} is invalid"

        # build dense network
        layers = []
        for i in range(self.num_layers):
            # fully connected layer
            layers.append(KL.Dense(
                units=hidden_units,
                activation="elu",
                #kernel_regularizer=K.regularizers.l2(1e-4),
                #bias_regularizer=K.regularizers.l2(1e-4),
            ))
            # dropout
            if dropout is not None:
                layers.append(KL.Dropout(dropout))

        layers.append(KL.Dense(self.output_units))

        self.f = K.Sequential(layers)

    def call(self, x, training=None):
        return self.f(x, training=training)

class ImageEncoder(KL.Layer):
    def __init__(self, units: int, dropout: T.Optional[float] = None, **kwargs):
        super(ImageEncoder, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout

        reg = None #K.regularizers.l2(1e-4)
        self.cnn = K.Sequential([
            KL.Conv2D(24, 7, 2, activation="elu", kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2D(24, 5, 1, activation="elu", kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2D(48, 5, 2, activation="elu", kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2D(48, 3, 1, activation="elu", kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2D(72, 3, 2, activation="elu", kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2D(72, 3, 1, activation="elu", kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2D(96, 3, 2, activation="elu", kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2D(96, 3, 1, activation="elu", kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2D(128, 3, 2, activation="elu", kernel_regularizer=reg, bias_regularizer=reg),
        ])
        self.mlp = MLP(units, hidden_units=512, num_layers=1, dropout=dropout)

    def call(self, x, training=None):
        x = self.cnn(x)
        x = KL.Flatten()(x)
        x = self.mlp(x, training=training)

        return x

class ImageDecoder(KL.Layer):
    def __init__(self, feat_shape: T.Tuple[int], dropout: T.Optional[float] = None, **kwargs):
        super(ImageDecoder, self).__init__(**kwargs)
        self.feat_shape = feat_shape
        self.dropout = dropout

        reg = None #K.regularizers.l2(1e-4)
        self.mlp = MLP(math.prod(feat_shape), hidden_units=512, num_layers=1, dropout=dropout)
        self.decnn = K.Sequential([
            KL.Conv2DTranspose(96, 3, 2, activation="elu", output_padding=(0, 1),
                               kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2DTranspose(96, 3, 1, activation="elu",
                               kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2DTranspose(72, 3, 2, activation="elu", output_padding=1,
                               kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2DTranspose(72, 3, 1, activation="elu",
                               kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2DTranspose(48, 3, 2, activation="elu",
                               kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2DTranspose(48, 3, 1, activation="elu",
                               kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2DTranspose(24, 5, 2, activation="elu",
                               kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2DTranspose(24, 5, 1, activation="elu",
                               kernel_regularizer=reg, bias_regularizer=reg),
            KL.Conv2DTranspose(3, 7, 2, activation=None,
                               kernel_regularizer=reg, bias_regularizer=reg),
        ])

    def call(self, x, training=None):
        x = self.mlp(x, training=training)
        x = KL.Reshape(self.feat_shape)(x)
        x = self.decnn(x)

        return x


class GRUDynamics(KL.Layer):

    MLP_HIDDEN_UNITS = 128
    MLP_NUM_LAYERS = 2
    MLP_DROPOUT = None

    def __init__(self,
                 state_units: int,
                 **kwargs):
        super(GRUDynamics, self).__init__(**kwargs)
        self.state_units = state_units
        self.gru = KL.GRU(2 * state_units, activation=None)

    def call(self, inputs, mask=None, training=None):
        ctrl_inputs_bt2, init_state = inputs
        gru_state = self.gru(
            ctrl_inputs_bt2, initial_state=init_state, mask=mask, training=training)
        z = gru_state[..., :self.state_units]
        z_log_var = gru_state[..., self.state_units:]
        return z, z_log_var

class KalmanMeasure(KL.Layer):
    def __init__(self, state_units: int, observe_units: int, **kwargs):
        super(KalmanMeasure, self).__init__(**kwargs)
        self.state_units = state_units
        self.observe_units = observe_units

    def build(self, input_shapes):
        self.C = self.add_weight(
            name="C",
            shape=(self.observe_units, self.state_units),
            trainable=True,
        )

    def call(self, inputs):
        # process prediction
        z, z_log_var, y = inputs
        z_vec = z[..., None]
        Pz = tf.linalg.diag(tf.exp(z_log_var))

        C = self.C[None]
        C_T = tf.linalg.adjoint(C)

        y_hat_vec = C @ z_vec
        y_hat = y_hat_vec[..., 0]

        r = y - y_hat
        S = C @ Pz @ C_T + tf.eye(self.observe_units)[None]
        K_T = tf.linalg.cholesky_solve(S, C @ Pz)
        K = tf.linalg.adjoint(K_T)

        Py = C @ Pz @ C_T + tf.eye(self.observe_units)[None]

        z_vec += K @ r[..., None]
        Pz = (tf.eye(self.state_units)[None] - K @ C) @ Pz

        # corrected measurements
        y_vec = C @ z_vec

        return {
            "z_meas": z_vec[..., 0],
            "z_log_var_meas": tf.math.log(tf.linalg.diag_part(Pz)),
            "y_meas": y_vec[..., 0],
            "y_proc": y_hat,
            "Py_proc": Py,
        }

class DPNet:

    DEFAULT_PARAMS = ParamDict(
        img_enc_units = 128,
        img_units = 128,
        state_enc_units = 32,
        state_units = 3,
        control_units = 2,
        z_units = 256,
        mlp_dropout = None,
    )

    def __init__(self,
                 params: ParamDict = DEFAULT_PARAMS,
                 data_params: ParamDict = NuScenesDataset.DEFAULT_PARAMS):
        self.p = params
        self.data_p = data_params

        self.img_encoder = ImageEncoder(self.p.img_enc_units, dropout=self.p.mlp_dropout)
        image_shape = (None,) + self.data_p.img_size + (3,)
        feat_shape = self.img_encoder.cnn.compute_output_shape(image_shape)
        self.img_decoder = ImageDecoder(feat_shape[1:], dropout=self.p.mlp_dropout)
        self.state_encoder = MLP(
            self.p.state_enc_units, 64, 1, self.p.mlp_dropout, name="state_encoder")
        self.state_decoder = MLP(
            self.p.state_units, 64, 1, self.p.mlp_dropout, name="state_decoder")
        self.z_init_mlp = MLP(self.p.z_units, 256, 1)

        self.dynamics = GRUDynamics(state_units=self.p.z_units)
        self.kalman_meas = KalmanMeasure(
            state_units=self.p.z_units,
            observe_units=self.p.img_enc_units + self.p.state_enc_units,
            name="kalman_meas",
        )
        self.model = self.build_model()

    def get_init_state(self, image_bhw3, state_b3):
        image_code = self.img_encoder(image_bhw3)
        state_code = self.state_encoder(state_b3)
        y = tf.concat([image_code, state_code], axis=-1)

        init_z = self.z_init_mlp(y)
        init_z_log_var = tf.zeros_like(init_z)

        return {"init_state": tf.concat([init_z, init_z_log_var], axis=-1)}

    def build_model(self) -> K.Model:
        img_bhw3 = KL.Input(self.data_p.img_size + (3,), batch_size=self.data_p.batch_size)
        state_b3 = KL.Input((self.p.state_units,), batch_size=self.data_p.batch_size)
        ctrl_mask_bt = KL.Input((None,), batch_size=self.data_p.batch_size, dtype=tf.bool)
        ctrl_inputs_bt2 = KL.Input((None, 2), batch_size=self.data_p.batch_size)
        init_state = KL.Input((self.p.z_units*2,), batch_size=self.data_p.batch_size)

        # process model
        num_steps = tf.shape(ctrl_mask_bt)[1]
        z, z_log_var = self.dynamics((ctrl_inputs_bt2, init_state), mask=ctrl_mask_bt)

        # observation
        img_code = self.img_encoder(img_bhw3)
        state_code = self.state_encoder(state_b3)
        img_recon = self.img_decoder(img_code)
        state_recon = self.state_decoder(state_code)
        y = KL.Concatenate()([img_code, state_code])

        corrected = self.kalman_meas((z, z_log_var, y))

        return K.Model(
            inputs={
                "image_bhw3": img_bhw3,
                "state_b3": state_b3,
                "ctrl_mask_bt": ctrl_mask_bt,
                "ctrl_inputs_bt2": ctrl_inputs_bt2,
                "init_state": init_state,
            },
            outputs={
                "z_proc": z,
                "z_log_var_proc": z_log_var,
                "y": y,
                "y_proc": corrected["y_proc"],
                "Py_proc": corrected["Py_proc"],
                "z_meas": corrected["z_meas"],
                "z_log_var_meas": corrected["z_log_var_meas"],
                # auxiliary outputs
                "img_recon": img_recon,
                "state_recon": state_recon,
                "init_z": self.z_init_mlp(y),
            },
        )

if __name__ == "__main__":
    dpnet = DPNet()
    dpnet.model.summary()
