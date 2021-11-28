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
                 has_bn: bool = False,
                 dropout: T.Optional[float] = None,
                 last_activation: T.Optional[str] = None,
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
            ))

            # batchnorm
            if has_bn:
                layers.append(KL.BatchNormalization())

            # activation
            layers.append(KL.ReLU())

            # dropout
            if dropout is not None:
                layers.append(KL.Dropout(dropout))

        layers.append(KL.Dense(self.output_units, activation=last_activation))

        self.f = K.Sequential(layers)

    def call(self, x, training=None):
        return self.f(x, training=training)

class ImageEncoder(KL.Layer):
    def __init__(self, units: int, dropout: T.Optional[float] = None, **kwargs):
        super(ImageEncoder, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout

        self.cnn = K.Sequential([
            KL.Conv2D(24, 7, 2, "same", activation="relu"),
            KL.Conv2D(48, 5, 2, "same", activation="relu"),
            KL.Conv2D(96, 5, 2, "same", activation="relu"),
            KL.Conv2D(144, 3, 2, "same", activation="relu"),
            KL.Conv2D(144, 3, 1, "valid", activation="relu"),
            KL.Conv2D(192, 3, 1, "valid", activation="relu"),
            KL.Conv2D(192, 3, 1, "valid", activation="relu"),
            KL.Conv2D(256, 3, 1, "valid", activation="relu"),
        ])
        self.mlp = MLP(units, hidden_units=1024, num_layers=1, dropout=dropout)

    def call(self, x, training=None):
        x = self.cnn(x, training=training)
        x = KL.Flatten()(x)
        x = self.mlp(x, training=training)

        return x

class ImageDecoder(KL.Layer):
    def __init__(self, feat_shape: T.Tuple[int], dropout: T.Optional[float] = None, **kwargs):
        super(ImageDecoder, self).__init__(**kwargs)
        self.feat_shape = feat_shape
        self.dropout = dropout

        self.mlp = MLP(math.prod(feat_shape), hidden_units=1024, num_layers=1, dropout=dropout, last_activation="relu")
        self.reshape = KL.Reshape(feat_shape)
        self.decnn = K.Sequential([
            KL.Conv2DTranspose(192, 3, 1, "valid", activation="relu"),
            KL.Conv2DTranspose(192, 3, 1, "valid", activation="relu"),
            KL.Conv2DTranspose(144, 3, 1, "valid", activation="relu"),
            KL.Conv2DTranspose(144, 3, 1, "valid", activation="relu"),
            KL.Conv2DTranspose(96, 3, 2, "same", activation="relu"),
            KL.Conv2DTranspose(48, 5, 2, "same", activation="relu"),
            KL.Conv2DTranspose(24, 5, 2, "same", activation="relu"),
            KL.Conv2DTranspose(3, 7, 2, "same"),
        ])

    def call(self, x, training=None):
        x = self.mlp(x, training=training)
        x = self.reshape(x)
        x = self.decnn(x, training=training)

        return x


class LTVDynamics(KL.Layer):

    MLP_HIDDEN_UNITS = 128
    MLP_NUM_LAYERS = 2
    MLP_DROPOUT = None

    def __init__(self,
                 state_units: int,
                 state_history_units: int,
                 control_units: int,
                 num_modes: int,
                 **kwargs):
        super(LTVDynamics, self).__init__(**kwargs)
        self.state_units = state_units
        self.state_history_units = state_history_units
        self.control_units = control_units
        self.num_modes = num_modes
        self.gru = KL.GRUCell(
            units=state_history_units,
        )
        self.alpha = MLP(
            output_units=num_modes,
            hidden_units=self.MLP_HIDDEN_UNITS,
            num_layers=self.MLP_NUM_LAYERS,
            dropout=self.MLP_DROPOUT,
        )

        self.state_size = [
            tf.TensorShape([state_units]),              # z
            tf.TensorShape([state_units, state_units]), # Pz
            tf.TensorShape([state_history_units]),      # z_hist
        ]
        self.output_size = [
            tf.TensorShape([state_units]),              # y
            tf.TensorShape([state_units, state_units]), # Py
        ]

    def build(self, input_shapes):
        A_eye = tf.eye(self.state_units, batch_shape=(self.num_modes,))
        A_eye = tf.reshape(A_eye, (self.num_modes, self.state_units * self.state_units))
        self.As = self.add_weight(
            name="As",
            shape=(self.num_modes, self.state_units * self.state_units),
            trainable=True,
            initializer=K.initializers.constant(A_eye),
            regularizer=K.regularizers.l1_l2(l1=1e-4, l2=1e-4),
        )
        self.Bs = self.add_weight(
            name="Bs",
            shape=(self.num_modes, self.state_units * self.control_units),
            trainable=True,
            initializer=K.initializers.random_normal(stddev=1.0),
            regularizer=K.regularizers.l1(1e-4),
        )

    def call(self, inputs, states, training=None):
        u, Pu = inputs
        z, Pz, z_hist = states

        z_vec = z[..., None]
        u_vec = u[..., None]

        y_vec = z_vec
        Py = Pz + tf.eye(self.state_units)[None]

        _, z_hist = self.gru(inputs=z, states=z_hist)

        alpha = self.alpha(z_hist, training=training)
        alpha = tf.nn.softmax(alpha)

        At = tf.reshape(alpha @ self.As, (-1, self.state_units, self.state_units))
        Bt = tf.reshape(alpha @ self.Bs, (-1, self.state_units, self.control_units))

        z_vec = At @ z_vec + Bt @ u_vec

        Pz = At @ Pz @ tf.linalg.adjoint(At) + \
             Bt @ Pu @ tf.linalg.adjoint(Bt) + \
             tf.eye(self.state_units)[None]
        Pz = Pz + tf.eye(self.state_units)[None]

        Pz = .5 * (Pz + tf.linalg.adjoint(Pz))

        return [y_vec[..., 0], Py], [z_vec[..., 0], Pz, z_hist]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [
            tf.zeros((batch_size, self.state_units)),               # z
            tf.eye(self.state_units, batch_shape=(batch_size,)),    # Pz
            tf.zeros((batch_size, self.state_history_units)),       # z_hist
        ]

class KalmanMeasure(KL.Layer):
    def __init__(self, state_units: int, **kwargs):
        super(KalmanMeasure, self).__init__(**kwargs)
        self.state_units = state_units

    def call(self, inputs):
        r = inputs["y"] - inputs["y_hat"]
        S = inputs["Pz"] + \
            tf.eye(self.state_units)[None]
        K_T = tf.linalg.solve(S, inputs["Pz"])
        K = tf.linalg.adjoint(K_T)

        z_vec = inputs["z"][..., None] + K @ r[..., None]
        Pz = (tf.eye(self.state_units)[None] - K) @ inputs["Pz"]

        # ensure symmetry
        Pz = .5 * (Pz + tf.linalg.adjoint(Pz))

        # post correction measurement stats
        y_vec = z_vec
        Py = Pz + tf.eye(self.state_units)[None]

        return {
            "z": z_vec[..., 0],
            "Pz": Pz,
            "y": y_vec[..., 0],
            "Py": Py
        }

class DPNet:

    DEFAULT_PARAMS = ParamDict(
        img_enc_units = 128,
        state_enc_units = 16,
        state_units = 3,
        control_units = 2,
        z_hist_units = 128,
        num_linear_modes = 8,
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
        print("Image feature shape:", feat_shape)
        self.img_decoder = ImageDecoder(feat_shape[1:], dropout=self.p.mlp_dropout)
        self.state_encoder = MLP(
            self.p.state_enc_units, 32, 4, self.p.mlp_dropout, name="state_encoder")
        self.state_decoder = MLP(
            self.p.state_units, 32 , 4, self.p.mlp_dropout, name="state_decoder")

        self.dynamics = KL.RNN(LTVDynamics(
            state_units=self.p.img_enc_units + self.p.state_enc_units,
            state_history_units=self.p.z_hist_units,
            control_units=self.p.control_units,
            num_modes=self.p.num_linear_modes,
            name="ltv_dynamics",
        ), return_state=True)
        self.kalman_meas = KalmanMeasure(
            state_units=self.p.img_enc_units + self.p.state_enc_units,
            name="kalman_meas",
        )
        self.model = self.build_model()

    def get_init_state(self, image_bhw3, state_b3, training=None):
        image_code = self.img_encoder(image_bhw3, training=training)
        state_code = self.state_encoder(state_b3, training=training)
        y = tf.concat([image_code, state_code], axis=-1)

        states = self.dynamics.cell.get_initial_state(batch_size=self.data_p.batch_size)
        return {
            "init_z": y,
            "init_Pz": states[1],
            "init_z_hist": states[2],
        }

    def build_model(self) -> K.Model:
        img_bhw3 = KL.Input(self.data_p.img_size + (3,), batch_size=self.data_p.batch_size)
        state_b3 = KL.Input((self.p.state_units,), batch_size=self.data_p.batch_size)
        ctrl_mask_bt = KL.Input((None,), batch_size=self.data_p.batch_size, dtype=tf.bool)
        ctrl_inputs_bt2 = KL.Input((None, 2), batch_size=self.data_p.batch_size)
        z_units = self.p.state_enc_units + self.p.img_enc_units
        init_z = KL.Input((z_units,), batch_size=self.data_p.batch_size)
        init_Pz = KL.Input((z_units, z_units),
                           batch_size=self.data_p.batch_size)
        init_z_hist = KL.Input((self.p.z_hist_units,), batch_size=self.data_p.batch_size)

        # process model
        num_steps = tf.shape(ctrl_mask_bt)[1]
        ctrl_inputs_cov_bt22 = tf.zeros((self.data_p.batch_size, num_steps, 2, 2))
        y_hat, Py, z, Pz, z_hist = self.dynamics(
            (ctrl_inputs_bt2, ctrl_inputs_cov_bt22),
            initial_state=[init_z, init_Pz, init_z_hist],
            mask=ctrl_mask_bt)

        # observation
        img_code = self.img_encoder(img_bhw3)
        state_code = self.state_encoder(state_b3)
        img_recon = self.img_decoder(img_code)
        state_recon = self.state_decoder(state_code)
        y = KL.Concatenate()([img_code, state_code])

        corrected_state = self.kalman_meas({
            "y": y,
            "y_hat": y_hat,
            "z": z,
            "Pz": Pz,
        })

        return K.Model(
            inputs={
                "image_bhw3": img_bhw3,
                "state_b3": state_b3,
                "ctrl_mask_bt": ctrl_mask_bt,
                "ctrl_inputs_bt2": ctrl_inputs_bt2,
                "init_z": init_z,
                "init_Pz": init_Pz,
                "init_z_hist": init_z_hist,
            },
            outputs={
                "z_proc": z,
                "Pz_proc": Pz,
                "y": y,
                "y_proc": y_hat,
                "Py_proc": Py,
                "z_meas": corrected_state["z"],
                "Pz_meas": corrected_state["Pz"],
                "y_meas": corrected_state["y"],
                "z_hist": z_hist,
                # auxiliary outputs
                "img_recon": img_recon,
                "state_recon": state_recon,
            },
        )

if __name__ == "__main__":
    dpnet = DPNet()
    dpnet.model.summary()
