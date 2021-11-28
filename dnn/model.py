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


class DPNet:

    DEFAULT_PARAMS = ParamDict(
        img_enc_units = 128,
        img_units = 128,
        state_enc_units = 32,
        state_units = 3,
        control_units = 2,
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
            self.p.state_enc_units, 32, 2, self.p.mlp_dropout, name="state_encoder")
        self.state_decoder = MLP(
            self.p.state_units, 32, 2, self.p.mlp_dropout, name="state_decoder")

        self.dynamics = KL.GRU(units=self.p.img_enc_units + self.p.state_enc_units)
        self.model = self.build_model()

    def encode(self, img_bhw3, state_b3):
        img_code = self.img_encoder(img_bhw3)
        state_code = self.state_encoder(state_b3)
        y = KL.Concatenate()([img_code, state_code])

        return y, img_code, state_code

    def build_model(self) -> K.Model:
        img_bhw3 = KL.Input(self.data_p.img_size + (3,), batch_size=self.data_p.batch_size)
        state_b3 = KL.Input((self.p.state_units,), batch_size=self.data_p.batch_size)
        ctrl_mask_bt = KL.Input((None,), batch_size=self.data_p.batch_size, dtype=tf.bool)
        ctrl_inputs_bt2 = KL.Input((None, 2), batch_size=self.data_p.batch_size)
        init_state = KL.Input((self.p.img_enc_units + self.p.state_enc_units,),
                              batch_size=self.data_p.batch_size)

        # process model
        num_steps = tf.shape(ctrl_mask_bt)[1]
        z = self.dynamics(ctrl_inputs_bt2, initial_state=init_state, mask=ctrl_mask_bt)

        # encoder-decoder reconstruction
        y, img_code, state_code = self.encode(img_bhw3, state_b3)
        img_recon = self.img_decoder(img_code)
        state_recon = self.state_decoder(state_code)

        return K.Model(
            inputs={
                "image_bhw3": img_bhw3,
                "state_b3": state_b3,
                "ctrl_mask_bt": ctrl_mask_bt,
                "ctrl_inputs_bt2": ctrl_inputs_bt2,
                "init_state": init_state,
            },
            outputs={
                "z": z,
                "y": y,
                # auxiliary outputs
                "img_recon": img_recon,
                "state_recon": state_recon,
            },
        )

if __name__ == "__main__":
    dpnet = DPNet()
    dpnet.model.summary()
