from utils.params import ParamDict

from dnn.data import NuScenesDataset
from dnn.model import DPNet
from dnn.train import Trainer

PARAMS = ParamDict(
    trainer = Trainer.DEFAULT_PARAMS,
    model = DPNet.DEFAULT_PARAMS,
    data = NuScenesDataset.DEFAULT_PARAMS,
)
