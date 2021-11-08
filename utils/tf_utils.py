import typing as T
import numpy as np
import tensorflow as tf
import tensorflow.python.framework.ops as tfops

def set_tf_memory_growth(mode: bool = True):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, mode)

def bytes_to_feature(value: T.Union[str, bytes, tfops.EagerTensor]) -> tf.train.Feature:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if type(value) == str:
        value = value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tensor_to_feature(tensor: T.Union[np.ndarray, tfops.EagerTensor]) -> tf.train.Feature:
    return bytes_to_feature(tf.io.serialize_tensor(tensor))

def const_to_feature(data: T.List[T.Any], dtype=None):
    return tensor_to_feature(tf.constant(data, dtype=dtype))
