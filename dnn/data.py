import os
import tensorflow as tf
import typing as T
import tensorflow_graphics.geometry.transformation as tfg_transform

from utils.params import ParamDict

class NuScenesDataset:

    DEFAULT_PARAMS = ParamDict(
        seq_len = 16,
        max_scene_len = 189,
        batch_size = 32,
        data_root = "/data/tfrecords/nuScenes/front_cam",
        img_size = (144, 256),
        img_noise = 5.0,
        prefetch = 16,
    )

    def __init__(self, params: ParamDict = DEFAULT_PARAMS):
        self.p = params
        self.dataset = self.build_dataset()

    def build_dataset(self):
        file_dataset = tf.data.Dataset.list_files(os.path.join(self.p.data_root, "*.tfrecord"))
        file_dataset = file_dataset.repeat().shuffle(100)

        interleaved_dataset = file_dataset.interleave(self._interleave_func,
            cycle_length=self.p.batch_size, num_parallel_calls=8, deterministic=True)

        batched_dataset = interleaved_dataset.padded_batch(self.p.batch_size, drop_remainder=True)

        return batched_dataset.prefetch(self.p.prefetch)

    def _interleave_func(self, file):
        raw_dataset = tf.data.TFRecordDataset(file)
        num_skip = tf.random.uniform(
                [], minval=0, maxval=self.p.max_scene_len - self.p.seq_len, dtype=tf.int64)
        skipped_dataset = raw_dataset.skip(num_skip)
        taked_dataset = skipped_dataset.take(self.p.seq_len)
        return taked_dataset.map(lambda x: self._preprocess(self._parse_func(x)))

    def _parse_func(self, example_proto):
        feature_desc = {
            "image_jpg":        tf.io.FixedLenFeature([], tf.string),
            "world_t_ego":      tf.io.FixedLenFeature([], tf.string),
            "world_R_ego":      tf.io.FixedLenFeature([], tf.string),
            "can_linear_accel": tf.io.FixedLenFeature([], tf.string),
            "can_angular_vel":  tf.io.FixedLenFeature([], tf.string),
        }

        return tf.io.parse_single_example(example_proto, feature_desc)

    def _preprocess(self, feature):
        image_hw3 = tf.io.decode_image(feature["image_jpg"])
        world_t_ego_3 = tf.io.parse_tensor(feature["world_t_ego"], out_type=tf.float32)
        world_R_ego_4 = tf.io.parse_tensor(feature["world_R_ego"], out_type=tf.float32)
        can_lin_accel_t3 = tf.io.parse_tensor(feature["can_linear_accel"], out_type=tf.float32)
        can_ang_vel_t3 = tf.io.parse_tensor(feature["can_angular_vel"], out_type=tf.float32)

        image_hw3.set_shape([None, None, 3])
        world_t_ego_3.set_shape([3])
        world_R_ego_4.set_shape([4])
        can_lin_accel_t3.set_shape([None, 3])
        can_ang_vel_t3.set_shape([None, 3])

        world_Rxyz_ego = tfg_transform.euler.from_quaternion(world_R_ego_4)

        image_hw3 = tf.image.resize(image_hw3, self.p.img_size)
        image_hw3 += tf.random.normal(shape=tf.shape(image_hw3), stddev=self.p.img_noise)
        image_hw3 = image_hw3 / 127.5 - 1

        state_b3 = tf.concat([world_t_ego_3[:-1], world_Rxyz_ego[-1, None]], axis=0)
        ctrl_inputs_t2 = tf.concat(
            [can_lin_accel_t3[:, 0, None], can_ang_vel_t3[:, -1, None]], axis=-1)

        return {
            "image_bhw3": image_hw3,
            "state_b3": state_b3,
            "ctrl_inputs_bt2": ctrl_inputs_t2,
            "ctrl_mask_bt": tf.ones(tf.shape(can_lin_accel_t3)[0]),
        }

