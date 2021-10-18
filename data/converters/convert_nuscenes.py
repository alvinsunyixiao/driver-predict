import argparse
import io
import os
import typing

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transform
from nuscenes.nuscenes import NuScenes

from utils.tf_utils import tensor_to_feature, bytes_to_feature

class NuScenesConverter:
    def __init__(self):
        self.args = self._parse_args()
        self.nusc = NuScenes("v1.0-trainval",
                             dataroot=self.args.data_root, verbose=self.args.verbose)

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--verbose", action="store_true",
                            help="verbose output")
        parser.add_argument("-o", "--output-dir", type=str, required=True,
                            help="directory to store the output tfrecords")
        parser.add_argument("--data-root", type=str, default="/mnt/data/nuScenes",
                            help="root directory to the nuScenes datastet")
        return parser.parse_args()

    def _parse_sample_data(self, sample_data: dict, dt: int) -> str:
        img_path = os.path.join(self.args.data_root, sample_data["filename"])
        with open(img_path, 'rb') as f:
            img_data = f.read()

        ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
        calibrated_sensor = self.nusc.get(
                "calibrated_sensor", sample_data["calibrated_sensor_token"])

        assert sample_data["timestamp"] == ego_pose["timestamp"], \
            "Timestamp mismatch between sample_data and ego_pose"

        world_t_ego = tf.convert_to_tensor(ego_pose["translation"], dtype=tf.float32)
        world_R_ego = tfg_transform.rotation_matrix_3d.from_quaternion(ego_pose["rotation"])
        ego_t_sensor = tf.convert_to_tensor(calibrated_sensor["translation"], dtype=tf.float32)

        world_t_sensor = tf.linalg.matvec(world_R_ego, ego_t_sensor) + world_t_ego

        feature = {
            "image_jpg":        bytes_to_feature(img_data),
            "world_t_ego":      tensor_to_feature(world_t_ego),
            "world_t_sensor":   tensor_to_feature(world_t_sensor),
            "dt":               tensor_to_feature(tf.constant(dt / 1e6)), #[us] -> [s]
            "intrinsics":       tensor_to_feature(calibrated_sensor["camera_intrinsic"]),
            "channel":          bytes_to_feature(sample_data["channel"]),
        }

        features = tf.train.Features(feature=feature)
        example_proto = tf.train.Example(features=features)

        return example_proto.SerializeToString()

    def _convert_scene(self, scene: dict):
        first_sample = self.nusc.get("sample", scene["first_sample_token"])
        cameras = [k for k in first_sample["data"].keys() if k.startswith("CAM")]
        data_dict = {cam: self.nusc.get("sample_data", first_sample["data"][cam])
                     for cam in cameras}

        tfrecord_file = os.path.join(self.args.output_dir,
                                     scene["name"] + ".tfrecord")
        prev_sample = None

        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            while data_dict:
                min_t_cam = min(data_dict, key=lambda c: data_dict[c]["timestamp"])
                min_t_sample = data_dict[min_t_cam]

                if prev_sample:
                    writer.write(self._parse_sample_data(
                        prev_sample, min_t_sample["timestamp"] - prev_sample["timestamp"]))
                prev_sample = min_t_sample
                if data_dict[min_t_cam]["next"]:
                    data_dict[min_t_cam] = self.nusc.get("sample_data", data_dict[min_t_cam]["next"])
                else:
                    data_dict.pop(min_t_cam)


    def main(self):
        self._convert_scene(self.nusc.scene[0])


if __name__ == "__main__":
    NuScenesConverter().main()
