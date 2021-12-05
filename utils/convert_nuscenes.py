import argparse
import io
import os
import random
import typing as T

import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transform
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from tqdm import tqdm

from utils.tf_utils import tensor_to_feature, bytes_to_feature, const_to_feature
from utils.tf_utils import set_tf_memory_growth

class NuScenesConverter:
    def __init__(self):
        self.args = self._parse_args()
        self.nusc = NuScenes("v1.0-trainval",
                             dataroot=self.args.data_root, verbose=self.args.verbose)
        self.nusc_can = NuScenesCanBus(dataroot=self.args.data_root)
        self._scene_black_list = set([
            "scene-{:04d}".format(sc) for sc in self.nusc_can.can_blacklist
        ])

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--verbose", action="store_true",
                            help="verbose output")
        parser.add_argument("--front-cam-only", action="store_true",
                            help="convert front camera only")
        parser.add_argument("-o", "--output-dir", type=str, required=True,
                            help="directory to store the output tfrecords")
        parser.add_argument("--data-root", type=str, default="/mnt/data/nuScenes",
                            help="root directory to the nuScenes datastet")
        parser.add_argument("--split", type=float, default=0.9,
                            help="the train percentage of a train-val split [default to 0.9]")
        return parser.parse_args()

    def _parse_sample_data(self,
            sample_data: T.Dict[str, T.Any], ctrls: T.List[T.Dict[str, T.Any]]) -> str:
        img_path = os.path.join(self.args.data_root, sample_data["filename"])
        with open(img_path, 'rb') as f:
            img_data = f.read()

        ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
        calibrated_sensor = self.nusc.get(
                "calibrated_sensor", sample_data["calibrated_sensor_token"])

        assert sample_data["timestamp"] == ego_pose["timestamp"], \
            "Timestamp mismatch between sample_data and ego_pose"

        world_t_ego = tf.convert_to_tensor(ego_pose["translation"], dtype=tf.float32)
        # [w, x, y, z] -> [x, y, z, w]
        world_R_ego = np.array(ego_pose["rotation"], dtype=np.float32)[[1,2,3,0]]
        world_R_ego_mat = tfg_transform.rotation_matrix_3d.from_quaternion(world_R_ego)
        ego_t_sensor = tf.convert_to_tensor(calibrated_sensor["translation"], dtype=tf.float32)

        world_t_sensor = tf.linalg.matvec(world_R_ego_mat, ego_t_sensor) + world_t_ego

        feature = {
            "image_jpg":        bytes_to_feature(img_data),
            "world_t_ego":      tensor_to_feature(world_t_ego),
            "world_R_ego":      tensor_to_feature(world_R_ego),
            "world_t_sensor":   tensor_to_feature(world_t_sensor),
            "timestamp":        const_to_feature(sample_data["timestamp"], dtype=tf.uint64),
            "intrinsics":       tensor_to_feature(calibrated_sensor["camera_intrinsic"]),
            "channel":          bytes_to_feature(sample_data["channel"]),
        }
        feature.update(self._convert_controls(ctrls))

        features = tf.train.Features(feature=feature)
        example_proto = tf.train.Example(features=features)

        return example_proto.SerializeToString()

    def _convert_controls(self, ctrls: T.List[T.Dict[str, T.Any]]):
        return {
            "can_position":     const_to_feature([ctrl["pos"] for ctrl in ctrls]),
            "can_orientation":  const_to_feature([ctrl["orientation"] for ctrl in ctrls]),
            "can_linear_vel":   const_to_feature([ctrl["vel"] for ctrl in ctrls]),
            "can_linear_accel": const_to_feature([ctrl["accel"] for ctrl in ctrls]),
            "can_angular_vel":  const_to_feature([ctrl["rotation_rate"] for ctrl in ctrls]),
            "can_timestamp":    const_to_feature([ctrl["utime"] for ctrl in ctrls], dtype=tf.uint64),
        }

    def _convert_scene(self, scene: T.Dict[str, T.Any], tag: str):
        first_sample = self.nusc.get("sample", scene["first_sample_token"])
        if self.args.front_cam_only:
            cameras = ["CAM_FRONT"]
        else:
            cameras = [k for k in first_sample["data"].keys() if k.startswith("CAM")]
        data_dict = {cam: self.nusc.get("sample_data", first_sample["data"][cam])
                     for cam in cameras}

        ctrl_inp = self.nusc_can.get_messages(scene["name"], "pose")
        ctrl_idx = 0

        tfrecord_root = os.path.join(self.args.output_dir, tag)
        os.makedirs(tfrecord_root, exist_ok=True)
        tfrecord_file = os.path.join(tfrecord_root,
                                     scene["name"] + ".tfrecord")
        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            while data_dict:
                # get min timestamp camera
                min_t_cam = min(data_dict, key=lambda c: data_dict[c]["timestamp"])
                min_t_sample = data_dict[min_t_cam]

                # gather all control inputs before the measurement time
                ctrls = []
                while ctrl_idx < len(ctrl_inp) and \
                      ctrl_inp[ctrl_idx]["utime"] < min_t_sample["timestamp"]:
                    ctrls.append(ctrl_inp[ctrl_idx])
                    ctrl_idx += 1

                # discard samples with no associated control inputs
                if len(ctrls) > 0:
                    writer.write(self._parse_sample_data(min_t_sample, ctrls))

                # traverse through the linked list
                if data_dict[min_t_cam]["next"]:
                    data_dict[min_t_cam] = self.nusc.get("sample_data", data_dict[min_t_cam]["next"])
                else:
                    data_dict.pop(min_t_cam)

    def main(self):
        scenes = [scene for scene in self.nusc.scene if scene["name"] not in self._scene_black_list]
        random.shuffle(scenes)
        num_train = int(len(scenes) * self.args.split)

        train_scenes = scenes[:num_train]
        val_scenes = scenes[num_train:]

        for scene in tqdm(train_scenes):
            self._convert_scene(scene, "train")

        for scene in tqdm(val_scenes):
            self._convert_scene(scene, "validation")

if __name__ == "__main__":
    set_tf_memory_growth()
    NuScenesConverter().main()
