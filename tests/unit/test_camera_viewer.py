#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple camera image viewer for testing camera topics.
Compatible with MuJoCo, ManiSkill, and other simulators.

Usage:
    # Show all cameras (default)
    python3 tests/unit/test_camera_viewer.py

    # Show specific cameras
    python3 tests/unit/test_camera_viewer.py --cameras head_rgbd left_wrist_rgbd
"""

import os
import sys
import cv2
import time
import threading
from cv_bridge import CvBridge

ros_version = os.getenv('ROS_VERSION')

if ros_version == '1':
    import rospy
    from sensor_msgs.msg import Image
elif ros_version == '2':
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import Image
else:
    raise RuntimeError("ROS_VERSION not set. Please source env.sh first.")


class CameraViewer:
    def __init__(self, camera_names=None):
        self.bridge = CvBridge()
        self.camera_images = {}

        # Default: all cameras
        if camera_names is None:
            self.camera_names = ['head_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd', 'astribot_global_camera']
        else:
            self.camera_names = camera_names

        self.last_update = {}
        self.update_lock = threading.Lock()

        if ros_version == '1':
            rospy.init_node('camera_viewer', anonymous=True)
            self.setup_ros1_subscribers()
        elif ros_version == '2':
            rclpy.init()
            self.node = Node('camera_viewer')
            self.setup_ros2_subscribers()
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)
            self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.spin_thread.start()

    def setup_ros1_subscribers(self):
        for camera_name in self.camera_names:
            topic = f'astribot_whole_body/camera/{camera_name}/image_raw'
            rospy.Subscriber(topic, Image, self.image_callback, callback_args=camera_name, queue_size=1, buff_size=2**24)

    def setup_ros2_subscribers(self):
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        for camera_name in self.camera_names:
            topic = f'astribot_whole_body/camera/{camera_name}/image_raw'
            self.node.create_subscription(Image, topic,
                                         lambda msg, name=camera_name: self.image_callback(msg, name), qos)

    def image_callback(self, msg, camera_name):
        current_time = time.time()
        if camera_name in self.last_update and (current_time - self.last_update[camera_name]) < 0.033:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.update_lock:
                self.camera_images[camera_name] = cv_image
                self.last_update[camera_name] = current_time
        except Exception as e:
            print(f"Error converting image from {camera_name}: {e}")

    def run(self):
        print("Camera Viewer Started")
        print("Press 'q' to quit")
        print(f"Listening to topics: {self.camera_names}")

        while True:
            with self.update_lock:
                for camera_name, image in self.camera_images.items():
                    if image is not None:
                        cv2.imshow(camera_name, image)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        if ros_version == '2':
            self.executor.shutdown()
            self.node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Camera image viewer')
    parser.add_argument('--cameras', nargs='+', help='Specific cameras to show')
    args = parser.parse_args()

    camera_names = args.cameras if args.cameras else None

    viewer = CameraViewer(camera_names)
    viewer.run()
