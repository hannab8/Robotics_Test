#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class StereoImagePanorama:
    def __init__(self):
        # Initialize the node
        rospy.init_node('stereo_image_panorama', anonymous=True)

        # Create a CV bridge
        self.bridge = CvBridge()

        # Subscribe to the left and right images
        self.left_sub = rospy.Subscriber("/left_image", Image, self.left_callback)
        self.right_sub = rospy.Subscriber("/right_image", Image, self.right_callback)

        # Initialize variables to hold left and right images
        self.left_image = None
        self.right_image = None

    def left_callback(self, data):
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.create_panorama()

        except CvBridgeError as e:
            print(e)

    def right_callback(self, data):
        try:
            self.right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.create_panorama()

        except CvBridgeError as e:
            print(e)

    def create_panorama(self):
        # Check if both left and right images are available
        if self.left_image is not None and self.right_image is not None:
            # Stitch the left and right images to create a panoramic image
            stitcher = cv2.createStitcher()  # Create a stitcher object
            result, panorama = stitcher.stitch([self.left_image, self.right_image])
            
            if result == cv2.Stitcher_OK:
                # Publish the panoramic image
                panorama_msg = self.bridge.cv2_to_imgmsg(panorama, "bgr8")
                self.panorama_pub.publish(panorama_msg)
            else:
                print("Stitching failed!")

    def run(self):
        # Publisher for the panoramic image
        self.panorama_pub = rospy.Publisher("/panoramic_image", Image, queue_size=10)
        rospy.spin()

if __name__ == '__main__':
    panorama_creator = StereoImagePanorama()
    panorama_creator.run()
