#!/usr/bin/env python

 

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

 

class StereoImageSplitter:
    def __init__(self):
        # Initialize the node
        rospy.init_node('stereo_image_splitter', anonymous=True)

 

        # Create a CV bridge
        self.bridge = CvBridge()

 

        # Subscribe to the combined stereo image
        self.stereo_sub = rospy.Subscriber("/stereo_image", Image, self.callback)

 

        # Publishers for the left and right images
        self.left_pub = rospy.Publisher("/left_image", Image, queue_size=10)
        self.right_pub = rospy.Publisher("/right_image", Image, queue_size=10)

 

    def callback(self, data):
        try:
            # Convert the ROS image message to OpenCV format
            stereo_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

 

            # Split the stereo image into left and right images
            height, width, _ = stereo_image.shape
            left_image = stereo_image[:, :width//2, :]
            right_image = stereo_image[:, width//2:, :]

 

            # Publish the separated left and right images
            self.left_pub.publish(self.bridge.cv2_to_imgmsg(left_image, "bgr8"))
            self.right_pub.publish(self.bridge.cv2_to_imgmsg(right_image, "bgr8"))

 

        except CvBridgeError as e:
            print(e)

 

if __name__ == '__main__':
    splitter = StereoImageSplitter()
    rospy.spin()


stitcher = cv2.createStitcher()  # Create a stitcher object
            result, panorama = stitcher.stitch([self.left_image, self.right_image])
            
            if result == cv2.Stitcher_OK:
                # Publish the panoramic image
                panorama_msg = self.bridge.cv2_to_imgmsg(panorama, "bgr8")
                self.panorama_pub.publish(panorama_msg)
