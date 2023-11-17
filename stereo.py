#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from dynamic_reconfigure.server import Server
from hsv_thresh.cfg import HSVConfig

br = CvBridge()

# Initialize HSV mask parameters
h_low, s_low, v_low = 0, 0, 177
h_high, s_high, v_high = 124, 15, 255

# Global variables to store left and right images
left_image = None
right_image = None

def callback(config, level):
    global h_low, s_low, v_low, h_high, s_high, v_high
    h_low, s_low, v_low = config.H_low, config.S_low, config.V_low
    h_high, s_high, v_high = config.H_high, config.S_high, config.V_high
    return config

def left_img_cb(msg):
    global left_image
    left_image = br.imgmsg_to_cv2(msg)

def right_img_cb(msg):
    global right_image
    right_image = br.imgmsg_to_cv2(msg)

def process_image(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower_white = np.array([h_low, s_low, v_low], np.uint8)
    upper_white = np.array([h_high, s_high, v_high], np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)

    h, w = mask.shape[:2]

    # Clear top fraction of the image
    f = 0.2
    mask[0:int(np.ceil(f * h)), 0:w] = 0

    # Set mask on robot body
    xs, xe = int(np.ceil(0.3 * w)), int(np.ceil(0.7 * w))
    ys = int(np.ceil(0.5 * h))
    mask[ys:h, xs:xe] = 0

    # Hough lines to remove spray
    mask_lanes = mask.copy()
    mask_lanes[0:h, 0:w] = 0

    hough = cv2.HoughLinesP(image=mask, rho=1, theta=10 * (np.pi) / 180, threshold=50, minLineLength=50, maxLineGap=100)
    if hough is not None:
        for points in hough:
            x0, y0, x1, y1 = points[0]
            cv2.line(mask_lanes, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=2)

    stacked_img = np.stack((mask_lanes,) * 3, axis=-1)
    return stacked_img

def stitch_images(left_img, right_img):
    # Use OpenCV's stitcher to combine images
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    (status, stitched) = stitcher.stitch([left_img, right_img])
    
    if status == cv2.Stitcher_OK:
        return stitched
    else:
        rospy.loginfo("Image stitching failed.")
        return None

def publish_panoramic():
    global left_image, right_image

    if left_image is not None and right_image is not None:
        left_processed = process_image(left_image)
        right_processed = process_image(right_image)

        panoramic_img = stitch_images(left_processed, right_processed)
        if panoramic_img is not None:
            pub.publish(br.cv2_to_imgmsg(panoramic_img, "bgr8"))

if __name__ == "__main__":
    rospy.init_node("hsv_thresh", anonymous=False)
    pub = rospy.Publisher("/panoramic_lane_img", Image, queue_size=10)
    cfg = Server(HSVConfig, callback)
    rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, left_img_cb)
    rospy.Subscriber("/zed2/zed_node/right/image_rect_color", Image, right_img_cb)

    rate = rospy.Rate(10)  # Adjust the rate as needed
    while not rospy.is_shutdown():
        publish_panoramic()
        rate.sleep()
