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


def callback(config, level):
    global h_low, s_low, v_low, h_high, s_high, v_high
    h_low, s_low, v_low = config.H_low, config.S_low, config.V_low
    h_high, s_high, v_high = config.H_high, config.S_high, config.V_high
    return config


def img_cb(msg):

    # hsv_channels = []
    img = br.imgmsg_to_cv2(msg)
    # print(img.shape)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([h_low, s_low, v_low], np.uint8)
    upper_white = np.array([h_high, s_high, v_high], np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)


# ------------------------------------------------------------------------------------------------------------------
    # Split the stereo image into left and right images
    # Hanna's attempt to merge Left and Right Image into Panorama

    # height, width, _ = img.shape
    # left_image = img[:, :width // 2, :]
    # right_image = img[:, width // 2:, :]

    # stitcher = cv2.createStitcher()  # Create a stitcher object
    # result, img = stitcher.stitch([left_image, right_image])

    # pan_pub.publish(br.cv2_to_imgmsg(img, "bgr8"))

# ------------------------------------------------------------------------------------------------------------------

    # window = np.zeros(img.shape[:2], dtype="uint8")

    # #Extract rectangles
    # er = 3
    # kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (er,er))
    # rect_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernel_rect, iterations=2)

    # #Remove sprays
    # kernel = np.ones((3,3),np.uint8)
    # mask_lanes = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # #Find Hough circles
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 5)
    # rows = gray.shape[0]
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 64,
    #                            param1=200, param2=20,
    #                            minRadius=20, maxRadius=30)

    # # print(circles)
    # # print(mask.shape)

    # #Create a mask of squares of same radius as that of detected circles
    # square_mask = 255-0*mask_open.copy()
    # #Draw circles
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         center = (i[0], i[1])
    #         #Start defining mask
    #         square_mask[i[1]-2*i[2]:i[1]+2*i[2],i[0]-2*i[2]:i[0]+2*i[2]] = 0
    #         # circle center
    #         cv2.circle(img, center, 1, (0, 100, 100), 3)
    #         # circle outline
    #         radius = i[2]
    #         cv2.circle(img, center, radius, (255, 0, 255), 3)

    # cv2.imshow("grayscale image",gray)
    # cv2.imshow("detected circles", img)
    # cv2.imshow("square mask",square_mask)
    # # print(square_mask)

    # #Bitwise_AND the square mask and lanes extracted
    # mask_lanes = cv2.bitwise_and(mask_open,square_mask)

    h = len(mask)
    w = len(mask[1])

    # Clear top f fraction
    f = 0.2
    mask[0:int(np.ceil(f * h)), 0:w] = 0
    # Set mask on robot body
    xs = int(np.ceil(0.4 * w))
    xe = int(np.ceil(0.6 * w))
    ys = int(np.ceil(0.8 * h))
    mask[ys:h, xs:xe] = 0

    # Hough lines to remove spray
    mask_lanes = mask.copy()

    # Set completely to zero
    mask_lanes[0:h, 0:w] = 0
    try:
        hough = cv2.HoughLinesP(image=mask, rho=1, theta=10 * (np.pi) / 180, threshold=50, minLineLength=50, maxLineGap=100)
        for points in hough:
            x0, y0, x1, y1 = points[0]
            cv2.line(mask_lanes, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=2)

            # mask_lanes[y0,x0] = 255
            # mask_lanes[y1,x1] = 255

        # Hough transforms for finer lane extraction

        # Show spray removed lane extraction
        cv2.imshow('Spray removed', mask_lanes)

        # stacked_img = np.stack((mask_lanes,) * 3, axis=-1)

        # # cv2.split(mask, hsv_channels)

        # # Publish image to \lane_img
        # pub.publish(br.cv2_to_imgmsg(stacked_img, "bgr8"))

        # # print(mask.shape)
        # cv2.imshow("image", img)
        # cv2.imshow("HSVed", mask)

        # cv2.waitKey(1)
    except:
        print("no lanes seen")
        mask_lanes[0:h, 0:w] = 0
        # cv2.destroyWindow('Spray removed')

    stacked_img = np.stack((mask_lanes,) * 3, axis=-1)
    pub.publish(br.cv2_to_imgmsg(stacked_img, "bgr8"))

    cv2.imshow("image", img)
    cv2.imshow("HSVed", mask)

    cv2.waitKey(1)


if __name__ == "__main__":

    rospy.init_node("hsv_thresh", anonymous=False)
    pub = rospy.Publisher("/lane_img", Image, queue_size=10)
    #pan_pub = rospy.Publisher("/pan_image", Image, queue_size=10)
    cfg = Server(HSVConfig, callback)
    rospy.Subscriber("/zed2i/zed_node/left/image_rect_color", Image, img_cb)
    #rospy.Subscriber("/zed2i/zed_node/stereo_raw/image_raw_color", Image, img_cb)
    # rospy.Subscriber("/camera_2d/image_raw",Image,img_cb)
    rospy.spin()

