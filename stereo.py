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
    # cv2.imshow("left image", left_image)
    # cv2.waitKey(1)
    if left_image is not None and right_image is not None:
        stitch_images()
        # print("Both filled")
        # cv2.imshow("right image", img)
        # cv2.waitKey(1)

def right_img_cb(msg):
    global right_image
    right_image = br.imgmsg_to_cv2(msg)
    # cv2.imshow ("left",left_image)
    # cv2.waitKey(1)

def img_cb(msg): 
    global left_image, right_image
    
    if left_image is not None and right_image is not None:
        img = stitch_images()
        cv2.imshow("image", img)
        cv2.waitKey(1)
    #img = br.imgmsg_to_cv2(msg)

    return
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([h_low, s_low, v_low], np.uint8)
    upper_white = np.array([h_high, s_high, v_high], np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)

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
            
        # Show spray removed lane extraction
        cv2.imshow('Spray removed', mask_lanes)

    except:
        print("no lanes seen")
        mask_lanes[0:h, 0:w] = 0
        # cv2.destroyWindow('Spray removed')

    stacked_img = np.stack((mask_lanes,) * 3, axis=-1)
    pub.publish(br.cv2_to_imgmsg(stacked_img, "bgr8"))

    cv2.imshow("image", img)
    cv2.imshow("HSVed", mask)

    cv2.waitKey(1)

def stitch_images():
    global left_image, right_image
    
    img1 = cv2.imread('/home/aasha/karina_ws/src/Karina/igvc_perception/hsv_thresh/src/left_image.jpg')
    img2 = cv2.imread('/home/aasha/karina_ws/src/Karina/igvc_perception/hsv_thresh/src/right_image.jpg')
    imgs = [img1,img2]
    # Use OpenCV's stitcher to combine images
    
    print("Stitching Images...")
    #imgs = [left_image,right_image]
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    (status, stitched) = stitcher.stitch(imgs)
    
    print("After Stitch...")
    if status == cv2.Stitcher_OK:
        print("STITCHED PROPERLY")
        cv2.imshow("image", stitched)
        cv2.waitKey(1)
        return stitched
    else:
        rospy.loginfo("Image stitching failed.")
        return None
    

if __name__ == "__main__":

    rospy.init_node("hsv_thresh", anonymous=False)
    pub = rospy.Publisher("/lane_img", Image, queue_size=10)
    cfg = Server(HSVConfig, callback)
    rospy.Subscriber("/zed2i/zed_node/left/image_rect_color", Image, left_img_cb)
    rospy.Subscriber("/zed2i/zed_node/right/image_rect_color", Image, right_img_cb)
    rospy.spin()

# THIS HAS PROVEN TO WORK

import cv2
import numpy as np
img1 = cv2.imread('/home/aasha/karina_ws/src/Karina/igvc_perception/hsv_thresh/src/left_image.jpg')
img2 = cv2.imread('/home/aasha/karina_ws/src/Karina/igvc_perception/hsv_thresh/src/right_image.jpg')

imgs = [img1,img2]
print("img1 size:",img1.shape)
print("img2 size:",img2.shape)

cv2.imshow("left",img1)
cv2.imshow("right",img2)

print("Creating stitcher...")
stitcher=cv2.Stitcher.create(cv2.Stitcher_SCANS)
#stitcher.setPanoConfidenceThresh(0.0) 
(status,output)=stitcher.stitch(imgs) 

print("Checking stitcher...")
if status != cv2.STITCHER_OK: 
  # checking if the stitching procedure is successful 
  # .stitch() function returns a true value if stitching is  
  # done successfully 
    print("stitching ain't successful") 
else:  
    print('Your Panorama is ready!!!') 
    
  
# final output 
cv2.imshow('final result',output) 
  
cv2.waitKey(0)
