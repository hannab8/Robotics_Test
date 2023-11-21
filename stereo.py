# #!/usr/bin/env python3

# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# from dynamic_reconfigure.server import Server
# from hsv_thresh.cfg import HSVConfig

# br = CvBridge()

# # Initialize HSV mask parameters
# h_low, s_low, v_low = 0, 0, 177
# h_high, s_high, v_high = 124, 15, 255

# # Global variables to store left and right images
# left_image = None
# right_image = None

# def callback(config, level):
#     global h_low, s_low, v_low, h_high, s_high, v_high
#     h_low, s_low, v_low = config.H_low, config.S_low, config.V_low
#     h_high, s_high, v_high = config.H_high, config.S_high, config.V_high
#     return config

# def left_img_cb(msg):
#     global left_image
#     left_image = br.imgmsg_to_cv2(msg)
#     # cv2.imshow("left image", left_image)
#     # cv2.waitKey(1)
#     if left_image is not None and right_image is not None:
#         stitch_images()
#         # print("Both filled")
#         # cv2.imshow("right image", img)
#         # cv2.waitKey(1)

# def right_img_cb(msg):
#     global right_image
#     right_image = br.imgmsg_to_cv2(msg)
#     # cv2.imshow ("left",left_image)
#     # cv2.waitKey(1)

# def img_cb(msg): 
#     global left_image, right_image
    
#     if left_image is not None and right_image is not None:
#         img = stitch_images()
#         cv2.imshow("image", img)
#         cv2.waitKey(1)
#     #img = br.imgmsg_to_cv2(msg)

#     return
#     blur = cv2.GaussianBlur(img, (3, 3), 0)
#     hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#     # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     lower_white = np.array([h_low, s_low, v_low], np.uint8)
#     upper_white = np.array([h_high, s_high, v_high], np.uint8)

#     mask = cv2.inRange(hsv, lower_white, upper_white)

#     h = len(mask)
#     w = len(mask[1])

#     # Clear top f fraction
#     f = 0.2
#     mask[0:int(np.ceil(f * h)), 0:w] = 0
#     # Set mask on robot body
#     xs = int(np.ceil(0.4 * w))
#     xe = int(np.ceil(0.6 * w))
#     ys = int(np.ceil(0.8 * h))
#     mask[ys:h, xs:xe] = 0

#     # Hough lines to remove spray
#     mask_lanes = mask.copy()

#     # Set completely to zero
#     mask_lanes[0:h, 0:w] = 0
#     try:
#         hough = cv2.HoughLinesP(image=mask, rho=1, theta=10 * (np.pi) / 180, threshold=50, minLineLength=50, maxLineGap=100)
#         for points in hough:
#             x0, y0, x1, y1 = points[0]
#             cv2.line(mask_lanes, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=2)
            
#         # Show spray removed lane extraction
#         cv2.imshow('Spray removed', mask_lanes)

#     except:
#         print("no lanes seen")
#         mask_lanes[0:h, 0:w] = 0
#         # cv2.destroyWindow('Spray removed')

#     stacked_img = np.stack((mask_lanes,) * 3, axis=-1)
#     pub.publish(br.cv2_to_imgmsg(stacked_img, "bgr8"))

#     cv2.imshow("image", img)
#     cv2.imshow("HSVed", mask)

#     cv2.waitKey(1)

# def stitch_images():
#     global left_image, right_image
    
#     img1 = cv2.imread('/home/aasha/karina_ws/src/Karina/igvc_perception/hsv_thresh/src/left_image.jpg')
#     img2 = cv2.imread('/home/aasha/karina_ws/src/Karina/igvc_perception/hsv_thresh/src/right_image.jpg')
#     imgs = [img1,img2]
#     # Use OpenCV's stitcher to combine images
    
#     print("Stitching Images...")
#     #imgs = [left_image,right_image]
#     stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
#     (status, stitched) = stitcher.stitch(imgs)
    
#     print("After Stitch...")
#     if status == cv2.Stitcher_OK:
#         print("STITCHED PROPERLY")
#         cv2.imshow("image", stitched)
#         cv2.waitKey(1)
#         return stitched
#     else:
#         rospy.loginfo("Image stitching failed.")
#         return None
    

# if __name__ == "__main__":

#     rospy.init_node("hsv_thresh", anonymous=False)
#     pub = rospy.Publisher("/lane_img", Image, queue_size=10)
#     cfg = Server(HSVConfig, callback)
#     rospy.Subscriber("/zed2i/zed_node/left/image_rect_color", Image, left_img_cb)
#     rospy.Subscriber("/zed2i/zed_node/right/image_rect_color", Image, right_img_cb)
#     rospy.spin()

# # THIS HAS PROVEN TO WORK

# import cv2
# import numpy as np
# img1 = cv2.imread('/home/aasha/karina_ws/src/Karina/igvc_perception/hsv_thresh/src/left_image.jpg')
# img2 = cv2.imread('/home/aasha/karina_ws/src/Karina/igvc_perception/hsv_thresh/src/right_image.jpg')

# imgs = [img1,img2]
# print("img1 size:",img1.shape)
# print("img2 size:",img2.shape)

# cv2.imshow("left",img1)
# cv2.imshow("right",img2)

# print("Creating stitcher...")
# stitcher=cv2.Stitcher.create(cv2.Stitcher_SCANS)
# #stitcher.setPanoConfidenceThresh(0.0) 
# (status,output)=stitcher.stitch(imgs) 

# print("Checking stitcher...")
# if status != cv2.STITCHER_OK: 
#   # checking if the stitching procedure is successful 
#   # .stitch() function returns a true value if stitching is  
#   # done successfully 
#     print("stitching ain't successful") 
# else:  
#     print('Your Panorama is ready!!!') 
    
  
# # final output 
# cv2.imshow('final result',output) 
  
# cv2.waitKey(0)


# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np

# class ImageStitcher:
#     def __init__(self):
#         self.bridge = CvBridge()
#         self.left_image = None
#         self.right_image = None

#         self.pub = rospy.Publisher("/lane_img", Image, queue_size=10)
#         rospy.Subscriber("/zed2i/zed_node/left/image_rect_color", Image, self.left_img_cb)
#         rospy.Subscriber("/zed2i/zed_node/right/image_rect_color", Image, self.right_img_cb)

#     def left_img_cb(self, msg):
#         self.left_image = self.bridge.imgmsg_to_cv2(msg)
#         self.try_stitching()

#     def right_img_cb(self, msg):
#         self.right_image = self.bridge.imgmsg_to_cv2(msg)
#         self.try_stitching()

#     def try_stitching(self):
#         if self.left_image is not None and self.right_image is not None:
#             stitched_image = self.stitch_images()
#             if stitched_image is not None:
#                 cv2.imshow("Stitched Image", stitched_image)
#                 cv2.waitKey(1)

#     def stitch_images(self):
#         # Ensure both images are color
#         if self.left_image.shape[2] != 3:
#             self.left_image = cv2.cvtColor(self.left_image, cv2.COLOR_GRAY2BGR)
#         if self.right_image.shape[2] != 3:
#             self.right_image = cv2.cvtColor(self.right_image, cv2.COLOR_GRAY2BGR)

#         try:
#             imgs = [self.left_image, self.right_image]
#             stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
#             (status, stitched) = stitcher.stitch(imgs)

#             if status == cv2.Stitcher_OK:
#                 print("Stitching successful.")
#                 return stitched
#             else:
#                 print("Stitching failed with status:", status)
#                 return None
#         except Exception as e:
#             print(f"Exception in stitching: {e}")
#             return None

# if __name__ == "__main__":
#     rospy.init_node("image_stitcher_node")
#     stitcher = ImageStitcher()
#     rospy.spin()


#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Initialize ROS node
rospy.init_node('image_stitcher')

# Initialize CvBridge
bridge = CvBridge()

# Global variables to hold the images from the callbacks
image1 = None
image2 = None

def stitch_images(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters and matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Minimum number of matches to consider
    MIN_MATCH_COUNT = 10
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        # Calculate Homography
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp image
        h, w, _ = img2.shape
        result = cv2.warpPerspective(img1, M, (w * 2, h))
        result[0:h, 0:w] = img2
        return result
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
        return None

def callback1(image_msg):
    global image1
    # Convert ROS image to OpenCV format
    image1 = bridge.imgmsg_to_cv2(image_msg, "bgr8")

def callback2(image_msg):
    global image2
    # Convert ROS image to OpenCV format
    image2 = bridge.imgmsg_to_cv2(image_msg, "bgr8")

# Subscribe to camera topics
sub1 = rospy.Subscriber('/camera1/image_raw', Image, callback1)
sub2 = rospy.Subscriber('/camera2/image_raw', Image, callback2)

while not rospy.is_shutdown():
    if image1 is not None and image2 is not None:
        # Perform stitching
        stitched_image = stitch_images(image1, image2)
        if stitched_image is not None:
            # Display the result
            cv2.imshow('Stitched Image', stitched_image)
            cv2.waitKey(1)

# Keep the node running
rospy.spin()


