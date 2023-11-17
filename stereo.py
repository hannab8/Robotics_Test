import cv2
import numpy as np

def load_and_split_stereo_image(image_path):
    # Load the stereo image
    stereo_img = cv2.imread(image_path)

    # Split the image into left and right
    height, width, _ = stereo_img.shape
    left_img = stereo_img[:, :width // 2]
    right_img = stereo_img[:, width // 2:]

    return left_img, right_img

def detect_and_match_features(left_img, right_img):
    # Convert images to grayscale
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    keypoints_left, descriptors_left = orb.detectAndCompute(gray_left, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(gray_right, None)

    # Match features using BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_left, descriptors_right)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    points_left = np.zeros((len(matches), 2), dtype=np.float32)
    points_right = np.zeros_like(points_left)

    for i, match in enumerate(matches):
        points_left[i, :] = keypoints_left[match.queryIdx].pt
        points_right[i, :] = keypoints_right[match.trainIdx].pt

    return points_left, points_right

def stitch_images(left_img, right_img, points_left, points_right):
    # Find the homography matrix
    H, _ = cv2.findHomography(points_right, points_left, method=cv2.RANSAC)

    # Warp the right image to align with the left image
    height, width, _ = left_img.shape
    warped_img = cv2.warpPerspective(right_img, H, (width, height))

    # Stitch the images together
    result = np.zeros((height, 2 * width, 3), dtype=np.uint8)
    result[0:height, 0:width] = left_img
    result[0:height, width:2*width] = warped_img

    return result

if __name__ == "__main__":
    image_path = '/Users/hannabulinda/Desktop/Stereo_Image.png'  

    left_img, right_img = load_and_split_stereo_image(image_path)
    points_left, points_right = detect_and_match_features(left_img, right_img)
    panoramic_img = stitch_images(left_img, right_img, points_left, points_right)

    cv2.imshow('Panoramic Image', panoramic_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
