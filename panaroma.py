"""
Image Stitching Problem

The goal of this problem is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.

For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching. 
"""
import cv2
import numpy as np
import random

def sift_Creator(image):
    sift = cv2.xfeatures2d.SIFT_create()
    ksp, features = sift.detectAndCompute(image,None)
    return ksp, features

def knn(feature_1, feature_2, size):
    match = cv2.BFMatcher()
    matches = match.knnMatch(feature_1,feature_2,k=size)
    return matches

def matching(matches):
    matched = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            matched.append(m)
    return matched

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    img1 = cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY)
    
    # find key points
    ksp_right, right_features = sift_Creator(img1)
    ksp_left, left_features = sift_Creator(img2)

    matches = knn(right_features,left_features,2)

    matched = matching(matches)
            
    match_condition = 10
    
    if len(matched) > match_condition:
        right_points = np.float32([ ksp_right[m.queryIdx].pt for m in matched ]).reshape(-1,1,2)
        left_points = np.float32([ ksp_left[m.trainIdx].pt for m in matched ]).reshape(-1,1,2)
        homography_Matrix, mask = cv2.findHomography(right_points, left_points, cv2.RANSAC, 5.0)
        height_img1,width_img1 = img1.shape
        points_new = np.float32([ [0,0],[0,height_img1-1],[width_img1-1,height_img1-1],[width_img1-1,0] ]).reshape(-1,1,2)
        stiched = cv2.perspectiveTransform(points_new, homography_Matrix)
    else:
        print("Not enought matches are found - %d/%d", (len(matched)/match_condition))

    stiched = cv2.warpPerspective(right_img,homography_Matrix,(left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
    stiched[0:left_img.shape[0],0:left_img.shape[1]] = left_img

    return stiched


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('/panaroma.jpg',result_image)


