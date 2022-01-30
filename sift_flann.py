
import cv2
from cv2 import resize
import numpy as np
from matplotlib import pyplot as plt
from traceback2 import print_tb
import dlib

from utils import rotate_bbox, rotate_bound, warpImg, findFaces, is_two_image_same
from utils import resizeImage, get_warpPerspective, get_angle_and_box_coord


def siftMatching(img1, img2):
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    print("Total good matches:", len(good))       
    good = good[:20]
    return kp1, kp2, good


def main():
    
    template = cv2.imread("test/test3.jpg")
    sample = cv2.imread("train/tc_ID_rot.jpg")

    MIN_MATCH_COUNT = 20

    img1 = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)         # trainImage
    img2 = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)           # queryImage
    img1 = resizeImage(img1)

    kp1, kp2, good = siftMatching(img1, img2)

    if len(good) >= MIN_MATCH_COUNT:

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        M2, mask2 = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
        h,w,_ = img1.shape
    
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        border = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 3)

        warped_image = get_warpPerspective(img2, M2, dst)
       
        (heigth_q, width_q) = img2.shape[:2]
        (cx, cy) = (width_q // 2, heigth_q // 2)

        angle, box = get_angle_and_box_coord(dst)
        
        rotated_img = rotate_bound(img2, angle)
        
        new_bbox = rotate_bbox(box, cx, cy, heigth_q, width_q, angle)
       
        warp_image = warpImg(rotated_img, new_bbox ,  heigth_q, width_q)
        
        face_crop_img_query = findFaces(warped_image)
        face_crop_img_target = findFaces(img1)
        
        if(img1 is not None):
            plt.title("rotated_image")
            plt.imshow(rotated_img)
            plt.show()
        
        if(warped_image  is not None):
            plt.title("homography transformed image")
            plt.imshow(warped_image)
            plt.show()
        
        if(face_crop_img_query is not None):
            plt.title("face_crop")
            plt.imshow(face_crop_img_query)
            plt.show()
        
        if(face_crop_img_target is not None and face_crop_img_query is not None):
            plt.title("face_crop_target")
            plt.imshow(face_crop_img_target)
            plt.show()
            is_two_image_same(face_crop_img_target, face_crop_img_query, 15)
        

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good, None,flags=2)

    fig = plt.figure(figsize=(16, 12))
    plt.title("Matched image")
    plt.imshow(img3, 'gray'),plt.show()

  

if __name__ == '__main__':
    main()