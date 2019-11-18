import cv2


# ORB Feature matcher
FLANN_INDEX_LSH = 6
index_params = {"algorithm": FLANN_INDEX_LSH,
                "table_number": 6,
                "key_size": 12,
                "multi_probe_level": 1}


def find_keypoints_and_descriptors(grayL, grayR):
    feature_object = cv2.ORB_create(5000)
    l_keypoints, l_descriptors = feature_object.detectAndCompute(grayL, None)
    r_keypoints, r_descriptors = feature_object.detectAndCompute(grayR, None)
    return l_keypoints, l_descriptors, r_keypoints, r_descriptors


def match(l_keypoints, l_descriptors, r_keypoints, r_descriptors):
    matcher = cv2.FlannBasedMatcher(index_params, {"checks": 50})
    matches = matcher.knnMatch(l_descriptors, trainDescriptors=r_descriptors, k=2)
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            yield l_keypoints[m.queryIdx], r_keypoints[m.trainIdx]


# imgL = cv2.imread('/home/eeojun/Downloads/TTBB-durham-02-10-17-sub10/left-images/1506942473.484027_L.png')
# imgR = cv2.imread('/home/eeojun/Downloads/TTBB-durham-02-10-17-sub10/right-images/1506942473.484027_R.png')
# grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
# grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
# for lk, rk in match(imgL, imgR):
#     lx, ly = lk.pt
#     rx, ry = rk.pt
#     imgL[int(ly), int(lx)] = (0, 0, 255)
#     imgR[int(ry), int(rx)] = (0, 0, 255)

# cv2.imshow("imgL", imgL)
# cv2.imshow("imgR", imgR)
# cv2.waitKey()
