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
    # Perform ratio test
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            yield l_keypoints[m.queryIdx], r_keypoints[m.trainIdx]
