import cv2
import numpy as np


def is_valid_match(class_name, left, top, right, bottom):
    if is_bonnet_bounding_box(left, top, right, bottom) \
            or class_name not in USEFUL_NAMES:
        return False
    return True


def is_bonnet_bounding_box(left, top, right, bottom):
    return (bottom >= 450 or top >= 350) and right - left >= 900


def compute_luma(img):
    B, G, R = cv2.split(img)
    M = np.maximum(B, np.maximum(G, R))
    return M


def preprocess_for_object_recognition(imgL):
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2YCrCb, dst=imgL)
    imgL[:, :, 0] = tiled_histogram_eq(imgL[:, :, 0])
    cv2.cvtColor(imgL, cv2.COLOR_YCrCb2BGR, dst=imgL)
    return imgL


def annotate_image(tags, img):
    # Helper function to annotate the image with bounding
    # boxes and distance information

    for depth, class_name, confidence, left, top, right, bottom in tags:
        # construct label
        label = '%s (%.2fm)' % (class_name, depth)

        # draw a bounding box around matched section
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        # display the label at the top of the bounding box
        labelsize, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, thickness=1)
        top = max(top, labelsize[1])
        cv2.rectangle(
            img,
            (left, top - round(1.5 * labelsize[1]) - 2),
            (left + round(1.5 * labelsize[0]), top + baseline - 2),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(img, label, (left, top - 2), cv2.FONT_HERSHEY_DUPLEX, fontScale=0.75, color=(0,0,0), thickness=1)


USEFUL_NAMES = {
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'parking meter',
    'skateboard',
}


def mode(array):
    (_, idx, counts) = np.unique(array, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]
    mode = array[index]
    return mode


def tiled_histogram_eq(gray_img, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe.apply(gray_img, gray_img)
    return gray_img
