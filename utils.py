import cv2
import numpy as np


def equalise_hist(img):
    # img must be a HSV image
    v = img[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    img[:, :, 2] = v
    return img


def sharpen(img, dst=None):
    # Convolves an image with the edge-sharpening filter.
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel, dst=dst)


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


def tiled_histogram_eq(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_img)
