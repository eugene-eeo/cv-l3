import cv2
import numpy as np


# Taken from https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    --------
        matched: np.ndarray
            The transformed output image
    """

    shape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float32)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float32)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(shape)


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


def kmeans(points, priors, maxiter=20):
    # Perform k-means on a 1D-array of points
    # Where number of centroids is k, and priors is a list of
    # centroid guesses.

    k = priors.shape[0]
    centroids = np.array(priors, dtype=np.float32).reshape(-1, 1)
    classes = np.zeros(points.shape[0], dtype=np.uint8)
    distances = np.zeros((k, points.shape[0]), dtype=np.float32)

    # distances is used as follows:
    # D = [ p1 p2 p3 ... pn  | for centroid 1
    #       p1 p2 p3 ... pn  | for centroid 2

    for _ in range(maxiter):
        # Distance function = (point - centroid)^2
        distances[:] = points
        distances -= centroids
        distances **= 2
        classes = np.argmin(distances, axis=0)
        for c in range(k):
            centroids[c] = np.mean(points[classes == c])
    return classes, centroids.ravel()


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
